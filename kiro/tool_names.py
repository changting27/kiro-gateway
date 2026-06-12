# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Transparent tool-name normalization for the Kiro API.

Kiro (AWS Bedrock Converse ``ToolSpecification.name``) rejects tool names that
exceed **64 characters** or contain characters outside ``[a-zA-Z0-9_-]``. The
length limit was verified empirically against the live API: a 64-character name
is accepted, a 65-character one returns ``{"message": "Improperly formed
request."}``. MCP / plugin clients routinely produce names that violate this, for
example Claude Code's plugin scheme::

    mcp__plugin_chrome-devtools-mcp_chrome-devtools__performance_analyze_insight  (76 chars)

Rather than rejecting such requests (gatekeeper behaviour), the gateway
transparently normalizes offending names to a deterministic, collision-resistant
alias before sending them to Kiro, and restores the original names in responses so
the client never sees the alias.

Design
------
* **Forward** (:func:`normalize_tool_name`) is a *pure, deterministic* function
  with no shared state. The same input always yields the same output, so it can be
  applied independently to tool *definitions* and to ``tool_use`` names embedded in
  conversation *history* and they stay mutually consistent without any map.
* **Reverse** (:func:`build_tool_name_restore_map` + :func:`restore_tool_name`)
  rebuilds a ``{normalized: original}`` map *per request* from the request's own
  tool list. No process-wide/global state is used, so concurrent requests never
  interfere and there is no unbounded memory growth.

Names that are already valid pass through unchanged, so well-behaved tools incur
zero behavioural change and zero restore overhead.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

from loguru import logger

# --- Kiro / AWS Bedrock Converse ToolSpecification.name constraints ---
# Maximum tool-name length accepted by Kiro (verified: 64 ok, 65 rejected).
MAX_TOOL_NAME_LENGTH: int = 64
# Character set Kiro accepts in a tool name. Claude models allow '-' and '_';
# names already within this set and length are passed through untouched.
_VALID_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,%d}$" % MAX_TOOL_NAME_LENGTH)
_INVALID_CHAR_RE = re.compile(r"[^A-Za-z0-9_-]")
# Length (in hex chars) of the deterministic hash suffix. 8 hex = 32 bits, which
# makes accidental collisions between distinct original names negligible.
_HASH_LEN: int = 8
# Room left for the readable prefix once "_" + hash is reserved (64 - 1 - 8 = 55).
_MAX_PREFIX_LEN: int = MAX_TOOL_NAME_LENGTH - 1 - _HASH_LEN


def is_valid_tool_name(name: str) -> bool:
    """
    Report whether a tool name already satisfies Kiro's length and charset rules.

    Args:
        name: Tool name to check.

    Returns:
        True if ``name`` is non-empty, at most ``MAX_TOOL_NAME_LENGTH`` characters,
        and contains only ``[A-Za-z0-9_-]``; otherwise False.
    """
    return bool(name) and bool(_VALID_TOOL_NAME_RE.match(name))


def normalize_tool_name(name: str) -> str:
    """
    Map a tool name to a Kiro-safe alias, deterministically.

    Already-valid names (see :func:`is_valid_tool_name`) are returned unchanged.
    Otherwise the name is charset-sanitized (invalid characters replaced with
    ``_``), truncated, and suffixed with an 8-character hash derived from the
    **full original** name. Using the full original for the hash means two distinct
    originals that share a truncated prefix still receive different aliases.

    The result is guaranteed to be at most ``MAX_TOOL_NAME_LENGTH`` characters and
    to contain only ``[A-Za-z0-9_-]``.

    This function is pure and deterministic: it holds no state and the same input
    always produces the same output. Empty input is returned unchanged so callers
    can handle missing names explicitly.

    Args:
        name: Original tool name (possibly too long or with invalid characters).

    Returns:
        A Kiro-safe tool name. Equal to ``name`` when no change is required.

    Examples:
        >>> normalize_tool_name("get_weather")
        'get_weather'
        >>> normalized = normalize_tool_name("a" * 70)
        >>> len(normalized)
        64
        >>> normalize_tool_name("a" * 70) == normalize_tool_name("a" * 70)
        True
    """
    if not name:
        return name
    if is_valid_tool_name(name):
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_LEN]
    sanitized = _INVALID_CHAR_RE.sub("_", name)
    prefix = sanitized[:_MAX_PREFIX_LEN]
    return f"{prefix}_{digest}"


def _iter_request_tool_names(request_tools: Optional[List[Any]]) -> List[str]:
    """
    Extract tool names from a heterogeneous ``request_tools`` collection.

    Tolerates the several shapes the original tool list can take across the code
    base: OpenAI nested ``{"type": "function", "function": {"name": ...}}``, flat /
    Anthropic ``{"name": ...}`` dicts, and the equivalent Pydantic-model objects
    (or their ``model_dump`` dicts).

    Args:
        request_tools: The request's original tool list, or None.

    Returns:
        List of non-empty tool-name strings, in order. Unrecognised entries are
        skipped.
    """
    names: List[str] = []
    for tool in request_tools or []:
        name: Optional[str] = None
        if isinstance(tool, dict):
            name = tool.get("name")
            if not name:
                fn = tool.get("function")
                if isinstance(fn, dict):
                    name = fn.get("name")
        else:
            name = getattr(tool, "name", None)
            if not name:
                fn = getattr(tool, "function", None)
                if isinstance(fn, dict):
                    name = fn.get("name")
                elif fn is not None:
                    name = getattr(fn, "name", None)
        if name:
            names.append(name)
    return names


def build_tool_name_restore_map(request_tools: Optional[List[Any]]) -> Dict[str, str]:
    """
    Build a ``{normalized_name: original_name}`` map for restoring response names.

    Only names that actually change under :func:`normalize_tool_name` are included,
    so a request whose tools are all valid yields an empty map and adds no restore
    overhead. The map is request-scoped — callers build one per request from that
    request's own tool list, so there is no global state and no cross-request
    interference.

    If two distinct originals normalize to the same alias (astronomically unlikely
    given the full-name hash), the first wins and a warning is logged.

    Args:
        request_tools: The request's original tool list, or None.

    Returns:
        Mapping from normalized alias to original tool name (possibly empty).
    """
    restore: Dict[str, str] = {}
    for original in _iter_request_tool_names(request_tools):
        normalized = normalize_tool_name(original)
        if normalized == original:
            continue
        existing = restore.get(normalized)
        if existing is not None and existing != original:
            logger.warning(
                f"Tool-name normalization collision: '{existing}' and '{original}' "
                f"both map to '{normalized}'; keeping the first. Responses for the "
                f"second tool could surface the first tool's name."
            )
            continue
        restore[normalized] = original
        logger.debug(f"Tool name normalized for Kiro: '{original}' -> '{normalized}'")
    return restore


def restore_tool_name(name: str, restore_map: Optional[Dict[str, str]]) -> str:
    """
    Restore a (possibly normalized) tool name to its original value.

    Args:
        name: Tool name as returned by Kiro (may be a normalized alias).
        restore_map: Map produced by :func:`build_tool_name_restore_map`, or None.

    Returns:
        The original tool name when ``name`` is a known alias; otherwise ``name``
        unchanged. Safe to call with an empty/None map (acts as a no-op).
    """
    if not restore_map:
        return name
    return restore_map.get(name, name)
