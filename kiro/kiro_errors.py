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
Kiro API error enhancement and user-friendly message formatting.

This module provides a centralized system for enhancing cryptic Kiro API errors
with clear, actionable, user-friendly messages.

Architecture:
- KiroErrorReason: Enum of known error reasons from Kiro API
- KiroErrorInfo: Structured information about an enhanced error
- enhance_kiro_error(): Analyzes error JSON and returns enhanced message
- parse_upstream_error(): Single entry point for any non-200 upstream response,
  including errors produced by network intermediaries instead of Kiro
- resolve_client_error_status(): Client-facing status for a parsed error

Example:
    >>> error_json = {"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
    >>> error_info = enhance_kiro_error(error_json)
    >>> print(error_info.user_message)
    "Model context limit reached. Conversation size exceeds model capacity."
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from loguru import logger

from kiro.config import DEFAULT_MAX_INPUT_TOKENS

# Synthetic reason assigned to errors produced by a network intermediary instead
# of the Kiro service. Account classification treats it as FATAL: every account
# shares the same network path, so failover cannot help, and the account itself
# is healthy, so it must not be penalized.
UPSTREAM_INTERMEDIARY_REASON: str = "UPSTREAM_INTERMEDIARY"

# Header that AWS JSON-protocol services attach to every response they generate.
# Its absence on an error is the primary signal that AWS never saw the request.
AWS_REQUEST_ID_HEADER: str = "x-amzn-requestid"

# Status codes whose client-side semantics stay correct even when an intermediary
# produced them: 429 keeps rate-limit backoff, 5xx keeps "server-side, retry".
# Every other intermediary status (404, 400, 401, 403, ...) would be misread as a
# statement about the model, request, or credentials and is remapped to 502.
_INTERMEDIARY_PRESERVED_STATUS_CODES: frozenset = frozenset({429})
INTERMEDIARY_CLIENT_STATUS_CODE: int = 502

# Number of characters of the intermediary body kept for debug logging only.
_INTERMEDIARY_BODY_SNIPPET_LENGTH: int = 200


@dataclass
class KiroErrorInfo:
    """
    Structured information about a Kiro API error.
    
    Contains both the enhanced user-friendly message and the original
    error details for logging and debugging.
    
    Attributes:
        reason: Error reason code from Kiro API (as string, e.g. "CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        user_message: Enhanced, user-friendly message for end users
        original_message: Original message from Kiro API (for logging)
        is_context_overflow: True when the error means the conversation exceeded the
            model's context window. Downstream clients (Claude Code, OpenAI SDKs)
            only auto-compact when they receive the canonical overflow error shape,
            so this flag drives the response builders below.
        is_upstream_intermediary: True when the error response was produced by a
            network intermediary (reverse proxy, firewall, captive portal) rather
            than by the Kiro/AWS service. The original status code is then not a
            statement about the request or model, so builders remap it.
    """
    reason: str
    user_message: str
    original_message: str
    is_context_overflow: bool = False
    is_upstream_intermediary: bool = False


def enhance_kiro_error(error_json: Dict[str, Any]) -> KiroErrorInfo:
    """
    Enhances Kiro API error with user-friendly message.
    
    Takes raw error JSON from Kiro API and returns structured information
    with enhanced, user-friendly messages that help users understand what
    went wrong without technical jargon.
    
    Args:
        error_json: Parsed JSON from Kiro API error response
                   Expected format: {"message": "...", "reason": "..."}
                   The "reason" field is optional.
    
    Returns:
        KiroErrorInfo with enhanced message and original details
    
    Example:
        >>> error_json = {"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
        >>> error_info = enhance_kiro_error(error_json)
        >>> print(error_info.user_message)
        "Model context limit reached. Conversation size exceeds model capacity."
        >>> print(error_info.original_message)
        "Input is too long."
    
    Example (unknown error):
        >>> error_json = {"message": "Something went wrong.", "reason": "UNKNOWN_REASON"}
        >>> error_info = enhance_kiro_error(error_json)
        >>> print(error_info.user_message)
        "Something went wrong. (reason: UNKNOWN_REASON)"
    """
    # Extract original message and reason from Kiro API response
    # Handle None values explicitly (preserve empty strings)
    original_message = error_json.get("message")
    if original_message is None:
        original_message = "Unknown error"
    
    reason = error_json.get("reason")
    if reason is None:
        reason = "UNKNOWN"
    
    # Map known reasons to user-friendly messages
    if reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD":
        # Context limit exceeded - conversation is too long
        user_message = "Model context limit reached. Conversation size exceeds model capacity."
    
    elif reason == "MONTHLY_REQUEST_COUNT":
        # Monthly request limit exceeded - account quota exhausted
        user_message = "Monthly request limit exceeded. Account has reached its monthly quota."
    
    elif reason == "INVALID_MODEL_ID":
        # Invalid model name or subscription tier insufficient
        user_message = "Invalid model ID or insufficient subscription level to use it."

    elif original_message == "Improperly formed request." and reason in (None, "UNKNOWN", "null"):
        # Generic 400 error
        user_message = (
            "Kiro API rejected the request. If problem persists, open issue with info and attached debug logs at:"
            "https://github.com/jwadow/kiro-gateway/issues"
        )

    # Future error enhancements can be added here:
    # elif reason == "RATE_LIMIT_EXCEEDED":
    #     user_message = "Rate limit exceeded. Too many requests in a short time."
    # elif reason == "INVALID_MODEL":
    #     user_message = "Invalid model specified. The requested model is not available."
    
    else:
        # Unknown error or no enhancement available
        # Keep original message and append reason if present
        if "reason" in error_json and reason != "UNKNOWN":
            user_message = f"{original_message} (reason: {reason})"
        else:
            user_message = original_message
    
    # Detect context-window overflow. Kiro signals this primarily via the reason
    # code, but older/edge responses only carry it in the free-text message, so we
    # add a conservative text fallback (only when the reason is unknown) to avoid
    # false positives on unrelated errors.
    is_context_overflow = reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
    if not is_context_overflow and reason == "UNKNOWN":
        lowered = original_message.lower()
        if (
            "content length exceeds" in lowered
            or "exceeds threshold" in lowered
            or "input is too long" in lowered
            or "prompt is too long" in lowered
        ):
            is_context_overflow = True

    return KiroErrorInfo(
        reason=reason,
        user_message=user_message,
        original_message=original_message,
        is_context_overflow=is_context_overflow,
    )


def _get_header(headers: Optional[Mapping[str, str]], name: str) -> Optional[str]:
    """
    Look up a header case-insensitively in any mapping.

    ``httpx.Headers`` is already case-insensitive, but plain dicts (tests, other
    transports) are not, so fall back to a manual scan.

    Args:
        headers: Response headers, or None.
        name: Lower-case header name to look up.

    Returns:
        The header value, or None if absent or if ``headers`` is not a mapping.
    """
    if not isinstance(headers, Mapping) or not headers:
        return None
    value = headers.get(name)
    if isinstance(value, str):
        return value
    for key, candidate in headers.items():
        if isinstance(key, str) and key.lower() == name and isinstance(candidate, str):
            return candidate
    return None


def is_upstream_intermediary_response(
    headers: Optional[Mapping[str, str]],
    error_text: str,
) -> bool:
    """
    Decide whether an error response came from a network intermediary, not AWS.

    The rule is deliberately conservative and requires both signals:

    1. The AWS request-id header (``x-amzn-requestid``) is absent, so AWS never
       processed the request.
    2. The body is markup: ``Content-Type`` contains ``text/html`` or the body
       starts with ``<``. Kiro's JSON-protocol service never emits markup errors.

    Plain-text or JSON bodies without the header are left untouched, so existing
    error handling (including the text-based context-overflow fallback) keeps
    working.

    Args:
        headers: Upstream response headers, or None if unavailable.
        error_text: Decoded upstream response body.

    Returns:
        True if the response should be treated as an intermediary error.
    """
    if _get_header(headers, AWS_REQUEST_ID_HEADER):
        return False
    content_type = (_get_header(headers, "content-type") or "").lower()
    if "text/html" in content_type:
        return True
    return error_text.lstrip().startswith("<")


def parse_upstream_error(
    status_code: int,
    headers: Optional[Mapping[str, str]],
    error_text: str,
) -> KiroErrorInfo:
    """
    Parse any non-200 upstream response into a :class:`KiroErrorInfo`.

    This is the single entry point every API surface uses for upstream errors.
    Kiro JSON errors (and non-JSON text, so the context-overflow text fallback can
    still fire) go through :func:`enhance_kiro_error`. Markup responses from a
    network intermediary become an actionable gateway error instead of leaking
    raw HTML to the client and letting its status code (for example 404) be
    misread as "model does not exist".

    Args:
        status_code: Upstream HTTP status code.
        headers: Upstream response headers, or None if unavailable.
        error_text: Decoded upstream response body.

    Returns:
        Structured error information for logging and client error shaping.
    """
    try:
        error_json = json.loads(error_text)
    except json.JSONDecodeError:
        error_json = None

    if isinstance(error_json, dict):
        error_info = enhance_kiro_error(error_json)
    else:
        error_info = enhance_kiro_error({"message": error_text})

    if error_info.is_context_overflow or not is_upstream_intermediary_response(
        headers, error_text
    ):
        return error_info

    server = _get_header(headers, "server") or "unknown"
    logger.warning(
        f"Upstream HTTP {status_code} was produced by a network intermediary "
        f"(no {AWS_REQUEST_ID_HEADER} header, markup body, server: {server}); "
        f"Kiro never received the request"
    )
    snippet = " ".join(error_text.split())[:_INTERMEDIARY_BODY_SNIPPET_LENGTH]
    user_message = (
        f"The request was rejected by a network intermediary (proxy, firewall, or "
        f"gateway) before reaching Kiro (upstream HTTP {status_code}, non-AWS HTML "
        f"response). The model and request were not evaluated. Check the proxy and "
        f"DNS configuration for the Kiro runtime host; if your network only allows "
        f"the legacy path, set KIRO_GENERATE_ASSISTANT_RESPONSE_PATH."
    )
    return KiroErrorInfo(
        reason=UPSTREAM_INTERMEDIARY_REASON,
        user_message=user_message,
        original_message=snippet,
        is_upstream_intermediary=True,
    )


def resolve_client_error_status(error_info: KiroErrorInfo, status_code: int) -> int:
    """
    Choose the HTTP status returned to the client for a non-overflow error.

    Kiro errors keep their original status. Intermediary errors keep 429 and 5xx
    (their retry semantics are still right) and map everything else to 502 Bad
    Gateway, because a proxy's 404/400/401/403 says nothing about the model,
    request, or credentials and clients misreport it (e.g. "model may not exist").

    Args:
        error_info: Parsed upstream error.
        status_code: Original upstream status code.

    Returns:
        The status code to send to the client.
    """
    if not error_info.is_upstream_intermediary:
        return status_code
    if status_code >= 500 or status_code in _INTERMEDIARY_PRESERVED_STATUS_CODES:
        return status_code
    return INTERMEDIARY_CLIENT_STATUS_CODE


def format_context_overflow_message(
    input_tokens: int,
    max_tokens: int,
    context_limit: int,
) -> str:
    """
    Return the canonical Anthropic context-overflow message.

    Claude Code (and OpenAI-compatible clients) detect this exact phrasing to
    trigger automatic conversation compaction and retry. Reproducing Anthropic's
    native wording verbatim is what makes downstream auto-compaction fire; a
    custom message such as "Model context limit reached" is silently ignored by
    the client, which is why the conversation dead-ends instead of compacting.

    Args:
        input_tokens: Estimated prompt/input token count.
        max_tokens: Requested completion budget (max_tokens).
        context_limit: The model's real input-token limit.

    Returns:
        A message string of the exact form Anthropic's API emits on overflow.

    Examples:
        >>> format_context_overflow_message(210000, 8192, 200000)
        'input length and max_tokens exceed context limit: 210000 + 8192 > 200000, decrease input length or max_tokens and try again'
    """
    return (
        f"input length and max_tokens exceed context limit: "
        f"{input_tokens} + {max_tokens} > {context_limit}, "
        f"decrease input length or max_tokens and try again"
    )


def _resolve_overflow_input_tokens(
    estimated_input_tokens: Optional[int],
    max_tokens: int,
    context_limit: int,
) -> int:
    """
    Clamp the input-token count so ``input + max_tokens > context_limit`` holds.

    Kiro already rejected the request for exceeding the window, so the true input
    size is at least the limit. When the local estimate is missing or under-counts
    (tokenizers differ from Kiro's accounting), we raise it to the smallest value
    that keeps the reported inequality self-consistent, so the client treats it as
    a genuine overflow rather than recomputing and dismissing it.

    Args:
        estimated_input_tokens: Best-effort local estimate, or None.
        max_tokens: Requested completion budget.
        context_limit: The model's real input-token limit.

    Returns:
        A positive input-token count that satisfies input + max_tokens > limit.
    """
    input_tokens = max(estimated_input_tokens or 0, 1)
    if input_tokens + max_tokens <= context_limit:
        input_tokens = max(1, context_limit - max_tokens + 1)
    return input_tokens


def build_anthropic_error_body(
    error_info: KiroErrorInfo,
    status_code: int,
    *,
    context_limit: int = DEFAULT_MAX_INPUT_TOKENS,
    max_tokens: int = 0,
    estimated_input_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Render a Kiro error as an Anthropic-format error body.

    Context-overflow errors are re-expressed as Anthropic's canonical
    ``invalid_request_error`` (HTTP 400) so Claude Code recognises the overflow and
    triggers auto-compaction + retry. All other errors keep the existing
    ``api_error`` envelope; their status comes from
    :func:`resolve_client_error_status` (original status for Kiro errors).

    Args:
        error_info: Result of :func:`enhance_kiro_error`.
        status_code: Original upstream status code (used for non-overflow errors).
        context_limit: Model input-token limit for the overflow message.
        max_tokens: Requested completion budget for the overflow message.
        estimated_input_tokens: Optional local input-token estimate.

    Returns:
        Tuple of (http_status_code, response_body_dict).
    """
    if error_info.is_context_overflow:
        input_tokens = _resolve_overflow_input_tokens(
            estimated_input_tokens, max_tokens, context_limit
        )
        message = format_context_overflow_message(
            input_tokens, max_tokens, context_limit
        )
        return 400, {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
    return resolve_client_error_status(error_info, status_code), {
        "type": "error",
        "error": {"type": "api_error", "message": error_info.user_message},
    }


def build_openai_error_body(
    error_info: KiroErrorInfo,
    status_code: int,
    *,
    context_limit: int = DEFAULT_MAX_INPUT_TOKENS,
    max_tokens: int = 0,
    estimated_input_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Render a Kiro error as an OpenAI-format error body.

    Context-overflow errors are re-expressed as OpenAI's canonical
    ``context_length_exceeded`` / ``invalid_request_error`` (HTTP 400) so
    OpenAI-compatible clients trigger their own history trimming. All other errors
    keep the existing ``kiro_api_error`` envelope; their status comes from
    :func:`resolve_client_error_status` (original status for Kiro errors).

    Args:
        error_info: Result of :func:`enhance_kiro_error`.
        status_code: Original upstream status code (used for non-overflow errors).
        context_limit: Model input-token limit for the overflow message.
        max_tokens: Requested completion budget for the overflow message.
        estimated_input_tokens: Optional local input-token estimate.

    Returns:
        Tuple of (http_status_code, response_body_dict).
    """
    if error_info.is_context_overflow:
        input_tokens = _resolve_overflow_input_tokens(
            estimated_input_tokens, max_tokens, context_limit
        )
        message = format_context_overflow_message(
            input_tokens, max_tokens, context_limit
        )
        return 400, {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "messages",
            }
        }
    client_status = resolve_client_error_status(error_info, status_code)
    return client_status, {
        "error": {
            "message": error_info.user_message,
            "type": "kiro_api_error",
            "code": client_status,
        }
    }
