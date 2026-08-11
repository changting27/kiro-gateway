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
Pydantic models for the OpenAI-compatible ``/v1/responses`` endpoint.

The Responses API is OpenAI's successor to Chat Completions and is the only wire
protocol supported by recent Codex CLI / IDE builds (``wire_api = "chat"`` was
removed in February 2026). This module defines the request schema; the response
objects and streaming events are assembled as plain dicts in
``streaming_responses.py`` (mirroring how ``streaming_openai.py`` builds chat
completion dicts) because their shape is highly polymorphic and driven by the
Kiro stream contents.

Key request-shape differences from Chat Completions:

* ``input`` replaces ``messages`` and may be a bare string or a list of typed
  *input items* (``message``, ``function_call``, ``function_call_output``,
  ``reasoning``, ...).
* ``instructions`` carries the system prompt (there is no ``system`` message).
* ``tools`` use the **flat** function shape ``{"type": "function", "name": ...,
  "parameters": ...}`` rather than the nested ``{"type": "function",
  "function": {...}}`` used by Chat Completions.
* ``reasoning`` is an object ``{"effort": ..., "summary": ...}`` rather than the
  scalar ``reasoning_effort``.
* ``max_output_tokens`` replaces ``max_tokens``.

All models use ``extra="allow"`` so unknown/less-common fields Codex sends
(``store``, ``include``, ``prompt_cache_key``, ``previous_response_id``,
``parallel_tool_calls``, ...) are accepted and preserved without validation
failures, honouring the gateway's pass-through philosophy.
"""

from typing import Any, Dict, List, Optional, Union

from typing_extensions import Annotated
from pydantic import BaseModel, Field


# ==================================================================================================
# Sub-models
# ==================================================================================================

class ResponsesReasoning(BaseModel):
    """
    Reasoning configuration for the Responses API.

    Attributes:
        effort: Reasoning effort level. Official values are ``minimal``, ``low``,
            ``medium``, ``high``; unknown values are tolerated and mapped by the
            converter. ``None`` means "not specified" (gateway default applies).
        summary: Requested reasoning-summary verbosity (``auto``/``concise``/
            ``detailed``). Accepted for compatibility; the gateway always streams
            the reasoning it derives from Kiro's thinking output.
    """
    effort: Optional[str] = None
    summary: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponsesFunctionTool(BaseModel):
    """
    Nested function description (Chat-Completions-style) accepted for robustness.

    The Responses API sends tools in a flat shape, but some clients reuse the
    nested Chat Completions shape. Both are supported by ``ResponsesTool``.

    Attributes:
        name: Function name.
        description: Human-readable description.
        parameters: JSON Schema for the function arguments.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ResponsesTool(BaseModel):
    """
    Tool definition for the Responses API.

    Supports two shapes:

    1. Flat (canonical Responses):
       ``{"type": "function", "name": "...", "description": "...", "parameters": {...}}``
    2. Nested (Chat Completions style, tolerated):
       ``{"type": "function", "function": {"name": "...", ...}}``

    Non-function tool types (e.g. ``web_search``, ``file_search``, ``mcp``) are
    represented too but are ignored by the converter, matching the Chat
    Completions adapter which only forwards function tools to Kiro.

    Attributes:
        type: Tool type. Defaults to ``function``.
        name: Function name (flat shape).
        description: Function description (flat shape).
        parameters: JSON Schema of arguments (flat shape).
        function: Nested function description (Chat Completions shape).
    """
    type: str = "function"
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    function: Optional[ResponsesFunctionTool] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Request model
# ==================================================================================================

class ResponsesRequest(BaseModel):
    """
    Request body for ``POST /v1/responses``.

    Only ``model`` and ``input`` are strictly required. ``input`` may be a plain
    string (shorthand for a single user message) or a list of typed input items.
    Every other field is optional and, where not explicitly modelled, accepted
    via ``extra="allow"`` so the full Codex field set passes through untouched.

    Attributes:
        model: Model ID for generation.
        input: Either a user-message string or a list of Responses input items.
        instructions: System prompt (Responses API has no ``system`` message).
        stream: Whether to stream Server-Sent Events (default False).
        tools: Available tools (flat or nested function shape).
        tool_choice: Tool selection strategy.
        reasoning: Reasoning configuration object.
        max_output_tokens: Maximum output tokens (Responses name for max_tokens).
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        parallel_tool_calls: Whether parallel tool calls are permitted.
        previous_response_id: Prior response id for server-side threading
            (accepted but not persisted — the gateway is stateless).
        store: Whether OpenAI would persist the response (accepted, ignored).
        metadata: Arbitrary client metadata (accepted, ignored).
    """
    model: str
    input: Union[str, List[Any]]

    instructions: Optional[str] = None
    stream: bool = False

    # Tools (function calling)
    tools: Optional[List[ResponsesTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Reasoning
    reasoning: Optional[ResponsesReasoning] = None

    # Generation parameters
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None

    # Stateful / bookkeeping fields (accepted for compatibility, not persisted)
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}
