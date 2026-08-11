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
Streaming and collection logic for the OpenAI Responses API.

Converts the unified Kiro event stream (see ``streaming_core.parse_kiro_stream``)
into the Responses server-sent-event protocol, and collects a full non-streaming
``response`` object. This is the Responses-API counterpart of
``streaming_openai.py``.

Event model
-----------
The Responses stream is a sequence of *typed* SSE events (each carrying a
monotonic ``sequence_number``) that describe a growing list of *output items*.
This module emits, in order:

1. ``response.created`` + ``response.in_progress`` (once, on the first Kiro event).
2. A ``reasoning`` output item (if the model produced ``<thinking>`` content):
   ``output_item.added`` → ``reasoning_summary_part.added`` →
   ``reasoning_summary_text.delta``* → ``reasoning_summary_text.done`` →
   ``reasoning_summary_part.done`` → ``output_item.done``.
3. A ``message`` output item (assistant text):
   ``output_item.added`` → ``content_part.added`` → ``output_text.delta``* →
   ``output_text.done`` → ``content_part.done`` → ``output_item.done``.
4. One ``function_call`` output item per tool call:
   ``output_item.added`` → ``function_call_arguments.delta`` →
   ``function_call_arguments.done`` → ``output_item.done``.
5. ``response.completed`` with the fully-populated response object and usage.

Ordering note: Kiro emits thinking then regular content, and tool calls are only
surfaced once the byte stream ends (``parse_kiro_stream`` yields ``tool_use``
events last). This maps cleanly onto the Responses "reasoning, then message, then
function calls" item order. Any thinking that (unusually) arrives after the
message has started is folded into the message text so no content is lost.

web_search scope
----------------
Unlike the Chat Completions path, this endpoint does **not** auto-inject or
intercept the ``web_search`` MCP-emulation tool. Injecting the tool without
performing the server-side interception would emit a tool call the client cannot
fulfil; wiring the MCP interception into the Responses event/item model is
deliberately deferred. Function tools supplied by the client are fully
supported. Truncation recovery *is* supported (applied in
``converters_responses.py``), keeping parity for that feature.
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.parsers import parse_bracket_tool_calls, deduplicate_tool_calls
from kiro.tool_names import build_tool_name_restore_map, restore_tool_name
from kiro.config import FIRST_TOKEN_TIMEOUT, FIRST_TOKEN_MAX_RETRIES
from kiro.tokenizer import (
    count_tokens,
    count_message_tokens,
    count_tools_tokens,
    serialize_tool_calls_for_tokens,
)
from kiro.observability import add_token_usage
from kiro.streaming_core import (
    parse_kiro_stream,
    collect_stream_to_result,
    FirstTokenTimeoutError,
    calculate_tokens_from_context_usage,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

try:
    from kiro.debug_logger import debug_logger
except ImportError:  # pragma: no cover
    debug_logger = None


__all__ = [
    "stream_kiro_to_responses",
    "stream_kiro_to_responses_with_retry",
    "collect_responses_response",
    "generate_response_id",
]


# ==================================================================================================
# ID + event helpers
# ==================================================================================================

def generate_response_id() -> str:
    """Generate a unique Responses object id (``resp_...``)."""
    return f"resp_{uuid.uuid4().hex}"


def _new_id(prefix: str) -> str:
    """Generate a unique prefixed id, e.g. ``msg_<hex>`` / ``fc_<hex>`` / ``rs_<hex>``."""
    return f"{prefix}_{uuid.uuid4().hex}"


class _Sequencer:
    """
    Monotonic sequence-number generator for Responses stream events.

    Every streamed event carries a ``sequence_number`` that increases by one in
    emission order; the SDK uses it to detect gaps/reordering.
    """

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        """Return the next sequence number (0-based, monotonically increasing)."""
        value = self._n
        self._n += 1
        return value


def _sse(event_type: str, seq: _Sequencer, **fields: Any) -> str:
    """
    Serialise one Responses SSE event.

    Emits both a named ``event:`` line (matching OpenAI's wire format) and the
    JSON ``data:`` line. The JSON always carries ``type`` and ``sequence_number``
    so clients that key off either the SSE event name or the payload ``type``
    both work.

    Args:
        event_type: The event type, e.g. ``response.output_text.delta``.
        seq: The per-stream sequence generator.
        **fields: Event-specific fields to include in the JSON payload.

    Returns:
        A ready-to-yield SSE frame string.
    """
    payload: Dict[str, Any] = {"type": event_type, "sequence_number": seq.next()}
    payload.update(fields)
    frame = f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    if debug_logger:
        debug_logger.log_modified_chunk(frame.encode("utf-8"))
    return frame


def _build_response_object(
    *,
    base_response: Dict[str, Any],
    response_id: str,
    model: str,
    created_at: int,
    status: str,
    output: List[Dict[str, Any]],
    usage: Optional[Dict[str, Any]],
    output_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble a Responses ``response`` object.

    Args:
        base_response: Echoed request fields (instructions, tools, reasoning, ...).
        response_id: The stable ``resp_...`` id shared by created/completed events.
        model: The client-facing model name.
        created_at: Unix creation timestamp.
        status: ``in_progress`` or ``completed``.
        output: The list of output items (empty until content is produced).
        usage: Usage object, or None while the response is in progress.
        output_text: Convenience aggregation of message text (completed only).

    Returns:
        A response object dict suitable for created/in_progress/completed events
        and the non-streaming JSON body.
    """
    obj: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage,
        "error": None,
        "incomplete_details": None,
    }
    # Echo request-derived fields (instructions, tools, tool_choice, temperature,
    # top_p, max_output_tokens, parallel_tool_calls, reasoning, metadata,
    # previous_response_id). base_response never overrides the dynamic keys above.
    for key, value in base_response.items():
        obj.setdefault(key, value)
    if output_text is not None:
        obj["output_text"] = output_text
    return obj


# ==================================================================================================
# Tool-call normalisation
# ==================================================================================================

def _normalize_tool_call(tc: Dict[str, Any], restore_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Normalise a raw Kiro tool-call dict into ``(call_id, name, arguments)``.

    Handles both shapes produced by the parser (``function``-nested and flat
    ``name``/``input``) and restores any tool name that was normalised for
    Kiro's 64-character limit.

    Args:
        tc: Raw tool-call dict from the Kiro stream.
        restore_map: Reverse map from normalised to original tool names.

    Returns:
        Dict with keys ``call_id``, ``name``, ``arguments`` (arguments is a str).
    """
    func = tc.get("function") or {}
    name = func.get("name") or tc.get("name") or ""
    name = restore_tool_name(name, restore_map)

    arguments = func.get("arguments")
    if arguments is None:
        arguments = tc.get("input")
    if arguments is None:
        arguments = "{}"
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            arguments = "{}"

    call_id = tc.get("id") or _new_id("call")
    return {"call_id": call_id, "name": name, "arguments": arguments}


def _compute_usage(
    *,
    full_content: str,
    full_thinking: str,
    tool_calls: List[Dict[str, Any]],
    context_usage_percentage: Optional[float],
    model_cache: "ModelInfoCache",
    model: str,
    request_messages: Optional[list],
    request_tools: Optional[list],
) -> Dict[str, Any]:
    """
    Compute the Responses ``usage`` object from stream contents.

    Uses Kiro's context-usage percentage for prompt/total tokens when available,
    falling back to tiktoken over the original request. Completion tokens are
    always counted with tiktoken over content + thinking + serialized tool calls,
    mirroring the Chat Completions accounting.

    Args:
        full_content: Accumulated assistant text.
        full_thinking: Accumulated reasoning text.
        tool_calls: Normalised tool calls (for token accounting).
        context_usage_percentage: Kiro's reported context usage, if any.
        model_cache: Model cache for max-input-token lookup.
        model: Model name.
        request_messages: Original request messages (fallback token counting).
        request_tools: Original request tools (fallback token counting).

    Returns:
        A Responses usage dict with input/output/total tokens and detail sub-objects.
    """
    completion_tokens = count_tokens(
        full_content + full_thinking + serialize_tool_calls_for_tokens(tool_calls)
    )

    prompt_tokens, total_tokens, prompt_source, _total_source = calculate_tokens_from_context_usage(
        context_usage_percentage, completion_tokens, model_cache, model
    )

    if prompt_source == "unknown" and request_messages:
        prompt_tokens = count_message_tokens(request_messages, apply_claude_correction=False)
        if request_tools:
            prompt_tokens += count_tools_tokens(request_tools, apply_claude_correction=False)
        total_tokens = prompt_tokens + completion_tokens

    reasoning_tokens = count_tokens(full_thinking) if full_thinking else 0

    # Observability: record token usage (no-op if disabled).
    add_token_usage(prompt_tokens, completion_tokens)

    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    }


# ==================================================================================================
# Streaming
# ==================================================================================================

async def stream_kiro_to_responses_internal(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    response_id: str,
    base_response: Dict[str, Any],
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert a Kiro stream into Responses SSE events (no retry).

    Raises ``FirstTokenTimeoutError`` if the first byte does not arrive within
    ``first_token_timeout`` so the retry wrapper can re-issue the request. The
    ``response.created`` / ``response.in_progress`` events are deliberately
    deferred until the first Kiro event is seen, so a first-token retry never
    double-emits them.

    Args:
        client: HTTP client (connection management / parity with chat signature).
        response: The upstream HTTP response to parse.
        model: Client-facing model name.
        model_cache: Model cache for token accounting.
        auth_manager: Account auth manager (unused here; parity with chat signature).
        response_id: Stable ``resp_...`` id for this response.
        base_response: Echoed request fields for the response object.
        first_token_timeout: First-token wait timeout (seconds).
        request_messages: Original request messages (fallback token counting).
        request_tools: Original request tools (fallback token counting).

    Yields:
        SSE frame strings terminated by ``data: [DONE]``.
    """
    seq = _Sequencer()
    created_at = int(time.time())
    restore_map = build_tool_name_restore_map(request_tools)

    started = False
    output_items: List[Dict[str, Any]] = []
    next_index = 0

    full_content = ""
    full_thinking = ""
    tool_calls_from_stream: List[Dict[str, Any]] = []
    context_usage_percentage: Optional[float] = None

    # Reasoning item state.
    reasoning_open = False
    reasoning_id: Optional[str] = None
    reasoning_index = 0
    reasoning_text = ""

    # Message item state.
    message_open = False
    message_id: Optional[str] = None
    message_index = 0
    message_text = ""

    def in_progress_response() -> Dict[str, Any]:
        """Build the in-progress response object snapshot."""
        return _build_response_object(
            base_response=base_response,
            response_id=response_id,
            model=model,
            created_at=created_at,
            status="in_progress",
            output=[],
            usage=None,
        )

    try:
        async for event in parse_kiro_stream(response, first_token_timeout):
            # Emit lifecycle events lazily on the very first Kiro event so that a
            # first-token retry (which happens before any event) never repeats them.
            if not started:
                started = True
                yield _sse("response.created", seq, response=in_progress_response())
                yield _sse("response.in_progress", seq, response=in_progress_response())

            if event.type == "thinking" and event.thinking_content:
                if message_open:
                    # Late reasoning after message started: fold into message text
                    # to avoid an invalid second reasoning item (rare with Kiro).
                    delta = event.thinking_content
                    message_text += delta
                    full_content += delta
                    yield _sse(
                        "response.output_text.delta", seq,
                        item_id=message_id, output_index=message_index,
                        content_index=0, delta=delta, logprobs=[],
                    )
                    continue

                if not reasoning_open:
                    reasoning_open = True
                    reasoning_id = _new_id("rs")
                    reasoning_index = next_index
                    next_index += 1
                    yield _sse(
                        "response.output_item.added", seq,
                        output_index=reasoning_index,
                        item={"id": reasoning_id, "type": "reasoning", "summary": []},
                    )
                    yield _sse(
                        "response.reasoning_summary_part.added", seq,
                        item_id=reasoning_id, output_index=reasoning_index,
                        summary_index=0, part={"type": "summary_text", "text": ""},
                    )

                reasoning_text += event.thinking_content
                full_thinking += event.thinking_content
                yield _sse(
                    "response.reasoning_summary_text.delta", seq,
                    item_id=reasoning_id, output_index=reasoning_index,
                    summary_index=0, delta=event.thinking_content,
                )

            elif event.type == "content" and event.content:
                # Close reasoning before the message item begins.
                if reasoning_open:
                    yield _sse(
                        "response.reasoning_summary_text.done", seq,
                        item_id=reasoning_id, output_index=reasoning_index,
                        summary_index=0, text=reasoning_text,
                    )
                    yield _sse(
                        "response.reasoning_summary_part.done", seq,
                        item_id=reasoning_id, output_index=reasoning_index,
                        summary_index=0,
                        part={"type": "summary_text", "text": reasoning_text},
                    )
                    yield _sse(
                        "response.output_item.done", seq,
                        output_index=reasoning_index,
                        item={
                            "id": reasoning_id, "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": reasoning_text}],
                        },
                    )
                    output_items.append({
                        "id": reasoning_id, "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": reasoning_text}],
                    })
                    reasoning_open = False

                if not message_open:
                    message_open = True
                    message_id = _new_id("msg")
                    message_index = next_index
                    next_index += 1
                    yield _sse(
                        "response.output_item.added", seq,
                        output_index=message_index,
                        item={
                            "id": message_id, "type": "message", "role": "assistant",
                            "status": "in_progress", "content": [],
                        },
                    )
                    yield _sse(
                        "response.content_part.added", seq,
                        item_id=message_id, output_index=message_index, content_index=0,
                        part={"type": "output_text", "text": "", "annotations": []},
                    )

                message_text += event.content
                full_content += event.content
                yield _sse(
                    "response.output_text.delta", seq,
                    item_id=message_id, output_index=message_index,
                    content_index=0, delta=event.content, logprobs=[],
                )

            elif event.type == "tool_use" and event.tool_use:
                tool_calls_from_stream.append(event.tool_use)

            elif event.type == "context_usage" and event.context_usage_percentage is not None:
                context_usage_percentage = event.context_usage_percentage

            # 'usage' events carry only Kiro credit metering; token usage is
            # computed from context_usage_percentage / tiktoken below.

        # ---- Stream ended: finalise open items ----
        if not started:
            # Empty upstream response: still emit a well-formed lifecycle.
            started = True
            yield _sse("response.created", seq, response=in_progress_response())
            yield _sse("response.in_progress", seq, response=in_progress_response())

        # Close a still-open message item.
        if message_open:
            yield _sse(
                "response.output_text.done", seq,
                item_id=message_id, output_index=message_index,
                content_index=0, text=message_text, logprobs=[],
            )
            yield _sse(
                "response.content_part.done", seq,
                item_id=message_id, output_index=message_index, content_index=0,
                part={"type": "output_text", "text": message_text, "annotations": []},
            )
            message_item = {
                "id": message_id, "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": message_text, "annotations": []}],
            }
            yield _sse(
                "response.output_item.done", seq,
                output_index=message_index, item=message_item,
            )
            output_items.append(message_item)
        elif reasoning_open:
            # Reasoning-only response (no regular content, no tools yet).
            yield _sse(
                "response.reasoning_summary_text.done", seq,
                item_id=reasoning_id, output_index=reasoning_index,
                summary_index=0, text=reasoning_text,
            )
            yield _sse(
                "response.reasoning_summary_part.done", seq,
                item_id=reasoning_id, output_index=reasoning_index,
                summary_index=0, part={"type": "summary_text", "text": reasoning_text},
            )
            reasoning_item = {
                "id": reasoning_id, "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            }
            yield _sse(
                "response.output_item.done", seq,
                output_index=reasoning_index, item=reasoning_item,
            )
            output_items.append(reasoning_item)
            reasoning_open = False

        # ---- Tool calls (bracket + stream), de-duplicated ----
        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = deduplicate_tool_calls(tool_calls_from_stream + bracket_tool_calls)

        normalized_tool_calls: List[Dict[str, Any]] = []
        for tc in all_tool_calls:
            norm = _normalize_tool_call(tc, restore_map)
            normalized_tool_calls.append(norm)

            fc_id = _new_id("fc")
            fc_index = next_index
            next_index += 1

            added_item = {
                "id": fc_id, "type": "function_call", "call_id": norm["call_id"],
                "name": norm["name"], "arguments": "", "status": "in_progress",
            }
            yield _sse(
                "response.output_item.added", seq,
                output_index=fc_index, item=added_item,
            )
            yield _sse(
                "response.function_call_arguments.delta", seq,
                item_id=fc_id, output_index=fc_index, delta=norm["arguments"],
            )
            yield _sse(
                "response.function_call_arguments.done", seq,
                item_id=fc_id, output_index=fc_index,
                arguments=norm["arguments"], name=norm["name"],
            )
            done_item = {
                "id": fc_id, "type": "function_call", "call_id": norm["call_id"],
                "name": norm["name"], "arguments": norm["arguments"], "status": "completed",
            }
            yield _sse(
                "response.output_item.done", seq,
                output_index=fc_index, item=done_item,
            )
            output_items.append(done_item)

        # ---- Completed ----
        usage = _compute_usage(
            full_content=full_content,
            full_thinking=full_thinking,
            tool_calls=normalized_tool_calls,
            context_usage_percentage=context_usage_percentage,
            model_cache=model_cache,
            model=model,
            request_messages=request_messages,
            request_tools=request_tools,
        )
        completed = _build_response_object(
            base_response=base_response,
            response_id=response_id,
            model=model,
            created_at=created_at,
            status="completed",
            output=output_items,
            usage=usage,
            output_text=message_text,
        )
        yield _sse("response.completed", seq, response=completed)
        yield "data: [DONE]\n\n"

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit, /v1/responses)")
        raise
    finally:
        try:
            await response.aclose()
        except (httpx.HTTPError, httpx.StreamError, RuntimeError) as close_error:
            logger.debug(f"Error closing response: {close_error}")


async def stream_kiro_to_responses(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    response_id: str,
    base_response: Dict[str, Any],
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Non-retrying wrapper over :func:`stream_kiro_to_responses_internal`.

    Retry on first-token timeout is layered on by
    :func:`stream_kiro_to_responses_with_retry`.

    Args:
        client: HTTP client.
        response: Upstream response.
        model: Client-facing model name.
        model_cache: Model cache.
        auth_manager: Account auth manager.
        response_id: Stable response id.
        base_response: Echoed request fields.
        request_messages: Original messages (fallback token counting).
        request_tools: Original tools (fallback token counting).

    Yields:
        SSE frame strings.
    """
    async for chunk in stream_kiro_to_responses_internal(
        client, response, model, model_cache, auth_manager,
        response_id=response_id, base_response=base_response,
        request_messages=request_messages, request_tools=request_tools,
    ):
        yield chunk


async def stream_kiro_to_responses_with_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    response_id: str,
    base_response: Dict[str, Any],
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream Responses SSE with automatic retry on first-token timeout.

    Delegates to the shared ``streaming_core.stream_with_first_token_retry``,
    supplying a Responses-specific stream processor. See the chat equivalent in
    ``streaming_openai.stream_with_first_token_retry`` for the retry contract.

    Args:
        make_request: Factory that issues a fresh upstream request.
        client: HTTP client.
        model: Client-facing model name.
        model_cache: Model cache.
        auth_manager: Account auth manager.
        response_id: Stable response id shared across created/completed events.
        base_response: Echoed request fields for the response object.
        initial_response: Pre-validated HTTP 200 response for the first attempt.
        max_retries: Maximum attempts.
        first_token_timeout: First-token wait timeout (seconds).
        request_messages: Original messages (fallback token counting).
        request_tools: Original tools (fallback token counting).

    Yields:
        SSE frame strings.

    Raises:
        HTTPException: After exhausting retries or on upstream HTTP errors.
    """
    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        return HTTPException(status_code=status_code, detail=f"Upstream API error: {error_text}")

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        return HTTPException(
            status_code=504,
            detail=f"Model did not respond within {timeout}s after {retries} attempts. Please try again.",
        )

    async def stream_processor(resp: httpx.Response) -> AsyncGenerator[str, None]:
        async for chunk in stream_kiro_to_responses_internal(
            client, resp, model, model_cache, auth_manager,
            response_id=response_id, base_response=base_response,
            first_token_timeout=first_token_timeout,
            request_messages=request_messages, request_tools=request_tools,
        ):
            yield chunk

    async for chunk in stream_with_first_token_retry_core(
        make_request=make_request,
        stream_processor=stream_processor,
        initial_response=initial_response,
        max_retries=max_retries,
        first_token_timeout=first_token_timeout,
        on_http_error=create_http_error,
        on_all_retries_failed=create_timeout_error,
    ):
        yield chunk


# ==================================================================================================
# Non-streaming collection
# ==================================================================================================

async def collect_responses_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    response_id: str,
    base_response: Dict[str, Any],
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Collect a full Responses ``response`` object (non-streaming mode).

    Consumes the entire Kiro stream via ``collect_stream_to_result`` and builds
    the terminal response object with reasoning / message / function_call output
    items and a usage block, identical in shape to the ``response.completed``
    payload emitted while streaming.

    Args:
        client: HTTP client.
        response: Upstream response to collect.
        model: Client-facing model name.
        model_cache: Model cache for token accounting.
        auth_manager: Account auth manager (parity with chat signature).
        response_id: Stable ``resp_...`` id.
        base_response: Echoed request fields.
        request_messages: Original messages (fallback token counting).
        request_tools: Original tools (fallback token counting).

    Returns:
        A Responses ``response`` object dict with ``status="completed"``.
    """
    created_at = int(time.time())
    restore_map = build_tool_name_restore_map(request_tools)

    try:
        result = await collect_stream_to_result(response)
    finally:
        try:
            await response.aclose()
        except (httpx.HTTPError, httpx.StreamError, RuntimeError) as close_error:
            logger.debug(f"Error closing response: {close_error}")

    output_items: List[Dict[str, Any]] = []
    next_index = 0

    # Reasoning item (if any thinking content).
    if result.thinking_content:
        output_items.append({
            "id": _new_id("rs"),
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": result.thinking_content}],
        })
        next_index += 1

    # Message item (if any assistant text).
    if result.content:
        output_items.append({
            "id": _new_id("msg"),
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": result.content, "annotations": []}],
        })
        next_index += 1

    # Function-call items.
    normalized_tool_calls: List[Dict[str, Any]] = []
    for tc in result.tool_calls:
        norm = _normalize_tool_call(tc, restore_map)
        normalized_tool_calls.append(norm)
        output_items.append({
            "id": _new_id("fc"),
            "type": "function_call",
            "call_id": norm["call_id"],
            "name": norm["name"],
            "arguments": norm["arguments"],
            "status": "completed",
        })
        next_index += 1

    usage = _compute_usage(
        full_content=result.content,
        full_thinking=result.thinking_content,
        tool_calls=normalized_tool_calls,
        context_usage_percentage=result.context_usage_percentage,
        model_cache=model_cache,
        model=model,
        request_messages=request_messages,
        request_tools=request_tools,
    )

    return _build_response_object(
        base_response=base_response,
        response_id=response_id,
        model=model,
        created_at=created_at,
        status="completed",
        output=output_items,
        usage=usage,
        output_text=result.content,
    )
