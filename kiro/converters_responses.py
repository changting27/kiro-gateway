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
Converters for transforming the OpenAI Responses API format to Kiro format.

This adapter mirrors ``converters_openai.py`` for the Responses API surface. It
translates a :class:`~kiro.models_responses.ResponsesRequest` into the API-agnostic
unified format consumed by :func:`kiro.converters_core.build_kiro_payload`.

Responses-specific handling:

* ``input`` may be a bare string (shorthand for one user message) or a list of
  typed input items. Supported item types:
    - ``message``            → a user/assistant/system/developer message
    - ``function_call``      → an assistant tool call (merged by the core layer)
    - ``function_call_output`` → a tool result (grouped onto a user message)
    - ``reasoning`` / ``item_reference`` → dropped (not round-tripped to Kiro)
  Unknown item types are handled best-effort (any embedded text becomes a user
  message) so unusual Codex items never crash the request.
* ``instructions`` becomes the system prompt.
* ``input_image`` content parts are normalised to the ``image_url`` shape the
  core image extractor understands, giving vision parity with Chat Completions.
* Tool/content truncation-recovery is applied to incoming ``function_call_output``
  and assistant ``message`` items, matching the Chat Completions route so a
  truncated tool result / assistant turn is transparently annotated on the next
  request (feature-consistency, AGENTS.md §10).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_responses import ResponsesRequest, ResponsesTool

# Reuse shared core logic and the OpenAI reasoning-budget mapping (same family).
from kiro.converters_core import (
    extract_text_content,
    extract_images_from_content,
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    build_kiro_payload as core_build_kiro_payload,
)
from kiro.converters_openai import reasoning_effort_to_budget

# Truncation-recovery hooks (same system used by the Chat Completions route).
from kiro.truncation_state import get_tool_truncation, get_content_truncation
from kiro.truncation_recovery import (
    generate_truncation_tool_result,
    generate_truncation_user_message,
)


# ==================================================================================================
# Input normalisation helpers
# ==================================================================================================

def _normalize_input(input_data: Any) -> List[Any]:
    """
    Normalise the ``input`` field into a list of input items.

    A bare string is shorthand for a single user message. A list is returned
    as-is. Any other type yields an empty list (the core layer will then raise
    "No messages to send", surfaced to the client as HTTP 400).

    Args:
        input_data: The request ``input`` (string or list of items).

    Returns:
        A list of input items (dicts and/or strings).
    """
    if isinstance(input_data, str):
        return [{"type": "message", "role": "user", "content": input_data}]
    if isinstance(input_data, list):
        return input_data
    return []


def _normalize_image_parts(content: Any) -> Any:
    """
    Rewrite Responses ``input_image`` parts into the core ``image_url`` shape.

    The core image extractor understands ``{"type": "image_url", "image_url":
    {"url": ...}}`` (OpenAI) and ``{"type": "image", ...}`` (Anthropic) but not
    the Responses ``{"type": "input_image", "image_url": "data:..."}`` part. This
    helper converts the latter so images can be reused via the shared extractor.

    Args:
        content: A message ``content`` value (string or list of parts).

    Returns:
        The content with ``input_image`` parts rewritten; non-list content and
        non-image parts are returned unchanged.
    """
    if not isinstance(content, list):
        return content

    normalized: List[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_image":
            url = part.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            normalized.append({"type": "image_url", "image_url": {"url": url or ""}})
        else:
            normalized.append(part)
    return normalized


def _extract_call_id(item: Dict[str, Any]) -> str:
    """
    Extract the tool-call linkage id from a function_call / function_call_output item.

    Responses links a call and its output by ``call_id``; some clients only send
    ``id``. Kiro links tool uses to tool results by this shared identifier.

    Args:
        item: A ``function_call`` or ``function_call_output`` input item.

    Returns:
        The call id, or an empty string if absent.
    """
    return item.get("call_id") or item.get("id") or ""


def _function_call_to_unified(item: Dict[str, Any]) -> UnifiedMessage:
    """
    Convert a ``function_call`` input item into an assistant UnifiedMessage.

    Consecutive assistant tool calls are merged by the core layer into a single
    assistantResponseMessage with multiple toolUses.

    Args:
        item: A ``function_call`` input item ``{call_id, name, arguments}``.

    Returns:
        An assistant UnifiedMessage carrying one tool call.
    """
    arguments = item.get("arguments", "{}")
    if not isinstance(arguments, str):
        # Responses always sends stringified JSON, but tolerate dicts defensively.
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            arguments = "{}"

    return UnifiedMessage(
        role="assistant",
        content="",
        tool_calls=[{
            "id": _extract_call_id(item),
            "type": "function",
            "function": {
                "name": item.get("name", ""),
                "arguments": arguments,
            },
        }],
    )


def _function_call_output_to_tool_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a ``function_call_output`` input item into a unified tool_result.

    Applies tool-level truncation recovery: if the referenced tool call was
    previously truncated by Kiro, the original output is prefixed with a
    synthetic notice so the model is informed on this turn (matches the Chat
    Completions route behaviour).

    Args:
        item: A ``function_call_output`` item ``{call_id, output}``.

    Returns:
        A unified tool_result dict ``{type, tool_use_id, content}``.
    """
    call_id = _extract_call_id(item)
    output = item.get("output", "")
    content = extract_text_content(output) or "(empty result)"

    truncation_info = get_tool_truncation(call_id)
    if truncation_info:
        synthetic = generate_truncation_tool_result(
            tool_name=truncation_info.tool_name,
            tool_use_id=call_id,
            truncation_info=truncation_info.truncation_info,
        )
        content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{content}"
        logger.debug(f"Applied truncation recovery to function_call_output {call_id}")

    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
    }


# ==================================================================================================
# Message conversion
# ==================================================================================================

def convert_responses_input_to_unified(
    instructions: Optional[str],
    input_data: Any,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert Responses ``instructions`` + ``input`` into unified messages.

    Args:
        instructions: The system prompt (or None).
        input_data: The request ``input`` (string or list of typed items).

    Returns:
        Tuple of (system_prompt, unified_messages).
    """
    system_prompt = (instructions or "").strip()
    items = _normalize_input(input_data)

    processed: List[UnifiedMessage] = []
    pending_tool_results: List[Dict[str, Any]] = []
    total_tool_calls = 0
    total_tool_results = 0
    total_images = 0

    def flush_tool_results() -> None:
        """Flush accumulated tool results into a single user message."""
        if pending_tool_results:
            processed.append(UnifiedMessage(
                role="user",
                content="",
                tool_results=pending_tool_results.copy(),
            ))
            pending_tool_results.clear()

    for item in items:
        # Bare string entries in the list are treated as user text.
        if isinstance(item, str):
            flush_tool_results()
            processed.append(UnifiedMessage(role="user", content=item))
            continue

        if not isinstance(item, dict):
            logger.debug(f"Skipping non-dict Responses input item: {type(item).__name__}")
            continue

        item_type = item.get("type")
        role = item.get("role")

        # --- Tool result ---
        if item_type == "function_call_output":
            pending_tool_results.append(_function_call_output_to_tool_result(item))
            total_tool_results += 1
            continue

        # Any non-tool-result item flushes pending results first (they attach to
        # the preceding user turn, matching the Chat Completions grouping).
        flush_tool_results()

        # --- Assistant tool call ---
        if item_type == "function_call":
            processed.append(_function_call_to_unified(item))
            total_tool_calls += 1
            continue

        # --- Dropped items (not round-tripped to Kiro) ---
        if item_type in ("reasoning", "item_reference"):
            logger.debug(f"Dropping Responses input item of type '{item_type}'")
            continue

        # --- Message item (explicit type, or a role-bearing item without a type) ---
        if item_type == "message" or (item_type is None and role is not None):
            content = item.get("content")
            text = extract_text_content(content)

            if role == "system":
                # Fold additional system content into the system prompt.
                system_prompt = f"{system_prompt}\n{text}".strip() if system_prompt else text.strip()
                continue

            images = extract_images_from_content(_normalize_image_parts(content)) or None
            if images:
                total_images += len(images)

            unified_role = role or "user"
            processed.append(UnifiedMessage(
                role=unified_role,
                content=text,
                images=images,
            ))

            # Content-level truncation recovery: if a prior assistant turn was
            # truncated, inform the model with a synthetic user message.
            if unified_role == "assistant" and text:
                content_truncation = get_content_truncation(text)
                if content_truncation:
                    processed.append(UnifiedMessage(
                        role="user",
                        content=generate_truncation_user_message(),
                    ))
                    logger.debug("Added truncation notice after assistant message (Responses)")
            continue

        # --- Unknown item type: best-effort text salvage ---
        salvaged = extract_text_content(item.get("content") or item.get("text"))
        if salvaged:
            logger.debug(f"Salvaged text from unknown Responses item type '{item_type}'")
            processed.append(UnifiedMessage(role="user", content=salvaged))
        else:
            logger.debug(f"Ignoring unsupported Responses input item type '{item_type}'")

    # Flush any trailing tool results.
    flush_tool_results()

    if total_tool_calls or total_tool_results or total_images:
        logger.debug(
            f"Converted Responses input: {total_tool_calls} tool_calls, "
            f"{total_tool_results} tool_results, {total_images} images, "
            f"{len(processed)} messages"
        )

    return system_prompt, processed


def convert_responses_tools_to_unified(
    tools: Optional[List[ResponsesTool]],
) -> Optional[List[UnifiedTool]]:
    """
    Convert Responses tools to unified format.

    Handles both the flat Responses shape (``name``/``description``/``parameters``
    on the tool) and the nested Chat Completions shape (``function`` sub-object).
    Only ``function`` tools are forwarded to Kiro; other tool types
    (``web_search``, ``file_search``, ``mcp``, ...) are ignored, matching the
    Chat Completions adapter.

    Args:
        tools: List of ResponsesTool objects, or None.

    Returns:
        List of UnifiedTool objects, or None if there are no function tools.
    """
    if not tools:
        return None

    unified_tools: List[UnifiedTool] = []
    for tool in tools:
        if tool.type != "function":
            logger.debug(f"Ignoring non-function Responses tool of type '{tool.type}'")
            continue

        if tool.function is not None:
            # Nested Chat-Completions shape.
            unified_tools.append(UnifiedTool(
                name=tool.function.name,
                description=tool.function.description,
                input_schema=tool.function.parameters,
            ))
        elif tool.name is not None:
            # Flat Responses shape.
            unified_tools.append(UnifiedTool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.parameters,
            ))
        else:
            logger.warning("Skipping invalid Responses tool: no 'name' or 'function' field")

    return unified_tools if unified_tools else None


# ==================================================================================================
# Thinking configuration
# ==================================================================================================

def extract_thinking_config_from_responses(
    reasoning: Any,
    max_output_tokens: Optional[int],
) -> ThinkingConfig:
    """
    Derive the thinking configuration from the Responses ``reasoning`` object.

    Mirrors :func:`kiro.converters_openai.extract_thinking_config_from_openai`:

    - No ``reasoning`` / no ``effort`` → enabled with the default budget.
    - ``effort == "none"`` → disabled (no thinking tags injected).
    - Known effort → enabled with a percentage-based budget of ``max_output_tokens``.
    - Unknown effort → enabled with the default budget (lenient pass-through).

    Args:
        reasoning: The request ``reasoning`` object (ResponsesReasoning) or None.
        max_output_tokens: The request ``max_output_tokens`` (or None).

    Returns:
        A ThinkingConfig for the core layer.
    """
    effort = getattr(reasoning, "effort", None) if reasoning is not None else None

    if not effort:
        return ThinkingConfig(enabled=True, budget_tokens=None)

    effort = effort.lower()
    if effort == "none":
        return ThinkingConfig(enabled=False, budget_tokens=None)

    # Known effort levels shared with the Chat Completions mapping.
    if effort in ("minimal", "low", "medium", "high", "xhigh"):
        max_tokens = max_output_tokens or 4096
        budget = reasoning_effort_to_budget(max_tokens, effort)
        logger.debug(
            f"Extracted thinking config from Responses: effort='{effort}', "
            f"max_output_tokens={max_tokens}, budget={budget}"
        )
        return ThinkingConfig(enabled=True, budget_tokens=budget)

    # Unknown effort → default budget (pass-through, do not reject).
    logger.debug(f"Unknown Responses reasoning effort '{effort}', using default budget")
    return ThinkingConfig(enabled=True, budget_tokens=None)


# ==================================================================================================
# Tokenizer inputs (fallback token counting)
# ==================================================================================================

def build_responses_tokenizer_inputs(
    request_data: ResponsesRequest,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Build tiktoken-friendly (messages, tools) inputs from a Responses request.

    These approximate role/content pairs are only used for *fallback* prompt-token
    counting when Kiro does not return a context-usage percentage (the primary
    source). They intentionally flatten the Responses input into the simple
    ``{"role": ..., "content": ...}`` shape the tokenizer expects.

    Args:
        request_data: The parsed Responses request.

    Returns:
        Tuple of (messages, tools) where ``tools`` is None when no tools were sent.
    """
    messages: List[Dict[str, Any]] = []
    if request_data.instructions:
        messages.append({"role": "system", "content": request_data.instructions})

    for item in _normalize_input(request_data.input):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "function_call":
            name = item.get("name", "")
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            messages.append({"role": "assistant", "content": f"{name}{arguments}"})
        elif item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "content": extract_text_content(item.get("output", "")),
            })
        elif item_type in ("reasoning", "item_reference"):
            continue
        elif item_type == "message" or (item_type is None and item.get("role") is not None):
            messages.append({
                "role": item.get("role", "user"),
                "content": extract_text_content(item.get("content")),
            })

    tools = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
    return messages, tools


# ==================================================================================================
# Main entry point
# ==================================================================================================

def build_kiro_payload_from_responses(
    request_data: ResponsesRequest,
    conversation_id: str,
    profile_arn: str,
) -> dict:
    """
    Build a complete Kiro API payload from a Responses request.

    This is the main entry point for Responses → Kiro conversion; it uses the
    shared core :func:`kiro.converters_core.build_kiro_payload` with
    Responses-specific adapters.

    Args:
        request_data: The parsed Responses request.
        conversation_id: Unique conversation ID for Kiro.
        profile_arn: AWS CodeWhisperer profile ARN.

    Returns:
        The payload dict for POSTing to Kiro's generateAssistantResponse.

    Raises:
        ValueError: If there are no messages to send (surfaced as HTTP 400).
    """
    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.instructions, request_data.input
    )
    unified_tools = convert_responses_tools_to_unified(request_data.tools)
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(
        request_data.reasoning, request_data.max_output_tokens
    )

    logger.debug(
        f"Converting Responses request: model={request_data.model} -> {model_id}, "
        f"messages={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, thinking_budget={thinking_config.budget_tokens}"
    )

    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )

    return result.payload
