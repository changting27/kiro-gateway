# -*- coding: utf-8 -*-

"""
Tests for the OpenAI Responses -> Kiro converter (kiro/converters_responses.py).

Covers input-item parsing (message / function_call / function_call_output /
reasoning-drop / unknown salvage), flat + nested tools, instructions -> system
prompt, input_image normalisation, reasoning-effort mapping, truncation
recovery parity, fallback tokenizer inputs, and empty-input error handling.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kiro.models_responses import ResponsesRequest
from kiro.converters_core import ThinkingConfig
from kiro.converters_responses import (
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    extract_thinking_config_from_responses,
    build_kiro_payload_from_responses,
    build_responses_tokenizer_inputs,
)


def _user_input_message(payload):
    """Extract the Kiro currentMessage.userInputMessage from a payload."""
    return payload["conversationState"]["currentMessage"]["userInputMessage"]


class TestConvertInputToUnifiedSuccess:
    """convert_responses_input_to_unified across item types."""

    def test_string_input_becomes_single_user_message(self):
        """
        What it does: A bare string yields one user message.
        Purpose: Shorthand input form.
        """
        system, messages = convert_responses_input_to_unified(None, "hello")
        assert system == ""
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "hello"

    def test_instructions_become_system_prompt(self):
        """
        What it does: instructions populate the system prompt.
        Purpose: Responses has no system message; instructions carry it.
        """
        system, _ = convert_responses_input_to_unified("You are a bot.", "hi")
        assert system == "You are a bot."

    def test_message_parts_and_system_role_folding(self):
        """
        What it does: input_text parts are extracted; a system message item folds
                      into the system prompt.
        Purpose: Verify content-part text extraction + system consolidation.
        """
        system, messages = convert_responses_input_to_unified(
            "base",
            [
                {"type": "message", "role": "system", "content": "extra sys"},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "question"}]},
            ],
        )
        assert "base" in system and "extra sys" in system
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "question"

    def test_function_call_becomes_assistant_tool_call(self):
        """
        What it does: A function_call item becomes an assistant message with tool_calls.
        Purpose: Prior assistant tool calls must be represented for Kiro linkage.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [{"type": "function_call", "call_id": "call_1", "name": "f",
              "arguments": "{\"a\":1}"}],
        )
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        tc = messages[0].tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "f"
        assert tc["function"]["arguments"] == "{\"a\":1}"

    def test_function_call_output_becomes_tool_result(self):
        """
        What it does: A function_call_output becomes a user message with tool_results.
        Purpose: Tool results must be linked by call_id for Kiro.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [{"type": "function_call_output", "call_id": "call_1", "output": "result text"}],
        )
        assert len(messages) == 1
        assert messages[0].role == "user"
        tr = messages[0].tool_results[0]
        assert tr["tool_use_id"] == "call_1"
        assert tr["content"] == "result text"

    def test_reasoning_and_item_reference_dropped(self):
        """
        What it does: reasoning / item_reference items are dropped.
        Purpose: We don't round-trip reasoning to Kiro.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "item_reference", "id": "x"},
                {"type": "message", "role": "user", "content": "hi"},
            ],
        )
        assert len(messages) == 1
        assert messages[0].content == "hi"

    def test_unknown_item_text_is_salvaged(self):
        """
        What it does: An unknown item type with text is salvaged as a user message.
        Purpose: Robustness — unusual Codex items must not crash the request.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [{"type": "some_future_item", "content": "salvage me"}],
        )
        assert len(messages) == 1
        assert messages[0].content == "salvage me"

    def test_multiple_function_call_outputs_group(self):
        """
        What it does: Consecutive function_call_output items group into one user message.
        Purpose: Matches Chat Completions grouping so Kiro links all tool results.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [
                {"type": "function_call_output", "call_id": "a", "output": "ra"},
                {"type": "function_call_output", "call_id": "b", "output": "rb"},
            ],
        )
        assert len(messages) == 1
        assert len(messages[0].tool_results) == 2


class TestConvertInputImages:
    """input_image normalisation to the core image extractor shape."""

    def test_input_image_data_url_extracted(self):
        """
        What it does: An input_image data URL is extracted into unified images.
        Purpose: Vision parity with Chat Completions.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image",
                 "image_url": "data:image/png;base64,QUJD"},
            ]}],
        )
        assert messages[0].images
        assert messages[0].images[0]["media_type"] == "image/png"
        assert messages[0].images[0]["data"] == "QUJD"

    def test_input_image_object_url_extracted(self):
        """
        What it does: input_image with an object image_url ({"url": ...}) is handled.
        Purpose: Tolerate both string and object image_url forms.
        """
        _, messages = convert_responses_input_to_unified(
            None,
            [{"type": "message", "role": "user", "content": [
                {"type": "input_image",
                 "image_url": {"url": "data:image/jpeg;base64,WFla"}},
            ]}],
        )
        assert messages[0].images[0]["media_type"] == "image/jpeg"
        assert messages[0].images[0]["data"] == "WFla"


class TestConvertTools:
    """convert_responses_tools_to_unified for flat + nested + ignored tool types."""

    def test_flat_tool(self):
        """What it does: Flat function tool → UnifiedTool. Purpose: canonical shape."""
        req = ResponsesRequest(model="m", input="x", tools=[
            {"type": "function", "name": "f", "description": "d",
             "parameters": {"type": "object", "properties": {}}}])
        unified = convert_responses_tools_to_unified(req.tools)
        assert unified and unified[0].name == "f"
        assert unified[0].description == "d"

    def test_nested_tool(self):
        """What it does: Nested tool → UnifiedTool. Purpose: Chat-style tolerance."""
        req = ResponsesRequest(model="m", input="x", tools=[
            {"type": "function", "function": {"name": "g", "parameters": {}}}])
        unified = convert_responses_tools_to_unified(req.tools)
        assert unified and unified[0].name == "g"

    def test_non_function_tools_ignored(self):
        """What it does: web_search tool is skipped. Purpose: only functions go to Kiro."""
        req = ResponsesRequest(model="m", input="x", tools=[
            {"type": "web_search"},
            {"type": "function", "name": "f", "parameters": {}}])
        unified = convert_responses_tools_to_unified(req.tools)
        assert len(unified) == 1
        assert unified[0].name == "f"

    def test_none_tools_returns_none(self):
        """What it does: None tools → None. Purpose: no-tools case."""
        assert convert_responses_tools_to_unified(None) is None


class TestThinkingConfig:
    """extract_thinking_config_from_responses mapping."""

    def test_no_reasoning_defaults_enabled(self):
        """What it does: No reasoning → enabled, default budget. Purpose: default behaviour."""
        cfg = extract_thinking_config_from_responses(None, None)
        assert cfg.enabled is True
        assert cfg.budget_tokens is None

    def test_effort_none_disables(self):
        """What it does: effort 'none' → disabled. Purpose: explicit opt-out."""
        cfg = extract_thinking_config_from_responses(SimpleNamespace(effort="none"), 4096)
        assert cfg.enabled is False

    def test_known_effort_sets_budget(self):
        """What it does: effort 'high' → enabled with budget. Purpose: budget mapping."""
        cfg = extract_thinking_config_from_responses(SimpleNamespace(effort="high"), 4096)
        assert cfg.enabled is True
        assert cfg.budget_tokens == int(4096 * 0.80)

    def test_unknown_effort_defaults_budget(self):
        """What it does: unknown effort → enabled, default budget. Purpose: pass-through."""
        cfg = extract_thinking_config_from_responses(SimpleNamespace(effort="ultra"), 4096)
        assert cfg.enabled is True
        assert cfg.budget_tokens is None


class TestTruncationRecovery:
    """Tool/content truncation recovery parity with the Chat Completions route."""

    def test_tool_truncation_recovery_applied(self):
        """
        What it does: A previously-truncated tool output is annotated on this turn.
        Purpose: Feature parity (AGENTS.md §10) with the chat path.
        """
        fake_info = SimpleNamespace(tool_name="f", truncation_info={"bytes": 10})
        with patch("kiro.converters_responses.get_tool_truncation", return_value=fake_info), \
             patch("kiro.converters_responses.generate_truncation_tool_result",
                   return_value={"content": "TRUNCATION NOTICE"}):
            _, messages = convert_responses_input_to_unified(
                None,
                [{"type": "function_call_output", "call_id": "call_1", "output": "orig"}],
            )
        content = messages[0].tool_results[0]["content"]
        assert "TRUNCATION NOTICE" in content
        assert "orig" in content

    def test_content_truncation_recovery_appends_user_message(self):
        """
        What it does: A truncated assistant message triggers a synthetic user notice.
        Purpose: Content-level truncation recovery parity.
        """
        with patch("kiro.converters_responses.get_content_truncation", return_value=SimpleNamespace()), \
             patch("kiro.converters_responses.generate_truncation_user_message",
                   return_value="PLEASE CONTINUE"):
            _, messages = convert_responses_input_to_unified(
                None,
                [{"type": "message", "role": "assistant", "content": "partial answer"}],
            )
        assert messages[-1].role == "user"
        assert messages[-1].content == "PLEASE CONTINUE"


class TestTokenizerInputs:
    """build_responses_tokenizer_inputs flattening."""

    def test_tokenizer_inputs_flatten_all_item_types(self):
        """
        What it does: instructions + message + function_call + function_call_output
                      flatten into role/content dicts; tools become dumped dicts.
        Purpose: Fallback token counting needs a simple message/tool view.
        """
        req = ResponsesRequest(
            model="m",
            instructions="sys",
            input=[
                {"type": "message", "role": "user", "content": "q"},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c", "output": "r"},
                {"type": "reasoning", "id": "rs", "summary": []},
            ],
            tools=[{"type": "function", "name": "f", "parameters": {}}],
        )
        messages, tools = build_responses_tokenizer_inputs(req)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert "user" in roles and "assistant" in roles and "tool" in roles
        assert tools and len(tools) == 1


class TestBuildKiroPayload:
    """build_kiro_payload_from_responses end-to-end shape."""

    def test_string_input_builds_payload(self):
        """What it does: string input → valid Kiro payload. Purpose: happy path."""
        req = ResponsesRequest(model="claude-sonnet-4-5", input="Hello")
        payload = build_kiro_payload_from_responses(req, "conv-1", "arn:test")
        uim = _user_input_message(payload)
        assert "Hello" in uim["content"]
        assert payload["profileArn"] == "arn:test"

    def test_tools_land_in_context(self):
        """What it does: tools appear in userInputMessageContext. Purpose: tool wiring."""
        req = ResponsesRequest(
            model="claude-sonnet-4-5", input="weather?",
            tools=[{"type": "function", "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}}}],
        )
        payload = build_kiro_payload_from_responses(req, "conv-1", "arn:test")
        ctx = _user_input_message(payload).get("userInputMessageContext", {})
        assert "tools" in ctx

    def test_empty_input_list_raises_value_error(self):
        """
        What it does: An empty input list raises ValueError.
        Purpose: The route maps this to HTTP 400 (no messages to send).
        """
        req = ResponsesRequest(model="m", input=[])
        with pytest.raises(ValueError):
            build_kiro_payload_from_responses(req, "conv-1", "arn:test")

    def test_model_name_normalized(self):
        """
        What it does: Dashed model name is normalized to Kiro form in modelId.
        Purpose: Reuses the shared model resolver.
        """
        req = ResponsesRequest(model="claude-sonnet-4-5", input="hi")
        payload = build_kiro_payload_from_responses(req, "conv-1", "arn:test")
        assert _user_input_message(payload)["modelId"] == "claude-sonnet-4.5"
