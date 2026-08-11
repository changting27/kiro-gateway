# -*- coding: utf-8 -*-

"""
Tests for OpenAI Responses API Pydantic models (kiro/models_responses.py).

Covers request validation, both tool shapes (flat + nested), reasoning config,
extra-field tolerance (pass-through philosophy), and required-field errors.
"""

import pytest
from pydantic import ValidationError

from kiro.models_responses import (
    ResponsesRequest,
    ResponsesTool,
    ResponsesFunctionTool,
    ResponsesReasoning,
)


class TestResponsesRequestSuccess:
    """Valid ResponsesRequest construction across supported input shapes."""

    def test_string_input_is_accepted(self):
        """
        What it does: Builds a request with a bare-string input.
        Purpose: The Responses API allows `input` as shorthand for one user message.
        """
        print("Setup: string input...")
        req = ResponsesRequest(model="claude-sonnet-4-5", input="Hello")
        assert req.input == "Hello"
        assert req.stream is False
        assert req.tools is None

    def test_list_input_is_accepted(self):
        """
        What it does: Builds a request with a list of typed input items.
        Purpose: The primary Responses input form is a list of items.
        """
        print("Setup: list input with a message item...")
        req = ResponsesRequest(
            model="claude-sonnet-4-5",
            input=[{"type": "message", "role": "user", "content": "Hi"}],
        )
        assert isinstance(req.input, list)
        assert req.input[0]["role"] == "user"

    def test_instructions_and_generation_params(self):
        """
        What it does: Accepts instructions, max_output_tokens, temperature, top_p.
        Purpose: Ensure Responses-specific generation fields are modelled.
        """
        req = ResponsesRequest(
            model="m", input="x", instructions="be terse",
            max_output_tokens=256, temperature=0.5, top_p=0.9, stream=True,
        )
        assert req.instructions == "be terse"
        assert req.max_output_tokens == 256
        assert req.temperature == 0.5
        assert req.stream is True

    def test_extra_fields_are_preserved(self):
        """
        What it does: Unknown fields (store, include, prompt_cache_key) are accepted.
        Purpose: Pass-through philosophy — Codex sends many fields we don't model.
        """
        print("Setup: request with unmodelled Codex fields...")
        req = ResponsesRequest(
            model="m", input="x",
            store=False, include=["reasoning.encrypted_content"],
            prompt_cache_key="abc", parallel_tool_calls=False,
        )
        assert req.store is False
        assert req.parallel_tool_calls is False
        # Extra field preserved via extra="allow"
        assert req.model_dump().get("include") == ["reasoning.encrypted_content"]


class TestResponsesToolShapes:
    """Both the flat (canonical) and nested (Chat-Completions) tool shapes parse."""

    def test_flat_function_tool(self):
        """
        What it does: Parses a flat Responses function tool.
        Purpose: `{"type":"function","name":...,"parameters":...}` is the canonical shape.
        """
        req = ResponsesRequest(
            model="m", input="x",
            tools=[{
                "type": "function", "name": "get_weather",
                "description": "w", "parameters": {"type": "object", "properties": {}},
            }],
        )
        tool = req.tools[0]
        assert tool.type == "function"
        assert tool.name == "get_weather"
        assert tool.function is None

    def test_nested_function_tool(self):
        """
        What it does: Parses a nested Chat-Completions-style tool.
        Purpose: Tolerate clients that reuse the nested shape.
        """
        req = ResponsesRequest(
            model="m", input="x",
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        )
        tool = req.tools[0]
        assert isinstance(tool.function, ResponsesFunctionTool)
        assert tool.function.name == "f"

    def test_non_function_tool_type_preserved(self):
        """
        What it does: A non-function tool (web_search) is accepted (type preserved).
        Purpose: The converter ignores these, but the model must not reject them.
        """
        req = ResponsesRequest(model="m", input="x", tools=[{"type": "web_search"}])
        assert req.tools[0].type == "web_search"


class TestResponsesReasoning:
    """Reasoning config parsing."""

    def test_reasoning_effort_and_summary(self):
        """
        What it does: Parses reasoning {effort, summary}.
        Purpose: Codex sends reasoning as an object, not a scalar.
        """
        req = ResponsesRequest(
            model="m", input="x", reasoning={"effort": "high", "summary": "auto"},
        )
        assert isinstance(req.reasoning, ResponsesReasoning)
        assert req.reasoning.effort == "high"
        assert req.reasoning.summary == "auto"

    def test_reasoning_unknown_effort_tolerated(self):
        """
        What it does: An unknown effort value is accepted (mapped later by converter).
        Purpose: Pass-through — do not reject unfamiliar effort levels.
        """
        req = ResponsesRequest(model="m", input="x", reasoning={"effort": "ultra"})
        assert req.reasoning.effort == "ultra"


class TestResponsesRequestErrors:
    """Required-field validation."""

    def test_missing_model_raises(self):
        """
        What it does: Omitting model raises ValidationError.
        Purpose: model is required.
        """
        with pytest.raises(ValidationError):
            ResponsesRequest(input="x")

    def test_missing_input_raises(self):
        """
        What it does: Omitting input raises ValidationError.
        Purpose: input is required.
        """
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m")

    def test_wrong_input_type_raises(self):
        """
        What it does: A non-str, non-list input raises ValidationError.
        Purpose: input must be a string or list.
        """
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input=12345)
