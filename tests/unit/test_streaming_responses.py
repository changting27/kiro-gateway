# -*- coding: utf-8 -*-

"""
Tests for Responses streaming + collection (kiro/streaming_responses.py).

Covers the SSE event state machine (reasoning -> message -> function_call), the
monotonic sequence_number contract, empty/tool/reasoning cases, the terminal
[DONE] marker, tool-call normalisation, usage computation, and the non-streaming
collector building an equivalent response object.
"""

import json

import pytest

from kiro.cache import ModelInfoCache
from kiro.streaming_responses import (
    stream_kiro_to_responses_internal,
    collect_responses_response,
    generate_response_id,
    _normalize_tool_call,
)

pytestmark = pytest.mark.asyncio


BASE_RESPONSE = {
    "instructions": None, "tools": [], "tool_choice": "auto",
    "temperature": None, "top_p": None, "max_output_tokens": None,
    "parallel_tool_calls": True, "reasoning": None, "metadata": None,
    "previous_response_id": None,
}


class FakeResponse:
    """Minimal httpx.Response stand-in yielding pre-baked Kiro byte chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def _parse_events(frames):
    """Parse SSE frames into a list of JSON event dicts (excluding [DONE])."""
    events = []
    for frame in frames:
        for line in frame.split("\n"):
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data and data != "[DONE]":
                    events.append(json.loads(data))
    return events


async def _run_stream(chunks, request_tools=None):
    frames = []
    async for frame in stream_kiro_to_responses_internal(
        None, FakeResponse(chunks), "claude-sonnet-4-5", ModelInfoCache(), None,
        response_id=generate_response_id(), base_response=BASE_RESPONSE,
        request_messages=[{"role": "user", "content": "hi"}],
        request_tools=request_tools,
    ):
        frames.append(frame)
    return frames


class TestIdHelpers:
    """ID generation."""

    async def test_response_id_prefix(self):
        """What it does: response id starts with resp_. Purpose: SDK expectation."""
        assert generate_response_id().startswith("resp_")


class TestNormalizeToolCall:
    """_normalize_tool_call across raw shapes."""

    async def test_nested_function_shape(self):
        """What it does: {function:{name,arguments}} normalises. Purpose: parser shape A."""
        norm = _normalize_tool_call(
            {"id": "call_1", "function": {"name": "f", "arguments": "{\"a\":1}"}}, {})
        assert norm == {"call_id": "call_1", "name": "f", "arguments": "{\"a\":1}"}

    async def test_flat_name_input_shape(self):
        """What it does: {name,input} normalises; input dict → JSON. Purpose: parser shape B."""
        norm = _normalize_tool_call({"id": "call_2", "name": "g", "input": {"x": 2}}, {})
        assert norm["call_id"] == "call_2"
        assert norm["name"] == "g"
        assert json.loads(norm["arguments"]) == {"x": 2}

    async def test_missing_id_generates_call_id(self):
        """What it does: missing id → generated call_id. Purpose: never emit empty call_id."""
        norm = _normalize_tool_call({"function": {"name": "f"}}, {})
        assert norm["call_id"].startswith("call_")
        assert norm["arguments"] == "{}"


class TestStreamingContent:
    """Text-only streaming."""

    async def test_content_event_sequence(self):
        """
        What it does: Verifies the full ordered event sequence for text output.
        Purpose: Codex relies on created→item.added→content_part→deltas→done→completed.
        """
        frames = await _run_stream(
            [b'{"content":"Hello"}', b'{"content":" World"}', b'{"usage":1.0}'])
        events = _parse_events(frames)
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert types[1] == "response.in_progress"
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert types.count("response.output_text.delta") == 2
        assert "response.output_text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        assert types[-1] == "response.completed"
        assert frames[-1] == "data: [DONE]\n\n"

    async def test_sequence_numbers_monotonic(self):
        """
        What it does: sequence_number increases by 1 across all events.
        Purpose: The SDK detects gaps/reordering via sequence_number.
        """
        frames = await _run_stream([b'{"content":"Hi"}', b'{"usage":1.0}'])
        seqs = [e["sequence_number"] for e in _parse_events(frames)]
        assert seqs == list(range(len(seqs)))

    async def test_completed_carries_usage_and_output_text(self):
        """
        What it does: response.completed carries usage + aggregated output_text.
        Purpose: Non-stream-aware clients read the final response object.
        """
        frames = await _run_stream(
            [b'{"content":"Hi"}', b'{"usage":1.0}', b'{"contextUsagePercentage":10.0}'])
        completed = _parse_events(frames)[-1]["response"]
        assert completed["status"] == "completed"
        assert completed["output_text"] == "Hi"
        assert completed["usage"]["output_tokens"] > 0
        assert completed["usage"]["total_tokens"] > 0
        assert completed["output"][0]["type"] == "message"


class TestStreamingTool:
    """Tool-call streaming."""

    async def test_function_call_events(self):
        """
        What it does: Tool calls emit function_call item + arg delta/done + item.done.
        Purpose: Codex executes tool calls from these events.
        """
        chunks = [
            b'{"name":"get_weather","toolUseId":"call_abc123"}',
            b'{"input":"{\\"location\\": \\"Moscow\\"}"}',
            b'{"stop":true}',
            b'{"usage":1.0}',
        ]
        frames = await _run_stream(
            chunks, request_tools=[{"type": "function", "function": {"name": "get_weather"}}])
        events = _parse_events(frames)
        types = [e["type"] for e in events]
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types
        done = [e for e in events
                if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"]
        assert done
        item = done[0]["item"]
        assert item["name"] == "get_weather"
        assert item["call_id"] == "call_abc123"
        assert "Moscow" in item["arguments"]
        assert item["status"] == "completed"
        completed = events[-1]["response"]
        assert any(o["type"] == "function_call" for o in completed["output"])


class TestStreamingReasoning:
    """Reasoning (thinking) streaming."""

    async def test_reasoning_then_message(self):
        """
        What it does: A <thinking> block produces reasoning summary events before the
                      message, then the message text events.
        Purpose: Reasoning must be surfaced as a reasoning item ahead of the message.
        """
        frames = await _run_stream(
            [b'{"content":"<thinking>let me think</thinking>Final answer"}', b'{"usage":1.0}'])
        events = _parse_events(frames)
        types = [e["type"] for e in events]
        assert "response.reasoning_summary_part.added" in types
        assert "response.reasoning_summary_text.delta" in types
        assert "response.reasoning_summary_text.done" in types
        # reasoning must precede the message text
        assert types.index("response.reasoning_summary_text.delta") < types.index("response.output_text.delta")
        completed = events[-1]["response"]
        kinds = [o["type"] for o in completed["output"]]
        assert kinds[0] == "reasoning"
        assert "message" in kinds
        # reasoning tokens counted
        assert completed["usage"]["output_tokens_details"]["reasoning_tokens"] > 0


class TestStreamingEmpty:
    """Empty upstream response still yields a well-formed lifecycle."""

    async def test_empty_stream_emits_created_and_completed(self):
        """
        What it does: An empty upstream still emits created/in_progress/completed/[DONE].
        Purpose: Never send a malformed empty SSE stream.
        """
        frames = await _run_stream([])
        events = _parse_events(frames)
        types = [e["type"] for e in events]
        assert types == ["response.created", "response.in_progress", "response.completed"]
        assert frames[-1] == "data: [DONE]\n\n"
        assert events[-1]["response"]["output"] == []


class TestCollectResponses:
    """Non-streaming collection."""

    async def test_collect_content(self):
        """What it does: content collects into a message item + usage. Purpose: non-stream text."""
        obj = await collect_responses_response(
            None, FakeResponse([b'{"content":"Hi there"}', b'{"usage":1.0}']),
            "claude-sonnet-4-5", ModelInfoCache(), None,
            response_id=generate_response_id(), base_response=BASE_RESPONSE,
            request_messages=[{"role": "user", "content": "hi"}], request_tools=None,
        )
        assert obj["object"] == "response"
        assert obj["status"] == "completed"
        assert obj["output"][0]["type"] == "message"
        assert obj["output"][0]["content"][0]["text"] == "Hi there"
        assert obj["output_text"] == "Hi there"
        assert obj["usage"]["output_tokens"] > 0

    async def test_collect_tool(self):
        """What it does: tool collects into a function_call item. Purpose: non-stream tools."""
        chunks = [
            b'{"name":"get_weather","toolUseId":"call_abc123"}',
            b'{"input":"{\\"location\\": \\"Moscow\\"}"}',
            b'{"stop":true}',
            b'{"usage":1.0}',
        ]
        obj = await collect_responses_response(
            None, FakeResponse(chunks), "claude-sonnet-4-5", ModelInfoCache(), None,
            response_id=generate_response_id(), base_response=BASE_RESPONSE,
            request_messages=[{"role": "user", "content": "hi"}],
            request_tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        fcs = [o for o in obj["output"] if o["type"] == "function_call"]
        assert fcs and fcs[0]["name"] == "get_weather"
        assert fcs[0]["call_id"] == "call_abc123"
        assert "Moscow" in fcs[0]["arguments"]

    async def test_collect_reasoning(self):
        """What it does: thinking collects into a reasoning item. Purpose: non-stream reasoning."""
        obj = await collect_responses_response(
            None, FakeResponse([b'{"content":"<thinking>ponder</thinking>Done"}', b'{"usage":1.0}']),
            "claude-sonnet-4-5", ModelInfoCache(), None,
            response_id=generate_response_id(), base_response=BASE_RESPONSE,
            request_messages=[{"role": "user", "content": "hi"}], request_tools=None,
        )
        kinds = [o["type"] for o in obj["output"]]
        assert kinds[0] == "reasoning"
        assert "message" in kinds
        assert obj["output"][0]["summary"][0]["text"] == "ponder"

    async def test_collect_empty(self):
        """What it does: empty upstream → completed with no output. Purpose: robustness."""
        obj = await collect_responses_response(
            None, FakeResponse([]), "claude-sonnet-4-5", ModelInfoCache(), None,
            response_id=generate_response_id(), base_response=BASE_RESPONSE,
            request_messages=None, request_tools=None,
        )
        assert obj["status"] == "completed"
        assert obj["output"] == []
