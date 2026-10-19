# -*- coding: utf-8 -*-

"""Cross-surface integration tests for upstream content-filter refusals.

Regression context: when Kiro's content filter blocks a conversation it ends the
AWS event stream with a ``metadataEvent`` carrying
``{"stopReason": "CONTENT_FILTERED", "stopDetails": {"refusal": {...}}}`` and no
assistant text. The gateway previously dropped this metadata, so clients received
an HTTP 200 with zero output and stopped silently with no explanation.

These tests drive the REAL streaming/collection functions and the REAL parser
(``parse_kiro_stream`` → ``AwsEventStreamParser``) with only the HTTP response
faked. They assert, for OpenAI Chat, OpenAI Responses and Anthropic Messages in
both streaming and non-streaming modes, that the refusal is surfaced with a
protocol-correct stop signal and a visible explanation instead of an empty 200.

No real network or credentials are used.
"""

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from kiro.cache import ModelInfoCache
from kiro.streaming_anthropic import (
    collect_anthropic_response,
    stream_kiro_to_anthropic,
)
from kiro.streaming_openai import collect_stream_response, stream_kiro_to_openai

# The real refusal frame as captured from Kiro (category + explanation), followed
# by the trailing contextUsageEvent that always accompanies it.
_EXPLANATION = (
    "The selected model cannot continue this conversation. Please select a "
    "different model, or start a new conversation."
)
_REFUSAL_FRAME = (
    b'{"stopDetails":{"refusal":{"category":"CYBER","explanation":"'
    + _EXPLANATION.encode("utf-8")
    + b'"}},"stopReason":"CONTENT_FILTERED"}'
    b'{"contextUsagePercentage":23.23}'
)

_MODEL = "claude-opus-4.8"

_BASE_RESPONSE = {
    "instructions": None,
    "tools": [],
    "tool_choice": "auto",
    "temperature": None,
    "top_p": None,
    "max_output_tokens": None,
    "parallel_tool_calls": True,
    "reasoning": None,
    "metadata": None,
    "previous_response_id": None,
}


class _FakeResponse:
    """Minimal httpx.Response stand-in yielding pre-baked Kiro byte chunks."""

    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks
        self.status_code = 200
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _mk_response() -> _FakeResponse:
    return _FakeResponse([_REFUSAL_FRAME])


def _parse_openai_chunks(frames: List[str]) -> List[Dict[str, Any]]:
    events = []
    for frame in frames:
        if not frame.startswith("data:"):
            continue
        data = frame[len("data:"):].strip()
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


def _parse_sse(frames: List[str]) -> List[Dict[str, Any]]:
    events = []
    for frame in frames:
        for line in frame.split("\n"):
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data and data != "[DONE]":
                    events.append(json.loads(data))
    return events


# ==================================================================================================
# OpenAI Chat Completions
# ==================================================================================================


class TestOpenAIChatRefusal:
    """OpenAI Chat surfaces CONTENT_FILTERED as finish_reason=content_filter."""

    @pytest.mark.asyncio
    async def test_streaming_sets_content_filter_and_explanation(self):
        """Goal: streaming emits the explanation text + finish_reason=content_filter."""
        frames = []
        async for frame in stream_kiro_to_openai(
            None, _mk_response(), _MODEL, ModelInfoCache(), Mock()
        ):
            frames.append(frame)

        events = _parse_openai_chunks(frames)
        finish_reasons = [
            c["choices"][0].get("finish_reason")
            for c in events
            if c.get("choices")
        ]
        assert "content_filter" in finish_reasons
        joined = "".join(
            c["choices"][0].get("delta", {}).get("content", "")
            for c in events
            if c.get("choices")
        )
        assert _EXPLANATION in joined

    @pytest.mark.asyncio
    async def test_non_streaming_sets_content_filter_and_explanation(self):
        """Goal: non-streaming collector inherits content_filter + explanation."""
        result = await collect_stream_response(
            None, _mk_response(), _MODEL, ModelInfoCache(), Mock()
        )

        assert result["choices"][0]["finish_reason"] == "content_filter"
        assert _EXPLANATION in result["choices"][0]["message"]["content"]


# ==================================================================================================
# Anthropic Messages
# ==================================================================================================


class TestAnthropicMessagesRefusal:
    """Anthropic Messages surfaces CONTENT_FILTERED as stop_reason=refusal."""

    @pytest.mark.asyncio
    async def test_streaming_sets_refusal_and_explanation(self):
        """Goal: streaming emits a text delta + message_delta stop_reason=refusal."""
        frames = []
        async for frame in stream_kiro_to_anthropic(
            _mk_response(), _MODEL, ModelInfoCache(), Mock()
        ):
            frames.append(frame)

        events = _parse_sse(frames)
        stop_reasons = [
            e.get("delta", {}).get("stop_reason")
            for e in events
            if e.get("type") == "message_delta"
        ]
        assert stop_reasons == ["refusal"]
        text = "".join(
            e.get("delta", {}).get("text", "")
            for e in events
            if e.get("type") == "content_block_delta"
        )
        assert _EXPLANATION in text

    @pytest.mark.asyncio
    async def test_non_streaming_sets_refusal_and_explanation(self):
        """Goal: non-streaming result has stop_reason=refusal + explanation block."""
        result = await collect_anthropic_response(
            _mk_response(), _MODEL, ModelInfoCache(), Mock()
        )

        assert result["stop_reason"] == "refusal"
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        assert any(_EXPLANATION in t for t in texts)


# ==================================================================================================
# OpenAI Responses
# ==================================================================================================


class TestOpenAIResponsesRefusal:
    """OpenAI Responses surfaces CONTENT_FILTERED as status=incomplete/content_filter."""

    @pytest.mark.asyncio
    async def test_streaming_sets_incomplete_and_explanation(self):
        """Goal: response.completed carries status=incomplete + content_filter reason."""
        from kiro.streaming_responses import (
            generate_response_id,
            stream_kiro_to_responses_internal,
        )

        frames = []
        async for frame in stream_kiro_to_responses_internal(
            None, _mk_response(), _MODEL, ModelInfoCache(), None,
            response_id=generate_response_id(), base_response=_BASE_RESPONSE,
            request_messages=[{"role": "user", "content": "hi"}],
        ):
            frames.append(frame)

        events = _parse_sse(frames)
        completed = [e for e in events if e.get("type") == "response.completed"]
        assert len(completed) == 1
        resp = completed[0]["response"]
        assert resp["status"] == "incomplete"
        assert resp["incomplete_details"] == {"reason": "content_filter"}
        assert _EXPLANATION in resp.get("output_text", "")

    @pytest.mark.asyncio
    async def test_non_streaming_sets_incomplete_and_explanation(self):
        """Goal: non-streaming response object has incomplete status + explanation."""
        from kiro.streaming_responses import (
            collect_responses_response,
            generate_response_id,
        )

        resp = await collect_responses_response(
            None, _mk_response(), _MODEL, ModelInfoCache(), None,
            response_id=generate_response_id(), base_response=_BASE_RESPONSE,
            request_messages=[{"role": "user", "content": "hi"}],
        )

        assert resp["status"] == "incomplete"
        assert resp["incomplete_details"] == {"reason": "content_filter"}
        assert _EXPLANATION in resp.get("output_text", "")
