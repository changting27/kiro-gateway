# -*- coding: utf-8 -*-

"""
Endpoint tests for the OpenAI Responses API (POST /v1/responses).

Fully network-isolated (see tests/conftest.py). Upstream Kiro responses are
mocked by patching ``kiro.routes_common.KiroHttpClient`` (where the executor
instantiates the client). Covers all four capability combinations required by
the feature — streaming/non-streaming × text/tool — plus reasoning, auth,
validation, upstream-error shaping, response-field echoes, and edge cases.
"""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest


# ---- Mock Kiro byte streams -------------------------------------------------

_CONTENT_CHUNKS = [b'{"content":"Hello"}', b'{"content":" World"}', b'{"usage":1.0}']
_TOOL_CHUNKS = [
    b'{"name":"get_weather","toolUseId":"call_abc123"}',
    b'{"input":"{\\"location\\": \\"Moscow\\"}"}',
    b'{"stop":true}',
    b'{"usage":1.0}',
]
_REASONING_CHUNKS = [
    b'{"content":"<thinking>let me think</thinking>Final answer"}',
    b'{"usage":1.0}',
]


@contextmanager
def _mock_upstream(chunks, status=200, error_body=b'{"message":"boom"}'):
    """
    Patch the executor's KiroHttpClient to return a mock upstream response.

    Args:
        chunks: Raw Kiro byte chunks to stream for a 200 response.
        status: Upstream HTTP status code.
        error_body: Body returned by aread() for non-200 responses.

    Yields:
        The mock KiroHttpClient instance (for call assertions).
    """
    upstream = AsyncMock()
    upstream.status_code = status

    async def _aiter():
        for chunk in chunks:
            yield chunk

    upstream.aiter_bytes = _aiter
    upstream.aclose = AsyncMock()
    upstream.aread = AsyncMock(return_value=error_body)

    mock_client = AsyncMock()
    mock_client.request_with_retry = AsyncMock(return_value=upstream)
    mock_client.client = AsyncMock()
    mock_client.close = AsyncMock()

    with patch("kiro.routes_common.KiroHttpClient", return_value=mock_client):
        yield mock_client


def _parse_sse(text):
    """Parse an SSE response body into a list of JSON event dicts (excluding [DONE])."""
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line[len("data:"):].strip()
            if data and data != "[DONE]":
                events.append(json.loads(data))
    return events


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


# =============================================================================
# Authentication
# =============================================================================

class TestResponsesAuth:
    """Authentication on /v1/responses."""

    def test_missing_api_key_rejected(self, test_client):
        """What it does: no key → 401. Purpose: endpoint must require auth."""
        print("Action: POST /v1/responses without auth...")
        resp = test_client.post("/v1/responses", json={"model": "claude-sonnet-4-5", "input": "hi"})
        assert resp.status_code == 401

    def test_invalid_api_key_rejected(self, test_client, auth_headers):
        """What it does: wrong key → 401. Purpose: reject invalid credentials."""
        resp = test_client.post(
            "/v1/responses",
            headers=auth_headers(invalid=True),
            json={"model": "claude-sonnet-4-5", "input": "hi"},
        )
        assert resp.status_code == 401


# =============================================================================
# Validation
# =============================================================================

class TestResponsesValidation:
    """Request-body validation."""

    def test_missing_input_is_422(self, test_client, valid_proxy_api_key):
        """What it does: omit input → 422. Purpose: input is required."""
        resp = test_client.post(
            "/v1/responses", headers=_headers(valid_proxy_api_key),
            json={"model": "claude-sonnet-4-5"},
        )
        assert resp.status_code == 422

    def test_missing_model_is_422(self, test_client, valid_proxy_api_key):
        """What it does: omit model → 422. Purpose: model is required."""
        resp = test_client.post(
            "/v1/responses", headers=_headers(valid_proxy_api_key),
            json={"input": "hi"},
        )
        assert resp.status_code == 422

    def test_empty_input_list_is_400(self, test_client, valid_proxy_api_key):
        """
        What it does: empty input list → 400.
        Purpose: The converter raises ValueError (no messages), mapped to 400.
        """
        with _mock_upstream(_CONTENT_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": []},
            )
        assert resp.status_code == 400


# =============================================================================
# Non-streaming
# =============================================================================

class TestResponsesNonStreaming:
    """Non-streaming (stream=false) responses."""

    def test_text_response_object(self, test_client, valid_proxy_api_key):
        """
        What it does: A text request returns a completed response object with a
                      message output item, usage, and aggregated output_text.
        Purpose: Core non-streaming text path (combination 1).
        """
        with _mock_upstream(_CONTENT_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "Hi", "stream": False},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["model"] == "claude-sonnet-4-5"
        msg = [o for o in body["output"] if o["type"] == "message"]
        assert msg and msg[0]["content"][0]["text"] == "Hello World"
        assert body["output_text"] == "Hello World"
        assert body["usage"]["output_tokens"] > 0
        assert body["id"].startswith("resp_")

    def test_tool_call_response_object(self, test_client, valid_proxy_api_key):
        """
        What it does: A tool request returns a function_call output item.
        Purpose: Core non-streaming tool path (combination 2).
        """
        with _mock_upstream(_TOOL_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={
                    "model": "claude-sonnet-4-5", "input": "weather?", "stream": False,
                    "tools": [{"type": "function", "name": "get_weather",
                               "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}],
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        fcs = [o for o in body["output"] if o["type"] == "function_call"]
        assert fcs, body["output"]
        assert fcs[0]["name"] == "get_weather"
        assert fcs[0]["call_id"] == "call_abc123"
        assert "Moscow" in fcs[0]["arguments"]

    def test_reasoning_response_object(self, test_client, valid_proxy_api_key):
        """
        What it does: A <thinking> response yields a reasoning output item.
        Purpose: Reasoning event conversion (non-streaming).
        """
        with _mock_upstream(_REASONING_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "think", "stream": False,
                      "reasoning": {"effort": "high"}},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        kinds = [o["type"] for o in body["output"]]
        assert kinds and kinds[0] == "reasoning"
        assert "message" in kinds
        assert body["usage"]["output_tokens_details"]["reasoning_tokens"] > 0

    def test_response_echoes_request_fields(self, test_client, valid_proxy_api_key):
        """
        What it does: instructions/temperature/reasoning are echoed on the response.
        Purpose: OpenAI clients read these back from the response object.
        """
        with _mock_upstream(_CONTENT_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "Hi", "stream": False,
                      "instructions": "be terse", "temperature": 0.5,
                      "reasoning": {"effort": "low"}},
            )
        body = resp.json()
        assert body["instructions"] == "be terse"
        assert body["temperature"] == 0.5
        assert body["reasoning"]["effort"] == "low"


# =============================================================================
# Streaming
# =============================================================================

class TestResponsesStreaming:
    """Streaming (stream=true) responses."""

    def test_text_stream_event_sequence(self, test_client, valid_proxy_api_key):
        """
        What it does: A streaming text request emits the ordered Responses SSE
                      events and a terminal [DONE].
        Purpose: Core streaming text path (combination 3).
        """
        with _mock_upstream(_CONTENT_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "Hi", "stream": True},
            )
        assert resp.status_code == 200, resp.text
        assert "text/event-stream" in resp.headers["content-type"]
        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        assert resp.text.rstrip().endswith("data: [DONE]")
        # sequence numbers monotonic
        seqs = [e["sequence_number"] for e in events]
        assert seqs == list(range(len(seqs)))

    def test_tool_stream_events(self, test_client, valid_proxy_api_key):
        """
        What it does: A streaming tool request emits function_call argument events.
        Purpose: Core streaming tool path (combination 4).
        """
        with _mock_upstream(_TOOL_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={
                    "model": "claude-sonnet-4-5", "input": "weather?", "stream": True,
                    "tools": [{"type": "function", "name": "get_weather",
                               "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}],
                },
            )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types
        done = [e for e in events
                if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"]
        assert done and done[0]["item"]["name"] == "get_weather"
        assert "Moscow" in done[0]["item"]["arguments"]

    def test_reasoning_stream_events(self, test_client, valid_proxy_api_key):
        """
        What it does: A streaming <thinking> response emits reasoning summary events
                      before the message text.
        Purpose: Reasoning event conversion (streaming).
        """
        with _mock_upstream(_REASONING_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "think", "stream": True,
                      "reasoning": {"effort": "medium"}},
            )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert "response.reasoning_summary_text.delta" in types
        assert types.index("response.reasoning_summary_text.delta") < types.index("response.output_text.delta")


# =============================================================================
# Upstream error shaping
# =============================================================================

class TestResponsesUpstreamErrors:
    """Non-200 upstream responses are surfaced as client errors, not hangs."""

    def test_upstream_400_surfaces_error(self, test_client, valid_proxy_api_key):
        """
        What it does: An upstream 400 yields a >=400 client response (not 200).
        Purpose: Errors must be surfaced, not swallowed.
        """
        with _mock_upstream([], status=400,
                            error_body=b'{"message":"Improperly formed request."}'):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "hi", "stream": False},
            )
        assert resp.status_code >= 400

    def test_upstream_context_overflow_maps_to_400(self, test_client, valid_proxy_api_key):
        """
        What it does: Kiro CONTENT_LENGTH_EXCEEDS_THRESHOLD maps to a 400 error body.
        Purpose: Shares the OpenAI-style context-overflow shaping via the executor.
        """
        overflow = (b'{"message":"Input content length exceeds threshold.",'
                    b'"reason":"CONTENT_LENGTH_EXCEEDS_THRESHOLD"}')
        with _mock_upstream([], status=400, error_body=overflow):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={"model": "claude-sonnet-4-5", "input": "hi", "stream": False},
            )
        assert resp.status_code == 400
        # OpenAI-style error envelope
        body = resp.json()
        assert "error" in body or "detail" in body


# =============================================================================
# Input shape handling
# =============================================================================

class TestResponsesInputShapes:
    """Different input forms are accepted end-to-end."""

    def test_list_input_with_prior_tool_turn(self, test_client, valid_proxy_api_key):
        """
        What it does: A multi-item input (message + function_call + output) is accepted.
        Purpose: Codex sends prior tool turns as input items on follow-up requests.
        """
        with _mock_upstream(_CONTENT_CHUNKS):
            resp = test_client.post(
                "/v1/responses", headers=_headers(valid_proxy_api_key),
                json={
                    "model": "claude-sonnet-4-5", "stream": False,
                    "instructions": "sys",
                    "input": [
                        {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "weather?"}]},
                        {"type": "function_call", "call_id": "c1", "name": "get_weather",
                         "arguments": "{}"},
                        {"type": "function_call_output", "call_id": "c1", "output": "Sunny"},
                    ],
                    "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
                },
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "completed"
