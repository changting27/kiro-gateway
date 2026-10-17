# -*- coding: utf-8 -*-

"""Cross-surface integration tests for the GenerateAssistantResponse request URL.

Regression context: some network paths put a path allow-listing reverse proxy in
front of ``runtime.{region}.kiro.dev``. That proxy forwards the canonical AWS JSON
1.0 request ``POST /`` (operation selected by ``x-amz-target``) but rejects the
legacy REST alias ``/generateAssistantResponse`` with a non-AWS HTML 404, which
Claude Code reports as "model does not exist".

These tests drive the real routes and the real ``KiroHttpClient`` with only the
transport mocked, and assert the exact URL and operation header sent upstream for
OpenAI Chat, OpenAI Responses, and Anthropic Messages, in streaming and
non-streaming modes, for both the legacy single-account path and the Account
System failover path. No real network or credentials are used.
"""

from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator, List
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from kiro.http_client import KiroHttpClient


_RUNTIME_HOST = "https://runtime.us-east-1.kiro.dev"
_CANONICAL_URL = f"{_RUNTIME_HOST}/"
_LEGACY_PATH = "/generateAssistantResponse"
_OPERATION_TARGET = "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"

_CONTENT_CHUNKS = [
    b'{"content":"OK"}',
    b'{"usage":1.0}',
]

_SURFACES = [
    (
        "/v1/chat/completions",
        {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    ),
    (
        "/v1/chat/completions",
        {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ),
    (
        "/v1/responses",
        {"model": "claude-sonnet-4-5", "input": "Hello", "stream": False},
    ),
    (
        "/v1/responses",
        {"model": "claude-sonnet-4-5", "input": "Hello", "stream": True},
    ),
    (
        "/v1/messages",
        {
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    ),
    (
        "/v1/messages",
        {
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ),
]

_SURFACE_IDS = [
    "openai-chat-nonstreaming",
    "openai-chat-streaming",
    "openai-responses-nonstreaming",
    "openai-responses-streaming",
    "anthropic-messages-nonstreaming",
    "anthropic-messages-streaming",
]


def _headers(api_key: str) -> dict[str, str]:
    """Build an authorization header accepted by every API surface.

    Args:
        api_key: Test gateway API key.

    Returns:
        Bearer authorization headers.
    """
    return {"Authorization": f"Bearer {api_key}"}


def _success_response() -> AsyncMock:
    """Build a minimal successful Kiro event-stream response.

    Returns:
        A mocked ``httpx.Response`` with status 200 and a short byte stream.
    """
    success = AsyncMock(spec=httpx.Response)
    success.status_code = 200

    async def iter_success_bytes() -> AsyncIterator[bytes]:
        """Yield a minimal successful Kiro event stream."""
        for chunk in _CONTENT_CHUNKS:
            yield chunk

    success.aiter_bytes = iter_success_bytes
    success.aclose = AsyncMock()
    return success


@contextmanager
def _recording_transport() -> Iterator[List[dict[str, Any]]]:
    """Mock the Kiro transport and record every outbound request.

    Yields:
        A list that receives one ``{"method", "url", "headers"}`` dict per
        upstream request, regardless of whether the client used the streaming
        (``build_request``/``send``) or non-streaming (``request``) code path.
    """
    sent: List[dict[str, Any]] = []

    def record(method: str, url: str, **kwargs: Any) -> Mock:
        """Record one outbound request and return a placeholder request object."""
        sent.append(
            {"method": method, "url": str(url), "headers": dict(kwargs.get("headers") or {})}
        )
        return Mock()

    async def record_request(method: str, url: str, **kwargs: Any) -> AsyncMock:
        """Record a non-streaming request and return a successful response."""
        record(method, url, **kwargs)
        return _success_response()

    transport = AsyncMock(spec=httpx.AsyncClient)
    transport.is_closed = False
    transport.build_request = Mock(side_effect=record)
    transport.send = AsyncMock(side_effect=lambda *args, **kwargs: _success_response())
    transport.request = AsyncMock(side_effect=record_request)

    with patch.object(
        KiroHttpClient, "_get_client", new=AsyncMock(return_value=transport)
    ):
        yield sent


def _post(test_client: Any, api_key: str, endpoint: str, body: dict) -> httpx.Response:
    """Send one request through the gateway, reading streaming bodies fully.

    Args:
        test_client: FastAPI TestClient.
        api_key: Test gateway API key.
        endpoint: Gateway endpoint path.
        body: JSON request body.

    Returns:
        The completed gateway response.
    """
    return test_client.post(endpoint, headers=_headers(api_key), json=body)


@pytest.mark.parametrize("account_system", [False, True], ids=["legacy", "account-system"])
@pytest.mark.parametrize(("endpoint", "body"), _SURFACES, ids=_SURFACE_IDS)
def test_generation_uses_canonical_root_path_on_every_surface(
    test_client: Any,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
    account_system: bool,
) -> None:
    """What it does: asserts every generation call targets ``POST {host}/``.

    Purpose: Prevent regression to the legacy ``/generateAssistantResponse`` path
    that path-allow-listing proxies reject with an HTML 404, across all API
    surfaces, both client modes, and both account execution paths.
    """
    # Arrange
    test_client.app.state.account_system = account_system

    # Act
    with _recording_transport() as sent:
        response = _post(test_client, valid_proxy_api_key, endpoint, body)

    # Assert
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    assert sent[0]["method"] == "POST"
    assert sent[0]["url"] == _CANONICAL_URL
    assert not sent[0]["url"].endswith(_LEGACY_PATH)
    assert sent[0]["headers"].get("x-amz-target") == _OPERATION_TARGET
    assert sent[0]["headers"].get("Content-Type") == "application/x-amz-json-1.0"


@pytest.mark.parametrize(("endpoint", "body"), _SURFACES, ids=_SURFACE_IDS)
def test_generation_path_override_applies_on_every_surface(
    test_client: Any,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What it does: honors ``KIRO_GENERATE_ASSISTANT_RESPONSE_PATH`` everywhere.

    Purpose: Operators whose network path needs the legacy alias can restore it
    via configuration, and the override must reach all API surfaces uniformly.
    """
    # Arrange
    monkeypatch.setattr(
        "kiro.config.KIRO_GENERATE_ASSISTANT_RESPONSE_PATH", _LEGACY_PATH
    )

    # Act
    with _recording_transport() as sent:
        response = _post(test_client, valid_proxy_api_key, endpoint, body)

    # Assert
    assert response.status_code == 200, response.text
    assert [entry["url"] for entry in sent] == [f"{_RUNTIME_HOST}{_LEGACY_PATH}"]
