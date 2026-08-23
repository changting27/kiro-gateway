# -*- coding: utf-8 -*-

"""Cross-surface integration tests for transient INVALID_MODEL_ID retries.

These tests use the real shared ``KiroHttpClient.request_with_retry`` loop while
mocking its transport. They prove consistent recovery across OpenAI Chat,
OpenAI Responses, and Anthropic Messages in client streaming and non-streaming
modes without any real network or credential access.
"""

import json
from contextlib import contextmanager
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from kiro.http_client import KiroHttpClient


_CONTENT_CHUNKS = [
    b'{"content":"Recovered"}',
    b'{"usage":1.0}',
]


def _headers(api_key: str) -> dict[str, str]:
    """Build an authorization header for every compatible API surface.

    Args:
        api_key: Test gateway API key.

    Returns:
        Bearer authorization headers accepted by all gateway routes.
    """
    return {"Authorization": f"Bearer {api_key}"}


@contextmanager
def _transient_invalid_model_transport() -> Iterator[
    tuple[AsyncMock, AsyncMock, AsyncMock, AsyncMock]
]:
    """Mock one transient INVALID_MODEL_ID followed by a healthy Kiro stream.

    Yields:
        Tuple containing the fake transport client and both upstream responses.
    """
    invalid = AsyncMock(spec=httpx.Response)
    invalid.status_code = 400
    invalid.aread = AsyncMock(
        return_value=json.dumps(
            {
                "message": "Invalid model ID. Please select a different model to continue.",
                "reason": "INVALID_MODEL_ID",
            }
        ).encode("utf-8")
    )
    invalid.aclose = AsyncMock()

    success = AsyncMock(spec=httpx.Response)
    success.status_code = 200

    async def iter_success_bytes() -> AsyncIterator[bytes]:
        """Yield a minimal successful Kiro event stream."""
        for chunk in _CONTENT_CHUNKS:
            yield chunk

    success.aiter_bytes = iter_success_bytes
    success.aclose = AsyncMock()

    transport = AsyncMock(spec=httpx.AsyncClient)
    transport.is_closed = False
    transport.build_request = Mock(side_effect=lambda *args, **kwargs: Mock())
    transport.send = AsyncMock(side_effect=[invalid, success])

    with patch.object(
        KiroHttpClient,
        "_get_client",
        new=AsyncMock(return_value=transport),
    ):
        with patch("kiro.http_client.random.uniform", return_value=0.0):
            with patch(
                "kiro.http_client.asyncio.sleep", new_callable=AsyncMock
            ) as sleep:
                yield transport, invalid, success, sleep


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
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
            {
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "stream": False,
            },
        ),
        (
            "/v1/responses",
            {
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "stream": True,
            },
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
    ],
    ids=[
        "openai-chat-nonstreaming",
        "openai-chat-streaming",
        "openai-responses-nonstreaming",
        "openai-responses-streaming",
        "anthropic-messages-nonstreaming",
        "anthropic-messages-streaming",
    ],
)
def test_transient_invalid_model_recovers_on_every_api_surface(
    test_client,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
) -> None:
    """What it does: recovers the same transient 400 on all API/mode pairs.

    Purpose: Enforce complete feature consistency and prove routes rely on the
    shared retry implementation rather than protocol-specific patches.
    """
    with _transient_invalid_model_transport() as (
        transport,
        invalid,
        _success,
        sleep,
    ):
        response = test_client.post(
            endpoint,
            headers=_headers(valid_proxy_api_key),
            json=body,
        )

    assert response.status_code == 200, response.text
    assert transport.send.await_count == 2
    assert transport.build_request.call_count == 2
    invalid.aread.assert_awaited_once()
    invalid.aclose.assert_awaited_once()
    sleep.assert_awaited_once()
