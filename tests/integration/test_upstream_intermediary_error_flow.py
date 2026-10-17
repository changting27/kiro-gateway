# -*- coding: utf-8 -*-

"""Cross-surface integration tests for errors produced by network intermediaries.

Regression context: a reverse proxy in front of the Kiro runtime host answered a
generation request with an HTML ``404 Not Found`` page (no AWS request id). The
gateway forwarded the 404, and Claude Code reported "model may not exist" even
though the model was valid and Kiro never saw the request.

These tests drive the real routes, the real ``KiroHttpClient`` and the real error
parser with only the transport mocked. They assert, for OpenAI Chat, OpenAI
Responses, and Anthropic Messages, in streaming and non-streaming modes, and for
both the legacy single-account path and the Account System path, that:

* a markup response without ``x-amzn-requestid`` becomes an actionable 502 that
  never leaks the raw HTML and never fails over or penalizes the account;
* genuine Kiro errors (JSON, or any body carrying ``x-amzn-requestid``) keep
  their original status code and message.

No real network or credentials are used.
"""

from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable, Iterator, List
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from kiro.account_errors import ErrorType
from kiro.http_client import KiroHttpClient
from kiro.kiro_errors import UPSTREAM_INTERMEDIARY_REASON


_HTML_404_BODY = (
    b"<html>\r\n<head><title>404 Not Found</title></head>\r\n<body>\r\n"
    b"<center><h1>404 Not Found</h1></center>\r\n<hr><center>openresty</center>\r\n"
    b"</body>\r\n</html>\r\n"
)
_HTML_HEADERS = {"content-type": "text/html", "server": "openresty"}
_AWS_JSON_404_BODY = (
    b'{"__type":"UnknownOperationException","message":"Unknown operation requested."}'
)
_AWS_REQUEST_ID = {"x-amzn-requestid": "00000000-0000-0000-0000-000000000000"}

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


def _error_response_factory(
    status_code: int, headers: dict[str, str], body: bytes
) -> Callable[..., httpx.Response]:
    """Build a factory that returns a fresh real ``httpx.Response`` per call.

    Args:
        status_code: Upstream HTTP status code.
        headers: Upstream response headers.
        body: Upstream response body.

    Returns:
        A callable usable as a mocked transport side effect.
    """

    def make(*_args: Any, **_kwargs: Any) -> httpx.Response:
        """Return a new upstream error response."""
        return httpx.Response(status_code, headers=headers, content=body)

    return make


@contextmanager
def _error_transport(
    make_response: Callable[..., httpx.Response],
) -> Iterator[List[str]]:
    """Mock the Kiro transport so every request returns the given error.

    Args:
        make_response: Factory producing the upstream error response.

    Yields:
        A list receiving the URL of every upstream request.
    """
    sent: List[str] = []

    def record(method: str, url: str, **_kwargs: Any) -> Mock:
        """Record one outbound streaming request."""
        sent.append(str(url))
        return Mock()

    async def send(*args: Any, **kwargs: Any) -> httpx.Response:
        """Return the configured upstream error for a streaming request."""
        return make_response(*args, **kwargs)

    async def request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Record and answer a non-streaming request."""
        sent.append(str(url))
        return make_response()

    transport = AsyncMock(spec=httpx.AsyncClient)
    transport.is_closed = False
    transport.build_request = Mock(side_effect=record)
    transport.send = AsyncMock(side_effect=send)
    transport.request = AsyncMock(side_effect=request)

    with patch.object(
        KiroHttpClient, "_get_client", new=AsyncMock(return_value=transport)
    ):
        yield sent


def _error_message(endpoint: str, payload: dict) -> str:
    """Extract the client-facing error message for a surface's envelope.

    Args:
        endpoint: Gateway endpoint that produced the error.
        payload: Parsed JSON error body.

    Returns:
        The error message string.
    """
    if endpoint == "/v1/messages":
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "api_error"
        return payload["error"]["message"]
    assert payload["error"]["type"] == "kiro_api_error"
    return payload["error"]["message"]


@pytest.mark.parametrize("account_system", [False, True], ids=["legacy", "account-system"])
@pytest.mark.parametrize(("endpoint", "body"), _SURFACES, ids=_SURFACE_IDS)
def test_intermediary_html_404_becomes_actionable_502(
    test_client: Any,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
    account_system: bool,
) -> None:
    """What it does: sends a proxy HTML 404 (no AWS request id) upstream.

    Purpose: The client must get a 502 that names the network intermediary and
    the configuration knob, instead of a 404 misread as "model does not exist",
    and the raw HTML must not leak. The request must not be retried or failed
    over, and the account must not be penalized.
    """
    # Arrange
    test_client.app.state.account_system = account_system
    account_manager = test_client.app.state.account_manager
    report_failure = AsyncMock()
    make_response = _error_response_factory(404, _HTML_HEADERS, _HTML_404_BODY)

    # Act
    with patch.object(account_manager, "report_failure", new=report_failure):
        with _error_transport(make_response) as sent:
            response = test_client.post(
                endpoint,
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json=body,
            )

    # Assert
    assert response.status_code == 502, response.text
    payload = response.json()
    message = _error_message(endpoint, payload)
    assert "network intermediary" in message
    assert "KIRO_GENERATE_ASSISTANT_RESPONSE_PATH" in message
    assert "upstream HTTP 404" in message
    assert "<html" not in response.text.lower()
    assert "openresty" not in response.text.lower()
    if endpoint != "/v1/messages":
        assert payload["error"]["code"] == 502
    assert len(sent) == 1, "intermediary errors must not be retried or failed over"
    if account_system:
        report_failure.assert_awaited_once()
        args = report_failure.await_args.args
        assert args[2] == ErrorType.FATAL
        assert args[3] == 404
        assert args[4] == UPSTREAM_INTERMEDIARY_REASON
    else:
        report_failure.assert_not_awaited()


@pytest.mark.parametrize(("endpoint", "body"), _SURFACES, ids=_SURFACE_IDS)
def test_intermediary_status_normally_recoverable_is_not_failed_over(
    test_client: Any,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
) -> None:
    """What it does: sends a proxy HTML 402 through the Account System path.

    Purpose: A Kiro 402 (quota) is RECOVERABLE and triggers account failover and
    a circuit-breaker penalty. The same status from an intermediary says nothing
    about the account, so it must be classified FATAL and surfaced as a 502.
    """
    # Arrange
    test_client.app.state.account_system = True
    account_manager = test_client.app.state.account_manager
    report_failure = AsyncMock()
    make_response = _error_response_factory(402, _HTML_HEADERS, _HTML_404_BODY)

    # Act
    with patch.object(account_manager, "report_failure", new=report_failure):
        with _error_transport(make_response) as sent:
            response = test_client.post(
                endpoint,
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json=body,
            )

    # Assert
    assert response.status_code == 502, response.text
    assert "network intermediary" in _error_message(endpoint, response.json())
    assert len(sent) == 1
    report_failure.assert_awaited_once()
    assert report_failure.await_args.args[2] == ErrorType.FATAL
    assert report_failure.await_args.args[4] == UPSTREAM_INTERMEDIARY_REASON


@pytest.mark.parametrize("account_system", [False, True], ids=["legacy", "account-system"])
@pytest.mark.parametrize(("endpoint", "body"), _SURFACES, ids=_SURFACE_IDS)
def test_aws_json_404_keeps_original_status(
    test_client: Any,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
    account_system: bool,
) -> None:
    """What it does: sends a genuine AWS JSON 404 with an AWS request id.

    Purpose: Regression guard; real Kiro errors keep their status and message,
    so the intermediary rule never masks what Kiro actually said.
    """
    # Arrange
    test_client.app.state.account_system = account_system
    make_response = _error_response_factory(
        404, {"content-type": "application/x-amz-json-1.0", **_AWS_REQUEST_ID},
        _AWS_JSON_404_BODY,
    )

    # Act
    with _error_transport(make_response) as sent:
        response = test_client.post(
            endpoint,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json=body,
        )

    # Assert
    assert response.status_code == 404, response.text
    message = _error_message(endpoint, response.json())
    assert message == "Unknown operation requested."
    assert len(sent) == 1


@pytest.mark.parametrize(("endpoint", "body"), _SURFACES, ids=_SURFACE_IDS)
def test_html_body_with_aws_request_id_is_not_treated_as_intermediary(
    test_client: Any,
    valid_proxy_api_key: str,
    endpoint: str,
    body: dict,
) -> None:
    """What it does: sends an HTML 404 that carries ``x-amzn-requestid``.

    Purpose: The AWS request id proves AWS processed the request, so the gateway
    must not reinterpret it as an intermediary error (conservative rule).
    """
    # Arrange
    make_response = _error_response_factory(
        404, {**_HTML_HEADERS, **_AWS_REQUEST_ID}, _HTML_404_BODY
    )

    # Act
    with _error_transport(make_response):
        response = test_client.post(
            endpoint,
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json=body,
        )

    # Assert
    assert response.status_code == 404, response.text
    assert "network intermediary" not in response.text
