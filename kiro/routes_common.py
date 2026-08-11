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
Shared request-execution machinery for the OpenAI-compatible endpoints.

Both ``/v1/chat/completions`` and ``/v1/responses`` translate a client request
into a single Kiro ``generateAssistantResponse`` call and then re-stream (or
collect) the Kiro event stream back to the client. The surrounding plumbing is
identical for both endpoints:

* Account-System failover loop (try each account, classify errors as FATAL vs
  RECOVERABLE, honour Circuit Breaker) **or** the legacy single-account path.
* Per-request ``KiroHttpClient`` selection (per-request client for streaming to
  avoid CLOSE_WAIT leaks, shared pooled client for non-streaming).
* Uniform error shaping (Kiro error → enhanced error → OpenAI-style body) and
  debug-log flushing.
* The streaming wrapper that owns client-disconnect handling, upstream cleanup,
  access logging, and debug-buffer flushing.

Only three things differ between the two endpoints: how the Kiro payload is
built, how the Kiro stream is formatted for the client (SSE), and how a
non-streaming response is collected. Those are injected as callables so this
module stays endpoint-agnostic. This is the single source of truth for the
failover/error behaviour — endpoints must not re-implement it.
"""

import functools
import json
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from kiro.config import PROFILE_ARN
from kiro.http_client import KiroHttpClient
from kiro.streaming_core import collect_nonstreaming_with_retry
from kiro.utils import generate_conversation_id
from kiro.kiro_errors import enhance_kiro_error

# Import debug_logger (optional — may be unavailable in minimal deployments)
try:
    from kiro.debug_logger import debug_logger
except ImportError:  # pragma: no cover - debug logger is always present in practice
    debug_logger = None


# ==================================================================================================
# Callback type aliases
# ==================================================================================================

# (conversation_id, profile_arn) -> Kiro payload dict. May raise ValueError for
# unrecoverable request-shape problems (surfaced to the client as HTTP 400).
PayloadBuilder = Callable[[str, str], Dict[str, Any]]

# Produces the client-facing SSE chunk stream for one Kiro response. Owns
# first-token-retry and API-specific formatting. Called with keyword arguments:
#   http_client, initial_response, make_retry_request, auth_manager, model_cache
StreamFactory = Callable[..., AsyncGenerator[str, None]]

# Collects a single (already status-checked) Kiro response into the final
# client JSON body. Called with (http_client, response, auth_manager, model_cache).
# Wrapped by the executor in collect_nonstreaming_with_retry for mid-stream recovery.
ResponseCollector = Callable[..., Awaitable[Dict[str, Any]]]

# (error_info, status_code, context_limit) -> (http_status, error_body_dict).
# Lets each endpoint attach its own token estimate / max-output value while
# sharing the executor's context-limit lookup.
ErrorResponseBuilder = Callable[[Any, int, int], Tuple[int, Dict[str, Any]]]


def _build_kiro_url(auth_manager: Any) -> str:
    """
    Build the Kiro generateAssistantResponse URL for an account's auth manager.

    Args:
        auth_manager: The account's KiroAuthManager (provides ``api_host``).

    Returns:
        Fully-qualified endpoint URL.
    """
    return f"{auth_manager.api_host}/generateAssistantResponse"


async def _read_error_text(response: httpx.Response) -> str:
    """
    Read an upstream error body defensively.

    Args:
        response: The non-200 upstream response.

    Returns:
        Decoded error text, or a placeholder if the body could not be read.
    """
    try:
        error_content = await response.aread()
    except (httpx.HTTPError, httpx.StreamError, RuntimeError):
        error_content = b"Unknown error"
    return error_content.decode("utf-8", errors="replace")


def _make_stream_wrapper(
    *,
    endpoint: str,
    http_client: KiroHttpClient,
    stream_source: AsyncGenerator[str, None],
) -> Callable[[], AsyncGenerator[str, None]]:
    """
    Wrap a client-facing SSE generator with the shared lifecycle handling.

    The wrapper is responsible for the concerns that are identical across every
    streaming endpoint: swallowing client disconnects (GeneratorExit), emitting a
    terminal ``data: [DONE]`` marker on error so clients do not hang, always
    closing the per-request HTTP client, writing the access log line, and
    flushing or discarding debug buffers.

    Args:
        endpoint: Endpoint label for log lines (e.g. ``/v1/responses``).
        http_client: The per-request Kiro HTTP client to close when done.
        stream_source: The already-constructed API-specific SSE generator.

    Returns:
        A zero-argument async generator function suitable for StreamingResponse.
    """

    async def stream_wrapper() -> AsyncGenerator[str, None]:
        streaming_error: Optional[Exception] = None
        client_disconnected = False
        try:
            async for chunk in stream_source:
                yield chunk
        except GeneratorExit:
            # Client disconnected - normal, not an error.
            client_disconnected = True
            logger.debug(f"Client disconnected during streaming (GeneratorExit, {endpoint})")
        except (httpx.HTTPError, HTTPException, ValueError, RuntimeError) as e:
            streaming_error = e
            # Best-effort terminal marker so the client stops waiting.
            try:
                yield "data: [DONE]\n\n"
            except (GeneratorExit, RuntimeError):
                pass
            raise
        finally:
            await http_client.close()
            if streaming_error:
                error_type = type(streaming_error).__name__
                error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                logger.error(
                    f"HTTP 500 - POST {endpoint} (streaming) - [{error_type}] {error_msg[:100]}"
                )
            elif client_disconnected:
                logger.info(f"HTTP 200 - POST {endpoint} (streaming) - client disconnected")
            else:
                logger.info(f"HTTP 200 - POST {endpoint} (streaming) - completed")
            if debug_logger:
                if streaming_error:
                    debug_logger.flush_on_error(500, str(streaming_error))
                else:
                    debug_logger.discard_buffers()

    return stream_wrapper


async def _dispatch_success(
    *,
    endpoint: str,
    stream: bool,
    http_client: KiroHttpClient,
    response: httpx.Response,
    make_retry_request: Callable[[], Awaitable[httpx.Response]],
    auth_manager: Any,
    model_cache: Any,
    make_stream: StreamFactory,
    collect_response: ResponseCollector,
) -> Any:
    """
    Turn a successful (HTTP 200) Kiro response into a client response.

    Streaming returns a StreamingResponse wrapping the API-specific SSE
    generator; non-streaming collects the full response (with mid-stream retry)
    and returns a JSONResponse. The per-request HTTP client is closed by the
    stream wrapper (streaming) or inline (non-streaming).

    Args:
        endpoint: Endpoint label for logging.
        stream: Whether the client requested streaming.
        http_client: The per-request Kiro HTTP client.
        response: The already-validated HTTP 200 upstream response.
        make_retry_request: Factory that re-issues the upstream POST for retries.
        auth_manager: Account auth manager (passed through to callbacks).
        model_cache: Account model cache (passed through to callbacks).
        make_stream: API-specific SSE stream factory.
        collect_response: API-specific single-response collector.

    Returns:
        A StreamingResponse (streaming) or JSONResponse (non-streaming).
    """
    if stream:
        stream_source = make_stream(
            http_client=http_client,
            initial_response=response,
            make_retry_request=make_retry_request,
            auth_manager=auth_manager,
            model_cache=model_cache,
        )
        wrapper = _make_stream_wrapper(
            endpoint=endpoint,
            http_client=http_client,
            stream_source=stream_source,
        )
        return StreamingResponse(wrapper(), media_type="text/event-stream")

    # Non-streaming: collect the whole response. A mid-response disconnect
    # (UpstreamStreamError) is safe to retry here because no bytes have reached
    # the client yet - the JSONResponse below is only built after full collection.
    client_body = await collect_nonstreaming_with_retry(
        initial_response=response,
        make_request=make_retry_request,
        collect=lambda _resp: collect_response(
            http_client=http_client,
            response=_resp,
            auth_manager=auth_manager,
            model_cache=model_cache,
        ),
    )
    await http_client.close()
    logger.info(f"HTTP 200 - POST {endpoint} (non-streaming) - completed")
    if debug_logger:
        debug_logger.discard_buffers()
    return JSONResponse(content=client_body)


async def execute_kiro_request(
    *,
    request: Request,
    model: str,
    stream: bool,
    endpoint: str,
    build_payload: PayloadBuilder,
    make_stream: StreamFactory,
    collect_response: ResponseCollector,
    build_error_response: ErrorResponseBuilder,
) -> Any:
    """
    Execute a Kiro generateAssistantResponse call with full failover + error handling.

    This is the shared engine behind ``/v1/chat/completions`` and
    ``/v1/responses``. It reproduces the Account-System failover loop and the
    legacy single-account path exactly, delegating only the three
    endpoint-specific concerns (payload build, SSE formatting, response
    collection) plus error-body shaping to the injected callables.

    Args:
        request: FastAPI request (for ``app.state`` — account manager, http client).
        model: Resolved client model name (used for account selection + limits).
        stream: Whether the client requested a streaming response.
        endpoint: Endpoint label used in every log line (e.g. ``/v1/responses``).
        build_payload: ``(conversation_id, profile_arn) -> kiro_payload``. Raising
            ``ValueError`` maps to HTTP 400.
        make_stream: API-specific SSE stream factory (owns first-token-retry).
        collect_response: API-specific single-response collector (wrapped in
            mid-stream retry by this executor).
        build_error_response: ``(error_info, status, context_limit) -> (status, body)``
            producing the OpenAI-style error payload for the client.

    Returns:
        A FastAPI ``StreamingResponse`` or ``JSONResponse``.

    Raises:
        HTTPException: For 400 (bad request shape), auth/availability failures,
            and unrecoverable upstream/internal errors, mirroring the original
            per-endpoint behaviour.
    """
    if request.app.state.account_system:
        return await _execute_with_account_system(
            request=request,
            model=model,
            stream=stream,
            endpoint=endpoint,
            build_payload=build_payload,
            make_stream=make_stream,
            collect_response=collect_response,
            build_error_response=build_error_response,
        )
    return await _execute_legacy(
        request=request,
        model=model,
        stream=stream,
        endpoint=endpoint,
        build_payload=build_payload,
        make_stream=make_stream,
        collect_response=collect_response,
        build_error_response=build_error_response,
    )


async def _execute_with_account_system(
    *,
    request: Request,
    model: str,
    stream: bool,
    endpoint: str,
    build_payload: PayloadBuilder,
    make_stream: StreamFactory,
    collect_response: ResponseCollector,
    build_error_response: ErrorResponseBuilder,
) -> Any:
    """
    Account-System failover path: try each available account until one succeeds.

    See ``execute_kiro_request`` for argument semantics. This mirrors the
    original ``chat_completions`` failover loop verbatim, including error
    classification (FATAL vs RECOVERABLE), Circuit Breaker success/failure
    reporting, and single-vs-multi-account terminal behaviour.
    """
    from kiro.account_errors import classify_error, ErrorType

    account_manager = request.app.state.account_manager
    all_accounts = list(account_manager._accounts.keys())
    max_attempts = len(all_accounts) * 2  # Full circle with margin

    last_error_message: Optional[str] = None
    last_error_status: Optional[int] = None
    tried_accounts = set()

    for _attempt in range(max_attempts):
        account = await account_manager.get_next_account(
            model, exclude_accounts=tried_accounts
        )

        if account is None:
            # All accounts unavailable in this loop.
            if len(all_accounts) == 1:
                raise HTTPException(
                    status_code=last_error_status or 503,
                    detail=last_error_message or "Account unavailable",
                )
            detail = "No available accounts for this model."
            if last_error_message:
                detail += f" Error from last account: {last_error_message}"
            raise HTTPException(status_code=503, detail=detail)

        tried_accounts.add(account.id)
        auth_manager = account.auth_manager
        model_cache = account.model_cache

        conversation_id = generate_conversation_id()
        profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

        try:
            kiro_payload = build_payload(conversation_id, profile_arn_for_payload)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        _log_kiro_payload(kiro_payload)

        url = _build_kiro_url(auth_manager)
        logger.debug(f"Kiro API URL: {url} (account: {account.id})")

        if stream:
            http_client = KiroHttpClient(auth_manager, shared_client=None)
        else:
            http_client = KiroHttpClient(
                auth_manager, shared_client=request.app.state.http_client
            )

        make_retry_request = functools.partial(
            http_client.request_with_retry, "POST", url, kiro_payload, stream=True
        )

        try:
            response = await http_client.request_with_retry(
                "POST", url, kiro_payload, stream=True
            )

            if response.status_code == 200:
                await account_manager.report_success(account.id, model)
                return await _dispatch_success(
                    endpoint=endpoint,
                    stream=stream,
                    http_client=http_client,
                    response=response,
                    make_retry_request=make_retry_request,
                    auth_manager=auth_manager,
                    model_cache=model_cache,
                    make_stream=make_stream,
                    collect_response=collect_response,
                )

            # Non-200: classify and decide failover vs. return.
            error_text = await _read_error_text(response)
            await http_client.close()

            error_info = _enhance_error(error_text)
            error_reason = error_info.reason
            last_error_message = error_info.user_message
            last_error_status = response.status_code
            logger.debug(
                f"Original Kiro error: {error_info.original_message} "
                f"(reason: {error_info.reason})"
            )

            error_type = classify_error(response.status_code, error_reason)

            if error_type == ErrorType.FATAL:
                await account_manager.report_failure(
                    account.id, model, error_type, response.status_code, error_reason
                )
                logger.warning(
                    f"HTTP {response.status_code} - POST {endpoint} - {last_error_message[:100]}"
                )
                if debug_logger:
                    debug_logger.flush_on_error(response.status_code, last_error_message)

                context_limit = account_manager.get_model_max_input_tokens(model)
                overflow_status, error_body = build_error_response(
                    error_info, response.status_code, context_limit
                )
                return JSONResponse(status_code=overflow_status, content=error_body)

            # RECOVERABLE - try next account.
            await account_manager.report_failure(
                account.id, model, error_type, response.status_code, error_reason
            )
            if len(all_accounts) == 1:
                break
            continue

        except HTTPException as e:
            await http_client.close()
            if e.status_code in (502, 504):
                await account_manager.report_failure(
                    account.id, model, ErrorType.RECOVERABLE, e.status_code, None
                )
                last_error_message = str(e.detail)
                last_error_status = e.status_code
                if len(all_accounts) == 1:
                    break
                logger.warning(f"Network error on account {account.id}, trying next account")
                continue
            logger.error(f"HTTP {e.status_code} - POST {endpoint} - {e.detail}")
            if debug_logger:
                debug_logger.flush_on_error(e.status_code, str(e.detail))
            raise
        except Exception as e:
            await http_client.close()
            logger.error(f"Internal error: {e}", exc_info=True)
            logger.error(f"HTTP 500 - POST {endpoint} - {str(e)[:100]}")
            if debug_logger:
                debug_logger.flush_on_error(500, str(e))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    # All attempts exhausted.
    if len(all_accounts) == 1:
        raise HTTPException(status_code=last_error_status, detail=last_error_message)
    detail = "All accounts failed after full circle."
    if last_error_message:
        detail += f" Error from last account: {last_error_message}"
    raise HTTPException(status_code=503, detail=detail)


async def _execute_legacy(
    *,
    request: Request,
    model: str,
    stream: bool,
    endpoint: str,
    build_payload: PayloadBuilder,
    make_stream: StreamFactory,
    collect_response: ResponseCollector,
    build_error_response: ErrorResponseBuilder,
) -> Any:
    """
    Legacy single-account path (Account System disabled): no failover.

    See ``execute_kiro_request`` for argument semantics. Mirrors the original
    ``chat_completions`` legacy branch, including error shaping and debug-log
    flushing.
    """
    account = request.app.state.account_manager.get_first_account()
    if not account.auth_manager:
        logger.error("No initialized accounts available (legacy mode)")
        raise HTTPException(503, "No initialized accounts available")

    auth_manager = account.auth_manager
    model_cache = account.model_cache

    conversation_id = generate_conversation_id()
    profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

    try:
        kiro_payload = build_payload(conversation_id, profile_arn_for_payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _log_kiro_payload(kiro_payload)

    url = _build_kiro_url(auth_manager)
    logger.debug(f"Kiro API URL: {url}")

    if stream:
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        http_client = KiroHttpClient(
            auth_manager, shared_client=request.app.state.http_client
        )

    make_retry_request = functools.partial(
        http_client.request_with_retry, "POST", url, kiro_payload, stream=True
    )

    try:
        response = await http_client.request_with_retry(
            "POST", url, kiro_payload, stream=True
        )

        if response.status_code != 200:
            error_text = await _read_error_text(response)
            await http_client.close()

            error_info = _enhance_error(error_text)
            error_message = error_info.user_message
            logger.debug(
                f"Original Kiro error: {error_info.original_message} "
                f"(reason: {error_info.reason})"
            )
            logger.warning(
                f"HTTP {response.status_code} - POST {endpoint} - {error_message[:100]}"
            )
            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)

            context_limit = request.app.state.account_manager.get_model_max_input_tokens(model)
            overflow_status, error_body = build_error_response(
                error_info, response.status_code, context_limit
            )
            return JSONResponse(status_code=overflow_status, content=error_body)

        return await _dispatch_success(
            endpoint=endpoint,
            stream=stream,
            http_client=http_client,
            response=response,
            make_retry_request=make_retry_request,
            auth_manager=auth_manager,
            model_cache=model_cache,
            make_stream=make_stream,
            collect_response=collect_response,
        )

    except HTTPException as e:
        await http_client.close()
        if e.status_code in (502, 504):
            logger.warning("Network error (legacy mode, no failover available)")
        logger.error(f"HTTP {e.status_code} - POST {endpoint} - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST {endpoint} - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


def _enhance_error(error_text: str) -> Any:
    """
    Parse and enhance a Kiro error body.

    Runs non-JSON bodies through the enhancer too, so the text-based
    context-overflow fallback can still fire.

    Args:
        error_text: Raw upstream error text.

    Returns:
        The enhanced error info object from ``enhance_kiro_error``.
    """
    try:
        error_json = json.loads(error_text)
        return enhance_kiro_error(error_json)
    except (json.JSONDecodeError, KeyError):
        return enhance_kiro_error({"message": error_text})


def _log_kiro_payload(kiro_payload: Dict[str, Any]) -> None:
    """
    Log the outgoing Kiro payload to the debug logger (best-effort).

    Args:
        kiro_payload: The assembled Kiro request payload.
    """
    try:
        kiro_request_body = json.dumps(
            kiro_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to log Kiro request: {e}")
