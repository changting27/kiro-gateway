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
HTTP client for Kiro API with retry logic support.

Handles:
- 400 + INVALID_MODEL_ID: bounded exponential backoff with jitter
- 403: automatic token refresh and retry
- 429: exponential backoff
- 5xx: exponential backoff
- Timeouts: exponential backoff

Supports both per-request clients and shared application-level client
with connection pooling for better resource management.
"""

import asyncio
import json
import random
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    BASE_RETRY_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    INVALID_MODEL_BASE_RETRY_DELAY,
    INVALID_MODEL_MAX_RETRIES,
    INVALID_MODEL_MAX_RETRY_DELAY,
    INVALID_MODEL_RETRY_JITTER_RATIO,
    MAX_RETRIES,
    SSL_VERIFY,
    STREAMING_READ_TIMEOUT,
)
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers
from kiro.network_errors import classify_network_error, get_short_error_message, NetworkErrorInfo


class KiroHttpClient:
    """
    HTTP client for Kiro API with retry logic support.
    
    Automatically handles errors and retries requests:
    - 403: refreshes token and retries
    - 429: waits with exponential backoff
    - 5xx: waits with exponential backoff
    - Timeouts: waits with exponential backoff
    
    Supports two modes of operation:
    1. Per-request client: Creates and owns its own httpx.AsyncClient
    2. Shared client: Uses an application-level shared client (recommended)
    
    Using a shared client reduces memory usage and enables connection pooling,
    which is especially important for handling concurrent requests.
    
    Attributes:
        auth_manager: Authentication manager for obtaining tokens
        client: httpx HTTP client (owned or shared)
    
    Example:
        >>> # Per-request client (legacy mode)
        >>> client = KiroHttpClient(auth_manager)
        >>> response = await client.request_with_retry(...)
        
        >>> # Shared client (recommended)
        >>> shared = httpx.AsyncClient(limits=httpx.Limits(...))
        >>> client = KiroHttpClient(auth_manager, shared_client=shared)
        >>> response = await client.request_with_retry(...)
    """
    
    def __init__(
        self,
        auth_manager: KiroAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None
    ):
        """
        Initializes the HTTP client.
        
        Args:
            auth_manager: Authentication manager
            shared_client: Optional shared httpx.AsyncClient for connection pooling.
                          If provided, this client will be used instead of creating
                          a new one. The shared client will NOT be closed by close().
        """
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client
    
    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """
        Returns or creates an HTTP client with proper timeouts.
        
        If a shared client was provided at initialization, it is returned as-is.
        Otherwise, creates a new client with appropriate timeout configuration.
        
        httpx timeouts:
        - connect: TCP handshake (DNS + TCP SYN/ACK)
        - read: waiting for data from server between chunks
        - write: sending data to server
        - pool: waiting for free connection from pool
        
        IMPORTANT: FIRST_TOKEN_TIMEOUT is NOT used here!
        It is applied in streaming_openai.py via asyncio.wait_for() to control
        the wait time for the first token from the model (retry business logic).
        
        Args:
            stream: If True, uses STREAMING_READ_TIMEOUT for read (only for new clients)
        
        Returns:
            Active HTTP client
        """
        # If using shared client, return it directly
        # Shared client should be pre-configured with appropriate timeouts
        if self._shared_client is not None:
            return self._shared_client
        
        # Create new client if needed (per-request mode)
        if self.client is None or self.client.is_closed:
            if stream:
                # For streaming:
                # - connect: 30 sec (TCP connection, usually < 1 sec)
                # - read: STREAMING_READ_TIMEOUT (300 sec) - model may "think" between chunks
                # - write/pool: standard values
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0
                )
                logger.debug(f"Creating streaming HTTP client (read_timeout={STREAMING_READ_TIMEOUT}s)")
            else:
                # For regular requests: single timeout of 300 sec
                timeout_config = httpx.Timeout(timeout=300.0)
                logger.debug("Creating non-streaming HTTP client (timeout=300s)")
            
            self.client = httpx.AsyncClient(
                timeout=timeout_config,
                follow_redirects=True,
                verify=SSL_VERIFY  # process-scoped extra CA trust (see config.get_ssl_verify)
            )
        return self.client
    
    async def close(self) -> None:
        """
        Closes the HTTP client if this instance owns it.
        
        If using a shared client, this method does nothing - the shared client
        should be closed by the application lifecycle manager.
        
        Uses graceful exception handling to prevent errors during cleanup
        from masking the original exception in finally blocks.
        """
        # Don't close shared clients - they're managed by the application
        if not self._owns_client:
            return
        
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                # Log but don't propagate - we're in cleanup code
                # Propagating here could mask the original exception
                logger.warning(f"Error closing HTTP client: {e}")
    
    @staticmethod
    async def _get_error_reason(response: httpx.Response) -> Optional[str]:
        """Read a Kiro error response and return its structured reason.

        Streaming error responses must be consumed before their JSON body can be
        inspected. ``httpx`` caches the consumed bytes, so route-level error shaping
        can safely read the response again after retries are exhausted.

        Args:
            response: Non-success response returned by Kiro.

        Returns:
            Top-level string ``reason`` value, or ``None`` for non-JSON/malformed
            responses.
        """
        try:
            content = await response.aread()
        except httpx.HTTPError as exc:
            logger.debug(
                f"Could not read Kiro error response while checking retry reason: {exc}"
            )
            return None

        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return None

        if not isinstance(payload, dict):
            return None

        reason = payload.get("reason")
        return reason if isinstance(reason, str) else None

    @staticmethod
    async def _close_response_for_retry(response: httpx.Response) -> None:
        """Close a failed response before opening the next retry connection.

        Args:
            response: Failed Kiro response that will not be returned to the caller.
        """
        try:
            await response.aclose()
        except httpx.HTTPError as exc:
            logger.debug(f"Could not close failed Kiro response before retry: {exc}")

    @staticmethod
    def _invalid_model_retry_delay(attempt: int) -> float:
        """Calculate bounded exponential backoff with additive jitter.

        Args:
            attempt: Zero-based request attempt that produced the transient error.

        Returns:
            Delay in seconds before the next request.
        """
        exponential_delay = min(
            INVALID_MODEL_BASE_RETRY_DELAY * (2 ** attempt),
            INVALID_MODEL_MAX_RETRY_DELAY,
        )
        jitter_limit = exponential_delay * INVALID_MODEL_RETRY_JITTER_RATIO
        jitter = random.uniform(0.0, jitter_limit) if jitter_limit > 0 else 0.0
        return min(exponential_delay + jitter, INVALID_MODEL_MAX_RETRY_DELAY)

    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        stream: bool = False
    ) -> httpx.Response:
        """Execute an HTTP request with bounded, reason-aware retries.

        Automatically handles various error types:
        - 400 + ``INVALID_MODEL_ID``: bounded jittered backoff without auth refresh
        - 403: refreshes token via ``auth_manager.force_refresh()`` and retries
        - 429: waits with exponential backoff
        - 5xx: waits with exponential backoff
        - Timeouts/network errors: waits with exponential backoff when retryable

        Other 400 responses return immediately. For streaming,
        ``STREAMING_READ_TIMEOUT`` is used for waiting between chunks; first-token
        timeout is controlled separately by the streaming layer.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Request URL.
            json_data: Optional JSON body for POST/PUT/PATCH.
            params: Optional query parameters.
            stream: Whether to open the upstream response as a stream.

        Returns:
            Successful response, or the final original HTTP error response after a
            status-code retry budget is exhausted.

        Raises:
            HTTPException: When retryable transport errors exhaust their budget.
        """
        general_max_attempts = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        max_attempts = max(general_max_attempts, INVALID_MODEL_MAX_RETRIES)

        client = await self._get_client(stream=stream)
        last_error_info: Optional[NetworkErrorInfo] = None

        for attempt in range(max_attempts):
            try:
                token = await self.auth_manager.get_access_token()
                headers = get_kiro_headers(self.auth_manager, token)

                request_kwargs = {"headers": headers}
                if json_data is not None:
                    request_kwargs["content"] = json.dumps(json_data).encode()
                if params is not None:
                    request_kwargs["params"] = params

                if stream:
                    # Per-request streaming clients plus Connection: close prevent
                    # CLOSE_WAIT leaks while still allowing a fresh retry request.
                    headers["Connection"] = "close"
                    req = client.build_request(method, url, **request_kwargs)
                    logger.debug("Sending request to Kiro API...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug("Sending request to Kiro API...")
                    response = await client.request(method, url, **request_kwargs)

                if response.status_code == 200:
                    return response

                if response.status_code == 400:
                    reason = await self._get_error_reason(response)
                    if reason == "INVALID_MODEL_ID":
                        if attempt >= INVALID_MODEL_MAX_RETRIES - 1:
                            logger.warning(
                                "Retries exhausted for transient Kiro "
                                f"INVALID_MODEL_ID ({attempt + 1}/"
                                f"{INVALID_MODEL_MAX_RETRIES}); returning final 400"
                            )
                            return response

                        delay = self._invalid_model_retry_delay(attempt)
                        logger.warning(
                            "Received transient Kiro INVALID_MODEL_ID; "
                            f"retrying in {delay:.2f}s (attempt {attempt + 1}/"
                            f"{INVALID_MODEL_MAX_RETRIES})"
                        )
                        await self._close_response_for_retry(response)
                        await asyncio.sleep(delay)
                        continue

                    return response

                if response.status_code == 403:
                    if attempt >= general_max_attempts - 1:
                        logger.warning(
                            f"Retries exhausted for HTTP 403 ({attempt + 1}/"
                            f"{general_max_attempts}); returning final response"
                        )
                        return response
                    logger.warning(
                        f"Received 403, refreshing token (attempt {attempt + 1}/"
                        f"{general_max_attempts})"
                    )
                    await self._close_response_for_retry(response)
                    await self.auth_manager.force_refresh()
                    continue

                if response.status_code == 429:
                    if attempt >= general_max_attempts - 1:
                        logger.warning(
                            f"Retries exhausted for HTTP 429 ({attempt + 1}/"
                            f"{general_max_attempts}); returning final response"
                        )
                        return response
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Received 429, waiting {delay}s (attempt {attempt + 1}/"
                        f"{general_max_attempts})"
                    )
                    await self._close_response_for_retry(response)
                    await asyncio.sleep(delay)
                    continue

                if 500 <= response.status_code < 600:
                    if attempt >= general_max_attempts - 1:
                        logger.warning(
                            f"Retries exhausted for HTTP {response.status_code} "
                            f"({attempt + 1}/{general_max_attempts}); returning final response"
                        )
                        return response
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Received {response.status_code}, waiting {delay}s "
                        f"(attempt {attempt + 1}/{general_max_attempts})"
                    )
                    await self._close_response_for_retry(response)
                    await asyncio.sleep(delay)
                    continue

                return response

            except httpx.TimeoutException as exc:
                error_info = classify_network_error(exc)
                last_error_info = error_info
                short_msg = get_short_error_message(error_info)

                if error_info.is_retryable and attempt < general_max_attempts - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"{short_msg} - waiting {delay}s "
                        f"(attempt {attempt + 1}/{general_max_attempts})"
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    f"{short_msg} - no more retries "
                    f"(attempt {attempt + 1}/{general_max_attempts})"
                )
                break

            except httpx.RequestError as exc:
                error_info = classify_network_error(exc)
                last_error_info = error_info
                short_msg = get_short_error_message(error_info)

                if error_info.is_retryable and attempt < general_max_attempts - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"{short_msg} - waiting {delay}s "
                        f"(attempt {attempt + 1}/{general_max_attempts})"
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    f"{short_msg} - no more retries "
                    f"(attempt {attempt + 1}/{general_max_attempts})"
                )
                break

        if last_error_info:
            error_message = last_error_info.user_message

            if last_error_info.troubleshooting_steps:
                error_message += "\n\nTroubleshooting:\n"
                for index, step in enumerate(
                    last_error_info.troubleshooting_steps, 1
                ):
                    error_message += f"{index}. {step}\n"

            error_message += (
                f"\nTechnical details: {last_error_info.technical_details}"
            )
            raise HTTPException(
                status_code=last_error_info.suggested_http_code,
                detail=error_message.strip()
            )

        if stream:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Streaming failed after {general_max_attempts} attempts. "
                    "Unknown error."
                )
            )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Request failed after {general_max_attempts} attempts. Unknown error."
            )
        )
    
    async def __aenter__(self) -> "KiroHttpClient":
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the client when exiting context."""
        await self.close()