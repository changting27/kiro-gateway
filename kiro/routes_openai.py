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
FastAPI routes for Kiro Gateway.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import (
    PROXY_API_KEY,
    APP_VERSION,
    HEALTH_CHECK_METHODS,
    DEFAULT_MAX_INPUT_TOKENS,
)
from kiro.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver
from kiro.converters_openai import build_kiro_payload
from kiro.streaming_openai import stream_kiro_to_openai, collect_stream_response, stream_with_first_token_retry
from kiro.observability import enrich_current_metrics
from kiro.config import WEB_SEARCH_ENABLED
from kiro.mcp_tools import handle_native_web_search
from typing import Optional
from kiro.tokenizer import estimate_request_tokens
from kiro.kiro_errors import build_openai_error_body

# Responses API (/v1/responses) — shares the executor with /v1/chat/completions
from kiro.routes_common import execute_kiro_request
from kiro.models_responses import ResponsesRequest
from kiro.converters_responses import (
    build_kiro_payload_from_responses,
    build_responses_tokenizer_inputs,
)
from kiro.streaming_responses import (
    stream_kiro_to_responses_with_retry,
    collect_responses_response,
    generate_response_id,
)

# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


def _estimate_openai_input_tokens(request_data) -> Optional[int]:
    """
    Best-effort input-token estimate for the context-overflow error message.

    OpenAI requests carry the system prompt as a role="system" message, so the
    tokenizer counts it as part of ``messages``. This is strictly cosmetic — the
    error builder clamps to a self-consistent value when this returns None — so
    any estimation failure is swallowed rather than masking the real API error.

    Args:
        request_data: The incoming ChatCompletionRequest.

    Returns:
        Estimated total input tokens, or None if estimation failed.
    """
    try:
        messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
        tools_for_tokenizer = (
            [tool.model_dump() for tool in request_data.tools]
            if getattr(request_data, "tools", None) else None
        )
        stats = estimate_request_tokens(
            messages=messages_for_tokenizer,
            tools=tools_for_tokenizer,
            system_prompt=None,
            apply_claude_correction=True,
        )
        return stats["total_tokens"]
    except Exception as exc:  # best-effort estimate; must never mask the real error
        logger.debug(
            f"OpenAI input-token estimate failed, using clamp fallback: {exc}"
        )
        return None


# --- Security scheme ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    Verify API key in Authorization header.
    
    Expects format: "Bearer {PROXY_API_KEY}"
    
    Args:
        auth_header: Authorization header value
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


@router.api_route("/", methods=HEALTH_CHECK_METHODS)
async def root():
    """
    Health check endpoint.

    Accepts GET and HEAD (see ``HEALTH_CHECK_METHODS``) so that uptime monitors,
    load balancers, and container health checks can probe with either method.
    For HEAD requests Starlette returns the same status and headers with an empty
    body, per HTTP semantics.

    Returns:
        Status and application version
    """
    return {
        "status": "ok",
        "message": "Kiro Gateway is running",
        "version": APP_VERSION
    }


@router.api_route("/health", methods=HEALTH_CHECK_METHODS)
async def health():
    """
    Detailed health check.

    Accepts GET and HEAD (see ``HEALTH_CHECK_METHODS``) for compatibility with
    monitoring and load-balancer probes that issue HEAD requests.

    Returns:
        Status, timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION
    }

@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    """
    Return list of available models, enriched with each model's real context window.
    
    Models are loaded at startup (blocking) and cached. This endpoint returns the
    cached list. Every model is annotated with Kiro's real ``maxInputTokens`` (exposed
    as both ``max_input_tokens`` and ``context_length`` on each OpenAIModel) so that
    OpenAI-compatible clients can align their auto-compaction threshold with the true
    upstream limit instead of a hardcoded assumption.
    
    Feature-consistency note (AGENTS.md section 10): ``/v1/models`` is a single shared
    GET endpoint. The real OpenAI and Anthropic APIs both expose model listings at this
    same path, so this one handler serves clients of both surfaces - there is no
    separate Anthropic models endpoint to keep in sync. Being a GET listing, it has no
    streaming variant. The Anthropic ``count_tokens`` endpoint is intentionally left
    untouched: it returns only ``input_tokens`` per the Anthropic schema, and the model
    listing is the correct, schema-compatible place to advertise context windows.
    
    Args:
        request: FastAPI Request for accessing app.state
    
    Returns:
        ModelList with available models in consistent format (with dots), each carrying
        its real context-window size.
    """
    logger.info("Request to /v1/models")
    
    # Resolve the available model IDs and a per-model context-window lookup that works
    # for whichever mode the gateway runs in. The lookup always returns a concrete int
    # (falling back to DEFAULT_MAX_INPUT_TOKENS) so the advertised window is never null.
    if request.app.state.account_system:
        # Account system: collect models from all initialized accounts and aggregate
        # each model's limit across every account that provides it.
        account_manager = request.app.state.account_manager
        available_model_ids = account_manager.get_all_available_models()
        
        def get_model_limit(model_id: str) -> int:
            return account_manager.get_model_max_input_tokens(model_id)
    else:
        # Legacy: use resolver and cache from the first initialized account.
        account = request.app.state.account_manager.get_first_account()
        available_model_ids = account.model_resolver.get_available_models()
        model_cache = account.model_cache
        
        def get_model_limit(model_id: str) -> int:
            if model_cache is None:
                return DEFAULT_MAX_INPUT_TOKENS
            return model_cache.get_max_input_tokens(model_id)
    
    # Build OpenAI-compatible model list, advertising the real context window per model.
    openai_models = []
    for model_id in available_model_ids:
        max_input_tokens = get_model_limit(model_id)
        openai_models.append(
            OpenAIModel(
                id=model_id,
                owned_by="anthropic",
                description="Claude model via Kiro API",
                max_input_tokens=max_input_tokens,
                context_length=max_input_tokens,
            )
        )
    
    logger.debug(
        f"Returning {len(openai_models)} models with context-window metadata "
        f"(account_system={request.app.state.account_system})"
    )
    
    return ModelList(data=openai_models)


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions endpoint - compatible with OpenAI API.
    
    Accepts requests in OpenAI format and translates them to Kiro API.
    Supports streaming and non-streaming modes.
    
    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in OpenAI ChatCompletionRequest format
    
    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode
    
    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")
    # Observability: attach request semantics to the per-request metrics (no-op if disabled).
    enrich_current_metrics(
        api="openai",
        model=request_data.model,
        stream=bool(request_data.stream),
        message_count=len(request_data.messages) if request_data.messages else 0,
        tool_count=len(request_data.tools) if request_data.tools else 0,
    )
    
    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)
    
    # Check for truncation recovery opportunities
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    from kiro.models_openai import ChatMessage
    
    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0
    
    for msg in request_data.messages:
        # Check if this is a tool_result for a truncated tool call
        if msg.role == "tool" and msg.tool_call_id:
            truncation_info = get_tool_truncation(msg.tool_call_id)
            if truncation_info:
                # Modify tool_result content to include truncation notice
                synthetic = generate_truncation_tool_result(
                    tool_name=truncation_info.tool_name,
                    tool_use_id=msg.tool_call_id,
                    truncation_info=truncation_info.truncation_info
                )
                # Prepend truncation notice to original content
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                
                # Create NEW ChatMessage object (Pydantic immutability)
                modified_msg = msg.model_copy(update={"content": modified_content})
                modified_messages.append(modified_msg)
                tool_results_modified += 1
                logger.debug(f"Modified tool_result for {msg.tool_call_id} to include truncation notice")
                continue  # Skip normal append since we already added modified version
        
        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
            truncation_info = get_content_truncation(msg.content)
            if truncation_info:
                # Add this message first
                modified_messages.append(msg)
                # Then add synthetic user message about truncation
                synthetic_user_msg = ChatMessage(
                    role="user",
                    content=generate_truncation_user_message()
                )
                modified_messages.append(synthetic_user_msg)
                content_notices_added += 1
                logger.debug(f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})")
                continue  # Skip normal append since we already added it
        
        modified_messages.append(msg)
    
    if tool_results_modified > 0 or content_notices_added > 0:
        request_data.messages = modified_messages
        logger.info(f"Truncation recovery: modified {tool_results_modified} tool_result(s), added {content_notices_added} content notice(s)")
    
    # ==============================================================================
    # WebSearch Support - Path B: Auto-Injection (MCP Tool Emulation)
    # ==============================================================================
    
    # Auto-inject web_search tool if enabled (Path B - MCP emulation)
    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []
        
        # Check if web_search already exists
        has_ws = any(
            getattr(tool, "type", None) == "function" and
            getattr(getattr(tool, "function", None), "name", None) == "web_search"
            for tool in request_data.tools
        )
        
        if not has_ws:
            from kiro.models_openai import Tool, ToolFunction
            web_search_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web for current information. Use when you need up-to-date data from the internet.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")

    # ------------------------------------------------------------------------------
    # Delegate to the shared Kiro request executor (routes_common.execute_kiro_request).
    # Only the payload builder, SSE formatter, response collector, and error-body
    # shaper are OpenAI-Chat-specific; the account-failover / legacy / error-handling
    # machinery is shared with /v1/responses (single source of truth).
    # ------------------------------------------------------------------------------
    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
    tools_for_tokenizer = (
        [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
    )

    def _build_payload(conversation_id: str, profile_arn: str) -> dict:
        """Build the Kiro payload from the OpenAI Chat request."""
        return build_kiro_payload(request_data, conversation_id, profile_arn)

    def _make_stream(*, http_client, initial_response, make_retry_request, auth_manager, model_cache):
        """Produce the OpenAI Chat SSE stream (with first-token retry)."""
        return stream_with_first_token_retry(
            make_request=make_retry_request,
            client=http_client.client,
            model=request_data.model,
            model_cache=model_cache,
            auth_manager=auth_manager,
            initial_response=initial_response,
            request_messages=messages_for_tokenizer,
            request_tools=tools_for_tokenizer,
        )

    async def _collect(*, http_client, response, auth_manager, model_cache):
        """Collect a full non-streaming OpenAI Chat response."""
        return await collect_stream_response(
            http_client.client,
            response,
            request_data.model,
            model_cache,
            auth_manager,
            request_messages=messages_for_tokenizer,
            request_tools=tools_for_tokenizer,
        )

    def _build_error(error_info, status_code: int, context_limit: int):
        """Shape a Kiro error into an OpenAI-style error body."""
        return build_openai_error_body(
            error_info,
            status_code,
            context_limit=context_limit,
            max_tokens=request_data.max_tokens or 0,
            estimated_input_tokens=_estimate_openai_input_tokens(request_data),
        )

    return await execute_kiro_request(
        request=request,
        model=request_data.model,
        stream=request_data.stream,
        endpoint="/v1/chat/completions",
        build_payload=_build_payload,
        make_stream=_make_stream,
        collect_response=_collect,
        build_error_response=_build_error,
    )


def _estimate_responses_input_tokens(request_data) -> Optional[int]:
    """
    Best-effort input-token estimate for the Responses context-overflow message.

    Mirrors ``_estimate_openai_input_tokens`` but flattens the Responses ``input``
    into tokenizer-friendly messages first. Strictly cosmetic (the error builder
    clamps to a self-consistent value when this returns None), so any estimation
    failure is swallowed rather than masking the real API error.

    Args:
        request_data: The incoming ResponsesRequest.

    Returns:
        Estimated total input tokens, or None if estimation failed.
    """
    try:
        messages_for_tokenizer, tools_for_tokenizer = build_responses_tokenizer_inputs(request_data)
        stats = estimate_request_tokens(
            messages=messages_for_tokenizer,
            tools=tools_for_tokenizer,
            system_prompt=None,
            apply_claude_correction=True,
        )
        return stats["total_tokens"]
    except Exception as exc:  # best-effort estimate; must never mask the real error
        logger.debug(
            f"Responses input-token estimate failed, using clamp fallback: {exc}"
        )
        return None


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def responses(request: Request, request_data: ResponsesRequest):
    """
    Responses endpoint - compatible with the OpenAI Responses API.

    This is the protocol required by current Codex CLI / IDE builds (support for
    ``wire_api = "chat"`` was removed in February 2026). It accepts a Responses
    request, translates it to a single Kiro generateAssistantResponse call, and
    returns either a streamed sequence of typed Responses SSE events or a
    collected ``response`` object — with full function-calling and reasoning
    (thinking → reasoning summary) support.

    Streaming and non-streaming share the same account-failover, retry, and
    error-handling machinery as ``/v1/chat/completions`` via
    ``routes_common.execute_kiro_request``.

    Args:
        request: FastAPI Request for accessing app.state.
        request_data: Request in OpenAI Responses format.

    Returns:
        StreamingResponse for streaming mode, JSONResponse for non-streaming mode.

    Raises:
        HTTPException: On validation or upstream API errors.
    """
    logger.info(
        f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream})"
    )

    # Observability: attach request semantics to the per-request metrics.
    input_count = len(request_data.input) if isinstance(request_data.input, list) else 1
    enrich_current_metrics(
        api="openai",
        model=request_data.model,
        stream=bool(request_data.stream),
        message_count=input_count,
        tool_count=len(request_data.tools) if request_data.tools else 0,
    )

    # Tokenizer inputs for FALLBACK prompt-token counting (primary source is
    # Kiro's context-usage percentage; used only when that is absent).
    messages_for_tokenizer, tools_for_tokenizer = build_responses_tokenizer_inputs(request_data)

    # Stable response id shared by response.created / response.completed events and
    # the non-streaming body, so one logical response carries one id.
    response_id = generate_response_id()

    # Request fields echoed back on the response object (OpenAI clients read these).
    base_response = {
        "instructions": request_data.instructions,
        "tools": (
            [tool.model_dump(exclude_none=True) for tool in request_data.tools]
            if request_data.tools else []
        ),
        "tool_choice": request_data.tool_choice if request_data.tool_choice is not None else "auto",
        "temperature": request_data.temperature,
        "top_p": request_data.top_p,
        "max_output_tokens": request_data.max_output_tokens,
        "parallel_tool_calls": (
            request_data.parallel_tool_calls if request_data.parallel_tool_calls is not None else True
        ),
        "reasoning": (
            request_data.reasoning.model_dump(exclude_none=True) if request_data.reasoning else None
        ),
        "metadata": request_data.metadata,
        "previous_response_id": request_data.previous_response_id,
    }

    def _build_payload(conversation_id: str, profile_arn: str) -> dict:
        """Build the Kiro payload from the Responses request."""
        return build_kiro_payload_from_responses(request_data, conversation_id, profile_arn)

    def _make_stream(*, http_client, initial_response, make_retry_request, auth_manager, model_cache):
        """Produce the Responses SSE stream (with first-token retry)."""
        return stream_kiro_to_responses_with_retry(
            make_request=make_retry_request,
            client=http_client.client,
            model=request_data.model,
            model_cache=model_cache,
            auth_manager=auth_manager,
            response_id=response_id,
            base_response=base_response,
            initial_response=initial_response,
            request_messages=messages_for_tokenizer,
            request_tools=tools_for_tokenizer,
        )

    async def _collect(*, http_client, response, auth_manager, model_cache):
        """Collect a full non-streaming Responses ``response`` object."""
        return await collect_responses_response(
            http_client.client,
            response,
            request_data.model,
            model_cache,
            auth_manager,
            response_id=response_id,
            base_response=base_response,
            request_messages=messages_for_tokenizer,
            request_tools=tools_for_tokenizer,
        )

    def _build_error(error_info, status_code: int, context_limit: int):
        """Shape a Kiro error into an OpenAI-style error body (shared envelope)."""
        return build_openai_error_body(
            error_info,
            status_code,
            context_limit=context_limit,
            max_tokens=request_data.max_output_tokens or 0,
            estimated_input_tokens=_estimate_responses_input_tokens(request_data),
        )

    return await execute_kiro_request(
        request=request,
        model=request_data.model,
        stream=request_data.stream,
        endpoint="/v1/responses",
        build_payload=_build_payload,
        make_stream=_make_stream,
        collect_response=_collect,
        build_error_response=_build_error,
    )
