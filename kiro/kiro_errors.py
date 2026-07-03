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
Kiro API error enhancement and user-friendly message formatting.

This module provides a centralized system for enhancing cryptic Kiro API errors
with clear, actionable, user-friendly messages.

Architecture:
- KiroErrorReason: Enum of known error reasons from Kiro API
- KiroErrorInfo: Structured information about an enhanced error
- enhance_kiro_error(): Analyzes error JSON and returns enhanced message

Example:
    >>> error_json = {"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
    >>> error_info = enhance_kiro_error(error_json)
    >>> print(error_info.user_message)
    "Model context limit reached. Conversation size exceeds model capacity."
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from kiro.config import DEFAULT_MAX_INPUT_TOKENS


@dataclass
class KiroErrorInfo:
    """
    Structured information about a Kiro API error.
    
    Contains both the enhanced user-friendly message and the original
    error details for logging and debugging.
    
    Attributes:
        reason: Error reason code from Kiro API (as string, e.g. "CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        user_message: Enhanced, user-friendly message for end users
        original_message: Original message from Kiro API (for logging)
        is_context_overflow: True when the error means the conversation exceeded the
            model's context window. Downstream clients (Claude Code, OpenAI SDKs)
            only auto-compact when they receive the canonical overflow error shape,
            so this flag drives the response builders below.
    """
    reason: str
    user_message: str
    original_message: str
    is_context_overflow: bool = False


def enhance_kiro_error(error_json: Dict[str, Any]) -> KiroErrorInfo:
    """
    Enhances Kiro API error with user-friendly message.
    
    Takes raw error JSON from Kiro API and returns structured information
    with enhanced, user-friendly messages that help users understand what
    went wrong without technical jargon.
    
    Args:
        error_json: Parsed JSON from Kiro API error response
                   Expected format: {"message": "...", "reason": "..."}
                   The "reason" field is optional.
    
    Returns:
        KiroErrorInfo with enhanced message and original details
    
    Example:
        >>> error_json = {"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
        >>> error_info = enhance_kiro_error(error_json)
        >>> print(error_info.user_message)
        "Model context limit reached. Conversation size exceeds model capacity."
        >>> print(error_info.original_message)
        "Input is too long."
    
    Example (unknown error):
        >>> error_json = {"message": "Something went wrong.", "reason": "UNKNOWN_REASON"}
        >>> error_info = enhance_kiro_error(error_json)
        >>> print(error_info.user_message)
        "Something went wrong. (reason: UNKNOWN_REASON)"
    """
    # Extract original message and reason from Kiro API response
    # Handle None values explicitly (preserve empty strings)
    original_message = error_json.get("message")
    if original_message is None:
        original_message = "Unknown error"
    
    reason = error_json.get("reason")
    if reason is None:
        reason = "UNKNOWN"
    
    # Map known reasons to user-friendly messages
    if reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD":
        # Context limit exceeded - conversation is too long
        user_message = "Model context limit reached. Conversation size exceeds model capacity."
    
    elif reason == "MONTHLY_REQUEST_COUNT":
        # Monthly request limit exceeded - account quota exhausted
        user_message = "Monthly request limit exceeded. Account has reached its monthly quota."
    
    elif reason == "INVALID_MODEL_ID":
        # Invalid model name or subscription tier insufficient
        user_message = "Invalid model ID or insufficient subscription level to use it."

    elif original_message == "Improperly formed request." and reason in (None, "UNKNOWN", "null"):
        # Generic 400 error
        user_message = (
            "Kiro API rejected the request. If problem persists, open issue with info and attached debug logs at:"
            "https://github.com/jwadow/kiro-gateway/issues"
        )

    # Future error enhancements can be added here:
    # elif reason == "RATE_LIMIT_EXCEEDED":
    #     user_message = "Rate limit exceeded. Too many requests in a short time."
    # elif reason == "INVALID_MODEL":
    #     user_message = "Invalid model specified. The requested model is not available."
    
    else:
        # Unknown error or no enhancement available
        # Keep original message and append reason if present
        if "reason" in error_json and reason != "UNKNOWN":
            user_message = f"{original_message} (reason: {reason})"
        else:
            user_message = original_message
    
    # Detect context-window overflow. Kiro signals this primarily via the reason
    # code, but older/edge responses only carry it in the free-text message, so we
    # add a conservative text fallback (only when the reason is unknown) to avoid
    # false positives on unrelated errors.
    is_context_overflow = reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
    if not is_context_overflow and reason == "UNKNOWN":
        lowered = original_message.lower()
        if (
            "content length exceeds" in lowered
            or "exceeds threshold" in lowered
            or "input is too long" in lowered
            or "prompt is too long" in lowered
        ):
            is_context_overflow = True

    return KiroErrorInfo(
        reason=reason,
        user_message=user_message,
        original_message=original_message,
        is_context_overflow=is_context_overflow,
    )


def format_context_overflow_message(
    input_tokens: int,
    max_tokens: int,
    context_limit: int,
) -> str:
    """
    Return the canonical Anthropic context-overflow message.

    Claude Code (and OpenAI-compatible clients) detect this exact phrasing to
    trigger automatic conversation compaction and retry. Reproducing Anthropic's
    native wording verbatim is what makes downstream auto-compaction fire; a
    custom message such as "Model context limit reached" is silently ignored by
    the client, which is why the conversation dead-ends instead of compacting.

    Args:
        input_tokens: Estimated prompt/input token count.
        max_tokens: Requested completion budget (max_tokens).
        context_limit: The model's real input-token limit.

    Returns:
        A message string of the exact form Anthropic's API emits on overflow.

    Examples:
        >>> format_context_overflow_message(210000, 8192, 200000)
        'input length and max_tokens exceed context limit: 210000 + 8192 > 200000, decrease input length or max_tokens and try again'
    """
    return (
        f"input length and max_tokens exceed context limit: "
        f"{input_tokens} + {max_tokens} > {context_limit}, "
        f"decrease input length or max_tokens and try again"
    )


def _resolve_overflow_input_tokens(
    estimated_input_tokens: Optional[int],
    max_tokens: int,
    context_limit: int,
) -> int:
    """
    Clamp the input-token count so ``input + max_tokens > context_limit`` holds.

    Kiro already rejected the request for exceeding the window, so the true input
    size is at least the limit. When the local estimate is missing or under-counts
    (tokenizers differ from Kiro's accounting), we raise it to the smallest value
    that keeps the reported inequality self-consistent, so the client treats it as
    a genuine overflow rather than recomputing and dismissing it.

    Args:
        estimated_input_tokens: Best-effort local estimate, or None.
        max_tokens: Requested completion budget.
        context_limit: The model's real input-token limit.

    Returns:
        A positive input-token count that satisfies input + max_tokens > limit.
    """
    input_tokens = max(estimated_input_tokens or 0, 1)
    if input_tokens + max_tokens <= context_limit:
        input_tokens = max(1, context_limit - max_tokens + 1)
    return input_tokens


def build_anthropic_error_body(
    error_info: KiroErrorInfo,
    status_code: int,
    *,
    context_limit: int = DEFAULT_MAX_INPUT_TOKENS,
    max_tokens: int = 0,
    estimated_input_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Render a Kiro error as an Anthropic-format error body.

    Context-overflow errors are re-expressed as Anthropic's canonical
    ``invalid_request_error`` (HTTP 400) so Claude Code recognises the overflow and
    triggers auto-compaction + retry. All other errors keep the existing
    ``api_error`` envelope and original status code.

    Args:
        error_info: Result of :func:`enhance_kiro_error`.
        status_code: Original upstream status code (used for non-overflow errors).
        context_limit: Model input-token limit for the overflow message.
        max_tokens: Requested completion budget for the overflow message.
        estimated_input_tokens: Optional local input-token estimate.

    Returns:
        Tuple of (http_status_code, response_body_dict).
    """
    if error_info.is_context_overflow:
        input_tokens = _resolve_overflow_input_tokens(
            estimated_input_tokens, max_tokens, context_limit
        )
        message = format_context_overflow_message(
            input_tokens, max_tokens, context_limit
        )
        return 400, {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
    return status_code, {
        "type": "error",
        "error": {"type": "api_error", "message": error_info.user_message},
    }


def build_openai_error_body(
    error_info: KiroErrorInfo,
    status_code: int,
    *,
    context_limit: int = DEFAULT_MAX_INPUT_TOKENS,
    max_tokens: int = 0,
    estimated_input_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Render a Kiro error as an OpenAI-format error body.

    Context-overflow errors are re-expressed as OpenAI's canonical
    ``context_length_exceeded`` / ``invalid_request_error`` (HTTP 400) so
    OpenAI-compatible clients trigger their own history trimming. All other errors
    keep the existing ``kiro_api_error`` envelope and original status code.

    Args:
        error_info: Result of :func:`enhance_kiro_error`.
        status_code: Original upstream status code (used for non-overflow errors).
        context_limit: Model input-token limit for the overflow message.
        max_tokens: Requested completion budget for the overflow message.
        estimated_input_tokens: Optional local input-token estimate.

    Returns:
        Tuple of (http_status_code, response_body_dict).
    """
    if error_info.is_context_overflow:
        input_tokens = _resolve_overflow_input_tokens(
            estimated_input_tokens, max_tokens, context_limit
        )
        message = format_context_overflow_message(
            input_tokens, max_tokens, context_limit
        )
        return 400, {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "messages",
            }
        }
    return status_code, {
        "error": {
            "message": error_info.user_message,
            "type": "kiro_api_error",
            "code": status_code,
        }
    }
