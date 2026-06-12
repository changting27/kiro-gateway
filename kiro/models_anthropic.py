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
Pydantic models for Anthropic Messages API.

Defines data schemas for requests and responses compatible with
Anthropic's Messages API specification.

Reference: https://docs.anthropic.com/en/api/messages
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ==================================================================================================
# Content Block Models
# ==================================================================================================


class TextContentBlock(BaseModel):
    """
    Text content block in Anthropic format.

    Used in both requests and responses for text content.
    """

    type: Literal["text"] = "text"
    text: str


class ThinkingContentBlock(BaseModel):
    """
    Thinking content block in Anthropic format.

    Represents the model's reasoning/thinking process.
    Used when extended thinking is enabled.

    Attributes:
        type: Always "thinking"
        thinking: The thinking/reasoning content
        signature: Cryptographic signature for verification (placeholder in our case)
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class ToolUseContentBlock(BaseModel):
    """
    Tool use content block in Anthropic format.

    Represents a tool call made by the assistant.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolReferenceContentBlock(BaseModel):
    """
    Tool reference content block (Claude Code deferred tools).

    Sent by Claude Code v2.1.69+ inside tool_result blocks to indicate
    which tools were loaded via the ToolSearch deferred tool mechanism.
    """

    type: Literal["tool_reference"] = "tool_reference"
    tool_name: str

    model_config = {"extra": "allow"}


class ToolResultContentBlock(BaseModel):
    """
    Tool result content block in Anthropic format.

    Represents the result of a tool call, sent by the user.
    Tool results can contain text, images, tool references, or a mix.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[
        Union[str, List[Union["TextContentBlock", "ImageContentBlock", "ToolReferenceContentBlock"]]]
    ] = None
    is_error: Optional[bool] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Image Content Block Models
# ==================================================================================================


class Base64ImageSource(BaseModel):
    """
    Base64-encoded image source in Anthropic format.

    Attributes:
        type: Always "base64"
        media_type: MIME type (e.g., "image/jpeg", "image/png", "image/gif", "image/webp")
        data: Base64-encoded image data
    """

    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class URLImageSource(BaseModel):
    """
    URL-based image source in Anthropic format.

    Note: URL images require fetching and converting to base64 for Kiro API.
    Currently logged as warning and skipped.

    Attributes:
        type: Always "url"
        url: HTTP(S) URL to the image
    """

    type: Literal["url"] = "url"
    url: str


class ImageContentBlock(BaseModel):
    """
    Image content block in Anthropic format.

    Represents an image in a message. Supports both base64-encoded
    images and URL references.

    Attributes:
        type: Always "image"
        source: Image source (base64 or URL)
    """

    type: Literal["image"] = "image"
    source: Union[Base64ImageSource, URLImageSource]


# Union type for all content blocks (including images and thinking)
class GenericContentBlock(BaseModel):
    """
    Forward-compatible fallback for content blocks the gateway does not model
    explicitly (e.g. ``document`` PDFs, ``redacted_thinking``, or future Anthropic
    block types). Captures the ``type`` and preserves all other fields so a request
    using such a block validates instead of failing with HTTP 422 (issues #176, #82).

    The converters handle only the block types they understand and skip the rest, so
    an unmodelled block is accepted and passed through without crashing the request.
    Note: such blocks are not translated to Kiro - their payload (e.g. a PDF) is not
    forwarded - this only prevents a hard 422 on the whole request.
    """

    type: str
    model_config = {"extra": "allow"}


ContentBlock = Union[
    TextContentBlock,
    ThinkingContentBlock,
    ImageContentBlock,
    ToolUseContentBlock,
    ToolResultContentBlock,
    ToolReferenceContentBlock,
    GenericContentBlock,
]


# ==================================================================================================
# Message Models
# ==================================================================================================


class AnthropicMessage(BaseModel):
    """
    Message in Anthropic format.

    Attributes:
        role: Message role (user or assistant)
        content: Message content (string or list of content blocks)
    """

    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

    model_config = {"extra": "allow"}


# ==================================================================================================
# Tool Models
# ==================================================================================================


class AnthropicTool(BaseModel):
    """
    Tool definition in Anthropic format.
    
    Supports both user-defined tools and server-side tools (Anthropic):
    - User-defined tools: require input_schema
    - Server-side tools: use type field (e.g., "web_search_20250305")
    
    Attributes:
        type: Tool type for server-side tools (e.g., "web_search_20250305")
        name: Tool name (must match pattern ^[a-zA-Z0-9_-]{1,64}$)
        description: Tool description (optional but recommended)
        input_schema: JSON Schema for tool parameters (required for user-defined tools)
        max_uses: Maximum uses per conversation (server-side tools, optional)
        allowed_domains: Allowed domains for web_search (optional)
        blocked_domains: Blocked domains for web_search (optional)
        user_location: User location for web_search (optional)
    """
    
    # Server-side tool fields (Anthropic spec)
    type: Optional[str] = None
    
    # Common fields
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None  # Now optional for server-side tools
    
    # Server-side tool parameters (Anthropic spec - accepted but not enforced)
    max_uses: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    user_location: Optional[Dict[str, Any]] = None
    
    model_config = {"extra": "allow"}  # Forward compatibility
    
    @model_validator(mode="after")
    def validate_tool_consistency(self):
        """Validate that user-defined tools have input_schema."""
        is_server_side = self.type is not None
        
        if not is_server_side:
            # User-defined tool: input_schema is required
            if self.input_schema is None:
                raise ValueError(
                    "input_schema is required for user-defined tools "
                    "(those without a 'type' field)"
                )
        return self


class ToolChoiceAuto(BaseModel):
    """Auto tool choice - model decides whether to use tools."""

    type: Literal["auto"] = "auto"


class ToolChoiceAny(BaseModel):
    """Any tool choice - model must use at least one tool."""

    type: Literal["any"] = "any"


class ToolChoiceTool(BaseModel):
    """Specific tool choice - model must use the specified tool."""

    type: Literal["tool"] = "tool"
    name: str


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool]


# ==================================================================================================
# Request Models
# ==================================================================================================


class SystemContentBlock(BaseModel):
    """
    System content block for prompt caching.

    Anthropic API supports system as a list of content blocks
    with optional cache_control for prompt caching.
    """

    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# System can be a string or list of content blocks (for prompt caching)
SystemPrompt = Union[str, List[SystemContentBlock], List[Dict[str, Any]]]


def _extract_system_text(content: Any) -> str:
    """
    Extract plain text from a message's content (string or list of blocks).

    Args:
        content: A message ``content`` value - a string, or a list of content
            blocks (dicts with a ``text`` field, or bare strings).

    Returns:
        The concatenated text, or an empty string when no text is present.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _merge_system_prompt(existing: Any, hoisted_texts: List[str]) -> Any:
    """
    Merge hoisted inline-system text into an existing top-level system prompt.

    Preserves a list-form system prompt (used for prompt caching) by appending the
    hoisted text as an extra text block; otherwise produces a single combined string.

    Args:
        existing: The current top-level ``system`` value (None, str, or list).
        hoisted_texts: Text extracted from inline system-role messages, in order.

    Returns:
        The merged system prompt (str or list), or the joined hoisted text when no
        system prompt existed.
    """
    hoisted = "\n\n".join(text for text in hoisted_texts if text)
    if not hoisted:
        return existing
    if existing is None or existing == "":
        return hoisted
    if isinstance(existing, str):
        return f"{existing}\n\n{hoisted}"
    if isinstance(existing, list):
        return list(existing) + [{"type": "text", "text": hoisted}]
    return hoisted


class AnthropicMessagesRequest(BaseModel):
    """
    Request to Anthropic Messages API (/v1/messages).

    Attributes:
        model: Model ID (e.g., "claude-sonnet-4-5")
        messages: List of conversation messages
        max_tokens: Maximum tokens in response (required)
        system: System prompt (optional, string or list of content blocks for caching)
        stream: Whether to stream the response
        tools: List of available tools
        tool_choice: Tool selection strategy
        temperature: Sampling temperature (0-1)
        top_p: Top-p sampling
        top_k: Top-k sampling
        stop_sequences: Custom stop sequences
        metadata: Request metadata
    """

    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)
    max_tokens: int

    # Optional parameters - system can be string or list of content blocks
    system: Optional[SystemPrompt] = None
    stream: bool = False

    # Extended thinking (official Anthropic parameter)
    thinking: Optional[Dict[str, Any]] = None

    # Tools
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[ToolChoice, Dict[str, Any]]] = None

    # Sampling parameters
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)

    # Other parameters
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _hoist_inline_system_messages(cls, data: Any) -> Any:
        """
        Hoist inline ``role: "system"`` messages into the top-level system prompt.

        Newer Claude Code clients place the system prompt as a system-role entry
        inside the ``messages`` array rather than the dedicated ``system`` field.
        The Anthropic spec only permits ``user``/``assistant`` roles in ``messages``,
        so such requests would otherwise fail validation with HTTP 422 (issue #190).
        This runs before field validation: it extracts the text of any system-role
        messages, merges it into ``system`` (preserving an existing prompt), and
        removes those entries so the remaining messages validate normally.

        Args:
            data: Raw request payload before field validation.

        Returns:
            The request payload with inline system messages hoisted into ``system``.
        """
        if not isinstance(data, dict):
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        hoisted_texts: List[str] = []
        kept_messages: List[Any] = []
        found_system = False
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                found_system = True
                text = _extract_system_text(message.get("content"))
                if text:
                    hoisted_texts.append(text)
                continue  # drop the system-role entry from messages
            kept_messages.append(message)

        if not found_system:
            return data

        merged = {**data, "messages": kept_messages}
        if hoisted_texts:
            merged["system"] = _merge_system_prompt(data.get("system"), hoisted_texts)
        return merged


class AnthropicCountTokensRequest(BaseModel):
    """
    Request to Anthropic Count Tokens API (/v1/messages/count_tokens).
    
    Similar to AnthropicMessagesRequest but without generation parameters.
    Used to estimate token count before making actual request.
    
    Attributes:
        model: Model ID (e.g., "claude-sonnet-4-5")
        messages: List of conversation messages
        system: System prompt (optional, string or list of content blocks)
        tools: List of available tools
    """
    
    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)
    
    # Optional parameters - only those that affect token count
    system: Optional[SystemPrompt] = None
    tools: Optional[List[AnthropicTool]] = None
    
    model_config = {"extra": "allow"}


# ==================================================================================================
# Response Models
# ==================================================================================================


class AnthropicUsage(BaseModel):
    """
    Token usage information in Anthropic format.

    Attributes:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_read_input_tokens: Tokens read from prompt cache (only forwarded when explicitly returned by upstream Kiro API)
        cache_creation_input_tokens: Tokens used to create prompt cache (only forwarded when explicitly returned by upstream Kiro API)
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None

    model_config = {"extra": "allow"}


class AnthropicMessagesResponse(BaseModel):
    """
    Response from Anthropic Messages API (non-streaming).

    Attributes:
        id: Unique message ID
        type: Always "message"
        role: Always "assistant"
        content: List of content blocks (may include thinking, text, tool_use)
        model: Model used
        stop_reason: Why generation stopped
        stop_sequence: Stop sequence that triggered stop (if any)
        usage: Token usage information
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[ThinkingContentBlock, TextContentBlock, ToolUseContentBlock]]
    model: str
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# ==================================================================================================
# Streaming Event Models
# ==================================================================================================


class MessageStartEvent(BaseModel):
    """
    Event sent at the start of a message stream.

    Contains the initial message object with empty content.
    """

    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]


class ContentBlockStartEvent(BaseModel):
    """
    Event sent at the start of a content block.

    Attributes:
        index: Index of the content block
        content_block: Initial content block (with empty text for text blocks)
    """

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Dict[str, Any]


class TextDelta(BaseModel):
    """Delta for text content."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    """Delta for thinking content."""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class InputJsonDelta(BaseModel):
    """Delta for tool input JSON."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    """
    Event sent when content block is updated.

    Attributes:
        index: Index of the content block being updated
        delta: The delta update (text_delta, thinking_delta, or input_json_delta)
    """

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[TextDelta, ThinkingDelta, InputJsonDelta, Dict[str, Any]]


class ContentBlockStopEvent(BaseModel):
    """
    Event sent when a content block is complete.
    """

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaUsage(BaseModel):
    """Usage information in message_delta event."""

    output_tokens: int


class MessageDeltaEvent(BaseModel):
    """
    Event sent near the end of the stream with final message data.

    Attributes:
        delta: Contains stop_reason and stop_sequence
        usage: Output token count
    """

    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: MessageDeltaUsage


class MessageStopEvent(BaseModel):
    """
    Event sent at the end of the message stream.
    """

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """
    Ping event sent periodically to keep connection alive.
    """

    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """
    Error event sent when an error occurs during streaming.
    """

    type: Literal["error"] = "error"
    error: Dict[str, Any]


# Union of all streaming events
StreamingEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
]


# ==================================================================================================
# Error Models
# ==================================================================================================


class AnthropicErrorDetail(BaseModel):
    """
    Error detail in Anthropic format.
    """

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """
    Error response in Anthropic format.
    """

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
