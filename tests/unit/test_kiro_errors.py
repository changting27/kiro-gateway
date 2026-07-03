# -*- coding: utf-8 -*-

"""
Unit tests for Kiro API error enhancement system.
Tests enhance_kiro_error() function and KiroErrorInfo dataclass.
"""

import pytest

from kiro.kiro_errors import (
    KiroErrorInfo,
    enhance_kiro_error
)


class TestEnhanceKiroErrorContentLength:
    """Tests for CONTENT_LENGTH_EXCEEDS_THRESHOLD error enhancement."""
    
    def test_content_length_error_enhanced_successfully(self):
        """
        What it does: Verifies CONTENT_LENGTH_EXCEEDS_THRESHOLD is enhanced with user-friendly message.
        Purpose: Ensure cryptic Amazon error is replaced with clear, actionable message (issue #63).
        """
        print("Setup: Creating error JSON with CONTENT_LENGTH_EXCEEDS_THRESHOLD...")
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: User message is enhanced...")
        print(f"Comparing user_message: Expected 'Model context limit reached...', Got '{error_info.user_message}'")
        assert error_info.user_message == "Model context limit reached. Conversation size exceeds model capacity."
        assert error_info.reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        assert error_info.original_message == "Input is too long."
    
    def test_content_length_error_preserves_original_message(self):
        """
        What it does: Verifies original message is preserved for logging.
        Purpose: Ensure debugging information is not lost during enhancement.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Original message preserved...")
        assert error_info.original_message == "Input is too long."
        assert error_info.original_message != error_info.user_message
    
    def test_content_length_error_reason_string_correct(self):
        """
        What it does: Verifies reason is correctly preserved as string.
        Purpose: Ensure reason field can be used for programmatic error handling.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Reason is correct string value...")
        assert error_info.reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        assert isinstance(error_info.reason, str)
    
    def test_content_length_error_no_reason_suffix(self):
        """
        What it does: Verifies enhanced message doesn't include (reason: ...) suffix.
        Purpose: Ensure clean, user-friendly message without technical codes.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: No (reason: ...) in user message...")
        assert "(reason:" not in error_info.user_message
        assert "CONTENT_LENGTH_EXCEEDS_THRESHOLD" not in error_info.user_message


class TestEnhanceKiroErrorMonthlyLimit:
    """Tests for MONTHLY_REQUEST_COUNT error enhancement."""
    
    def test_monthly_limit_error_enhanced_successfully(self):
        """
        What it does: Verifies MONTHLY_REQUEST_COUNT is enhanced with user-friendly message.
        Purpose: Ensure monthly quota error is clear and actionable.
        """
        print("Setup: Creating error JSON with MONTHLY_REQUEST_COUNT...")
        error_json = {
            "message": "You have reached the limit.",
            "reason": "MONTHLY_REQUEST_COUNT"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: User message is enhanced...")
        assert error_info.user_message == "Monthly request limit exceeded. Account has reached its monthly quota."
        assert error_info.reason == "MONTHLY_REQUEST_COUNT"
        assert error_info.original_message == "You have reached the limit."
    
    def test_monthly_limit_error_no_reason_suffix(self):
        """
        What it does: Verifies enhanced message doesn't include (reason: ...) suffix.
        Purpose: Ensure clean, user-friendly message without technical codes.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "You have reached the limit.",
            "reason": "MONTHLY_REQUEST_COUNT"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: No (reason: ...) in user message...")
        assert "(reason:" not in error_info.user_message
        assert "MONTHLY_REQUEST_COUNT" not in error_info.user_message


class TestEnhanceKiroErrorInvalidModelId:
    """Tests for INVALID_MODEL_ID error enhancement."""
    
    def test_invalid_model_id_enhanced_successfully(self):
        """
        What it does: Verifies INVALID_MODEL_ID is enhanced with user-friendly message.
        Purpose: Ensure model availability error clearly indicates both possible causes.
        """
        print("Setup: Creating error JSON with INVALID_MODEL_ID...")
        error_json = {
            "message": "Invalid model ID. Please select a different model to continue.",
            "reason": "INVALID_MODEL_ID"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: User message is enhanced...")
        print(f"Comparing user_message: Expected 'Invalid model ID or insufficient subscription level to use it.', Got '{error_info.user_message}'")
        assert error_info.user_message == "Invalid model ID or insufficient subscription level to use it."
        assert error_info.reason == "INVALID_MODEL_ID"
        assert error_info.original_message == "Invalid model ID. Please select a different model to continue."
    
    def test_invalid_model_id_preserves_original_message(self):
        """
        What it does: Verifies original message is preserved for logging.
        Purpose: Ensure debugging information is not lost during enhancement.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Invalid model ID. Please select a different model to continue.",
            "reason": "INVALID_MODEL_ID"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Original message preserved...")
        assert error_info.original_message == "Invalid model ID. Please select a different model to continue."
        assert error_info.original_message != error_info.user_message
    
    def test_invalid_model_id_reason_string_correct(self):
        """
        What it does: Verifies reason is correctly preserved as string.
        Purpose: Ensure reason field can be used for programmatic error handling.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Invalid model ID. Please select a different model to continue.",
            "reason": "INVALID_MODEL_ID"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Reason is correct string value...")
        assert error_info.reason == "INVALID_MODEL_ID"
        assert isinstance(error_info.reason, str)
    
    def test_invalid_model_id_no_reason_suffix(self):
        """
        What it does: Verifies enhanced message doesn't include (reason: ...) suffix.
        Purpose: Ensure clean, user-friendly message without technical codes.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Invalid model ID. Please select a different model to continue.",
            "reason": "INVALID_MODEL_ID"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: No (reason: ...) in user message...")
        assert "(reason:" not in error_info.user_message
        assert "INVALID_MODEL_ID" not in error_info.user_message
    
    def test_invalid_model_id_mentions_both_causes(self):
        """
        What it does: Verifies message mentions both possible causes (invalid ID and subscription).
        Purpose: Ensure users understand both potential reasons for the error.
        """
        print("Setup: Creating error JSON...")
        error_json = {
            "message": "Invalid model ID. Please select a different model to continue.",
            "reason": "INVALID_MODEL_ID"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Message mentions both causes...")
        message_lower = error_info.user_message.lower()
        # Should mention "invalid" or "model id"
        assert "invalid" in message_lower or "model id" in message_lower
        # Should mention "subscription" or "level"
        assert "subscription" in message_lower or "level" in message_lower
    
    def test_invalid_model_id_with_different_original_message(self):
        """
        What it does: Verifies enhancement works regardless of original message text.
        Purpose: Ensure enhancement is based on reason code, not message text.
        """
        print("Setup: Creating error JSON with different original message...")
        error_json = {
            "message": "Model not found.",
            "reason": "INVALID_MODEL_ID"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Same enhanced message regardless of original...")
        assert error_info.user_message == "Invalid model ID or insufficient subscription level to use it."
        assert error_info.original_message == "Model not found."


class TestEnhanceKiroErrorUnknown:
    """Tests for unknown error handling."""
    
    def test_unknown_reason_keeps_original_with_suffix(self):
        """
        What it does: Verifies unknown reasons keep original message with (reason: ...) suffix.
        Purpose: Ensure unrecognized errors are passed through with context.
        """
        print("Setup: Creating error JSON with unknown reason...")
        error_json = {
            "message": "Something went wrong.",
            "reason": "UNKNOWN_FUTURE_ERROR"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Original message with reason suffix...")
        print(f"User message: {error_info.user_message}")
        assert error_info.user_message == "Something went wrong. (reason: UNKNOWN_FUTURE_ERROR)"
        assert error_info.reason == "UNKNOWN_FUTURE_ERROR"
        assert error_info.original_message == "Something went wrong."
    
    def test_unknown_reason_preserved_as_string(self):
        """
        What it does: Verifies unknown reasons are preserved as-is (not mapped to "UNKNOWN").
        Purpose: Ensure graceful handling of future Amazon error codes without information loss.
        """
        print("Setup: Creating error JSON with unknown reason...")
        error_json = {
            "message": "Error occurred.",
            "reason": "RATE_LIMIT_EXCEEDED"  # Future error not yet enhanced
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Reason preserved as original string...")
        assert error_info.reason == "RATE_LIMIT_EXCEEDED"
        assert error_info.user_message == "Error occurred. (reason: RATE_LIMIT_EXCEEDED)"
    
    def test_missing_reason_field_uses_unknown(self):
        """
        What it does: Verifies missing reason field defaults to UNKNOWN.
        Purpose: Ensure graceful handling of errors without reason field.
        """
        print("Setup: Creating error JSON without reason field...")
        error_json = {
            "message": "An error occurred."
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Reason is UNKNOWN, no suffix in message...")
        assert error_info.reason == "UNKNOWN"
        assert error_info.user_message == "An error occurred."
        assert "(reason:" not in error_info.user_message
    
    def test_reason_unknown_string_no_suffix(self):
        """
        What it does: Verifies reason="UNKNOWN" doesn't add redundant suffix.
        Purpose: Ensure clean message when reason is explicitly UNKNOWN.
        """
        print("Setup: Creating error JSON with reason='UNKNOWN'...")
        error_json = {
            "message": "Unknown error.",
            "reason": "UNKNOWN"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: No redundant (reason: UNKNOWN) suffix...")
        assert error_info.user_message == "Unknown error."
        assert "(reason: UNKNOWN)" not in error_info.user_message


class TestEnhanceKiroErrorEdgeCases:
    """Tests for edge cases and malformed input."""
    
    def test_empty_error_json_uses_defaults(self):
        """
        What it does: Verifies empty error JSON uses default values.
        Purpose: Ensure graceful handling of malformed API responses.
        """
        print("Setup: Creating empty error JSON...")
        error_json = {}
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Default values used...")
        assert error_info.original_message == "Unknown error"
        assert error_info.reason == "UNKNOWN"
        assert error_info.user_message == "Unknown error"
    
    def test_missing_message_field_uses_default(self):
        """
        What it does: Verifies missing message field uses "Unknown error" default.
        Purpose: Ensure graceful handling when message field is absent.
        """
        print("Setup: Creating error JSON without message field...")
        error_json = {
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Default message used, but enhancement still applied...")
        assert error_info.original_message == "Unknown error"
        # Enhancement should still work based on reason
        assert error_info.user_message == "Model context limit reached. Conversation size exceeds model capacity."
    
    def test_empty_message_string_preserved(self):
        """
        What it does: Verifies empty message string is preserved (not replaced with default).
        Purpose: Ensure explicit empty strings are distinguished from missing fields.
        """
        print("Setup: Creating error JSON with empty message...")
        error_json = {
            "message": "",
            "reason": "SOME_ERROR"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Empty string preserved...")
        assert error_info.original_message == ""
        assert error_info.user_message == " (reason: SOME_ERROR)"
    
    def test_none_values_handled_gracefully(self):
        """
        What it does: Verifies None values in error JSON are handled gracefully.
        Purpose: Ensure robustness against unexpected None values.
        """
        print("Setup: Creating error JSON with None values...")
        error_json = {
            "message": None,
            "reason": None
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Defaults used for None values...")
        assert error_info.original_message == "Unknown error"
        assert error_info.reason == "UNKNOWN"
    
    def test_extra_fields_ignored(self):
        """
        What it does: Verifies extra fields in error JSON are ignored.
        Purpose: Ensure forward compatibility with future API changes.
        """
        print("Setup: Creating error JSON with extra fields...")
        error_json = {
            "message": "Error occurred.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD",
            "extra_field": "extra_value",
            "another_field": 123
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Extra fields don't affect enhancement...")
        assert error_info.user_message == "Model context limit reached. Conversation size exceeds model capacity."
        assert error_info.reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
    
    def test_case_sensitive_reason_matching(self):
        """
        What it does: Verifies reason matching is case-sensitive.
        Purpose: Ensure exact string matching (Amazon API uses uppercase).
        """
        print("Setup: Creating error JSON with lowercase reason...")
        error_json = {
            "message": "Error.",
            "reason": "content_length_exceeds_threshold"  # lowercase
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Lowercase reason not matched, passed through as-is...")
        assert error_info.reason == "content_length_exceeds_threshold"
        assert error_info.user_message == "Error. (reason: content_length_exceeds_threshold)"


class TestEnhanceKiroErrorMessageQuality:
    """Tests for message quality and user experience."""
    
    def test_enhanced_message_is_user_friendly(self):
        """
        What it does: Verifies enhanced message is clear and non-technical.
        Purpose: Ensure end users can understand the error without technical knowledge.
        """
        print("Setup: Creating CONTENT_LENGTH_EXCEEDS_THRESHOLD error...")
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Message is user-friendly...")
        message = error_info.user_message
        # Should mention "context limit" (technical but understandable)
        assert "context limit" in message.lower()
        # Should mention "conversation" (user-facing term)
        assert "conversation" in message.lower()
        # Should NOT contain technical jargon
        assert "threshold" not in message.lower()
        assert "input" not in message.lower()
    
    def test_enhanced_message_indicates_model_limitation(self):
        """
        What it does: Verifies message indicates it's a model limitation, not gateway error.
        Purpose: Ensure users understand the error is from the model, not our service.
        """
        print("Setup: Creating CONTENT_LENGTH_EXCEEDS_THRESHOLD error...")
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Message indicates model limitation...")
        message = error_info.user_message
        assert "model" in message.lower()
        assert "limit" in message.lower()
    
    def test_unknown_error_preserves_original_context(self):
        """
        What it does: Verifies unknown errors preserve original message for context.
        Purpose: Ensure users get Amazon's original error when we can't enhance it.
        """
        print("Setup: Creating unknown error...")
        error_json = {
            "message": "Service temporarily unavailable.",
            "reason": "SERVICE_UNAVAILABLE"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Original message preserved with reason...")
        assert "Service temporarily unavailable" in error_info.user_message
        assert "SERVICE_UNAVAILABLE" in error_info.user_message


class TestKiroErrorInfoDataclass:
    """Tests for KiroErrorInfo dataclass."""
    
    def test_kiro_error_info_creation(self):
        """
        What it does: Verifies KiroErrorInfo can be created with all fields.
        Purpose: Ensure dataclass structure is correct.
        """
        print("Setup: Creating KiroErrorInfo...")
        error_info = KiroErrorInfo(
            reason="CONTENT_LENGTH_EXCEEDS_THRESHOLD",
            user_message="Test message",
            original_message="Original message"
        )
        
        print("Verification: All fields accessible...")
        assert error_info.reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        assert error_info.user_message == "Test message"
        assert error_info.original_message == "Original message"
    
    def test_kiro_error_info_fields_accessible(self):
        """
        What it does: Verifies KiroErrorInfo fields can be accessed.
        Purpose: Ensure error info structure is usable.
        """
        print("Setup: Creating KiroErrorInfo...")
        error_info = KiroErrorInfo(
            reason="UNKNOWN",
            user_message="Message",
            original_message="Original"
        )
        
        print("Verification: Fields are accessible...")
        assert hasattr(error_info, 'reason')
        assert hasattr(error_info, 'user_message')
        assert hasattr(error_info, 'original_message')


class TestEnhanceKiroErrorIntegration:
    """Integration tests for real-world scenarios."""
    
    def test_real_world_content_length_error(self):
        """
        What it does: Verifies enhancement works with real Kiro API error format.
        Purpose: Ensure compatibility with actual Amazon API responses (issue #63).
        """
        print("Setup: Creating real-world error JSON from Kiro API...")
        # This is the actual format from issue #63
        error_json = {
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        }
        
        print("Action: Enhancing error...")
        error_info = enhance_kiro_error(error_json)
        
        print("Verification: Real-world error enhanced correctly...")
        assert "Model context limit reached" in error_info.user_message
        assert "Conversation size exceeds model capacity" in error_info.user_message
        assert error_info.reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
    
    def test_multiple_errors_enhanced_independently(self):
        """
        What it does: Verifies multiple errors can be enhanced independently.
        Purpose: Ensure stateless enhancement (no side effects between calls).
        """
        print("Setup: Creating multiple error JSONs...")
        error1 = {"message": "Error 1", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
        error2 = {"message": "Error 2", "reason": "UNKNOWN_ERROR"}
        error3 = {"message": "Error 3"}
        
        print("Action: Enhancing all errors...")
        info1 = enhance_kiro_error(error1)
        info2 = enhance_kiro_error(error2)
        info3 = enhance_kiro_error(error3)
        
        print("Verification: Each error enhanced independently...")
        assert info1.user_message == "Model context limit reached. Conversation size exceeds model capacity."
        assert info2.user_message == "Error 2 (reason: UNKNOWN_ERROR)"
        assert info3.user_message == "Error 3"
        # Verify no cross-contamination
        assert info1.original_message == "Error 1"
        assert info2.original_message == "Error 2"
        assert info3.original_message == "Error 3"


class TestEnhanceImproperlyFormedRequest:
    """Tests for 'Improperly formed request.' error enhancement (issue #73)."""

    def test_enhance_improperly_formed_request_null_reason(self):
        """
        'Improperly formed request.' with null/UNKNOWN reason is enhanced
        to a clear payload-size message.
        """
        error_json = {
            "message": "Improperly formed request.",
            "reason": None,
        }
        error_info = enhance_kiro_error(error_json)

        assert "problem persists" in error_info.user_message
        assert "jwadow/kiro-gateway" in error_info.user_message
        assert error_info.original_message == "Improperly formed request."

    def test_enhance_improperly_formed_request_unknown_reason(self):
        """Same enhancement when reason field is missing (defaults to UNKNOWN)."""
        error_json = {"message": "Improperly formed request."}
        error_info = enhance_kiro_error(error_json)

        assert "problem persists" in error_info.user_message

    def test_improperly_formed_with_real_reason_not_enhanced(self):
        """If reason is a real code, don't apply the size-limit enhancement."""
        error_json = {
            "message": "Improperly formed request.",
            "reason": "VALIDATION_ERROR",
        }
        error_info = enhance_kiro_error(error_json)

        # Should fall through to generic handler, not the size-limit message
        assert "payload size exceeded" not in error_info.user_message


# ---------------------------------------------------------------------------
# Context-window overflow → canonical client-recognized error
#
# Claude Code (and OpenAI SDKs) only auto-compact when they receive the exact
# overflow error shape the native APIs emit. These tests lock that contract:
# the detection flag, the verbatim message, the number clamping, and both
# per-API response envelopes.
# ---------------------------------------------------------------------------

from kiro.config import DEFAULT_MAX_INPUT_TOKENS
from kiro.kiro_errors import (
    build_anthropic_error_body,
    build_openai_error_body,
    format_context_overflow_message,
    _resolve_overflow_input_tokens,
)


class TestContextOverflowDetection:
    """is_context_overflow flag on KiroErrorInfo."""

    def test_reason_code_sets_overflow_flag(self):
        """The canonical Kiro reason marks the error as a context overflow."""
        info = enhance_kiro_error({
            "message": "Input content length exceeds threshold.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD",
        })
        assert info.is_context_overflow is True

    def test_text_fallback_when_reason_unknown(self):
        """Overflow phrased only in the message (unknown reason) still trips the flag."""
        for message in (
            "Input content length exceeds threshold.",
            "Input is too long.",
            "Prompt is too long",
            "The request exceeds threshold for this model.",
        ):
            info = enhance_kiro_error({"message": message})
            assert info.is_context_overflow is True, message

    def test_no_false_positive_with_real_reason(self):
        """Overflow-ish text must NOT trip the flag when a real reason is present."""
        info = enhance_kiro_error({
            "message": "Input is too long.",
            "reason": "VALIDATION_ERROR",
        })
        assert info.is_context_overflow is False

    def test_unrelated_errors_are_not_overflow(self):
        """Quota / model / generic errors are never flagged as overflow."""
        for error_json in (
            {"message": "Monthly limit.", "reason": "MONTHLY_REQUEST_COUNT"},
            {"message": "Bad model.", "reason": "INVALID_MODEL_ID"},
            {"message": "Improperly formed request."},
            {"message": "Something else went wrong.", "reason": "UNKNOWN"},
        ):
            info = enhance_kiro_error(error_json)
            assert info.is_context_overflow is False, error_json


class TestFormatContextOverflowMessage:
    """The message string must match Anthropic's native wording verbatim."""

    def test_exact_wording(self):
        """Claude Code pattern-matches this phrasing to trigger auto-compaction."""
        msg = format_context_overflow_message(210000, 8192, 200000)
        assert msg == (
            "input length and max_tokens exceed context limit: "
            "210000 + 8192 > 200000, decrease input length or max_tokens and try again"
        )


class TestResolveOverflowInputTokens:
    """Number clamping guarantees input + max_tokens > context_limit."""

    def test_estimate_above_limit_used_verbatim(self):
        """A realistic estimate that already overflows is preserved for display."""
        assert _resolve_overflow_input_tokens(500000, 8192, 200000) == 500000

    def test_estimate_below_limit_clamped_up(self):
        """An under-count is raised so the reported inequality stays true."""
        result = _resolve_overflow_input_tokens(1000, 8192, 200000)
        assert result + 8192 > 200000
        assert result == 200000 - 8192 + 1

    def test_none_estimate_clamps_to_just_over_limit(self):
        """Missing estimate yields the smallest self-consistent overflow."""
        result = _resolve_overflow_input_tokens(None, 0, 200000)
        assert result == 200001

    def test_max_tokens_equal_or_greater_than_limit(self):
        """When max_tokens alone meets/exceeds the limit, input stays >= 1 and overflows."""
        result = _resolve_overflow_input_tokens(None, 200000, 200000)
        assert result >= 1
        assert result + 200000 > 200000

    def test_zero_and_negative_estimate_floored_to_one(self):
        """Degenerate estimates never produce a non-positive token count."""
        assert _resolve_overflow_input_tokens(0, 500000, 200000) >= 1
        assert _resolve_overflow_input_tokens(-50, 500000, 200000) >= 1


class TestBuildAnthropicErrorBody:
    """Anthropic error envelope: overflow → invalid_request_error 400."""

    def test_overflow_returns_canonical_invalid_request_error(self):
        info = enhance_kiro_error({
            "message": "Input content length exceeds threshold.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD",
        })
        status, body = build_anthropic_error_body(
            info, 400, context_limit=200000, max_tokens=8192, estimated_input_tokens=210000
        )
        assert status == 400
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == (
            "input length and max_tokens exceed context limit: "
            "210000 + 8192 > 200000, decrease input length or max_tokens and try again"
        )

    def test_overflow_forces_400_even_if_upstream_status_differs(self):
        """Overflow is always a 400 regardless of the upstream status code."""
        info = enhance_kiro_error({"reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD", "message": "x"})
        status, _ = build_anthropic_error_body(info, 500, context_limit=200000)
        assert status == 400

    def test_overflow_uses_default_limit_when_unspecified(self):
        info = enhance_kiro_error({"reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD", "message": "x"})
        _, body = build_anthropic_error_body(info, 400)
        assert f"> {DEFAULT_MAX_INPUT_TOKENS}," in body["error"]["message"]

    def test_non_overflow_preserves_api_error_shape_and_status(self):
        info = enhance_kiro_error({"message": "Improperly formed request."})
        status, body = build_anthropic_error_body(info, 400)
        assert status == 400
        assert body["error"]["type"] == "api_error"
        assert body["error"]["message"] == info.user_message


class TestBuildOpenAIErrorBody:
    """OpenAI error envelope: overflow → context_length_exceeded 400."""

    def test_overflow_returns_context_length_exceeded(self):
        info = enhance_kiro_error({
            "message": "Input content length exceeds threshold.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD",
        })
        status, body = build_openai_error_body(
            info, 400, context_limit=200000, max_tokens=4096, estimated_input_tokens=205000
        )
        assert status == 400
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "context_length_exceeded"
        assert body["error"]["param"] == "messages"
        assert "exceed context limit" in body["error"]["message"]

    def test_non_overflow_preserves_kiro_api_error_shape(self):
        info = enhance_kiro_error({"message": "Monthly limit.", "reason": "MONTHLY_REQUEST_COUNT"})
        status, body = build_openai_error_body(info, 402)
        assert status == 402
        assert body["error"]["type"] == "kiro_api_error"
        assert body["error"]["code"] == 402
        assert body["error"]["message"] == info.user_message
