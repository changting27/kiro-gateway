# -*- coding: utf-8 -*-

"""
Unit tests for kiro/tool_names.py — transparent tool-name normalization.

Covers the forward (normalize) and reverse (restore-map) halves of the system
that works around Kiro's 64-character + ``[A-Za-z0-9_-]`` tool-name limit:

- normalize_tool_name: passthrough of valid names, length truncation, charset
  sanitization, determinism, idempotency, boundary (64/65), and empties.
- is_valid_tool_name: length/charset/empty rules.
- build_tool_name_restore_map: only-changed entries, heterogeneous tool shapes,
  collision handling.
- restore_tool_name: hit, miss, and no-op for empty/None maps.
- Property test: the result is ALWAYS within the Kiro constraints.
"""

import hashlib
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from kiro.tool_names import (
    MAX_TOOL_NAME_LENGTH,
    is_valid_tool_name,
    normalize_tool_name,
    build_tool_name_restore_map,
    restore_tool_name,
)

# A real offending name observed in production logs (Claude Code plugin scheme).
REAL_LONG_NAME = "mcp__plugin_chrome-devtools-mcp_chrome-devtools__performance_analyze_insight"  # 76 chars


# =============================================================================
# normalize_tool_name
# =============================================================================

class TestNormalizeToolNameSuccess:
    """Tests for normalize_tool_name on names that need no change or simple changes."""

    def test_short_valid_name_unchanged(self):
        """
        What it does: A short, valid name passes through untouched.
        Purpose: Well-behaved tools must incur zero behavioural change.
        """
        assert normalize_tool_name("get_weather") == "get_weather"

    def test_hyphenated_name_unchanged(self):
        """
        What it does: A name with hyphens/underscores within 64 chars is unchanged.
        Purpose: MCP names like mcp__chrome-devtools__click already work on Kiro.
        """
        name = "mcp__chrome-devtools__click"
        assert normalize_tool_name(name) == name

    def test_exactly_64_chars_unchanged(self):
        """
        What it does: A 64-character valid name is the boundary and stays unchanged.
        Purpose: 64 is accepted by Kiro (verified), so we must not normalize it.
        """
        name = "a" * 64
        assert normalize_tool_name(name) == name
        assert len(normalize_tool_name(name)) == 64

    def test_empty_string_unchanged(self):
        """
        What it does: Empty input returns empty.
        Purpose: Callers handle missing names explicitly; we must not crash/hash "".
        """
        assert normalize_tool_name("") == ""


class TestNormalizeToolNameLength:
    """Tests for normalize_tool_name length handling."""

    def test_65_chars_is_truncated_to_64(self):
        """
        What it does: A 65-char name (just over the limit) is shortened to <=64.
        Purpose: 65 is rejected by Kiro; the boundary must trigger normalization.
        """
        out = normalize_tool_name("a" * 65)
        assert out != "a" * 65
        assert len(out) == 64

    def test_real_plugin_name_normalized_to_64(self):
        """
        What it does: The real 76-char Claude Code plugin name becomes a valid 64.
        Purpose: This is the exact production failure case.
        """
        out = normalize_tool_name(REAL_LONG_NAME)
        assert len(out) == 64
        assert is_valid_tool_name(out)

    def test_truncated_name_preserves_readable_prefix(self):
        """
        What it does: The normalized name keeps the leading characters of the original.
        Purpose: Readability/debuggability — the prefix identifies the tool.
        """
        out = normalize_tool_name(REAL_LONG_NAME)
        # Prefix (first 55 chars) of the original is retained before the hash suffix.
        assert out.startswith(REAL_LONG_NAME[:55])

    def test_suffix_is_hash_of_full_original(self):
        """
        What it does: The hash suffix is derived from the FULL original name.
        Purpose: Distinct originals sharing a prefix must not collide.
        """
        expected = hashlib.sha1(REAL_LONG_NAME.encode("utf-8")).hexdigest()[:8]
        assert normalize_tool_name(REAL_LONG_NAME).endswith("_" + expected)

    def test_same_prefix_different_tail_do_not_collide(self):
        """
        What it does: Two long names that share the first 55 chars get different aliases.
        Purpose: Truncation alone would collide; the full-name hash prevents it.
        """
        base = "x" * 60
        a = base + "_alpha_aaaaaaaaaa"
        b = base + "_beta_bbbbbbbbbbb"
        assert len(a) > 64 and len(b) > 64
        assert normalize_tool_name(a) != normalize_tool_name(b)


class TestNormalizeToolNameCharset:
    """Tests for normalize_tool_name charset defense."""

    def test_invalid_chars_are_sanitized(self):
        """
        What it does: A short name with invalid chars (dot, slash, colon) is sanitized.
        Purpose: Charset defense — only [A-Za-z0-9_-] may reach Kiro.
        """
        out = normalize_tool_name("bad.name/with:chars")
        assert is_valid_tool_name(out)

    def test_unicode_name_is_sanitized(self):
        """
        What it does: A name with non-ASCII characters is sanitized to a valid alias.
        Purpose: Defend against exotic MCP tool names.
        """
        out = normalize_tool_name("工具_naïve")
        assert is_valid_tool_name(out)

    def test_charset_change_alone_triggers_normalization(self):
        """
        What it does: A short (<=64) name with an invalid char is still normalized.
        Purpose: Length is not the only constraint; charset alone must trigger it.
        """
        name = "tool.with.dots"  # 14 chars, valid length, invalid charset
        out = normalize_tool_name(name)
        assert out != name
        assert is_valid_tool_name(out)


class TestNormalizeToolNameDeterminism:
    """Tests for determinism and idempotency."""

    def test_deterministic_same_input_same_output(self):
        """
        What it does: normalize is a pure function — same input, same output.
        Purpose: Forward direction needs no shared state; defs and history stay consistent.
        """
        assert normalize_tool_name(REAL_LONG_NAME) == normalize_tool_name(REAL_LONG_NAME)

    def test_idempotent_on_already_normalized(self):
        """
        What it does: Re-normalizing an already-normalized (valid) name is a no-op.
        Purpose: Restoring then re-sending must not double-mangle names.
        """
        once = normalize_tool_name(REAL_LONG_NAME)
        assert normalize_tool_name(once) == once


# =============================================================================
# is_valid_tool_name
# =============================================================================

class TestIsValidToolName:
    """Tests for is_valid_tool_name."""

    def test_valid_name(self):
        """What it does: Accepts a normal name. Purpose: baseline."""
        assert is_valid_tool_name("get_weather-2") is True

    def test_64_is_valid_65_is_not(self):
        """What it does: Boundary at 64. Purpose: matches Kiro's verified limit."""
        assert is_valid_tool_name("a" * 64) is True
        assert is_valid_tool_name("a" * 65) is False

    def test_invalid_charset(self):
        """What it does: Rejects invalid characters. Purpose: charset rule."""
        assert is_valid_tool_name("has space") is False
        assert is_valid_tool_name("dot.name") is False

    def test_empty_is_invalid(self):
        """What it does: Empty is not valid. Purpose: avoid empty tool names."""
        assert is_valid_tool_name("") is False


# =============================================================================
# build_tool_name_restore_map
# =============================================================================

class TestBuildToolNameRestoreMap:
    """Tests for build_tool_name_restore_map."""

    def test_empty_for_all_valid_tools(self):
        """
        What it does: All-valid tool list yields an empty map.
        Purpose: No restore overhead when nothing was normalized.
        """
        req = [{"name": "get_weather"}, {"name": "mcp__server__do_thing"}]
        assert build_tool_name_restore_map(req) == {}

    def test_only_changed_names_included(self):
        """
        What it does: Only tools whose name changed appear in the map.
        Purpose: Keep the reverse map minimal and precise.
        """
        req = [{"name": REAL_LONG_NAME}, {"name": "short_ok"}]
        m = build_tool_name_restore_map(req)
        assert m == {normalize_tool_name(REAL_LONG_NAME): REAL_LONG_NAME}

    def test_none_and_empty_inputs(self):
        """
        What it does: None or empty tool list yields an empty map.
        Purpose: Safety for tool-less requests.
        """
        assert build_tool_name_restore_map(None) == {}
        assert build_tool_name_restore_map([]) == {}

    def test_extracts_openai_nested_function_name(self):
        """
        What it does: Handles OpenAI {"type":"function","function":{"name":...}}.
        Purpose: request_tools come from model_dump of OpenAI tools.
        """
        req = [{"type": "function", "function": {"name": REAL_LONG_NAME}}]
        m = build_tool_name_restore_map(req)
        assert m[normalize_tool_name(REAL_LONG_NAME)] == REAL_LONG_NAME

    def test_extracts_pydantic_like_object(self):
        """
        What it does: Handles object-style tools (name attribute).
        Purpose: Tolerate non-dict tool entries.
        """
        req = [SimpleNamespace(name=REAL_LONG_NAME)]
        m = build_tool_name_restore_map(req)
        assert m[normalize_tool_name(REAL_LONG_NAME)] == REAL_LONG_NAME

    def test_extracts_object_with_function_attr(self):
        """
        What it does: Handles object with a .function.name (object or dict).
        Purpose: Tolerate OpenAI pydantic tool objects pre-dump.
        """
        req = [SimpleNamespace(name=None, function={"name": REAL_LONG_NAME})]
        m = build_tool_name_restore_map(req)
        assert m[normalize_tool_name(REAL_LONG_NAME)] == REAL_LONG_NAME

    def test_duplicate_identical_originals_single_entry(self):
        """
        What it does: The same original appearing twice yields one entry, no error.
        Purpose: Idempotent map building.
        """
        req = [{"name": REAL_LONG_NAME}, {"name": REAL_LONG_NAME}]
        m = build_tool_name_restore_map(req)
        assert m == {normalize_tool_name(REAL_LONG_NAME): REAL_LONG_NAME}

    def test_collision_keeps_first(self, monkeypatch):
        """
        What it does: Two DISTINCT originals mapping to the same alias keep the first.
        Purpose: Exercise the collision branch deterministically (forced via patch).
        """
        from kiro import tool_names as tn
        monkeypatch.setattr(tn, "normalize_tool_name", lambda name: "X" * 64)
        req = [{"name": "a" * 70}, {"name": "b" * 70}]
        m = tn.build_tool_name_restore_map(req)
        assert m == {"X" * 64: "a" * 70}


# =============================================================================
# restore_tool_name
# =============================================================================

class TestRestoreToolName:
    """Tests for restore_tool_name."""

    def test_restores_known_alias(self):
        """What it does: Maps an alias back to the original. Purpose: core reverse op."""
        m = build_tool_name_restore_map([{"name": REAL_LONG_NAME}])
        alias = normalize_tool_name(REAL_LONG_NAME)
        assert restore_tool_name(alias, m) == REAL_LONG_NAME

    def test_unknown_name_passthrough(self):
        """What it does: Unknown names pass through. Purpose: built-ins like web_search."""
        m = build_tool_name_restore_map([{"name": REAL_LONG_NAME}])
        assert restore_tool_name("web_search", m) == "web_search"

    def test_none_map_is_noop(self):
        """What it does: None map returns the name unchanged. Purpose: safety."""
        assert restore_tool_name("anything", None) == "anything"

    def test_empty_map_is_noop(self):
        """What it does: Empty map returns the name unchanged. Purpose: safety."""
        assert restore_tool_name("anything", {}) == "anything"


# =============================================================================
# Property-based: result always satisfies Kiro constraints
# =============================================================================

class TestNormalizeToolNameProperty:
    """Property tests that try to break normalize_tool_name with arbitrary input."""

    @given(st.text())
    def test_result_always_within_kiro_constraints(self, name):
        """
        What it does: For ANY input, the output is <=64 chars and either empty or
                      a valid Kiro tool name.
        Purpose: Guarantee we never emit a name Kiro would reject, whatever a client sends.
        """
        out = normalize_tool_name(name)
        assert len(out) <= MAX_TOOL_NAME_LENGTH
        assert out == "" or is_valid_tool_name(out)

    @given(st.text(min_size=1))
    def test_result_is_deterministic(self, name):
        """
        What it does: Output is stable across calls for any non-empty input.
        Purpose: Determinism underpins the map-free forward direction.
        """
        assert normalize_tool_name(name) == normalize_tool_name(name)
