# -*- coding: utf-8 -*-

"""
Unit tests for the observability module.

Covers:
- RequestMetrics serialization and summary formatting
- outcome_from_status mapping
- MetricsRegistry aggregation, snapshot isolation, reset
- Context-bound enrichment helpers (enrich/mark_first_token/add_token_usage/note_retry)
- JSONL sink (rolling path + append)
- record_request behaviour (enabled/disabled, fail-safe)
- ObservabilityMiddleware (http instrumentation, passthrough, error path, context reset)

All tests are network-isolated and reset the global registry between cases.
"""

import json
import glob

import pytest

import kiro.observability as obs
from kiro.observability import (
    RequestMetrics,
    MetricsRegistry,
    ObservabilityMiddleware,
    outcome_from_status,
    new_request_id,
    enrich_current_metrics,
    mark_first_token,
    add_token_usage,
    note_retry,
    record_request,
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    LATENCY_BUCKETS_MS,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the global registry before and after each test for isolation."""
    obs.registry.reset()
    yield
    obs.registry.reset()


def _make_metrics(**overrides) -> RequestMetrics:
    """Build a RequestMetrics with sensible defaults, applying overrides."""
    base = dict(
        request_id="rid000000001",
        ts="2026-06-12T00:00:00+00:00",
        method="POST",
        path="/v1/messages",
    )
    base.update(overrides)
    return RequestMetrics(**base)


# ==================================================================================================
# RequestMetrics
# ==================================================================================================

class TestRequestMetrics:
    """Tests for the RequestMetrics dataclass."""

    def test_to_dict_excludes_internal_and_rounds_timings(self):
        """
        What it does: to_dict drops start_monotonic and rounds float timings.
        Purpose: JSONL output must be stable and free of internal timing state.
        """
        m = _make_metrics(total_ms=1234.5678, first_token_ms=12.3456)
        data = m.to_dict()

        assert "start_monotonic" not in data
        assert data["total_ms"] == 1234.6
        assert data["first_token_ms"] == 12.3
        # None fields are preserved for a stable schema.
        assert "prompt_tokens" in data and data["prompt_tokens"] is None

    def test_to_dict_is_json_serializable(self):
        """
        What it does: to_dict produces JSON-serializable output.
        Purpose: The JSONL sink calls json.dumps on this dict.
        """
        m = _make_metrics(api="openai", model="m", stream=True, status_code=200)
        json.dumps(m.to_dict())  # must not raise

    def test_summary_line_omits_unset_fields(self):
        """
        What it does: summary_line includes set fields and omits unset ones.
        Purpose: Keep the per-request log line compact and noise-free.
        """
        m = _make_metrics(api="anthropic", model="claude", status_code=200, total_ms=20.0)
        line = m.summary_line()

        assert "rid=rid000000001" in line
        assert "api=anthropic" in line
        assert "model=claude" in line
        assert "status=200" in line
        assert "prompt_tokens=" not in line  # unset -> omitted
        assert "retries=" not in line  # zero retries omitted

    def test_summary_line_includes_zero_token_counts_but_not_zero_retries(self):
        """
        What it does: zero token counts are shown; zero retries are hidden.
        Purpose: completion_tokens=0 is meaningful signal (e.g. empty response),
                 while retries=0 is just noise.
        """
        m = _make_metrics(prompt_tokens=0, completion_tokens=0, retries=0)
        line = m.summary_line()

        assert "prompt_tokens=0" in line
        assert "completion_tokens=0" in line
        assert "retries=" not in line

    def test_elapsed_ms_is_non_negative(self):
        """
        What it does: elapsed_ms returns a non-negative duration.
        Purpose: Sanity of the monotonic timing base.
        """
        m = _make_metrics()
        assert m.elapsed_ms() >= 0.0


# ==================================================================================================
# outcome_from_status
# ==================================================================================================

class TestOutcomeFromStatus:
    """Tests for outcome_from_status."""

    @pytest.mark.parametrize("status,expected", [
        (200, OUTCOME_COMPLETED),
        (204, OUTCOME_COMPLETED),
        (302, OUTCOME_COMPLETED),
        (399, OUTCOME_COMPLETED),
        (400, OUTCOME_ERROR),
        (404, OUTCOME_ERROR),
        (500, OUTCOME_ERROR),
        (None, OUTCOME_COMPLETED),
    ])
    def test_mapping(self, status, expected):
        """
        What it does: Maps status codes to outcomes; >=400 is error.
        Purpose: Consistent outcome classification for the registry.
        """
        assert outcome_from_status(status) == expected


# ==================================================================================================
# MetricsRegistry
# ==================================================================================================

class TestMetricsRegistry:
    """Tests for the in-process metrics registry."""

    def test_record_aggregates_counters_and_tokens(self):
        """
        What it does: record folds status/api/outcome/model/tokens/retries in.
        Purpose: Verify the core aggregation used by /metrics.
        """
        reg = MetricsRegistry()
        reg.record(_make_metrics(
            api="openai", model="m1", status_code=200, outcome=OUTCOME_COMPLETED,
            prompt_tokens=100, completion_tokens=20, retries=2, total_ms=120.0,
        ))
        reg.record(_make_metrics(
            api="openai", model="m1", status_code=500, outcome=OUTCOME_ERROR,
            error_category="connection_closed", total_ms=60.0,
        ))
        snap = reg.snapshot()

        assert snap["requests_total"] == 2
        assert snap["by_status_class"] == {"2xx": 1, "5xx": 1}
        assert snap["by_api"] == {"openai": 2}
        assert snap["by_model"] == {"m1": 2}
        assert snap["by_outcome"] == {OUTCOME_COMPLETED: 1, OUTCOME_ERROR: 1}
        assert snap["by_error_category"] == {"connection_closed": 1}
        assert snap["prompt_tokens_total"] == 100
        assert snap["completion_tokens_total"] == 20
        assert snap["retries_total"] == 2
        assert snap["latency_ms_count"] == 2
        assert snap["latency_ms_sum"] == 180.0

    def test_latency_buckets_are_cumulative(self):
        """
        What it does: A latency lands in every bucket whose bound it is <=.
        Purpose: Prometheus histogram buckets are cumulative ("le").
        """
        reg = MetricsRegistry()
        reg.record(_make_metrics(total_ms=300.0))  # between 250 and 500
        snap = reg.snapshot()

        for bound in LATENCY_BUCKETS_MS:
            expected = 1 if 300.0 <= bound else 0
            assert snap["latency_buckets"][bound] == expected, bound

    def test_status_class_unknown_when_missing(self):
        """
        What it does: A None status code is classed as 'unknown'.
        Purpose: Requests that never sent a status (rare) are still counted.
        """
        reg = MetricsRegistry()
        reg.record(_make_metrics(status_code=None))
        assert reg.snapshot()["by_status_class"] == {"unknown": 1}

    def test_none_keys_are_skipped(self):
        """
        What it does: None api/model/outcome are not counted under a 'None' key.
        Purpose: Avoid polluting label dimensions with null buckets.
        """
        reg = MetricsRegistry()
        reg.record(_make_metrics(api=None, model=None, outcome=None))
        snap = reg.snapshot()
        assert snap["by_api"] == {}
        assert snap["by_model"] == {}
        assert snap["by_outcome"] == {}

    def test_snapshot_is_isolated_copy(self):
        """
        What it does: Mutating a snapshot does not affect the live registry.
        Purpose: /metrics rendering must not corrupt counters.
        """
        reg = MetricsRegistry()
        reg.record(_make_metrics(api="openai"))
        snap = reg.snapshot()
        snap["by_api"]["openai"] = 999
        snap["requests_total"] = 999

        assert reg.snapshot()["by_api"]["openai"] == 1
        assert reg.snapshot()["requests_total"] == 1

    def test_reset_clears_all(self):
        """
        What it does: reset returns all aggregates to zero.
        Purpose: Test isolation and operational reset.
        """
        reg = MetricsRegistry()
        reg.record(_make_metrics(api="openai", prompt_tokens=10, total_ms=5.0))
        reg.reset()
        snap = reg.snapshot()
        assert snap["requests_total"] == 0
        assert snap["by_api"] == {}
        assert snap["prompt_tokens_total"] == 0
        assert snap["latency_ms_count"] == 0


# ==================================================================================================
# Context-bound helpers
# ==================================================================================================

class TestContextHelpers:
    """Tests for enrichment helpers that operate on the current request context."""

    def test_helpers_are_noops_without_context(self):
        """
        What it does: All enrichment helpers no-op when no request is in flight.
        Purpose: Calling them outside a request (or when disabled) must be safe.
        """
        # Ensure no current metrics.
        assert obs.current_metrics() is None
        enrich_current_metrics(api="openai", model="m")
        mark_first_token()
        add_token_usage(1, 2)
        note_retry()
        # Nothing to assert beyond "did not raise"; current metrics still None.
        assert obs.current_metrics() is None

    def test_enrich_sets_known_fields_and_ignores_unknown_and_none(self):
        """
        What it does: enrich sets known non-None fields, ignores unknown keys/None.
        Purpose: Callers may pass a superset; bad keys must not crash.
        """
        m = _make_metrics()
        token = obs._current_metrics.set(m)
        try:
            enrich_current_metrics(api="anthropic", model=None, not_a_field="x", tool_count=7)
        finally:
            obs._current_metrics.reset(token)

        assert m.api == "anthropic"
        assert m.model is None  # None skipped
        assert m.tool_count == 7
        assert not hasattr(m, "not_a_field")

    def test_mark_first_token_is_idempotent(self):
        """
        What it does: Only the first mark_first_token call sets first_token_ms.
        Purpose: Subsequent chunks must not overwrite the first-token timing.
        """
        m = _make_metrics()
        token = obs._current_metrics.set(m)
        try:
            mark_first_token()
            first = m.first_token_ms
            assert first is not None
            mark_first_token()
            assert m.first_token_ms == first
        finally:
            obs._current_metrics.reset(token)

    def test_add_token_usage_assigns_and_skips_none(self):
        """
        What it does: add_token_usage sets provided values, leaves None untouched.
        Purpose: Allow partial updates without clobbering existing values.
        """
        m = _make_metrics(prompt_tokens=5, completion_tokens=5)
        token = obs._current_metrics.set(m)
        try:
            add_token_usage(None, 42)
        finally:
            obs._current_metrics.reset(token)
        assert m.prompt_tokens == 5  # None left unchanged
        assert m.completion_tokens == 42

    def test_note_retry_accumulates(self):
        """
        What it does: note_retry increments the retry counter.
        Purpose: Multiple re-issues accumulate.
        """
        m = _make_metrics()
        token = obs._current_metrics.set(m)
        try:
            note_retry()
            note_retry(2)
        finally:
            obs._current_metrics.reset(token)
        assert m.retries == 3


# ==================================================================================================
# new_request_id
# ==================================================================================================

class TestRequestId:
    """Tests for request id generation."""

    def test_id_is_12_hex_chars_and_unique(self):
        """
        What it does: new_request_id returns distinct 12-char hex strings.
        Purpose: Compact, collision-resistant correlation ids.
        """
        ids = {new_request_id() for _ in range(1000)}
        assert len(ids) == 1000  # no collisions in a small sample
        for rid in list(ids)[:20]:
            assert len(rid) == 12
            int(rid, 16)  # valid hex (raises ValueError otherwise)


# ==================================================================================================
# JSONL sink
# ==================================================================================================

class TestJsonlSink:
    """Tests for the rolling JSONL request log."""

    def test_append_writes_one_line_per_call(self, tmp_path, monkeypatch):
        """
        What it does: _append_jsonl writes one JSON line per call into a dated file.
        Purpose: The offline analysis dataset must be valid JSONL.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_LOG_DIR", str(tmp_path))
        obs._append_jsonl({"a": 1})
        obs._append_jsonl({"b": 2})

        files = glob.glob(str(tmp_path / "requests-*.jsonl"))
        assert len(files) == 1
        lines = open(files[0], encoding="utf-8").read().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_jsonl_path_creates_directory(self, tmp_path, monkeypatch):
        """
        What it does: _jsonl_path creates the directory if missing.
        Purpose: First write must not fail on a fresh deployment.
        """
        target = tmp_path / "nested" / "obs"
        monkeypatch.setattr(obs, "OBSERVABILITY_LOG_DIR", str(target))
        path = obs._jsonl_path()
        assert target.exists()
        assert path.name.startswith("requests-") and path.name.endswith(".jsonl")


# ==================================================================================================
# record_request
# ==================================================================================================

class TestRecordRequest:
    """Tests for record_request orchestration."""

    def test_records_into_registry_and_jsonl_when_enabled(self, tmp_path, monkeypatch):
        """
        What it does: record_request aggregates and writes JSONL when enabled.
        Purpose: The end-to-end recording path.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", True)
        monkeypatch.setattr(obs, "OBSERVABILITY_JSONL_ENABLED", True)
        monkeypatch.setattr(obs, "OBSERVABILITY_LOG_DIR", str(tmp_path))

        record_request(_make_metrics(api="openai", status_code=200, outcome=OUTCOME_COMPLETED, total_ms=10.0))

        assert obs.registry.snapshot()["requests_total"] == 1
        files = glob.glob(str(tmp_path / "requests-*.jsonl"))
        assert files and open(files[0]).read().strip()

    def test_disabled_records_nothing(self, tmp_path, monkeypatch):
        """
        What it does: When disabled, record_request neither aggregates nor writes.
        Purpose: The master switch must fully silence observability.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", False)
        monkeypatch.setattr(obs, "OBSERVABILITY_LOG_DIR", str(tmp_path))

        record_request(_make_metrics(api="openai", status_code=200))

        assert obs.registry.snapshot()["requests_total"] == 0
        assert glob.glob(str(tmp_path / "requests-*.jsonl")) == []

    def test_is_fail_safe_when_registry_raises(self, monkeypatch):
        """
        What it does: record_request swallows internal failures.
        Purpose: Instrumentation must never crash a request, even on a bug.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", True)
        monkeypatch.setattr(obs, "OBSERVABILITY_JSONL_ENABLED", False)

        def _boom(_metrics):
            raise RuntimeError("registry exploded")

        monkeypatch.setattr(obs.registry, "record", _boom)
        # Must not raise despite the registry blowing up.
        record_request(_make_metrics())


# ==================================================================================================
# ObservabilityMiddleware
# ==================================================================================================

async def _drive(middleware, scope):
    """Drive a middleware once and return the list of sent ASGI messages."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)
    return sent


class TestObservabilityMiddleware:
    """Tests for the pure-ASGI observability middleware."""

    @pytest.mark.asyncio
    async def test_http_request_records_and_injects_header(self, monkeypatch):
        """
        What it does: An HTTP request is timed, status-captured, header-injected,
                      enriched in-context, and recorded once.
        Purpose: The core middleware happy path across the full lifecycle.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", True)
        monkeypatch.setattr(obs, "OBSERVABILITY_JSONL_ENABLED", False)

        async def app(scope, receive, send):
            # Enrichment from "inside" the request must see the bound metrics.
            assert obs.current_metrics() is not None
            enrich_current_metrics(api="anthropic", model="claude")
            mark_first_token()
            await send({"type": "http.response.start", "status": 200, "headers": [(b"x", b"y")]})
            await send({"type": "http.response.body", "body": b"{}"})

        scope = {
            "type": "http", "method": "POST", "path": "/v1/messages",
            "headers": [(b"content-length", b"2048")],
        }
        sent = await _drive(ObservabilityMiddleware(app), scope)

        start = next(m for m in sent if m["type"] == "http.response.start")
        headers = dict(start["headers"])
        assert b"x-kiro-gateway-request-id" in headers
        assert len(headers[b"x-kiro-gateway-request-id"]) == 12

        snap = obs.registry.snapshot()
        assert snap["requests_total"] == 1
        assert snap["by_api"] == {"anthropic": 1}
        assert snap["by_status_class"] == {"2xx": 1}
        assert snap["latency_ms_count"] == 1

    @pytest.mark.asyncio
    async def test_resets_context_after_request(self, monkeypatch):
        """
        What it does: request_id_var returns to its default after the request.
        Purpose: No context leakage between requests.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", True)
        monkeypatch.setattr(obs, "OBSERVABILITY_JSONL_ENABLED", False)

        async def app(scope, receive, send):
            assert obs.get_request_id() != "-"
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        scope = {"type": "http", "method": "GET", "path": "/health", "headers": []}
        await _drive(ObservabilityMiddleware(app), scope)

        assert obs.get_request_id() == "-"
        assert obs.current_metrics() is None

    @pytest.mark.asyncio
    async def test_app_exception_records_error_and_reraises(self, monkeypatch):
        """
        What it does: When the app raises, the request is recorded as an error and
                      the exception propagates.
        Purpose: Failures must be observable and not swallowed.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", True)
        monkeypatch.setattr(obs, "OBSERVABILITY_JSONL_ENABLED", False)

        async def app(scope, receive, send):
            raise ValueError("boom")

        scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
        with pytest.raises(ValueError, match="boom"):
            await _drive(ObservabilityMiddleware(app), scope)

        snap = obs.registry.snapshot()
        assert snap["requests_total"] == 1
        assert snap["by_outcome"] == {OUTCOME_ERROR: 1}

    @pytest.mark.asyncio
    async def test_non_http_scope_is_passthrough(self, monkeypatch):
        """
        What it does: Non-HTTP scopes (lifespan/websocket) are passed through.
        Purpose: The middleware must not record or alter non-HTTP traffic.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", True)
        called = {"n": 0}

        async def app(scope, receive, send):
            called["n"] += 1

        await ObservabilityMiddleware(app)({"type": "lifespan"}, None, None)
        assert called["n"] == 1
        assert obs.registry.snapshot()["requests_total"] == 0

    @pytest.mark.asyncio
    async def test_disabled_is_passthrough_without_header(self, monkeypatch):
        """
        What it does: When disabled, the middleware adds no header and records nothing.
        Purpose: The master switch fully disables the middleware.
        """
        monkeypatch.setattr(obs, "OBSERVABILITY_ENABLED", False)

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        scope = {"type": "http", "method": "GET", "path": "/health", "headers": []}
        sent = await _drive(ObservabilityMiddleware(app), scope)

        start = next(m for m in sent if m["type"] == "http.response.start")
        assert dict(start["headers"]).get(b"x-kiro-gateway-request-id") is None
        assert obs.registry.snapshot()["requests_total"] == 0


# ==================================================================================================
# Prometheus exposition + /metrics endpoint
# ==================================================================================================

class TestPrometheus:
    """Tests for render_prometheus and the /metrics endpoint."""

    def test_render_contains_core_series_after_recording(self):
        """
        What it does: Rendered text reflects recorded requests with HELP/TYPE.
        Purpose: Counters must surface in valid Prometheus exposition.
        """
        obs.registry.record(_make_metrics(
            api="openai", model="m1", status_code=200, outcome=OUTCOME_COMPLETED,
            prompt_tokens=100, completion_tokens=20, total_ms=300.0,
        ))
        text = obs.render_prometheus()

        assert "# HELP kiro_gateway_requests_total" in text
        assert "# TYPE kiro_gateway_requests_total counter" in text
        assert "kiro_gateway_requests_total 1" in text
        assert 'kiro_gateway_requests_by_api{api="openai"} 1' in text
        assert 'kiro_gateway_requests_by_status{status_class="2xx"} 1' in text
        assert "kiro_gateway_prompt_tokens_total 100" in text
        assert "kiro_gateway_completion_tokens_total 20" in text
        assert text.endswith("\n")

    def test_render_latency_histogram_is_cumulative_with_inf_and_count(self):
        """
        What it does: Histogram emits cumulative buckets, +Inf, _sum and _count.
        Purpose: Conform to the Prometheus histogram contract.
        """
        obs.registry.record(_make_metrics(total_ms=300.0))
        text = obs.render_prometheus()

        # 300ms lands in le="500" (and above), not in le="250".
        assert 'kiro_gateway_request_latency_ms_bucket{le="250"} 0' in text
        assert 'kiro_gateway_request_latency_ms_bucket{le="500"} 1' in text
        assert 'kiro_gateway_request_latency_ms_bucket{le="+Inf"} 1' in text
        assert "kiro_gateway_request_latency_ms_count 1" in text
        assert "kiro_gateway_request_latency_ms_sum 300.000" in text

    def test_render_escapes_label_values(self):
        """
        What it does: Model names with quotes/backslashes are escaped in labels.
        Purpose: Prevent malformed exposition (and label injection) from odd names.
        """
        obs.registry.record(_make_metrics(model='weird"\\name', status_code=200))
        text = obs.render_prometheus()

        assert 'kiro_gateway_requests_by_model{model="weird\\"\\\\name"} 1' in text

    def test_render_is_stable_when_empty(self):
        """
        What it does: Rendering an empty registry still yields valid output.
        Purpose: /metrics must work before any traffic.
        """
        text = obs.render_prometheus()
        assert "kiro_gateway_requests_total 0" in text
        assert 'kiro_gateway_request_latency_ms_bucket{le="+Inf"} 0' in text

    def test_metrics_endpoint_returns_prometheus_text_unauthenticated(self):
        """
        What it does: GET /metrics returns 200 text/plain with the exposition and
                      requires no authentication.
        Purpose: Local Prometheus scraping must work without an API key.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        obs.registry.record(_make_metrics(api="anthropic", status_code=200))

        app = FastAPI()
        app.include_router(obs.metrics_router)
        client = TestClient(app)

        response = client.get("/metrics")  # no Authorization / x-api-key header
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "kiro_gateway_requests_total" in response.text
        assert 'kiro_gateway_requests_by_api{api="anthropic"}' in response.text
