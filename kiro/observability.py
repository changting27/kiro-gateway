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
Full-link observability for Kiro Gateway.

This module adds end-to-end, per-request observability without changing request
behaviour. It provides:

- A request-id propagated through a ``ContextVar`` so every log line of a request
  can be correlated, and returned to the client via the
  ``x-kiro-gateway-request-id`` response header.
- A ``RequestMetrics`` record accumulated per request (timings, token counts,
  model, message/tool counts, upstream status, outcome, etc.).
- An async-safe in-process ``MetricsRegistry`` aggregating counters and latency
  histograms (consumed by the Prometheus ``/metrics`` endpoint).
- A rolling daily JSONL sink (``observability/requests-YYYYMMDD.jsonl``) for
  offline analysis.
- ``ObservabilityMiddleware`` (pure ASGI) which owns the per-request lifecycle:
  it assigns the id, times the request, captures the response status, injects the
  response header, and records the metrics on completion.

Design rules:
- **Fail-safe**: instrumentation must never break request handling. Every recording
  path is wrapped so an observability error degrades to a debug log, not a 500.
- **Consistency**: works identically for OpenAI and Anthropic, streaming and
  non-streaming, because the middleware wraps every request and the enrichment
  helpers operate on a ``ContextVar`` that propagates down the await chain.
- The collection of rich fields (model, tokens, counts) is done by lightweight
  enrichment helpers (:func:`enrich_current_metrics`, :func:`mark_first_token`,
  :func:`add_token_usage`) called from the routes and the shared streaming core.
"""

import json
import time
import uuid
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from fastapi import APIRouter, Response

from kiro.config import (
    OBSERVABILITY_ENABLED,
    OBSERVABILITY_JSONL_ENABLED,
    OBSERVABILITY_LOG_DIR,
)


# ==================================================================================================
# Context variables (per-request, propagate down the await chain)
# ==================================================================================================

# Current request id. Defaults to "-" so log records outside any request still format.
request_id_var: ContextVar[str] = ContextVar("kiro_request_id", default="-")

# Current request's metrics object. None when no request is in flight.
_current_metrics: ContextVar[Optional["RequestMetrics"]] = ContextVar(
    "kiro_request_metrics", default=None
)


def new_request_id() -> str:
    """
    Generate a short, unique request id.

    Returns:
        A 12-character hexadecimal id (truncated UUID4), compact enough for logs
        and headers while remaining collision-resistant for per-process traffic.
    """
    return uuid.uuid4().hex[:12]


def get_request_id() -> str:
    """
    Return the request id bound to the current context.

    Returns:
        The current request id, or "-" when no request is in flight.
    """
    return request_id_var.get()


# ==================================================================================================
# Per-request metrics record
# ==================================================================================================

# Outcome classification for a finished request.
OUTCOME_COMPLETED = "completed"
OUTCOME_ERROR = "error"
OUTCOME_CLIENT_DISCONNECTED = "client_disconnected"


@dataclass
class RequestMetrics:
    """
    Mutable per-request observability record.

    Populated incrementally: the middleware sets transport-level fields (status,
    timings, payload size), while route/streaming enrichment helpers fill in the
    semantic fields (api, model, tokens, message/tool counts, upstream status).

    Attributes:
        request_id: Correlation id for this request.
        ts: ISO-8601 UTC timestamp captured at request start.
        method: HTTP method.
        path: Request path.
        api: Logical API surface ("openai" or "anthropic"); None for other paths.
        model: Client-requested model name.
        stream: Whether the client requested streaming.
        status_code: Final HTTP status code sent to the client.
        outcome: One of OUTCOME_COMPLETED / OUTCOME_ERROR / OUTCOME_CLIENT_DISCONNECTED.
        error_category: Network-error category when the request failed, else None.
        total_ms: Wall-clock duration of the whole request in milliseconds.
        first_token_ms: Time from request start to first upstream token (streaming).
        prompt_tokens: Estimated prompt tokens.
        completion_tokens: Generated completion tokens.
        message_count: Number of messages in the request.
        tool_count: Number of tools declared in the request.
        payload_bytes: Request body size from the Content-Length header.
        account_id: Account used (multi-account mode), else None.
        upstream_status: HTTP status returned by the upstream Kiro API.
        retries: Number of upstream re-attempts performed for this request.
    """

    request_id: str
    ts: str
    method: str
    path: str
    api: Optional[str] = None
    model: Optional[str] = None
    stream: Optional[bool] = None
    status_code: Optional[int] = None
    outcome: Optional[str] = None
    error_category: Optional[str] = None
    total_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    message_count: Optional[int] = None
    tool_count: Optional[int] = None
    payload_bytes: Optional[int] = None
    account_id: Optional[str] = None
    upstream_status: Optional[int] = None
    retries: int = 0

    # Internal monotonic start time (seconds); excluded from serialization.
    start_monotonic: float = field(default_factory=time.monotonic, repr=False)

    def elapsed_ms(self) -> float:
        """Return milliseconds elapsed since the request started."""
        return (time.monotonic() - self.start_monotonic) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-friendly dict (excludes internal timing state).

        Returns:
            A mapping suitable for JSONL output, with None fields kept so
            downstream tooling sees a stable schema.
        """
        data = asdict(self)
        data.pop("start_monotonic", None)
        # Round float timings for compact, stable output.
        for key in ("total_ms", "first_token_ms"):
            if isinstance(data.get(key), float):
                data[key] = round(data[key], 1)
        return data

    def summary_line(self) -> str:
        """
        Build a compact single-line human-readable summary for the application log.

        Returns:
            A space-separated ``key=value`` string omitting unset fields.
        """
        parts: List[str] = [f"rid={self.request_id}"]
        if self.api:
            parts.append(f"api={self.api}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.stream is not None:
            parts.append(f"stream={self.stream}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.outcome:
            parts.append(f"outcome={self.outcome}")
        if self.error_category:
            parts.append(f"error={self.error_category}")
        if self.total_ms is not None:
            parts.append(f"total_ms={self.total_ms:.0f}")
        if self.first_token_ms is not None:
            parts.append(f"first_token_ms={self.first_token_ms:.0f}")
        if self.prompt_tokens is not None:
            parts.append(f"prompt_tokens={self.prompt_tokens}")
        if self.completion_tokens is not None:
            parts.append(f"completion_tokens={self.completion_tokens}")
        if self.message_count is not None:
            parts.append(f"messages={self.message_count}")
        if self.tool_count is not None:
            parts.append(f"tools={self.tool_count}")
        if self.payload_bytes is not None:
            parts.append(f"payload_bytes={self.payload_bytes}")
        if self.upstream_status is not None:
            parts.append(f"upstream={self.upstream_status}")
        if self.retries:
            parts.append(f"retries={self.retries}")
        if self.account_id:
            parts.append(f"account={self.account_id}")
        return " ".join(parts)


def outcome_from_status(status_code: Optional[int]) -> str:
    """
    Derive a default outcome from an HTTP status code.

    Args:
        status_code: Final HTTP status code, or None when unknown.

    Returns:
        OUTCOME_COMPLETED for < 400 (and unknown), OUTCOME_ERROR otherwise.
    """
    if status_code is not None and status_code >= 400:
        return OUTCOME_ERROR
    return OUTCOME_COMPLETED


# ==================================================================================================
# In-process metrics registry
# ==================================================================================================

# Upper bounds (milliseconds) for the latency histogram buckets. The implicit
# "+Inf" bucket is represented by total counts.
LATENCY_BUCKETS_MS: List[float] = [
    50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 30000, 60000, 120000
]


class MetricsRegistry:
    """
    Thread-safe in-process aggregation of request metrics.

    Holds monotonic counters (by status class, api, outcome, error category, model)
    plus latency/first-token histograms and token totals. Designed for a single
    process; a ``threading.Lock`` guards updates so it is safe even if a future
    sync endpoint records from a worker thread.
    """

    def __init__(self) -> None:
        """Initialize all counters and histograms to zero."""
        self._lock = threading.Lock()
        self.requests_total: int = 0
        self.by_status_class: Dict[str, int] = {}
        self.by_api: Dict[str, int] = {}
        self.by_outcome: Dict[str, int] = {}
        self.by_error_category: Dict[str, int] = {}
        self.by_model: Dict[str, int] = {}
        self.prompt_tokens_total: int = 0
        self.completion_tokens_total: int = 0
        self.retries_total: int = 0
        self.latency_ms_sum: float = 0.0
        self.latency_ms_count: int = 0
        self.latency_buckets: Dict[float, int] = {b: 0 for b in LATENCY_BUCKETS_MS}
        self.first_token_ms_sum: float = 0.0
        self.first_token_ms_count: int = 0

    @staticmethod
    def _status_class(status_code: Optional[int]) -> str:
        """Map a status code to a class label like '2xx' (or 'unknown')."""
        if status_code is None:
            return "unknown"
        return f"{status_code // 100}xx"

    def _bump(self, mapping: Dict[str, int], key: Optional[str]) -> None:
        """Increment ``mapping[key]`` by one, skipping None keys."""
        if key is None:
            return
        mapping[key] = mapping.get(key, 0) + 1

    def record(self, metrics: "RequestMetrics") -> None:
        """
        Fold a finished request's metrics into the aggregates.

        Args:
            metrics: The completed per-request record.
        """
        with self._lock:
            self.requests_total += 1
            self._bump(self.by_status_class, self._status_class(metrics.status_code))
            self._bump(self.by_api, metrics.api)
            self._bump(self.by_outcome, metrics.outcome)
            self._bump(self.by_error_category, metrics.error_category)
            self._bump(self.by_model, metrics.model)

            if metrics.prompt_tokens:
                self.prompt_tokens_total += metrics.prompt_tokens
            if metrics.completion_tokens:
                self.completion_tokens_total += metrics.completion_tokens
            if metrics.retries:
                self.retries_total += metrics.retries

            if metrics.total_ms is not None:
                self.latency_ms_sum += metrics.total_ms
                self.latency_ms_count += 1
                for bound in LATENCY_BUCKETS_MS:
                    if metrics.total_ms <= bound:
                        self.latency_buckets[bound] += 1

            if metrics.first_token_ms is not None:
                self.first_token_ms_sum += metrics.first_token_ms
                self.first_token_ms_count += 1

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a consistent copy of all aggregates for rendering or testing.

        Returns:
            A copy (nested dicts duplicated) of the current counters/histograms.
        """
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "by_status_class": dict(self.by_status_class),
                "by_api": dict(self.by_api),
                "by_outcome": dict(self.by_outcome),
                "by_error_category": dict(self.by_error_category),
                "by_model": dict(self.by_model),
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
                "retries_total": self.retries_total,
                "latency_ms_sum": self.latency_ms_sum,
                "latency_ms_count": self.latency_ms_count,
                "latency_buckets": dict(self.latency_buckets),
                "first_token_ms_sum": self.first_token_ms_sum,
                "first_token_ms_count": self.first_token_ms_count,
            }

    def reset(self) -> None:
        """Reset all aggregates to zero (primarily for tests)."""
        with self._lock:
            self.__init__()


# Global registry instance shared across the process.
registry = MetricsRegistry()


# ==================================================================================================
# JSONL sink
# ==================================================================================================

def _jsonl_path() -> Path:
    """
    Compute today's rolling JSONL file path, creating the directory if needed.

    Returns:
        Path to ``<OBSERVABILITY_LOG_DIR>/requests-YYYYMMDD.jsonl`` (UTC day).
    """
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    directory = Path(OBSERVABILITY_LOG_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"requests-{day}.jsonl"


def _append_jsonl(record: Dict[str, Any]) -> None:
    """
    Append one record as a JSON line to today's rolling file.

    Args:
        record: The serialized request metrics.
    """
    line = json.dumps(record, ensure_ascii=False)
    path = _jsonl_path()
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# ==================================================================================================
# Recording
# ==================================================================================================

def record_request(metrics: "RequestMetrics") -> None:
    """
    Record a finished request: aggregate, log a summary line, and append JSONL.

    Every step is individually guarded so an observability failure never propagates
    into request handling.

    Args:
        metrics: The completed per-request record.
    """
    if not OBSERVABILITY_ENABLED:
        return

    # Aggregate into the in-process registry.
    try:
        registry.record(metrics)
    except Exception as exc:  # instrumentation must never crash a request
        logger.debug(f"[obs] registry record failed: {exc}")

    # Emit the human-readable per-request summary line.
    try:
        logger.info(f"[obs] {metrics.summary_line()}")
    except Exception as exc:  # instrumentation must never crash a request
        logger.debug(f"[obs] summary log failed: {exc}")

    # Append the structured record to the rolling JSONL file.
    if OBSERVABILITY_JSONL_ENABLED:
        try:
            _append_jsonl(metrics.to_dict())
        except Exception as exc:  # instrumentation must never crash a request
            logger.debug(f"[obs] jsonl append failed: {exc}")


# ==================================================================================================
# Enrichment helpers (called from routes and the streaming core)
# ==================================================================================================

def current_metrics() -> Optional["RequestMetrics"]:
    """
    Return the metrics record bound to the current request context.

    Returns:
        The active :class:`RequestMetrics`, or None when no request is in flight
        (e.g. observability disabled, or called outside a request).
    """
    return _current_metrics.get()


def enrich_current_metrics(**fields: Any) -> None:
    """
    Update fields on the current request's metrics record.

    Unknown field names are ignored so callers can pass a superset safely. None
    values are skipped. No-ops when there is no active request.

    Args:
        **fields: Attribute name/value pairs to set on :class:`RequestMetrics`.
    """
    metrics = _current_metrics.get()
    if metrics is None:
        return
    for name, value in fields.items():
        if value is None:
            continue
        if hasattr(metrics, name):
            setattr(metrics, name, value)


def mark_first_token() -> None:
    """
    Record the time-to-first-token, measured from request start.

    Idempotent: only the first call per request takes effect. No-ops when there is
    no active request. Intended to be called from the streaming parser when the
    first upstream chunk arrives.
    """
    metrics = _current_metrics.get()
    if metrics is None or metrics.first_token_ms is not None:
        return
    metrics.first_token_ms = metrics.elapsed_ms()


def add_token_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    """
    Set prompt/completion token counts on the current request's metrics.

    No-ops when there is no active request. Values are assigned (not accumulated)
    because each request finalizes its usage exactly once.

    Args:
        prompt_tokens: Estimated prompt tokens, or None to leave unchanged.
        completion_tokens: Generated completion tokens, or None to leave unchanged.
    """
    metrics = _current_metrics.get()
    if metrics is None:
        return
    if prompt_tokens is not None:
        metrics.prompt_tokens = prompt_tokens
    if completion_tokens is not None:
        metrics.completion_tokens = completion_tokens


def note_retry(count: int = 1) -> None:
    """
    Increment the upstream retry counter on the current request's metrics.

    No-ops when there is no active request.

    Args:
        count: Number of retries to add (default 1).
    """
    metrics = _current_metrics.get()
    if metrics is None:
        return
    metrics.retries += count


# ==================================================================================================
# Loguru request-id integration
# ==================================================================================================

def _request_id_patcher(record: Dict[str, Any]) -> None:
    """
    Loguru patcher injecting the current request id into every log record.

    Args:
        record: The loguru record whose ``extra`` mapping is updated in place.
    """
    record["extra"]["request_id"] = request_id_var.get()


def install_request_id_logging() -> None:
    """
    Install a global loguru patcher so every log record carries ``request_id``.

    Sets a default ``request_id`` of "-" and a patcher reading the context variable.
    Safe to call multiple times. Guarded so a loguru API mismatch cannot prevent
    startup; on failure, logging simply continues without the id.
    """
    try:
        logger.configure(extra={"request_id": "-"}, patcher=_request_id_patcher)
    except Exception as exc:  # never block startup on logging configuration
        logger.debug(f"[obs] request-id logging patch not installed: {exc}")


# ==================================================================================================
# ASGI middleware
# ==================================================================================================

class ObservabilityMiddleware:
    """
    Pure-ASGI middleware owning the per-request observability lifecycle.

    For every HTTP request it: assigns a request id and binds it (and a fresh
    :class:`RequestMetrics`) to context variables that propagate down the await
    chain; measures wall-clock duration; captures the response status; injects the
    ``x-kiro-gateway-request-id`` response header; and records the metrics once the
    response completes (or the app raises).

    A pure-ASGI implementation is used (rather than Starlette's BaseHTTPMiddleware)
    so the context variables reliably propagate to the route handlers and the
    streaming core running within the same task context.
    """

    def __init__(self, app: Any) -> None:
        """
        Store the wrapped ASGI application.

        Args:
            app: The next ASGI application in the stack.
        """
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        """
        Process one ASGI event scope, instrumenting HTTP requests.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        if scope.get("type") != "http" or not OBSERVABILITY_ENABLED:
            await self.app(scope, receive, send)
            return

        rid = new_request_id()
        metrics = RequestMetrics(
            request_id=rid,
            ts=datetime.now(timezone.utc).isoformat(),
            method=scope.get("method", ""),
            path=scope.get("path", ""),
        )
        metrics.payload_bytes = _content_length(scope)

        rid_token = request_id_var.set(rid)
        metrics_token = _current_metrics.set(metrics)

        rid_header = (b"x-kiro-gateway-request-id", rid.encode("ascii"))

        async def send_wrapper(message: Dict[str, Any]) -> None:
            """Capture the status code and inject the request-id response header."""
            if message["type"] == "http.response.start":
                metrics.status_code = message.get("status")
                headers = list(message.get("headers") or [])
                headers.append(rid_header)
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            metrics.total_ms = metrics.elapsed_ms()
            if metrics.outcome is None:
                metrics.outcome = OUTCOME_ERROR
            record_request(metrics)
            raise
        else:
            metrics.total_ms = metrics.elapsed_ms()
            if metrics.outcome is None:
                metrics.outcome = outcome_from_status(metrics.status_code)
            record_request(metrics)
        finally:
            request_id_var.reset(rid_token)
            _current_metrics.reset(metrics_token)


def _content_length(scope: Dict[str, Any]) -> Optional[int]:
    """
    Extract the request body size from the Content-Length header, if present.

    Args:
        scope: The ASGI connection scope.

    Returns:
        The parsed content length in bytes, or None when absent/invalid.
    """
    for key, value in scope.get("headers", []):
        if key == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


# ==================================================================================================
# Prometheus exposition
# ==================================================================================================

# Metric name prefix for all exported series.
_PROM_PREFIX = "kiro_gateway"


def _prom_escape_label(value: str) -> str:
    """
    Escape a Prometheus label value per the exposition format.

    Args:
        value: Raw label value.

    Returns:
        The value with backslash, double-quote, and newline escaped.
    """
    return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")


def _format_le(bound: float) -> str:
    """
    Format a histogram bucket bound as a Prometheus ``le`` value.

    Args:
        bound: Upper bound in milliseconds.

    Returns:
        An integer string when the bound is whole (e.g. "500"), else its repr.
    """
    return str(int(bound)) if float(bound).is_integer() else repr(bound)


def render_prometheus() -> str:
    """
    Render the current registry as Prometheus text exposition (version 0.0.4).

    Produces labelled counters for the request dimensions, a cumulative latency
    histogram, token/retry counters, and a first-token summary (sum/count). The
    output is a consistent snapshot taken under the registry lock.

    Returns:
        The exposition text, terminated by a trailing newline.
    """
    snap = registry.snapshot()
    lines: List[str] = []

    def counter(name: str, help_text: str, value: int) -> None:
        """Emit a single unlabelled counter series."""
        lines.append(f"# HELP {_PROM_PREFIX}_{name} {help_text}")
        lines.append(f"# TYPE {_PROM_PREFIX}_{name} counter")
        lines.append(f"{_PROM_PREFIX}_{name} {value}")

    def labelled(name: str, help_text: str, label: str, mapping: Dict[str, int]) -> None:
        """Emit a labelled counter series (one line per label value)."""
        lines.append(f"# HELP {_PROM_PREFIX}_{name} {help_text}")
        lines.append(f"# TYPE {_PROM_PREFIX}_{name} counter")
        for key, value in sorted(mapping.items()):
            lines.append(
                f'{_PROM_PREFIX}_{name}{{{label}="{_prom_escape_label(key)}"}} {value}'
            )

    counter("requests_total", "Total HTTP requests observed.", snap["requests_total"])
    labelled("requests_by_status", "Requests by HTTP status class.", "status_class", snap["by_status_class"])
    labelled("requests_by_api", "Requests by logical API surface.", "api", snap["by_api"])
    labelled("requests_by_outcome", "Requests by outcome.", "outcome", snap["by_outcome"])
    labelled("requests_by_error", "Requests by network error category.", "category", snap["by_error_category"])
    labelled("requests_by_model", "Requests by requested model.", "model", snap["by_model"])
    counter("prompt_tokens_total", "Total estimated prompt tokens.", snap["prompt_tokens_total"])
    counter("completion_tokens_total", "Total generated completion tokens.", snap["completion_tokens_total"])
    counter("upstream_retries_total", "Total upstream re-issues performed.", snap["retries_total"])

    # Latency histogram. Buckets are cumulative ("le"); the implicit +Inf bucket
    # equals the observation count.
    lines.append(f"# HELP {_PROM_PREFIX}_request_latency_ms Request wall-clock latency in milliseconds.")
    lines.append(f"# TYPE {_PROM_PREFIX}_request_latency_ms histogram")
    buckets = snap["latency_buckets"]
    for bound in LATENCY_BUCKETS_MS:
        lines.append(
            f'{_PROM_PREFIX}_request_latency_ms_bucket{{le="{_format_le(bound)}"}} {buckets.get(bound, 0)}'
        )
    lines.append(f'{_PROM_PREFIX}_request_latency_ms_bucket{{le="+Inf"}} {snap["latency_ms_count"]}')
    lines.append(f'{_PROM_PREFIX}_request_latency_ms_sum {snap["latency_ms_sum"]:.3f}')
    lines.append(f'{_PROM_PREFIX}_request_latency_ms_count {snap["latency_ms_count"]}')

    # First-token latency as a summary (sum/count only; no quantiles).
    lines.append(f"# HELP {_PROM_PREFIX}_first_token_ms Time-to-first-token in milliseconds (streaming).")
    lines.append(f"# TYPE {_PROM_PREFIX}_first_token_ms summary")
    lines.append(f'{_PROM_PREFIX}_first_token_ms_sum {snap["first_token_ms_sum"]:.3f}')
    lines.append(f'{_PROM_PREFIX}_first_token_ms_count {snap["first_token_ms_count"]}')

    return "\n".join(lines) + "\n"


# Router exposing the Prometheus endpoint; included by the application in main.py.
metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def metrics_endpoint() -> Response:
    """
    Expose the in-process metrics in Prometheus text exposition format.

    Unauthenticated by design: the gateway is intended to bind to a local
    interface and this endpoint exposes only aggregate counters (no request
    content and no credentials).

    Returns:
        A ``text/plain`` response with the Prometheus exposition body.
    """
    return Response(
        content=render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
