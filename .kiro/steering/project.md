# Project: Kiro Gateway

Python 3.10+ **FastAPI reverse proxy** that exposes OpenAI-compatible
(`/v1/chat/completions`, `/v1/responses`, `/v1/models`) and Anthropic-compatible
(`/v1/messages`) APIs on top of **Kiro API (Amazon Q Developer / AWS
CodeWhisperer)**. It translates request/response formats, manages the auth-token
lifecycle across four credential sources, resolves model names, and re-streams
AWS event-stream SSE in OpenAI, Responses, or Anthropic wire format.

- Entry point: `main.py` · Package: `kiro/` · License: AGPL-3.0
- **`AGENTS.md` is the authoritative long-form contract.** These steering files are
  the always-loaded distillation; read `AGENTS.md` before non-trivial work.

## Architecture map (parallel API adapters over a shared core)

- **Routes** — `routes_openai.py`, `routes_anthropic.py`, `routes_common.py`:
  endpoints, API-key auth, validation, and shared request execution.
- **Converters** — `converters_openai.py`, `converters_anthropic.py`,
  `converters_responses.py`, `converters_core.py`: client format → Kiro payload.
- **Streaming** — `streaming_openai.py`, `streaming_anthropic.py`,
  `streaming_responses.py`, `streaming_core.py`: Kiro event stream → client SSE.
  Shared parsing and first-token retry live in `streaming_core.py`.
- **Parsers** — `parsers.py` (AWS event framing), `thinking_parser.py` (FSM
  `<thinking>` extraction).
- **Core services** — `auth.py` (four auth types), `account_manager.py` +
  `account_errors.py` (failover/circuit breaker), `http_client.py` (shared HTTP
  retries and response lifecycle), `model_resolver.py` (normalize → dynamic cache
  → hidden models → pass-through), `cache.py`, `tokenizer.py`.
- **Hardening** — `network_errors.py`, `kiro_errors.py`, `payload_guards.py`,
  `truncation_state.py` + `truncation_recovery.py`, `exceptions.py`.
- **Observability** — `observability.py`, `debug_logger.py`, `debug_middleware.py`.
- **Models** — `models_openai.py`, `models_anthropic.py`, `models_responses.py`.

## Shared transient model-error recovery

`KiroHttpClient.request_with_retry()` retries only HTTP 400 responses whose parsed
JSON object has the exact top-level string `reason == "INVALID_MODEL_ID"`. The
default is four total attempts with exponential `1s/2s/4s` backoff, bounded
jitter, and an 8-second delay cap. Discarded responses are closed; the final
response remains caller-owned. This branch never calls `force_refresh()` because
of `INVALID_MODEL_ID`; normal token-expiry handling in `get_access_token()` remains
unchanged. Exhaustion returns the final original 400 for existing error shaping
and account failover.

The shared layer covers OpenAI Chat, OpenAI Responses, and Anthropic Messages in
both streaming and non-streaming modes. Other 400 responses return immediately.

## Current verified state (2026-08-23)

- Implementation commit: `8a62feb` (`fix(retry): recover transient invalid model errors`).
- Full isolated regression: **1,987 passed, 0 failed**; focused affected suites:
  **116 passed**; six-surface integration matrix: **6 passed**.
- Manual live verification observed three transient failures followed by HTTP 200
  on attempt four, then restarted the deployed gateway with the change loaded.
- Deployment discovered 19 models, passed `/health`, and returned a valid HTTP 200
  OpenAI Responses result for `claude-opus-4.8`.
- Credential and kiro-cli auth-row content fingerprints remained unchanged.
- A real upstream rejection window can exceed four attempts; bounded exhaustion
  deliberately returns the final 400. Operators may tune the retry budget.

## Common commands

```bash
python main.py                       # run server (default 0.0.0.0:8000)
python main.py --host 127.0.0.1 --port 9000
pytest -v                            # full suite (config in pytest.ini)
pytest tests/unit/ -v                # unit only
pytest -x                            # stop on first failure
pytest --cov=kiro --cov-report=html  # coverage (needs pytest-cov)
python manual_api_test.py            # deliberate REAL-API smoke (excluded from pytest)
pip install -r requirements.txt
docker-compose up -d
```

## Configuration surface

Loaded from `.env` (template: `.env.example`). Required: `PROXY_API_KEY` plus one
credential source — `KIRO_CREDS_FILE`, `REFRESH_TOKEN`, `KIRO_CLI_DB_FILE`, or
`credentials.json` via `ACCOUNT_SYSTEM=true`. Important optional settings include
`PROFILE_ARN`, `KIRO_REGION`/`KIRO_API_REGION`, `SERVER_HOST`, `SERVER_PORT`,
`VPN_PROXY_URL`, `KIRO_EXTRA_CA_CERTS`, `DEBUG_MODE`, and:

```env
INVALID_MODEL_MAX_RETRIES=4
INVALID_MODEL_BASE_RETRY_DELAY=1.0
INVALID_MODEL_MAX_RETRY_DELAY=8.0
INVALID_MODEL_RETRY_JITTER_RATIO=0.25
```

Priority: CLI args > environment variables > defaults.
