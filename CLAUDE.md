# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Primary reference: AGENTS.md

**Read `AGENTS.md` first.** It is the authoritative guide for AI agents working in this repo and covers project philosophy, coding conventions, architecture details, testing discipline, and gotchas in depth. This file is a short orientation layer — AGENTS.md is the long-form contract, and it overrides anything ambiguous here.

## What this project is

Kiro Gateway is a Python 3.10+ FastAPI reverse proxy that exposes OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) and Anthropic-compatible (`/v1/messages`) APIs on top of Kiro API (Amazon Q Developer / AWS CodeWhisperer). It translates request/response formats, handles auth token lifecycle across four credential sources, resolves model names, and streams AWS event-stream SSE back in either OpenAI or Anthropic wire format.

Entry point: `main.py`. Package: `kiro/`. License: AGPL-3.0.

## Common commands

```bash
# Run server
python main.py                              # default 0.0.0.0:8000
python main.py --host 127.0.0.1 --port 9000

# Tests (pytest config in pytest.ini — testpaths=tests, pythonpath=.)
pytest -v                                   # full suite
pytest tests/unit/ -v                       # unit only
pytest tests/unit/test_auth_manager.py -v   # one file
pytest tests/unit/test_auth_manager.py::TestKiroAuthManagerInitialization::test_initialization_stores_credentials -v
pytest -x                                   # stop on first failure
pytest --cov=kiro --cov-report=html         # coverage (needs pytest-cov)

# Manual real-API smoke test (excluded from pytest auto-collection)
python manual_api_test.py

# Dependencies
pip install -r requirements.txt

# Docker
docker-compose up -d
docker-compose logs -f
```

## Architecture at a glance

The codebase is laid out as parallel OpenAI/Anthropic stacks sharing a core:

- **Routes** (`routes_openai.py`, `routes_anthropic.py`) — FastAPI endpoints, API-key auth, request validation.
- **Converters** (`converters_openai.py`, `converters_anthropic.py`, `converters_core.py`) — client format → Kiro payload. Shared UnifiedMessage/tool-sanitization logic lives in `converters_core.py`.
- **Streaming** (`streaming_openai.py`, `streaming_anthropic.py`, `streaming_core.py`) — Kiro AWS event-stream SSE → client SSE. Shared parse/first-token-retry logic lives in `streaming_core.py`.
- **Parsers** (`parsers.py`, `thinking_parser.py`) — AWS event-stream framing and FSM-based `<thinking>` block extraction.
- **Core services** — `auth.py` (KiroAuthManager, 4 auth types auto-detected), `account_manager.py` + `account_errors.py` (multi-account failover with circuit breaker), `http_client.py` (retry on 403/429/5xx with token refresh), `model_resolver.py` (4-layer: normalize → dynamic cache → hidden models → pass-through), `cache.py`, `tokenizer.py`.
- **Hardening** — `network_errors.py`, `kiro_errors.py`, `payload_guards.py`, `truncation_state.py` + `truncation_recovery.py`, `exceptions.py`.
- **Observability** — `debug_logger.py` + `debug_middleware.py` write per-request artifacts to `debug_logs/` when `DEBUG_MODE` is `errors` or `all`.
- **Models** — Pydantic request/response schemas in `models_openai.py` / `models_anthropic.py`.

**Feature-consistency rule (from AGENTS.md §10):** any new functionality MUST land in both OpenAI and Anthropic paths AND in both streaming and non-streaming modes, with tests for all four combinations. One-API-only changes are rejected.

**Gateway-not-gatekeeper rule:** Unknown model names pass through to Kiro; we fix API quirks, never user content.

## Non-obvious gotchas

- **Streaming must use per-request `httpx.AsyncClient`s.** Reusing a shared client for streaming causes CLOSE_WAIT leaks. Non-streaming requests do use the pooled shared client.
- **Model name normalization** (`model_resolver.normalize_model_name`): dashes in minor versions become dots, date suffixes are stripped. `claude-haiku-4-5-20251001` → `claude-haiku-4.5` before resolution.
- **Auth type is auto-detected** from credential contents: presence of `clientId`/`clientSecret` ⇒ AWS SSO OIDC; absence ⇒ Kiro Desktop Auth. Four credential sources supported: JSON file (`KIRO_CREDS_FILE`), env var (`REFRESH_TOKEN`), kiro-cli SQLite (`KIRO_CLI_DB_FILE`), and folder-of-files via the multi-account `credentials.json` (`ACCOUNT_SYSTEM=true`).
- **Tool calls may arrive as bracket-wrapped JSON** `[{...}]` instead of a proper array; the parser already handles this — don't "fix" it upstream.
- **Thinking blocks** are extracted by an FSM in `thinking_parser.py`, not by regex. Preserve FSM semantics when editing.
- **"Improperly formed request"** from Kiro is deliberately ambiguous upstream — debug it via `DEBUG_MODE=errors` artifacts in `debug_logs/`, not by guessing.

## Testing discipline (non-negotiable)

`tests/conftest.py` installs a session-wide `block_all_network_calls` fixture that fails any test which attempts a real HTTP call. All external services must be mocked.

- Unit tests: `tests/unit/`. Integration tests: `tests/integration/`.
- Reuse the existing `test_<module>.py` files — AGENTS.md explicitly asks you to check `tests/README.md` before creating a new test file. Only add a new file for a genuinely new module.
- Tests must attempt to break the code. Happy-path-only PRs get rejected (AGENTS.md §7 and "Code Review Reality Check").
- Test classes: `Test*Success`, `Test*Errors`, `Test*EdgeCases`. Arrange-Act-Assert layout.
- `manual_api_test.py` hits the real API and is excluded from `pytest` via `pytest.ini`; run it only deliberately.

## Code conventions (enforced)

- Python 3.10+, `snake_case` / `PascalCase` / `UPPER_SNAKE_CASE`, private prefix `_`.
- Mandatory type hints on every parameter and return.
- Google-style docstrings with Args/Returns/Raises on every function.
- Logging via `loguru` at decision points — INFO for business logic, DEBUG for technical detail, ERROR for failures. Never log tokens or credentials.
- Never use bare `except:` or `except Exception:` — catch specific exceptions with context.
- All code, comments, and identifiers in English. Non-English identifiers are a rejection signal.
- All I/O is `async`.

## Configuration surface

Loaded from `.env` (template in `.env.example`). Required: `PROXY_API_KEY` (your chosen gateway password) plus one of the four credential sources above. Useful optional knobs: `PROFILE_ARN`, `KIRO_REGION` / `KIRO_API_REGION`, `SERVER_HOST`, `SERVER_PORT`, `VPN_PROXY_URL` (HTTP or SOCKS5, used for restricted networks), `ACCOUNT_SYSTEM=true` (activates `credentials.json` multi-account mode), `DEBUG_MODE=off|errors|all`. Priority: CLI args > env vars > defaults.

## Commit style

`<type>(<scope>): <description>` — types `feat|fix|docs|test|refactor|chore`. See `git log --oneline -20` for recent examples. There is a CLA (`CLA.md`) that contributors agree to on PR submission.

## Transient `INVALID_MODEL_ID` recovery (implemented and verified)

Kiro can intermittently return HTTP 400 with a top-level string `reason: "INVALID_MODEL_ID"` for a valid model. The shared `KiroHttpClient.request_with_retry()` layer now treats only that exact structured response as transient, so the behavior applies consistently to OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages in both streaming and non-streaming modes.

- Default budget: four total attempts, including the initial request.
- Backoff: exponential `1s / 2s / 4s` by default, with bounded jitter and an 8-second delay cap.
- Discarded responses are closed before retry to avoid connection leaks; the final response remains caller-owned.
- This error path never calls `force_refresh()` and does not refresh or mutate credentials.
- Other HTTP 400 responses return immediately. Malformed, nested, non-string, or otherwise different `reason` values are not retried.
- When the retry budget is exhausted, the final original 400 continues through existing error shaping and multi-account failover. A genuinely invalid or unavailable model therefore remains bounded rather than looping forever.

The public tuning variables are documented in `.env.example`:

```env
INVALID_MODEL_MAX_RETRIES=4
INVALID_MODEL_BASE_RETRY_DELAY=1.0
INVALID_MODEL_MAX_RETRY_DELAY=8.0
INVALID_MODEL_RETRY_JITTER_RATIO=0.25
```

Automated verification completed on 2026-08-23:

- Full isolated regression suite: **1,987 passed, 0 failed**.
- Focused config, HTTP-client, and cross-surface suites: **116 passed**.
- Cross-surface retry integration matrix: **6 passed**, covering all three APIs and both response modes with the real shared retry loop and a mocked transport.
- Independent scoped review found no defects in exact matching, retry bounds, response ownership, credential safety, or structural compatibility with the existing account-failover path. Retry exhaustion returns the final response unchanged; no new end-to-end multi-account exhaustion case was added.

Manual operational verification recorded on the same date (historical evidence, not reproduced by pytest):

- Live Kiro validation captured three consecutive transient `INVALID_MODEL_ID` responses followed by HTTP 200 on the fourth attempt, demonstrating retry-to-success with the configured `1s / 2s / 4s` backoff.
- The deployed gateway was restarted with the implementation loaded, discovered 19 models, passed `/health`, and returned a valid HTTP 200 OpenAI Responses result for `claude-opus-4.8`.
- Credential-content fingerprints and kiro-cli authentication-row fingerprints were unchanged by verification. Legacy startup may rewrite `credentials.json` with identical content, changing only its modification time.

A live upstream rejection window can outlast the default four-attempt budget. In that case the gateway deliberately returns the final 400; operators who prefer a longer availability window over lower worst-case latency can raise `INVALID_MODEL_MAX_RETRIES` and/or the delay settings without changing code.
