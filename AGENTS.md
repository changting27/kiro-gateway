# AGENTS.md - Kiro Gateway Development Guide

This file is the authoritative guide for AI coding agents working in this repository. It covers the project boundaries, architecture, coding conventions, testing discipline, and operational gotchas that must be respected when making changes.

## Project purpose

Kiro Gateway is a Python 3.10+ FastAPI reverse proxy that exposes OpenAI-compatible and Anthropic-compatible APIs on top of the Kiro API (Amazon Q Developer / AWS CodeWhisperer). It translates request and response formats, handles authentication across multiple credential sources, resolves model names, and converts AWS event streams into client-compatible SSE.

- Entry point: `main.py`
- Package: `kiro/`
- Tests: `tests/`
- License: AGPL-3.0
- OpenAI endpoints: `/v1/chat/completions`, `/v1/responses`, `/v1/models`
- Anthropic endpoints: `/v1/messages`, `/v1/messages/count_tokens`

## Project philosophy

Kiro Gateway is a transparent proxy with minimal, purposeful modifications. It is also a reverse-engineering project for an undocumented upstream API, so compatibility fixes must be systematic and well tested.

1. Preserve the user's original request structure and intent.
2. Make only surgical changes required for validation, compatibility, authentication, or explicitly configured enhancements.
3. Never remove user content or decide which conversation content is important.
4. Keep optional behavior configurable so users can recover native Kiro behavior.
5. Let Kiro arbitrate unknown models; the gateway is not a model gatekeeper.
6. Build reusable systems for classes of upstream quirks instead of one-off patches.
7. Keep API-level concerns in the gateway and content-level decisions in the client.
8. Preserve original upstream data when enriching responses with derived fields.

## Common commands

```bash
# Run the server
python main.py
python main.py --host 127.0.0.1 --port 9000
uvicorn main:app --host 0.0.0.0 --port 8000

# Run tests
pytest -v
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/unit/test_auth_manager.py -v
pytest tests/unit/test_auth_manager.py::TestKiroAuthManagerInitialization::test_initialization_stores_credentials -v
pytest -x
pytest --cov=kiro --cov-report=html

# Manual real-API smoke test; deliberately excluded from pytest collection
python manual_api_test.py

# Install dependencies
pip install -r requirements.txt

# Docker
docker-compose up -d
docker-compose logs -f
docker-compose down
docker-compose up -d --build
```

If the system `pytest` environment has incompatible globally installed plugins, use the repository virtual environment, for example `.venv/bin/python -m pytest tests/unit/test_parsers.py -v`.

## Architecture at a glance

The API implementations are parallel adapters over shared core services:

- Routes: `routes_openai.py`, `routes_anthropic.py`, and `routes_common.py` handle FastAPI endpoints, API-key authentication, validation, and common request behavior.
- Converters: `converters_openai.py`, `converters_anthropic.py`, `converters_responses.py`, and `converters_core.py` translate client formats into Kiro payloads. Shared message and tool-schema behavior belongs in `converters_core.py`.
- Streaming: `streaming_openai.py`, `streaming_anthropic.py`, `streaming_responses.py`, and `streaming_core.py` translate Kiro AWS event streams into client SSE. Shared parsing and first-token retry behavior belongs in `streaming_core.py`.
- Parsers: `parsers.py` handles AWS event framing and tool calls; `thinking_parser.py` uses a finite-state machine for `<thinking>` blocks.
- Models: `models_openai.py`, `models_anthropic.py`, and `models_responses.py` define Pydantic request and response schemas.
- Authentication: `auth.py` manages token loading, refresh, persistence, and authentication-type detection.
- Accounts: `account_manager.py` and `account_errors.py` implement multi-account failover, sticky selection, circuit breaking, and lazy initialization.
- HTTP: `http_client.py` handles token refresh and retry behavior for 403, 429, 5xx, timeouts, and upstream disconnects.
- Model resolution: `model_resolver.py` normalizes names, checks dynamic cache entries, checks hidden models, and finally passes unknown names through.
- Hardening: `network_errors.py`, `kiro_errors.py`, `payload_guards.py`, `truncation_state.py`, `truncation_recovery.py`, and `exceptions.py` provide validation and recovery behavior.
- Observability: `observability.py`, `debug_logger.py`, and `debug_middleware.py` provide metrics and request artifacts.
- MCP: `mcp_tools.py` implements Kiro-backed tools such as `web_search`.

## Complete feature consistency

New behavior must be implemented consistently across all applicable surfaces:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages
- Streaming and non-streaming modes

Tests must cover every applicable combination. If a feature is intentionally limited to one API or mode, document the protocol reason explicitly. Do not allow capability fragmentation between equivalent API surfaces.

## Important implementation constraints

### Streaming clients

Use a per-request `httpx.AsyncClient` for streaming requests to prevent `CLOSE_WAIT` leaks. Non-streaming requests should use the pooled shared client where the existing architecture permits it.

### TLS and proxy handling

All outbound HTTP clients must honor the central proxy and TLS verification configuration. `KIRO_EXTRA_CA_CERTS` references a local PEM file and augments trust only for this process. Never embed or commit a real corporate root CA, private key, credential, employee identity, or local certificate path.

### Model resolution

Normalize model names before resolution. Minor-version dashes become dots and date suffixes are stripped, for example `claude-haiku-4-5-20251001` becomes `claude-haiku-4.5`. Unknown models pass through to Kiro.

### Authentication

Authentication type is detected from credential contents. The supported sources are:

- Kiro IDE JSON credentials via `KIRO_CREDS_FILE`
- A direct refresh token via `REFRESH_TOKEN`
- A kiro-cli SQLite database via `KIRO_CLI_DB_FILE`
- Multi-account JSON, SQLite, or folder entries via `credentials.json`

The presence of `clientId` and `clientSecret` indicates AWS SSO OIDC; their absence indicates Kiro Desktop Auth. Never log tokens, secrets, credential payloads, or authenticated proxy URLs.

### Tool calls and thinking blocks

Kiro may return tool calls as bracket-wrapped JSON text such as `[{...}]`; the parser already supports this. Thinking blocks are extracted by the finite-state machine in `thinking_parser.py`, not by regular expressions. Preserve these semantics when editing adjacent code.

### Upstream validation errors

Kiro's `Improperly formed request` response is ambiguous and may represent message ordering, tool schemas, malformed content, authentication, or undocumented constraints. Diagnose it systematically with isolated tests and `DEBUG_MODE=errors` artifacts rather than guessing.

## Coding standards

- Use English for code, comments, docstrings, and identifiers, except in explicit Unicode or multilingual tests.
- Add type hints to every function parameter and return value.
- Write Google-style docstrings with applicable `Args`, `Returns`, and `Raises` sections.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Use `loguru` at key decisions: INFO for business events, DEBUG for technical detail, WARNING for recoverable concerns, and ERROR for failures.
- Catch specific exceptions and add context. Never use bare `except:` or broad `except Exception:` handlers.
- Keep I/O asynchronous.
- Extract duplicated logic and hard-coded values into appropriate constants, functions, or modules.
- Do not leave placeholders or partially implemented functions.
- Make errors actionable: explain what failed and how the user can fix it without exposing internal secrets.

## Privacy and repository hygiene

- Never commit real `.env` files, `credentials.json`, account state, debug logs, packet captures, databases, tokens, private keys, or corporate certificates.
- Never commit company names, internal domains, employee email addresses, workstation usernames, or real absolute development paths unless they are intentional public project data.
- Use neutral test paths such as `/workspace/project`, `/home/user`, or `/Users/example`.
- Use `example.com`, documentation IP ranges, and obvious test credentials in fixtures.
- Treat `certs/`, local scripts, patches, IDE state, and generated diagnostics as local-only unless explicitly reviewed for publication.
- Before committing, inspect both `git diff --cached` and the complete list of staged files. Do not use `git add .` in a dirty worktree.

## Testing discipline

Every behavior change must include tests. Tests are expected to exercise failure paths, malformed inputs, boundaries, and regressions rather than only confirming the happy path.

`tests/conftest.py` installs a global `block_all_network_calls` fixture. All tests must remain completely isolated from real networks and must mock external services.

- Unit tests belong in `tests/unit/`.
- Integration tests belong in `tests/integration/`.
- Read `tests/README.md` before adding tests.
- Add tests to the existing `test_<module>.py` file unless introducing a genuinely new module.
- Follow Arrange-Act-Assert structure.
- Prefer descriptive names such as `test_<behavior>_<expected_result>`.
- Use test classes such as `Test*Success`, `Test*Errors`, and `Test*EdgeCases` where they improve organization.
- Validate changed behavior with targeted tests, then run broader checks in proportion to the change.
- Run `git diff --check` before committing.

`manual_api_test.py` performs real network operations and must only be run deliberately with authorized credentials.

## Configuration

Configuration is loaded from `.env`; `.env.example` is the public template. A deployment requires `PROXY_API_KEY` and one supported credential source. Important optional settings include:

- `PROFILE_ARN`
- `KIRO_REGION` and `KIRO_API_REGION`
- `SERVER_HOST` and `SERVER_PORT`
- `VPN_PROXY_URL`
- `KIRO_EXTRA_CA_CERTS`
- `ACCOUNT_SYSTEM`
- `DEBUG_MODE`

Configuration priority is CLI arguments, then environment variables, then defaults.

## Git workflow

- Read relevant files and recent commits before editing.
- Preserve unrelated user changes in a dirty worktree.
- Stage explicit files instead of using `git add .`.
- Create commits only when requested.
- Do not amend, force-push, delete branches, or rewrite history without explicit permission.
- Follow `<type>(<scope>): <description>` using types such as `feat`, `fix`, `docs`, `test`, `refactor`, and `chore`.
- Keep credentials and private data out of commit messages and author metadata.
- Contributors must follow the terms in `CLA.md`.

## Definition of done

A change is complete only when it follows the shared architecture, covers all applicable API and streaming modes, includes adversarial tests, passes relevant validation, avoids private or environment-specific information, and includes user-facing documentation when configuration or behavior changes.
