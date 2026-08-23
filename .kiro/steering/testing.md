# Testing Discipline (non-negotiable)

Every behavior change ships with adversarial tests. Happy-path-only or "tests
later" work is rejected.

## Complete network and credential isolation

`tests/conftest.py` installs a session-wide `block_all_network_calls` fixture that
fails any real HTTP call. Mock every external service (`httpx.AsyncClient`, Kiro,
OIDC). Tests must not read, refresh, or write real credentials; use fake tokens
and temporary account/state paths.

## Write tests that try to break the code

Cover error paths, malformed inputs, boundaries, cleanup, ownership, and
exhaustion—not only recovery. For shared HTTP behavior, verify both the returned
response and side effects such as close, sleep, retry count, and auth refresh.

## Structure and layout

- Unit tests: `tests/unit/test_<module>.py`; integration: `tests/integration/`.
- Reuse an existing module test file unless a genuinely cross-surface flow needs
  its own integration file. Update `tests/README.md` when adding one.
- Test names: `test_<behavior>_<expected_result>`; use Arrange-Act-Assert.
- Add purpose docstrings. Async tests use `@pytest.mark.asyncio`.
- Every applicable feature must cover OpenAI Chat, OpenAI Responses, and Anthropic
  Messages in streaming and non-streaming modes.

## `INVALID_MODEL_ID` regression contract

- `tests/unit/test_http_client.py` verifies transient recovery, bounded exhaustion,
  `1s/2s/4s` delays, jitter bounds, discarded-response close, final-response
  ownership, and no `force_refresh()`.
- Negative cases prove that context overflow, wrong/nested/non-string reasons,
  arrays, malformed JSON, and empty bodies are not retried.
- `tests/unit/test_config.py` verifies retry configuration types and safety bounds.
- `tests/integration/test_invalid_model_retry_flow.py` is the six-case matrix for
  Chat/Responses/Messages × streaming/non-streaming using the real shared retry
  loop with only the transport mocked.

## Current verification baseline (2026-08-23)

- Full isolated suite: **1,987 passed, 0 failed**.
- Focused config + HTTP client + cross-surface suite: **116 passed**.
- Retry integration matrix: **6 passed**.
- `py_compile`, LSP diagnostics, and `git diff --check` passed.
- Manual live verification is deliberately separate from pytest: it captured
  three real transient failures followed by HTTP 200 on attempt four.

## Run

```bash
pytest tests/unit/test_<module>.py -v
pytest tests/integration/test_invalid_model_retry_flow.py -v
pytest -v
pytest --cov=kiro --cov-report=html
```

`manual_api_test.py` hits the real API and is excluded by `pytest.ini`; run it only
with explicit authorization and never as part of automated tests.
