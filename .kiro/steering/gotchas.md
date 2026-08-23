# Core Rules & Non-Obvious Gotchas

## Two rules that gate every change

- **Feature consistency:** new functionality MUST cover OpenAI Chat, OpenAI
  Responses, and Anthropic Messages in both streaming and non-streaming modes,
  with tests for every applicable combination. If a feature applies to only one
  surface, document the protocol reason explicitly.
- **Gateway, not gatekeeper:** fix Kiro API quirks, never the user's content or
  context decisions. Unknown model names pass through to Kiro; Kiro remains the
  final arbiter. Optional behavior stays configurable.

## Gotchas (don't "fix" these — they're intentional)

- **Streaming uses per-request `httpx.AsyncClient`.** Reusing a pooled client for
  streaming causes CLOSE_WAIT leaks. Non-streaming requests use the shared pool.
- **Transient `INVALID_MODEL_ID` is an exact-match exception.** Retry only HTTP
  400 whose parsed JSON object contains the top-level string
  `reason == "INVALID_MODEL_ID"`. Do not broaden this to nested values,
  non-string reasons, malformed JSON, arbitrary 400s, or message-text matching.
- **The model-error retry is bounded and does not force an auth refresh.** Defaults
  are four total attempts with `1s/2s/4s` exponential backoff plus bounded jitter.
  Close each discarded response before retry. Never call `force_refresh()` for
  this reason; normal token-expiry handling in `get_access_token()` still applies,
  and forced refresh belongs only to authentication failures such as 403.
- **Final response ownership matters.** After retry exhaustion, return the final
  original response without closing it so route error shaping and existing
  multi-account classification can read it. A true invalid model, or a transient
  window longer than the budget, must finish with a bounded 400.
- **Model-name normalization** (`model_resolver.normalize_model_name`): minor
  version dashes become dots and date suffixes are stripped, e.g.
  `claude-haiku-4-5-20251001` → `claude-haiku-4.5` before resolution.
- **Auth type is auto-detected** from credential contents: `clientId` plus
  `clientSecret` means AWS SSO OIDC; absence means Kiro Desktop Auth. Sources are
  JSON, `REFRESH_TOKEN`, kiro-cli SQLite, and multi-account `credentials.json`.
- **Tool calls may arrive as bracket-wrapped JSON** `[{...}]`; the parser already
  handles this. Do not rewrite it upstream.
- **Thinking blocks** are extracted by an FSM in `thinking_parser.py`, not regex.
- **"Improperly formed request"** is ambiguous. Diagnose it using isolated tests
  and `DEBUG_MODE=errors` artifacts, never by guessing.

## Security

Never log credentials. Validate input through Pydantic models. Sanitize client
errors. Automated tests must not read real credentials or use real networks.
