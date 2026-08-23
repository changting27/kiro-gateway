# Code Conventions (enforced)

These are hard rules. Violating them is a PR rejection signal (see AGENTS.md
"Code Review Reality Check").

- **Language:** Python 3.10+. All code, comments, docstrings, and identifiers in
  **English only** — non-English identifiers are an explicit rejection signal
  (sole exceptions: Unicode/multilingual test fixtures).
- **Naming:** `snake_case` functions/variables, `PascalCase` classes,
  `UPPER_SNAKE_CASE` constants, `_leading_underscore` for private members.
- **Type hints:** mandatory on **every** parameter and return value.
- **Docstrings:** Google-style with `Args:` / `Returns:` / `Raises:` on every function.
- **Async:** all I/O is `async`/`await`.
- **Logging:** use `loguru` at decision points — `INFO` for business logic,
  `DEBUG` for technical detail, `ERROR` for failures.
  **Never log tokens, API keys, or credentials.**
- **Exceptions:** never `except:` or bare `except Exception:`. Catch specific
  exceptions and add context. Client-facing errors must be sanitized and actionable
  (raise `HTTPException` with a helpful `detail`), never leak internals.
- **No placeholders:** every function ships complete and production-ready — no
  `TODO`-stub bodies, no "I'll add it later".
- **Systems over patches:** when fixing a class of problem, build the abstraction /
  dedicated module rather than a one-off `if`/`else`. Extract hardcoded values and
  duplicated code into constants/functions/modules as you see them.

## Commit style

`<type>(<scope>): <description>` where type ∈ `feat|fix|docs|test|refactor|chore`.
Inspect `git log --oneline -20` for live examples. A CLA (`CLA.md`) applies to contributions.

## Editing workflow

1. **Read before editing** — view the file and nearby siblings first.
2. **Match existing patterns** — check the parallel OpenAI/Anthropic module for style.
3. Add/extend tests, then run `pytest -v` before declaring done.
