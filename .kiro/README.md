# `.kiro/` — Kiro CLI workspace config

Project configuration for [Kiro CLI](https://kiro.dev/cli/). Commit this directory;
it is shared team configuration.

## Layout

```
.kiro/
├── steering/            # Always-loaded rules (every turn, every session)
│   ├── project.md       # Orientation: what the project is + architecture map + commands
│   ├── conventions.md   # Enforced code conventions (naming, types, docstrings, logging)
│   ├── testing.md       # Testing discipline + complete network isolation
│   └── gotchas.md       # Core rules (feature consistency, gateway-not-gatekeeper) + gotchas
└── agents/
    └── kiro-gateway.json # Project-tuned dev agent (Python/FastAPI tooling)
```

## Steering files

`.md` files in `.kiro/steering/` are loaded into the agent's context automatically
(for the built-in default agent) and shape its behavior every turn. They are a concise
**distillation** of `AGENTS.md` — the long-form authoritative contract, which the agent
reads on demand for full depth. Keep steering files short: each line costs context on
every turn.

## Agent

`agents/kiro-gateway.json` defines a project-scoped dev agent that:

- enables the dev toolset (`read`, `write`, `shell`, `grep`, `glob`, `code`,
  `introspect`, `knowledge`, `subagent`);
- auto-approves read-only tools and a curated set of `pytest` / `git` read commands;
- restricts `write` to source/test/doc paths and **blocks** secret files
  (`.env`, `credentials.json`, `state.json`);
- explicitly loads the steering files via `resources` (custom agents do **not** load
  steering automatically).

### Use it

```bash
kiro-cli agent list                                  # confirm it's discovered
kiro-cli agent validate --path .kiro/agents/kiro-gateway.json
kiro-cli chat --agent kiro-gateway                   # start a session with it
```

Without `--agent`, the built-in default agent still picks up `.kiro/steering/`
automatically.
