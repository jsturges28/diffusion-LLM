# AGENTS.md: working conventions for this repo

Durable guidance for any agent (or human) picking up work on **diffusion-LLM**.
Read this first, then `HANDOFF.md` (the living, per-session handoff), then
`README.md` and `ROADMAP.md`. Take a look, coarse through the necessary areas of the repo to familiarize yourself with the project, and let the user know what you think of the overall plan for this session. They prefer to discuss the overall direction, then ensure both sides are aligned sufficiently before going into Plan mode at the beginning of the session.

## What this project is

A local FastAPI + WebSocket visual playground and analytics suite for **LLMs**,
oriented toward explainability (XAI). The depth is in **discrete diffusion**
(LLaDA-8B-Instruct and DiffusionGemma-26B-A4B), with SmolLM3-3B alongside as an
autoregressive baseline and room for further model classes. It runs in the
browser (localhost) and as an optional native desktop app (`desktop.py`,
pywebview). Architecture, features, and the public roadmap live in `README.md`;
the deeper living roadmap in `ROADMAP.md`.

## Environments (never touch system Python)

Models need incompatible `transformers` versions, so there are three venvs:

- `.venv`: supervisor + LLaDA worker (`transformers==4.38.2`). Use
  `.venv/bin/python` and `.venv/bin/pip`.
- `.venv-dgemma`: DiffusionGemma worker (`transformers` v5). Use
  `.venv-dgemma/bin/python`.
- `.venv-ar`: SmolLM3 autoregressive worker (`transformers` >= 4.53,
  CUDA torch wheel that also runs on CPU). Use `.venv-ar/bin/python`.

Dependency files: `requirements.txt` (core `.venv`), `requirements-dgemma.txt`
(the DiffusionGemma env), `requirements-ar.txt` (the SmolLM3 env),
`requirements-desktop.txt` (optional `pywebview[qt]` desktop add-on for
`.venv`). Pin versions; do not install to system/user Python.

## Workflow cadence

The maintainer prefers **deliberate → Plan → Agent**:

1. For non-trivial work, read `README.md` + `ROADMAP.md`, skim the relevant code,
   and **discuss/deliberate** the approach and trade-offs before planning.
2. Use **Plan mode** to write a concise, grounded plan (cite files) and get
   confirmation.
3. Then implement in **Agent mode**.

Small, unambiguous fixes can skip straight to implementation. When a design has
multiple valid paths, surface the decision instead of guessing.

**A session brief overrides this cadence and the commit discipline below.**
When `HANDOFF.md` points you at one, read it first and let it win for that
session. `AUDIT_BRIEF.md` governed the read-only audit that produced
`AUDIT_REPORT.md`; `IMPLEMENTATION_BRIEF.md` governs the sessions working
through its findings, and narrows the session-end documentation habit at the
bottom of this file, because forty findings must not become forty rounds of
documentation churn.

## Coding standards

Follow the repo's **TigerStyle** rules (assertions, precise types, small
functions, explicit control flow, `pathlib`, pinned deps, etc.) and the rules in
`.cursor/rules/`. Match existing conventions in each file. Do not rewrite working
code without reason.

**No em-dashes in copy.** Do not use em-dashes (`\u2014`, `&mdash;`, `&#8212;`)
in user-facing text, and avoid them in the frontend (`src/web/static/`) and prose
generally (including this file). Replace each with a comma, semicolon, colon,
period, or nothing, whichever reads best (for example, prefer "Frame 5: click
tokens to remask" over an em-dash separator). If you happen to find or come across em-dashes present from previous sessions, feel free to remove them unless they're strictly necessary.

## Verification (before handing back)

- Python: `.venv/bin/python -m pytest` (tests live in `tests/`, mirroring `src/`).
  Compile-check changed modules with `.venv/bin/python -m py_compile <files>`.
- Lint: `.venv/bin/python scripts/lint_ratchet.py`. Ruff findings are
 recorded per file and per rule in `lint_baseline.json` and may only go
 down; the gate fails on any increase. When you reduce a count, rerun with
 `--update` to lock it in. Do not raise the ceiling to make a change fit.
- JS: `node --check` on each changed `.js` file, and
 `node --test tests/web/static/*.test.js` for the browser modules that have
 tests. Those load the shipped file into a `vm` context, so a testable helper
 stays a plain classic script with no export tail.
- Always run the linter (ReadLints) on changed files and fix what you introduced.
- **GUI/GPU can't be exercised in the sandbox** (no display, no CUDA). When work
  touches the desktop window or model inference, hand back with a short
  **manual-verification checklist** for the maintainer.

## Frontend cache-busting (do not regress)

The server auto-stamps `?v=<mtime>` onto local CSS/JS references at serve time
(`src/web/server.py`: `serve_index` / `serve_analytics_page` /
`_stamp_asset_versions`), and `_NoCacheStaticFiles` serves assets `no-store`. So
**edit CSS/JS and just reload; never add or bump manual `?v=` query strings.**

## Commit discipline (maintainer pushes manually)

The maintainer wants a **commit per validated feature** (the boundary is
subjective; err toward one cohesive, reviewable change per commit).

- After a feature is validated, propose a commit with a concise, imperative
  subject (<=72 chars) and a body that explains the **why**. Create it once the
  maintainer greenlights.
- **Never push**; the maintainer pushes manually.
- Git safety: never change git config; no `--force`; never amend a commit that
  has been pushed; don't commit secrets (`.env`, credentials, model weights, the
  venvs). Respect `.gitignore`.

## Session-end handoff (keep HANDOFF.md alive)

Before wrapping up a session, **update `HANDOFF.md`** so the next session can pick
up cold. Mirror its structure:

- Refresh **"Recently shipped"** with what changed this session.
- Rewrite **"Where to pick up"** with the next candidate work, each item with its
  settled decisions, sensible defaults, and grounding pointers (file paths /
  functions).
- Keep the orientation (what it is / models / architecture / conventions) current
  if it drifted.

Also update `README.md` (feature overview + Implementation Status) and
`ROADMAP.md` (shipped vs backlog) as features land.

Whenever you touch `HANDOFF.md` at session end, also check the in-app **About**
and **Help** modals (`src/web/static/index.html`, `#modal-about` /
`#modal-help`). If the session landed anything a user would notice: a new model,
a new page, changed overlays or settings, a new hyperparameter, or a workflow
change, update the matching About/Help copy in the same pass so the in-app docs
never fall behind. Small internal-only fixes do not require an About/Help edit.
Follow the no-em-dash rule there too.
