# TigerStyle: coding standards for this repo

These rules are mandatory for generated, modified, and reviewed code.
`AGENTS.md` points here; this file is the canonical text. It used to live only
in one maintainer's editor configuration, which meant a clone got the name of
the standard and none of its content.

**The numbers live in `pyproject.toml`, not here.** Line length, cyclomatic
complexity, and nesting depth are configured for Ruff and enforced by
`scripts/lint_ratchet.py`. This document explains the intent; that file is the
authority on the values, so the two cannot drift apart.

---

## 1. Safety

### 1.1 Types

- **Python:** type hints on every function signature, and on any variable
  whose type is not obvious from its assignment. Never `Any` unless wrapping
  an untyped third-party API, and then with a comment saying why.
- **JavaScript:** this repository ships classic browser scripts with no build
  step, so there is no TypeScript to be strict about. The equivalent
  discipline is to keep functions small enough that their argument shapes are
  obvious, and to document any non-obvious payload shape in a comment.
- Prefer narrow types over broad ones. Make invalid states unrepresentable.

### 1.2 Assertions and contracts

Assertions catch **programmer errors**, not user or operating errors. The
only correct response to corrupt internal state is to crash loudly.

- Assert arguments, return values, preconditions, postconditions, and
  invariants. Aim for at least two meaningful assertions in any non-trivial
  function.
- **Pair your assertions.** Check a critical property in two different places,
  for example before writing data and after reading it back.
- Split compound assertions. `assert a` then `assert b` says which one failed;
  `assert a and b` does not.
- Assert the positive space you expect **and** the negative space you do not.
  Bugs cluster at the boundary between them.
- Assert relationships between constants at module level. It documents the
  relationship and catches an edit that breaks it.

**Python specifically:** `assert` is stripped by `python -O`. For checks that
must survive in production, meaning argument validation at API boundaries and
data-integrity checks, use an explicit `if ... raise ValueError`. Reserve
`assert` for internal invariants.

### 1.3 Error handling

- **Every error is handled.** No bare `except:` in Python, no empty `catch {}`
  in JavaScript. Every handler logs, re-raises, or handles explicitly.
- Prefer specific exception types. Use context managers for resource cleanup.
- `contextlib.suppress` is for a failure that genuinely does not matter, such
  as best-effort cleanup whose real error is reraised or already logged. It is
  not a tidier spelling of ignoring a problem. If a handler is swallowing a
  fault nobody will ever see, that is a bug to fix, not syntax to compress.
- Separate **operating errors** (bad input, a network timeout, a missing file)
  from **programmer errors** (a broken invariant). Handle the first
  gracefully. Crash on the second.

### 1.4 Control flow

- Simple and explicit. Avoid recursion unless the data structure is itself
  recursive.
- **Put a limit on everything.** Every loop has a bounded iteration count.
  Every queue, buffer, and cache has a fixed upper bound.
- **Push `if`s up, push `for`s down.** Centralize branching in the caller and
  keep helpers branchless and ideally pure. One function owns the control
  flow; the rest compute.
- Every `if` should have an `else` that either handles the negative case or
  asserts it cannot happen.
- State invariants positively: prefer `if index < length` over
  `if index >= length`.

### 1.5 Scope and lifetime

- Declare variables at the smallest possible scope, as late as possible, and
  let them go as early as possible.
- One source of truth. Do not alias mutable references.
- Minimize the distance between where a value is checked and where it is used.
- **Python:** prefer immutable data where practical. Copy defensively across
  boundaries.
- **JavaScript:** `const` by default, `let` only when reassignment is real,
  never `var` in new module-scope code.

---

## 2. Performance

The large wins come from algorithmic and architectural choices, not
micro-optimization.

- Sketch resource usage before writing code: network, disk, memory, CPU, and
  for each, bandwidth and latency. A back-of-the-envelope estimate beats
  profiling after the fact.
- Optimize the slowest resource first, adjusted for how often it is touched.
- **Batch.** Amortize I/O by batching network calls, queries, and file
  operations.
- Distinguish the control plane (setup, configuration, once per run) from the
  data plane (per request, per frame, per token). Validation and assertions
  belong on the control plane. The data plane stays lean.

### Python, machine learning specifics

- **Validate tensor shapes early**, right after creation, reshape, or a
  function return. Name or comment the expected shape:
  `# (batch, seq_len, hidden_dim)`.
- Prefer vectorized operations. A Python loop over array elements in a hot
  path needs a justification.
- **Pin dependency versions.** These environments are fragile and
  reproducibility depends on exact versions.
- Separate configuration from code. Hyperparameters and paths belong in typed
  config objects, not literals in a loop.
- Log reproducibility metadata: seeds, library versions, commit hash.
- **Guard GPU memory.** Delete large tensors explicitly and call
  `torch.cuda.empty_cache()` between pipeline stages. Wrap all inference in
  `torch.no_grad()`.
- Use `pathlib.Path` for file operations, never string concatenation.

### Browser specifics

- This frontend has no framework and no bundler, by an explicit decision
  recorded in the audit. Keep it that way unless that decision is revisited.
- Touch the DOM as little as possible. Update in place rather than rebuilding;
  the token renderers in `app.js` exist because rebuilding every frame was too
  slow at LLaDA's default length.
- Prefer web-standard APIs over libraries. Anything vendored has to earn its
  place in `src/web/static/vendor/` and carry its license.

---

## 3. Developer experience

### 3.1 Naming

- **Get the nouns and verbs right.** Good names remove the need for comments.
- **Python:** `snake_case` for functions, variables, and modules.
  `PascalCase` for classes. `UPPER_SNAKE_CASE` for module constants.
- **JavaScript:** `camelCase` for functions and variables, `PascalCase` for
  classes and types, `snake_case` for file names, `UPPER_SNAKE_CASE` for
  module constants.
- **Do not abbreviate.** Spell out `configuration`, not `cfg`. Spell out
  `response`, not `resp`. Single letters are for loop counters in tight
  numerical code.
- **Units and qualifiers go last**, most significant first: `latency_ms_max`,
  `timeout_seconds_default`, `batch_size_train`. Related names then sort and
  align together.
- Match character length across related names so parallel code lines up:
  `source` and `target`, not `src` and `dest`.
- Prefix a helper with its caller: `process_batch` and
  `process_batch_validate`.
- Callbacks go last in a parameter list, mirroring invocation order.
- Put entry points and public API near the top of a file, helpers below their
  callers.

### 3.2 Functions

- **Soft target 70 lines, hard limit 100.** Between the two is a
  deliberate-use zone: allowed when splitting would genuinely hurt
  readability, for example an orchestration function coordinating sequential
  phases whose helpers would each have exactly one caller. Hot-path functions
  should stay under 70 regardless; the extra budget is for control-plane
  orchestration, not numerical kernels.
- **Complexity and nesting limits are enforced by the linter**, configured in
  `pyproject.toml`. A long flat function is safer than a short deeply nested
  one; line count is a proxy, complexity and nesting predict cognitive load
  directly.
- Good shape: few parameters, simple return type, meaty logic inside.
- Centralize state changes in the parent. Helpers compute what should change;
  the parent applies it.
- Prefer simpler return types, in order: `None`, `bool`, `int`,
  `Optional[int]`, `tuple`.
- **Python:** keyword-only arguments (`*`) once a function has four or more
  parameters, or two of the same type.
- **JavaScript:** a single options object once a function has three or more
  parameters, or whenever argument order could be confused.

### 3.3 Comments and documentation

- **Say why.** The code shows what. A comment explains why this approach, and
  what was rejected.
- **Say how at the top of a test:** what is under test, the strategy, and what
  a passing run proves.
- Comments are complete sentences. Inline end-of-line notes may be fragments.
- Pass options to library functions explicitly at the call site rather than
  relying on defaults. It documents intent and survives an upstream change.
- Do not write a comment that narrates the next line, records where code came
  from, or argues that a change is correct. That is talking to the reviewer,
  and it is noise once the change is merged.
- **Commit messages:** imperative subject, 72 characters or fewer, and a body
  explaining motivation rather than mechanics.

### 3.4 Formatting

- **Python:** `ruff` for formatting and linting. Four-space indentation. The
  line limit is in `pyproject.toml` and the sources are hand-wrapped to it.
- **JavaScript:** two-space indentation, same line limit.
- Nothing should be hidden behind a horizontal scrollbar.
- No semicolon-joined statements in Python. Single-line `if` only for
  assertions. Avoid clever one-liners; readable beats compact.
- **JavaScript:** always brace `if`/`else`/`for`/`while` unless the whole
  statement fits on one line.
- **No em-dashes anywhere.** See `AGENTS.md` for the full rule.

### 3.5 Dependencies

Every dependency is a liability: supply chain risk, version conflicts, and
cognitive overhead.

- Before adding one, ask whether the standard library solves it in 50 lines or
  fewer. If so, write those lines.
- When a dependency is justified, pin the exact version.
- **Python:** the machine learning core (`torch`, `numpy`, `transformers`,
  `pytest`) is expected. Justify anything beyond it, and remember it has to be
  added to the right one of three environments.
- **Browser:** there is no package manager here. Vendored assets live in
  `src/web/static/vendor/` with their licenses and a recorded hash, fetched by
  `scripts/vendor_assets.py`.

### 3.6 Project structure

The layout in place is the layout to follow: `src/` by concern (`backends/`,
`inference/`, `web/`, `analytics/`), `tests/` mirroring it, `scripts/` for
entry points, and `docs/` for prose. Do not reorganize without asking.

### 3.7 Testing

- `pytest` for Python. Browser modules are tested with `node --test`, loading
  the shipped classic script into a `vm` context so no test-only code enters
  the file the browser gets.
- **Test the boundary**, not just the happy path: valid input, invalid input,
  and the point where one becomes the other. Test error paths.
- One assertion concept per test. Several `assert` statements are fine if they
  check facets of one expectation.
- Name tests for the behavior:
  `test_train_step_raises_on_mismatched_batch_dims`, not `test_train_1`.
- **A regression test must fail against the bug it describes.** If it passes
  both before and after the fix, it is testing something else.
- For models: assert output shapes, and that transforms round-trip when they
  should. For UI: test behavior, not implementation detail.

---

## 4. Async

- **Python:** `asyncio` for I/O concurrency. Wrap blocking calls with
  `asyncio.to_thread` rather than mixing in threads by hand. Preconditions may
  not hold across an `await`, so re-check what matters after one.
- **JavaScript:** always handle promise rejections. Prefer `async`/`await`
  over `.then` chains. No fire-and-forget promises.

---

## 5. Working in an existing codebase

- **Read before writing.** Match the conventions already in the file, then
  improve incrementally.
- **Do not rewrite working code** unless asked. Make targeted changes.
- Never delete or weaken a test without saying why.
- When a design choice is genuinely open, surface it instead of picking
  silently. A `# TODO(review):` comment explaining the trade-off is the
  minimum; asking is better.
