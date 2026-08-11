---
name: Audit stage 1
overview: "Land the audit's five isolated safety fixes as five reviewable commits: transactional LLaDA resume state, loopback binding, an absolute configurable data root, a fenced Analytics detail modal, and vendored front-end assets. Each carries its own tests and its own ledger update."
todos:
  - id: life-07
    content: "LIFE-07: stage the LLaDA resume history and commit only after the accepted terminal outcome, mirroring dgemma_worker.py:312-323; add tests/backends/test_llada_resume_state.py with the four failure injections plus cancel and success cases; update the ledger in the same commit"
    status: completed
  - id: trust-01
    content: "TRUST-01: default main.py --host to 127.0.0.1, add _is_loopback plus a non-loopback warning, update the README quickstart, add tests/test_main.py; record the auth/origin narrowing in the ledger"
    status: completed
  - id: data-03
    content: "DATA-03: resolve one absolute results root from REPO_ROOT with DIFFUSION_LLM_RESULTS_DIR and a --results-dir flag, log it at startup, correct the desktop.py comment and the hardcoded results/ label in the delete modal, add tests/web/test_results_root.py; move ORG-01 to ready in the ledger"
    status: completed
  - id: analytics-01
    content: "ANALYTICS-01: add src/web/static/detail_requests.js as a classic global seam, wire epochs and AbortController through showDetail, loadRunCharts, loadRunOverlays, hideDetail and showComparison, centralize response.ok handling, add tests/web/static/detail_requests.test.js run via node --test, and add that command to AGENTS.md"
    status: completed
  - id: trust-02
    content: "TRUST-02: vendor Chart.js 4.4.7, Hammer.js 2.0.8, chartjs-plugin-zoom 2.2.0 and JetBrains Mono woff2 with licenses and provenance, drop the CDN and Google Fonts tags, feature-detect Chart so the table survives, add tests/web/test_no_external_assets.py; fall back to handing over curl commands if the proxy blocks downloads"
    status: completed
  - id: handback
    content: "Stage boundary: confirm pytest, ruff (156 ceiling), node --check and node --test all pass, finalize the ledger including the four noted-not-fixed items, and hand back with the manual verification checklist"
    status: completed
isProject: false
---

# Audit remediation, stage 1: isolated safety fixes

Governed by [IMPLEMENTATION_BRIEF.md](IMPLEMENTATION_BRIEF.md), state tracked in [IMPLEMENTATION_LEDGER.md](IMPLEMENTATION_LEDGER.md). Five findings, five commits, in the order agreed: `LIFE-07`, `TRUST-01`, `DATA-03`, `ANALYTICS-01`, `TRUST-02`. `DATA-03` moves ahead of `ANALYTICS-01` so stage 3 unblocks earlier if budget thins.

Decisions already settled and baked in below: the ANALYTICS-01 seam plus `node --test`, vendoring the webfont rather than a system stack, `--results-dir` and an environment variable with the flag winning, and TRUST-01 narrowed to the default flip.

## 1. LIFE-07: commit LLaDA retained state only after resume succeeds

`handle_resume` in [src/backends/llada_worker.py](src/backends/llada_worker.py) truncates at line 374, before the `try` opens at 378, so any failure below leaves the worker's only resumable state shortened.

The fix is convergence on an existing in-repo pattern: [src/backends/dgemma_worker.py](src/backends/dgemma_worker.py) lines 312 to 323 forwards frames, sends its terminal message inside `_forward_resume`, and only then assigns `state["frame_history"] = base_history + kept`.

- Stage instead of truncate: keep `base_tensor`, add `base_history = state["tensor_history"][:frame_index]`, and leave `state` untouched through the whole `try`.
- Commit `state["tensor_history"] = base_history + resume_history` after the accepted terminal outcome. On the `done` path that is after `stream.run` returns; on the `max_frames` path it must be after the worker's own `await ws.send_json` at line 413, since the Verification injects a failure during terminal send.
- No tensor copying: `streaming_resume` builds a fresh `x` via `torch.cat` and mutates only that ([src/inference/streaming_sampler.py](src/inference/streaming_sampler.py) lines 490 to 498), so staging is one list of existing references.
- Cancellation stays a commit, not a rollback. `streaming_resume` returns without yielding `done` when cancelled (lines 549 to 553), and the browser keeps the frames it received, so discarding would recreate the same disagreement in the other direction. A test pins this.

New file `tests/backends/test_llada_resume_state.py`, modeled on the harness in [tests/backends/test_smollm3_substitute.py](tests/backends/test_smollm3_substitute.py): stub WebSocket, a stub streamer reproducing `FrameStreamer.run` semantics including the `max_frames` break and the `done` return, and a monkeypatched `llada_worker.streaming_resume`. Verified that the module imports cleanly under `.venv` with no GPU.

Cases: failure before the first frame, midway, during the worker's terminal send on the `max_frames` path, and during the sampler's `done`; after each, the original history and `total_steps` are unchanged and a second resume from both the first and last original frame succeeds. Positive space: a full resume commits and sets `total_steps = len - 1`; a `max_frames` resume commits and leaves `total_steps` alone; a cancelled resume commits its partial history.

Ledger: `needs hardware` pending one real guided edit on LLaDA. The Verification clause itself is fully met by the automated tests; nothing in this stage depends on it.

## 2. TRUST-01: bind to loopback unless exposure is explicit

`0.0.0.0` appears in exactly three places: [main.py](main.py) lines 12 and 13, and [README.md](README.md) line 264.

- Default `--host` to `127.0.0.1`.
- Add `_is_loopback(host: str) -> bool` handling `127.0.0.0/8`, `::1`, and `localhost`, and print a warning from `main()` on any non-loopback host naming what is exposed: model activation, saves, and permanent deletion, with no authentication.
- Update the README quickstart so the plain command is loopback and the remote form carries the warning.

Explicitly out of scope and recorded in the ledger as a deliberate narrowing: authentication and WebSocket origin policy. The Direction mentions both, but they are a design project, not a stage 1 commit.

New `tests/test_main.py`: the default is loopback, `_is_loopback` accepts `127.0.0.1`, `localhost`, and `::1`, and rejects `0.0.0.0`, a LAN address, and the empty string, and the warning fires only for non-loopback. The maintainer proves the listener is unreachable from another interface.

## 3. DATA-03: resolve the data directory independently of CWD

`RESULTS_DIR = Path("results")` at [src/web/server.py](src/web/server.py) line 75 has 12 read sites in that file, plus `list_runs` in [src/analytics/metrics.py](src/analytics/metrics.py) and the four functions in [src/web/ui_state.py](src/web/ui_state.py).

- Replace the constant with a resolver run at import, defaulting to `(REPO_ROOT / "results").resolve()` and honoring `DIFFUSION_LLM_RESULTS_DIR`. It stays a module-level global so the two existing monkeypatch sites keep working unchanged: `tests/web/test_ui_state_reconcile.py:28` and `tests/web/test_collections_state.py:47`.
- `main.py` gains `--results-dir` and sets the environment variable before `uvicorn.run` imports the server by string. The flag wins over a pre-existing environment value.
- `desktop.py` keeps its `os.chdir(REPO_ROOT)`, which also pins the worker cwd per its own comment at lines 32 to 36; that comment gets corrected since results no longer depend on it.
- Log the resolved root once from the existing startup hook at server.py lines 811 to 815, saying whether it came from the environment or the default.
- The four `.resolve()` containment guards at lines 1266, 1585, and 1667 are left alone; they become idempotent on an already-absolute root.
- One small UI correction, because the finding's whole point is that the app must not assert a folder it is not using: [src/web/static/analytics.js](src/web/static/analytics.js) line 5106 hardcodes `"results/" + runId` in the delete confirmation. Carry a display root on the existing `/api/analytics/system` response, which `fetchSystemInfo` already fetches, and use it there.

Nothing migrates. `results/` is gitignored, holds 178 runs, and a repo-root launch resolves to the same directory.

New `tests/web/test_results_root.py`: the default is absolute and equals `REPO_ROOT/results`; empty and whitespace environment values fall back; a relative value resolves to absolute; `~` expands; negative space asserts no relative path can ever be returned. Plus a `_display_run_path` case for a root outside the repo, which already falls back to the full path and is what names an alternate root in the UI.

Landing this moves `ORG-01` from blocked to ready and opens stage 3.

## 4. ANALYTICS-01: fence detail responses to their run

`showDetail` at [src/web/static/analytics.js](src/web/static/analytics.js) line 1710 fires two independent fetches. `loadRunCharts` destroys charts inside its `.then` at 2558, so a slow response leaves the previous run's charts up; `loadRunOverlays` already tears down before its fetch at 2834, which is half the pattern already in the file. Neither callback rechecks `activeRunId`, and `hideDetail` at 1934 nulls it without aborting anything.

New `src/web/static/detail_requests.js`, following the [src/web/static/overlays.js](src/web/static/overlays.js) convention exactly: classic global script, `"use strict"`, prefixed names, no page state, loaded before `analytics.js` in the tag list at analytics.html lines 544 to 550. No ES module, no build step, no `package.json`. It exposes a DOM-free factory with `begin(runId)` (bump epoch, abort in-flight, return a token carrying the signal), `accepts(token)`, `cancel()`, and `isAbort(error)`.

Wiring:

- `showDetail` calls `begin` once and passes the token to both loaders.
- `loadRunCharts` moves its teardown at 2558 to 2565 ahead of the fetch and guards its `.then` with `accepts`.
- `loadRunOverlays` adds `clearOverlay()` to the pre-fetch teardown so the previous run's tokens do not linger, and guards its `.then`.
- `fetchMetrics` and `fetchFrames` (lines 588 to 607) take a signal, check `response.ok`, and return one normalized error shape. Today neither checks status and neither catches, so failures become unhandled rejections; the shared handler swallows aborts via `isAbort` and renders real failures once.
- `cancel()` is called from `hideDetail` and from `showComparison` at 4786 to 4788, which today hides the detail panel and nulls `activeRunId` without aborting.

Scope line: the fence covers every transition out of a detail view (close button, Escape, backdrop, post-delete at 5226, compare navigation). It does not fence `fetchCompare` at 595 to 600, whose own stale-response race belongs to `ANALYTICS-04`. Recorded in the ledger rather than opportunistically fixed.

New `tests/web/static/detail_requests.test.js`, run with `node --test tests/web/static/`. It loads the browser file in a `vm` context so no test-only code enters the shipped script, which also gives `QUALITY-01` a harness that generalizes to `overlays.js`. Cases: a superseded epoch is rejected while the current one is accepted; a matching runId with a stale epoch is rejected, which is the reopen-same-run case; everything after `cancel()` is rejected; `begin` aborts the previous controllers; `isAbort` distinguishes an abort from a real error.

One documentation line: add `node --test tests/web/static/` to the verification list in [AGENTS.md](AGENTS.md). This is the stage's only doc edit; `META-01` and `META-02` fold it in properly in stage 2.

## 5. TRUST-02: make every page work without third-party networks

Four pages load Google Fonts, and analytics.html lines 544 to 546 load Chart.js 4.4.7, Hammer.js 2.0.8, and chartjs-plugin-zoom 2.2.0 from jsDelivr. `analytics.js` line 177 dereferences `Chart` at top level, which is why a blocked CDN kills the entire script rather than just the charts.

- Vendor the three libraries under `src/web/static/vendor/`, each with its MIT license, plus a `vendor/README.md` recording exact source URL, version, and integrity hash. That provenance habit anticipates `TRUST-03` without doing it.
- Vendor JetBrains Mono woff2 at weights 300, 400, and 500 with the OFL license, add `@font-face` rules, and drop the four `<link>` tags. `--font-mono` at [src/web/static/style.css](src/web/static/style.css) line 28 already carries fallbacks, so no other CSS changes.
- Guard the `Chart` dereference and make each render function a no-op with a visible note, so the run table, detail metadata, overlays, and deletion keep working when charts cannot initialize.
- New `tests/web/test_no_external_assets.py` scans every static HTML and CSS file for external asset references and fails on any. That is the guard that stops this silently reverting.

Asset acquisition is blocked in this environment: the npm registry fails at the proxy with `FETCH_ERROR`, and jsDelivr and `fonts.gstatic.com` are very unlikely to be reachable. I will attempt once with network permission. If it is refused, I land the entire code half plus the guard test and hand over four `curl` commands. Ledger: `needs hardware` until the offline pass.

## Verification

Per commit: `.venv/bin/python -m pytest` (baseline 265, must only grow), `.venv/bin/python -m ruff check src tests` (baseline confirmed at 156 today, must not increase), `node --check` on every changed JS file, `node --test tests/web/static/` once it exists, and ReadLints on everything touched.

## Ledger updates, in the same commit as each change

Status transitions for all five, `ORG-01` to ready after `DATA-03`, and four notes recorded rather than fixed: the LLaDA partial-resume `total_steps` inconsistency, the missing containment guard at `_compute_run_metrics` (server.py line 1521), the compare panel's own stale-response race, and TRUST-01's deliberate narrowing.

No About or Help edit is needed. Nothing here changes a model, page, overlay, hyperparameter, or workflow a user would notice; the one new user-facing surface is `--results-dir`, which belongs in the README setup section.

## Manual checklist handed back at the stage boundary

- One guided LLaDA edit: remask, run to a frame, confirm a second edit from an original frame works.
- Confirm the server is unreachable from another machine on the default command, and that `--host 0.0.0.0` still serves and warns.
- Launch from the repo, from `/tmp`, and from the desktop entry; confirm the same runs appear each time, then confirm `--results-dir /tmp/isolated` is isolated and named in the log.
- With DevTools throttling, open run A, open run B before A responds, and close during each phase; confirm title, metadata, charts, and overlays always share one run or are empty.
- Block outbound traffic, open all four pages from a cold profile, and exercise the table, detail, charts, zoom, and deletion.