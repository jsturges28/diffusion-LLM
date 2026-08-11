# AUDIT_BRIEF: a read-only sweep of diffusion-LLM

This brief governs one session and one session only. That session reads the
whole repository and produces `AUDIT_REPORT.md`. It changes nothing else.

The app works and has grown quickly. The point of this pass is to find where it
has accumulated friction, duplication, fragility, and drift before the next
model class lands and makes all of it more expensive to fix. The maintainer
will take the report into a separate session that deliberates, plans, and
implements. So the deliverable is understanding, and the bandwidth is better
spent reading deeply than repairing shallowly.

## The contract

1. **The only path you may create or modify is `AUDIT_REPORT.md`.** No source
   file, no other document, no configuration, no dependency file. No commits,
   no branches, no `git add`.
2. **Do not fix anything, however small.** A one-line fix you spot in passing
   is a finding, not an edit. Fixing as you go destroys the reviewability of
   the report and pre-empts the planning session.
3. **Subagents are bound by the same contract**, and they do not inherit this
   file. Restate the read-only rule in every subagent prompt you write.
4. **Write findings into the report as you confirm them**, rather than holding
   them in context until the end. A long sweep can be compacted; a file on
   disk cannot. Append first, organize last.
5. **Every finding cites evidence as `path:line`.** A claim without a citation
   does not go in the report.
6. **Mark your confidence, and say when you are unsure.** A finding flagged
   low confidence is useful. A confident wrong finding costs the next session
   a wasted plan and some trust.
7. **You cannot exercise the GPU or a display here.** Static reading and the
   test suite are what you have. Anything about runtime behavior that you have
   not measured is reasoning, and must be labelled as such. Collect the
   measurements worth taking into the report's last section so the maintainer
   can run them on hardware.

Before starting the sweep, tell the maintainer how you intend to divide the
work and roughly what you expect it to cost, and let them redirect you. That
is the one check-in. After it, run to completion rather than surfacing
findings one at a time.

Commands you may run: `.venv/bin/python -m pytest`, `.venv/bin/python -m ruff
check src tests`, `node --check`, and read-only shell inspection. Note that
`ruff check src tests` reports **156 known findings** today and `ruff check .`
reports 211, because the second includes `scripts/` and the root files. Whether
that baseline should be burned down or formally accepted is itself a finding.

## What you are auditing

Seven areas, in the maintainer's own framing. They overlap; that is fine.

1. **Code organization.** Structure, monoliths, and what should be broken up
   or moved. The frontend is roughly 24,900 lines across 14 files with no
   module system and no build step, and `src/web/server.py` is 1,978 lines
   holding 24 routes and about 70 functions. Where are the seams, what is the
   cheapest first cut, and what would breaking things up cost in indirection?
2. **Tangled or friction-causing logic**, from both the implementation side
   and the runtime side. Race conditions, lifecycle and ordering hazards,
   state that can disagree with itself, and the places where a user doing
   something reasonable but unanticipated leaves the app wedged. Model
   activation and eviction, worker spawn and termination, the WebSocket proxy,
   request and response correlation, and two browser windows sharing
   server-side state are the obvious hunting grounds.
3. **Runtime speed and smoothness.** Streaming frame handling, canvas and DOM
   work per token, the analytics table at scale, load-time behavior, memory
   growth over a long session. Sketch the resource picture before proposing
   anything, and prefer findings with an argued magnitude over vague ones.
4. **Robustness and trust for a future user.** How the app is served, how
   models are downloaded and cached, what happens when something is missing,
   offline, interrupted, or half-written, and whether failures explain
   themselves. This is not about OS portability: Ubuntu-only is accepted and
   is not a finding. It is about whether someone who is not the author can
   trust the thing not to lose their work or lie to them.
5. **Duplication that can be simplified without breaking the app.** Prefer
   duplication with evidence of cost: a change that had to be made in three
   places is worth more than three functions that merely look alike.
6. **Positioning for the ROADMAP.** Read the future model work against what
   the code assumes today: state-space models at `ROADMAP.md:823` and
   `HANDOFF.md:3160`, Phase 3 multimodal image input at `ROADMAP.md:995`, and
   entropy and top-k for the diffusion models at `HANDOFF.md:3182`. Where will
   the next model class fight the architecture, and where will technical debt
   accrue if nothing changes first? Name the debt that is already accruing,
   too.
7. **Meta: docs against code, and routing for future agents.** Do the
   documents say true things? Are they organized so an agent finds what it
   needs without paying for what it does not? This includes `AGENTS.md`,
   `HANDOFF.md`, `README.md`, `ROADMAP.md`, and the rules under
   `.cursor/rules/`. Treat the routing cost as a real cost: every session pays
   it before it does anything useful.

## How to route yourself

Read these first, in this order, and stop where indicated. This inventory
exists so you do not spend budget rediscovering the shape of the repo.

- `AGENTS.md` (115 lines): all of it. It is the conventions contract.
- `HANDOFF.md` (3,223 lines): **lines 1 to 73 only** for orientation, which
  covers what the app is, the three models, and the architecture. Lines 74 to
  2,030 are "Recently shipped" and are a per-session narrative; read into them
  only when a specific finding needs the history, and cite the line range when
  you do. Lines 2,061 to 3,213 are "Where to pick up" and include a 132-item
  manual verification checklist running to line 3,107.
- `ROADMAP.md` (1,139 lines): "Current status (orientation)" starts at line 20,
  the future work at lines 818, 846, 943, 995, and 1030, and the quick map at
  line 1096. Read the map early even though it sits at the end.
- `README.md` (494 lines): all of it. It is the public description and the
  Implementation Status table is a claims surface worth checking.

Source inventory, largest first, so you can plan coverage:

| File | Lines |
|---|---|
| `src/web/static/app.js` | 7,795 |
| `src/web/static/analytics.js` | 5,586 |
| `src/web/static/style.css` | 4,193 |
| `src/web/server.py` | 1,978 |
| `src/web/static/overlays.js` | 1,667 |
| `src/web/static/menu.js` | 1,506 |
| `src/inference/ar_sampler.py` | 1,429 |
| `src/web/static/analytics.css` | 1,257 |
| `src/web/static/settings.js` | 706 |
| `src/backends/worker_base.py` | 702 |
| `src/inference/streaming_sampler.py` | 612 |
| `src/backends/smollm3_worker.py` | 553 |
| `src/inference/load_progress.py` | 487 |
| `src/inference/dgemma_sampler.py` | 458 |
| `src/backends/llada_worker.py` | 427 |
| `src/inference/dgemma_nf4.py` | 422 |
| `src/backends/dgemma_worker.py` | 375 |
| `src/backends/registry.py` | 336 |
| `src/analytics/metrics.py` | 328 |

Python source totals 9,002 lines against 4,459 lines of tests in `tests/`. The
frontend's 24,900 lines have no automated tests at all: there is no
`package.json`, so `node --check` is the entire JavaScript safety net.

## How to split the work

No single context holds this repo. Run parallel exploration tracks and have
each one return findings in the schema below, already written in the report's
voice. Five tracks that divide cleanly:

- **Supervisor and process lifecycle**: `src/web/server.py`, `desktop.py`,
  `main.py`, `src/web/ui_state.py`, `src/backends/run_worker.py`.
- **Worker contract and samplers**: `src/backends/*`, `src/inference/*`.
- **Generation frontend**: `index.html`, `app.js`, `overlays.js`, `menu.js`,
  `settings.js`, `download_toast.js`, `custom_select.js`, `style.css`.
- **Analytics frontend and metrics**: `analytics.html`, `analytics.js`,
  `analytics.css`, `src/analytics/metrics.py`, and the saved-run format.
- **Docs, tests, tooling, and meta**: the four documents, `.cursor/rules/`,
  `tests/`, `scripts/`, the `requirements-*.txt` set, `pyproject.toml`.

Cross-cutting questions (duplication, extensibility, race conditions) are yours
to synthesize after the tracks return, since no single track can see them.

## The report

One file, `AUDIT_REPORT.md`, at the repository root. Structure it as:

1. **Executive summary**, one screen. What is healthy, what is urgent, what
   the three highest-leverage moves are, and what the report deliberately does
   not recommend.
2. **Findings index**: a table of ID, area, severity, effort, one-line title.
   Written so the planning session can pick a subset and page in only those.
3. **Findings in full**, grouped by area, in the schema below.
4. **Sequencing**: a dependency-ordered build order with the reasoning. Which
   findings unblock others, which must not be attempted together, and what a
   sensible first commit looks like. This is information gathering, not
   implementation, so it belongs here.
5. **Measurements to take on hardware**: the specific things the maintainer
   could run or observe to confirm the runtime findings you could not.

Cap the report at roughly 40 findings. If you have more, the ones you cut are
the ones that would not change a decision. A report nobody can page into
repeats the problem this audit is partly about.

### Finding schema

```
### [AREA-01] Short imperative title

- **Severity**: critical | high | medium | low
- **Confidence**: high | medium | low
- **Evidence**: `path:line`, `path:line`
- **What is true today**: the mechanism, stated plainly.
- **Why it matters**: the consequence, in terms of a user, a future agent, or
  a future change. A finding with no consequence is a style opinion; cut it.
- **Direction**: the shape of a fix and the trade-off it carries. Not a diff,
  not a final design. Name the alternative you rejected.
- **Blast radius**: what else moves if this is changed.
- **Verification**: what would have to be true to believe the change worked,
  including tests that do not exist yet.
- **Blocks**: which ROADMAP work this stands in front of, if any.
```

If a recommendation is large enough to reshape the app, such as introducing a
module system, a build step, or a frontend framework, it is not forbidden, but
it must come with an incremental and reversible migration path and an honest
cheaper alternative alongside it. A recommendation that can only be done as a
big bang will not get done.

## Verified seeds

These are confirmed, and they exist so you start warm. They are a floor, not a
ceiling, and each still needs its "why it matters" and its direction worked
out. Do not let them anchor the sweep.

- **The analytics page depends on a CDN.** `analytics.html:544-546` loads
  Chart.js, Hammer.js, and the zoom plugin from jsdelivr, so a local-first app
  loses its charts offline.
- **The frontend is globals over plain script tags** (`index.html:597-600`,
  `analytics.html:547-550`), with no module system, no bundler, and no tests.
- **The ROADMAP's own trigger for consolidating dependencies has fired.**
  `ROADMAP.md:1115-1125` names two triggers: more than four or five
  requirements files, or a new incompatible environment. There are now four
  files and `.venv-ar` exists.
- **The quick map at `ROADMAP.md:1096` is stale.** It omits `smollm3_worker`,
  `ar_sampler`, `.venv-ar`, `requirements-ar.txt`, `overlays.js`, the menu and
  settings pages, `download_toast.js`, `custom_select.js`, and `analytics.css`.
- **`registry.py:1` calls itself a registry of diffusion models** while
  holding an autoregressive one, and `ModelInfo.venv_python` hardcodes a venv
  path per model, which is the seam every new model class will meet.
- **`src/web/ui_state.py:60` guards durable user state with a
  `threading.Lock`** inside an async server, writing one JSON file, while two
  windows can be open at once.
- **`worker_base.py:544` has a `gen_lock` with a deliberate lockless dispatch
  set** for tokenizer reads. Worth checking that every member of that set is
  genuinely safe to answer mid-generation.
- **Three samplers repeat the same shape.** `streaming_sampler.py`,
  `dgemma_sampler.py`, and `ar_sampler.py` each needed the same `prompt_len`
  addition to their `done` frame in one recent change, which is duplication
  with a receipt.
- **`.cursor/` is gitignored** (`.gitignore:1`), so the always-applied rules do
  not travel with the repository, and `ROADMAP.md:1137` points at
  `.cursor/plans/`, which no clone will have.
- **`HANDOFF.md` has structural drift**: 1,957 lines sit under "Recently
  shipped (this session)", and two different pick-up items are both numbered
  0, at lines 2,188 and 3,114.
- **`tests/web/` covers saved signals, UI state, reconciliation, and run
  paths, but no test names the model manager, activation and eviction, or the
  `/ws` proxy**, which is the part of the server most able to strand a user.

## Non-goals

- OS portability. Ubuntu-only is a settled constraint, not a finding.
- Designing the implementation. Directions and trade-offs, not designs.
- Editing any file other than the report, including the documents you are
  auditing. Recommend the reorganization; do not perform it.
- Grading. The maintainer does not need a score, and severity plus sequencing
  already carries the judgement.

## When you are done

Hand back with the report written and a short message that names the three
highest-leverage findings and anything you could not reach. If you ran out of
budget in an area, say which one and what you would have looked at, so the
next session knows where the map has a blank rather than a clean bill.

**The report is this session's handoff.** `AGENTS.md` asks every session to
update `HANDOFF.md`, the README, and the roadmap on the way out. That habit is
suspended here, because rewriting the documents you just audited would bury
the findings about them. Recommend those edits in the report and leave them
for the session that implements.
