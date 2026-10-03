# Repository Audit Report: 2026-10

Audit date: 2026-10-02

Governed by `docs/audit/AUDIT_BRIEF_2026-10.md`. This report audits the
current repository and leaves the 2026-08 report immutable.

## Executive summary

The repository is substantially healthier than it was at the first audit.
All 2,456 Python tests and 816 browser-module tests pass, and the lint
ratchet is at zero. Most completed 2026-08 contracts still hold. Two do not
hold end to end: disconnected-run recovery advertises a save it cannot
perform, and saved signal axes do not reach their Analytics consumer. Run
publication also retained a one-process lock after the product began
allowing a second supervisor. The broader process-isolation design,
versioned run format, bounded frame queues, model axes, offline assets, and
documentation inventory have held through Mamba-3, Vision, diffusion
candidates, revisions, and adaptive stopping.

This audit found **21 decision-changing items: 6 high, 12 medium, and 3
low**. There is no critical finding. The high-severity issues cluster at
ownership handoffs rather than in model numerics:

- a supervisor releases the machine-wide lease in the middle of switching
  models, so another supervisor can take it after the working model has
  already been evicted;
- saved-run identity and revision publication are transactional only inside
  one process even though two supervisors may share the default data root;
- a disconnected run advertises Save but cannot save because it never
  received terminal text;
- DiffusionGemma can commit an empty resume candidate and truncate retained
  history;
- saved signal axes never reach the Analytics frames reader, so diffusion
  entropy falls back to pre-manifest semantics; and
- the unauthenticated save endpoint has no aggregate request budget.

The monolith concern is real but should follow those corrections. Since the
first audit, `app.js`, `analytics.js`, `server.py`, `worker_base.py`, and
`overlays.js` accumulated the highest code churn as well as the highest line
counts. They now expose useful seams: an injectable model manager, a tested
worker dispatch shell, stable session snapshot keys, page-neutral frame
adapters, and a clear divide between visual overlays and browser persistence.
Those are reversible cuts. Splitting routes, comment sections, or globals
before their owners are explicit would produce a more fragmented version of
the same architecture.

The three highest-leverage moves are:

1. **Close cross-process ownership**: hold the residency lease through a
   switch, make lease degradation visible, and serialize run publication
   across supervisors.
2. **Repair observable XAI state with boundary tests**: make interrupted
   runs savable, pin DiffusionGemma backend resume behavior, and carry signal
   manifests into the real Analytics response.
3. **Then decompose by owner**: extract `ModelManager`, the worker socket
   shell, the generator snapshot codec, and Analytics frame semantics before
   considering route files or native modules.

This report deliberately does **not** recommend a frontend framework,
bundler, database, universal sampler, mass formatting pass, or big-bang ES
module conversion. It also does not relabel multi-canvas resume,
ROADMAP-04's future image artifact lifecycle, or the four old hardware
queues as newly discovered defects.

## Coverage and regression matrix

### Snapshot and verification

The audited tree is commit
`cada1e23b6e57e16adfe0d9bfe9fe7758ef618c8`, dated 2026-10-02. The
only audit-created paths were this report and its versioned brief. Thirty-two
pre-existing untracked plans under `.cursor/plans/` were left untouched.

The first-party inventory contains 286 tracked paths and 142,017 text lines
after excluding `reference/`, vendored static assets, media bytes, and local
Cursor plans from the line total. Its largest production files are
`app.js` (9,906 lines), `analytics.js` (7,570), `style.css` (4,917),
`server.py` (4,399),
`overlays.js` (2,823), `menu.js` (1,846), `ar_sampler.py` (1,669), and
`worker_base.py` (1,653). The tree has 59,846 lines of tests against
57,845 text lines under `src/`. These counts are navigation evidence, not
defect counts.

The automated baseline is clean:

- `.venv/bin/python -m pytest`: 2,456 passed, 6 skipped, 77 warnings in
  50.35 seconds.
- `.venv/bin/python scripts/lint_ratchet.py`: zero findings.
- `node --check` over every tracked first-party static JavaScript file:
  passed.
- `node --test tests/web/static/*.test.js`: 816 passed in 1.76 seconds.

The warnings are 58 FastAPI `on_event` deprecations, one expected no-CUDA
warning, two Hugging Face
`resume_download` future warnings, and 16 Python `fork`-from-multithreaded
process warnings. Their architectural consequences are assessed in the
findings rather than treated as test failures.

From 2026-08-10 through this snapshot, 211 commits changed the repository.
The highest application-code churn was `analytics.js` (4,611 changed lines
over 35 commits), `app.js` (4,013 over 54), `server.py` (4,004 over 42),
`worker_base.py` (1,619 over 16), and `overlays.js` (1,427 over 21).
`app.js` and `analytics.js` changed together in 20 commits; each also changed
with `overlays.js` in 15 or 16. This co-change evidence matters when judging
whether a shared seam reduces or merely relocates coordination.

The review covered all tracked first-party source, tests, scripts, dependency
metadata, and orientation documents. Vendored libraries and the two reference
implementations were checked at their integration boundaries, not reviewed
as owned production code.

### Regression contracts from the 2026-08 audit

The old report named forty findings
(`docs/audit/AUDIT_REPORT.md:59-105`). This matrix groups every one by the
invariant its remediation established. A clean automated suite is supporting
evidence only; each status was also checked against current code and focused
tests.

| Contract | Old findings | Status | Current evidence |
|---|---|---|---|
| Transactional run publication, schema, provenance, explicit data root, bounded GIFs, conflict-aware UI intent | ORG-01, DATA-01 to DATA-05, RUNTIME-02 | **Holds inside one supervisor; publication scope is incomplete across processes** | `run_store.py` still owns path and publication policy (`src/web/run_store.py:1-24`) and serializes identity through publication with a process-local lock (`src/web/run_store.py:410-442`); collections remain server-owned (`src/web/collections.py:1-5`). See A2-DATA-01. |
| Process termination, activation and run identity, host lease, pre-eviction validation, scoped protocol | LIFE-01 to LIFE-03, LIFE-05 to LIFE-07, PROTOCOL-01, ORG-04 | **Holds automatically; not fully verified on hardware** | The lease still guards residency across supervisors (`src/web/model_lease.py:1-39`); activation and termination fixtures remain in `tests/web/test_activation_identity.py:20-52` and `tests/web/test_worker_lifecycle.py:227-370`. LIFE-02 items 143-144 and LIFE-05 items 310-314 remain open (`docs/audit/IMPLEMENTATION_LEDGER.md:8-16`). |
| Disconnect, cancellation, bounded queues, append frames, session core, custom selects | LIFE-04, RUNTIME-01, RUNTIME-03, ORG-02 | **Partial regression in disconnected-run recovery; other automated contracts hold** | The producer queue is bounded at 32 frames and every frame put is bounded (`src/inference/frame_queue.py:35-75`, `src/inference/frame_queue.py:98-129`); `run_frames.js` and `run_phases.js` remain in the real page order (`src/web/static/index.html:810-821`). The disconnected client cannot execute its advertised save, A2-LIFE-02. Native ES modules remain explicitly deferred (`docs/audit/IMPLEMENTATION_LEDGER.md:52-61`). |
| Analytics response coherence, token-based convergence, lightweight catalog, bounded compare | ANALYTICS-01 to ANALYTICS-04 | **Holds** | Detail attempts are fenced by epoch and abort controller (`src/web/static/detail_requests.js:1-20`, `src/web/static/detail_requests.js:31-59`); catalog and compare fixtures pass. |
| Loopback serving, offline assets, pinned artifacts, owned downloads | TRUST-01 to TRUST-04 | **Holds** | Static dependencies remain vendored, worker loads remain local-only, and the download path remains a child process; `tests/web/test_no_external_assets.py` and offline-load fixtures pass. |
| Model axes, registry parameters, text adapters, signal manifests, one LLaDA kernel, consolidated environment intent | ROADMAP-01 to ROADMAP-03, ROADMAP-05, ORG-03, DEPS-01 | **Declarations hold; Analytics consumption does not hold end to end; one hardware reading remains** | Registry declarations still carry signal axes, units, and budgets (`src/backends/registry.py:38-48`, `src/backends/registry.py:164-179`), and the differential kernel and lock-manifest tests pass. The saved manifest is omitted from the frames API, A2-XAI-03. ROADMAP-03 item 296 remains open (`docs/audit/IMPLEMENTATION_LEDGER.md:86-89`). |
| Faithful intervention checkpoints | XAI-01 | **Holds at the tested checkpoint layers** | LLaDA's backend failure and repeatability paths pass in `tests/backends/test_llada_resume_state.py`; DiffusionGemma's sampler checkpoint tests pass, while A2-QUALITY-01 records its missing backend layer. |
| Lifecycle/browser fixtures and lint ratchet | QUALITY-01, QUALITY-02 | **Holds as a gate and ongoing obligation** | The lint baseline is zero; QUALITY-01 remains attached to each new seam rather than closed as a standalone task (`docs/audit/IMPLEMENTATION_LEDGER.md:136-140`). |
| Bounded documentation, tracked agent contract, derived inventory | META-01 to META-03 | **Holds automatically; walkthrough remains** | The bounded-document and derived-inventory tests pass (`tests/test_docs_inventory.py:1-18`). META-03 item 326 remains open (`docs/audit/IMPLEMENTATION_LEDGER.md:8-12`). |
| Multimodal artifact lifecycle | ROADMAP-04 | **Known remainder, not started** | The ledger still marks it ready only when an image is saved or conditions generation (`docs/audit/IMPLEMENTATION_LEDGER.md:115-124`, `docs/audit/IMPLEMENTATION_LEDGER.md:408-409`). |

Most old contracts hold. LIFE-04's disconnected client recovery and
ROADMAP-03's saved Analytics consumer do not hold end to end, and DATA-01's
publication lock did not expand when a second supervisor became supported.
The four old hardware queues remain LIFE-02 items 143-144, LIFE-05 items
310-314, ROADMAP-03 item 296, and META-03 item 326. ORG-02's native-module
conversion is deferred, ROADMAP-04 is untaken by design, and QUALITY-01
remains a companion obligation.

The implementation ledger dates its campaign baseline to 2026-09-28: 2,081
Python tests, 486 browser tests, and 70 Ruff findings
(`docs/audit/IMPLEMENTATION_LEDGER.md:90-94`). Today's figures are 2,456,
816, and zero; the old numbers are historical rather than current claims.
The manual ledger also says items 217-266 have no recorded outcome
(`docs/MANUAL_VERIFICATION.md:137-142`), so claims depending on those
hardware scenarios are classified as unverified rather than failed.

## Findings index

| ID | Type | Area | Severity | Effort | Title |
|---|---|---|---|---|---|
| A2-LIFE-01 | defect | lifecycle | high | M | Keep the residency lease across a model switch |
| A2-LIFE-02 | regression | lifecycle | high | S | Make disconnected partial runs actually savable |
| A2-LIFE-03 | architecture | lifecycle | medium | M | Use the resident operation epoch on reconnect |
| A2-LIFE-04 | defect | Analytics lifecycle | medium | S | Retire comparison work when detail view takes over |
| A2-DATA-01 | architecture | persistence | high | M | Serialize run publication across supervisors |
| A2-TRUST-01 | trust | residency | medium | S | Make lease-less operation an explicit failure mode |
| A2-TRUST-02 | trust | save API | high | M | Put an explicit budget on save requests |
| A2-XAI-01 | defect | DiffusionGemma resume | high | S | Stage resume state until a frame lands |
| A2-XAI-02 | defect | DiffusionGemma resume | medium | S | Preserve branch text at a guided stop |
| A2-XAI-03 | regression | Analytics signals | high | S | Carry saved signal semantics into the frames API |
| A2-XAI-04 | defect | Analytics entropy | medium | S | Make entropy bar fills respect signal axes |
| A2-QUALITY-01 | meta | resume coverage | medium | M | Test DiffusionGemma resume at the backend boundary |
| A2-QUALITY-02 | meta | proxy coverage | medium | S | Automate one supervisor-to-worker proxy round trip |
| A2-DEPS-01 | architecture | FastAPI lifecycle | low | M | Move application lifecycle hooks to lifespan contexts |
| A2-META-01 | meta | status routing | low | S | Remove shipped work from the public Next up list |
| A2-ORG-01 | organization | supervisor | medium | M | Extract model ownership before splitting routes |
| A2-ORG-02 | organization | worker runtime | medium | M | Separate the worker socket shell from backend semantics |
| A2-ORG-03 | organization | generator | medium | M | Extract a pure generator session snapshot codec |
| A2-ORG-04 | organization | Analytics | medium | L | Canonicalize wiring before splitting the controller |
| A2-ORG-05 | organization | shared frontend | medium | M | Split visual overlays from browser persistence services |
| A2-ORG-06 | organization | CSS | low | S | Move the Main Menu tail into a page stylesheet |

## Monolith and decomposition map

Line count opened this inquiry; ownership, co-change, and test seams decide
it. The table distinguishes first cuts worth taking from large files whose
current cohesion is stronger than the benefit of another module.

| File | Responsibilities and state | Decision | First reversible step | Main hazard |
|---|---|---|---|---|
| `src/web/static/app.js` (9,906) | Page DOM, model form, socket and run state, live/scrub rendering, edits, save, session/form persistence, boot | **Split in stages**, A2-ORG-03 | Pure session snapshot codec; parent still applies state | Moving global-reading functions verbatim creates files without owners; fix interrupted/reconnect state first |
| `src/web/static/analytics.js` (7,570) | Catalog, collections, detail transport, charts/plugins, token viewer, comparison, delete, boot | **Split after wiring is canonical**, A2-ORG-04 | One test script manifest, then pure frame/signal adapter | Twelve copied script arrays can make the harness disagree with shipped order |
| `src/web/static/style.css` (4,917) | Shared variables/layout/header plus generator controls/output/edit/model picker and a menu-only tail | **One navigation cut now**, A2-ORG-06 | Move lines 4,239 onward to `menu.css` | Broader splits can change cascade order and shared selector precedence |
| `src/web/server.py` (4,399) | Process/resource ownership, all HTTP/WS routes, save schema and bundle, Vision, Analytics, durable UI APIs, page serving | **Extract owner before routers**, A2-ORG-01 | `model_manager.py` with injectable lifecycle contract | Route-first work creates circular access to module-global manager and results root |
| `src/web/static/overlays.js` (2,823) | Token visual math/builders, popover, metrics, stopping, revisions, persistence/settings, new-run state, activation progress, drawer | **Split by lifetime**, A2-ORG-05 | Durable browser-state service, then activation progress | Generator and Analytics intentionally share visual semantics; do not fork them |
| `src/web/static/menu.js` (1,846) | Landing-page video, model rows, activation/download selection and boot | **Leave intact for now** | No cut until another responsibility arrives | Activation and download transports are already extracted; remaining code is one page controller |
| `src/inference/ar_sampler.py` (1,669) | Sampling numerics, append-frame building, decode, substitution, forced/probe paths, cache reuse, queue drain | **Revisit after higher-value cuts** | Candidate future `ar_intervention.py` for forced/probe/substitute paths | Decode and intervention deliberately share cache, trace, and sampling numerics; a universal sampler is rejected |
| `src/backends/worker_base.py` (1,653) | Backend contract, streaming/provenance/resources, tokenizer/context helpers, worker FastAPI/socket runtime | **Split transport**, A2-ORG-02 | `worker_socket.py` or `worker_app.py` | Coordinate with lifespan migration and preserve assigned-environment imports |
| `src/web/static/analytics.css` (1,572) | One page's toolbar, table, modal, charts, viewer and comparison | **Leave intact** | None until JS controllers settle | It is page-local already; splitting adds cascade files without ownership gain |
| `src/analytics/metrics.py` (1,148) | Saved-run schema adapters/readers, catalog projection, convergence and timing calculations | **Defer a semantic split** | On schema v3, move readers/adapters from pure metric computation | It is the declared read boundary and has one production consumer; splitting today mainly moves constants |
| `src/inference/dgemma_sampler.py` (860) | Streamer, logits/signals, thread queue bridge, generate and resume | **Leave intact while resume is repaired** | No structural move before backend tests in A2-QUALITY-01 | Current defects sit at worker/sampler boundary; moving code first obscures them |
| `src/inference/streaming_sampler.py` (789) | LLaDA input building, generation/resume orchestration, checkpoints and frame emission | **Leave intact** | None | Numerical step and schedule are already consolidated in `llada_kernel.py` |
| `docs/MANUAL_VERIFICATION.md` (4,294) | Numbered hardware regression ledger plus outcome ranges | **Leave as one searchable ledger** | Improve status bookkeeping, not file topology | Splitting breaks global item numbers and makes one scenario's authoritative outcome harder to locate |

### Cross-file conclusions

- The first audit's extracted seams paid off. `run_store.py`,
  `worker_process.py`, `model_lease.py`, collections, frame/candidate/phase
  reducers, API clients, text adapters, and `llada_kernel.py` should not be
  reopened merely to make a new directory tree.
- Native ES modules remain a poor first move. They would require replacing
  the shared-scope VM loading model that now carries 816 tests, while the
  underlying state owners would remain undecided. Extract explicit pure or
  namespaced contracts under classic scripts first; revisit modules when
  the entrypoints have little global state left to expose.
- The 20 commits in which `app.js` and `analytics.js` changed together are
  mostly evidence of one feature reaching both pages, not evidence that the
  pages should share one controller. Pure signal, candidate, color, and
  token-building semantics belong in shared helpers; page state and DOM
  lifecycles should remain separate.
- Server routers are a second-stage navigation change. The manager owner,
  save input budget, and cross-process publication boundary should settle
  before route files freeze dependencies on them.
- CSS and documentation cuts need a lower bar for benefit because they can
  be navigation-only, but only when selector order or ledger identity is
  demonstrably unchanged.

## Findings in full

### Process lifecycle and shared state

### [A2-LIFE-01] Keep the residency lease across a model switch

- **Type**: defect
- **Severity**: high
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/server.py:904-929`,
  `src/web/server.py:1489-1552`,
  `tests/web/test_residency_lease.py:213-228`
- **What is true today**: `activate()` takes the manager's process-local
  lock, stops the current worker, and then claims the machine-wide lease
  before launching the replacement. `_stop_locked()` reaches `_finalize()`,
  which unconditionally releases the lease for the current handle. The
  comment in `activate()` says a switching supervisor still owns the claim
  and that re-claiming is a no-op, but the terminal path has already given
  it away. Process termination awaits and therefore creates a real scheduling
  window in which another supervisor can claim the machine. The existing
  switch test has no competing claimant during that window.
- **Why it matters**: A browser supervisor switching models can evict its
  working model, lose the lease to the desktop supervisor, and then have the
  replacement refused. The user pays for an eviction even though the product
  promise is that one supervisor's switch is not self-competition. In a
  tighter interleaving, ownership and the free-VRAM preflight can describe
  different supervisors.
- **Direction**: Distinguish switch finalization from full unload. Keep the
  lease handle across a verified in-manager switch and rewrite its owner
  record for the target; release it only when the manager becomes genuinely
  idle or a launch fails with no worker. Add a two-manager interleaving test
  that pauses after termination and lets the peer attempt activation.
- **Rejected alternative**: Merely moving `_claim_residency()` before
  `_stop_locked()`. `_finalize()` would still release the just-confirmed
  claim.
- **Blast radius**: `ModelManager.activate`, `_stop_locked`, `_finalize`,
  `PrimaryModelLease`, residency tests, and two-supervisor manual checks.
- **Verification**: Under a deterministic interleaving, the switching
  manager retains the claim, the peer is refused throughout, the new worker
  becomes ready, and an explicit stop then hands the claim over.
- **Depends on**: none
- **Roadmap impact**: Required for trustworthy browser-plus-desktop and
  multi-window operation.

### [A2-LIFE-02] Make disconnected partial runs actually savable

- **Type**: regression
- **Severity**: high
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/web/static/app.js:2461-2482`,
  `src/web/static/app.js:2490-2503`,
  `src/web/static/app.js:7692-7694`,
  `src/web/static/app.js:8089-8098`,
  `src/web/static/app.js:9239-9245`
- **What is true today**: A socket close during generation enters an honest
  interrupted state, keeps the received frames, and enables Save. The only
  ordinary assignment of `lastFinalText`, however, is the terminal `done`
  handler, and a disconnect has no terminal frame. `saveRun()`,
  `saveSessionState()`, and the model-switch rescue all return immediately
  when `lastFinalText` is absent. Save is visibly enabled but clicking it
  does nothing and shows no explanation.
- **Why it matters**: The partial frames the disconnect work was designed
  to preserve cannot be persisted to disk, cannot survive a trip to
  Analytics, and are skipped by the rescue path if another window changes
  the resident model. The implementation and its comments both promise that
  these frames are savable.
- **Direction**: Derive provisional final text from the latest accepted
  frame when entering the interrupted state. `runFramesTextAt()` already
  handles both snapshot and append shapes. Persist the partial snapshot and
  make a failed save precondition visible rather than a silent resolved
  promise.
- **Rejected alternative**: Disable Save for every disconnected run. That
  avoids the false affordance by discarding the useful work the interruption
  contract intentionally kept.
- **Blast radius**: disconnect handling, save/session/rescue paths, partial
  run tests, and the saved `partial` flag.
- **Verification**: Feed frames, close the socket before `done`, click Save,
  and assert a request containing latest text and `partial: true`. Restore
  the same run from session storage and open it from Analytics.
- **Depends on**: none
- **Roadmap impact**: Enables honest analysis of interrupted runs and safe
  recovery during model switches.

### [A2-LIFE-03] Use the resident operation epoch on reconnect

- **Type**: architecture
- **Severity**: medium
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/server.py:2060-2074`,
  `src/web/static/app.js:2049-2068`,
  `src/backends/worker_base.py:696-708`
- **What is true today**: The supervisor sends the activation operation ID
  in every leading `resident` frame. The generator compares only model and
  device, then returns without reading the operation. Replacing a worker
  with the same model on the same device therefore leaves the page's run
  token, cached output, and interaction affordances looking live. The
  worker's nonce correctly prevents an operation from being answered from
  the wrong retained run, so this is no longer a silent correctness error;
  the first edit or probe is refused after the user acts.
- **Why it matters**: An activation epoch was added precisely to identify a
  worker replacement. Ignoring it at the last client boundary turns a
  recoverable reconnect into a stale page that fails only on the next
  stateful action. Unsaved display data also misses the automatic rescue
  path used for a model or device mismatch.
- **Direction**: Record the first resident operation for the page and compare
  it on later connections. On change, freeze stateful controls, save any
  display-only run that can be rescued, and reload or offer an explicit
  restart. Persist the epoch with the session snapshot if a navigation must
  distinguish returning to the same worker from a replacement.
- **Rejected alternative**: Rely only on the worker nonce. The nonce keeps
  state safe, but it cannot repair or explain the stale browser state.
- **Blast radius**: `handleResident`, generator session snapshots,
  reconnect tests, and model-switch rescue.
- **Verification**: Send two resident frames with the same model and device
  but different operations around a reconnect. No resume, substitution, or
  probe control may remain active against the old run.
- **Depends on**: A2-LIFE-02 if automatic rescue includes interrupted runs
- **Roadmap impact**: Improves crash recovery and same-model reactivation.

### [A2-LIFE-04] Retire comparison work when detail view takes over

- **Type**: defect
- **Severity**: medium
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/web/static/analytics.js:2263-2277`,
  `src/web/static/analytics.js:6660-6688`,
  `src/web/static/analytics.js:6834-6839`
- **What is true today**: Entering comparison explicitly cancels detail
  requests and closing comparison cancels comparison requests. Entering
  detail hides the compare panel but does not cancel an in-flight compare.
  Its response can still build a Chart.js graph and omission rows behind
  the detail dialog.
- **Why it matters**: The page performs work for a view the user left, keeps
  stale chart state alive, and makes the two request fences asymmetric. A
  later compare can briefly inherit the completed hidden state.
- **Direction**: Cancel `compareRequests` at the start of `showDetail`, before
  opening the dialog, mirroring `showComparison` and `hideComparison`.
- **Rejected alternative**: Treat `hidden` as cancellation. It changes
  visibility, not network, chart, or state lifetime.
- **Blast radius**: Analytics detail/compare lifecycle and one browser test.
- **Verification**: Hold a compare response, open a run, then release the
  response. No comparison render or chart allocation may occur.
- **Depends on**: none
- **Roadmap impact**: None; it closes a current lifecycle asymmetry.

### Persistence and trust boundaries

### [A2-DATA-01] Serialize run publication across supervisors

- **Type**: architecture
- **Severity**: high
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/run_store.py:410-442`,
  `src/web/model_lease.py:10-17`,
  `src/web/ui_state.py:94-112`
- **What is true today**: Run-token resolution, revision checking, and
  publication are protected by a module-local `threading.Lock`. The comment
  beside it says one supervisor owns the data root and points to an
  interprocess lock if that changes. The host lease now deliberately allows
  multiple supervisors to serve pages and Analytics while only residency is
  exclusive, and both use the same default results root. UI state already
  uses a sidecar `flock`; run publication does not.
- **Why it matters**: Two supervisors can both fail to find a run token and
  publish duplicate rows. More seriously, two replacements can both read
  one revision and publish different successors, with the later metadata
  move silently winning. Transactional files do not make the decision that
  chooses their target transactional across processes.
- **Direction**: Put resolution, revision check, and publication under a
  data-root sidecar `flock`, following `ui_state.py`. Keep the in-process
  lock if it simplifies thread ordering, but make the file lock the
  authoritative cross-process boundary.
- **Rejected alternative**: Declare that only one supervisor may use the
  default results root. Current lifecycle policy and UI-state code
  deliberately support the opposite.
- **Blast radius**: `run_store.save`, save race tests, and filesystem
  failure handling.
- **Verification**: Race two processes creating under one run token and two
  processes replacing one revision. The first case yields one run; the
  second yields one success and one revision conflict.
- **Depends on**: none
- **Roadmap impact**: Protects every future saved artifact and per-run note
  from cross-launcher lost updates.

### [A2-TRUST-01] Make lease-less operation an explicit failure mode

- **Type**: trust
- **Severity**: medium
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/web/model_lease.py:118-148`,
  `tests/web/test_model_lease.py:278-303`,
  `README.md:188-191`
- **What is true today**: If the runtime directory cannot create the lease
  file, `PrimaryModelLease.acquire()` logs a warning and returns `True`
  without holding any lock. A test pins this as a deliberate fail-open
  choice. The application continues to present the public guarantee that a
  host-wide lease prevents two instances from loading models, and no page
  exposes that enforcement is absent.
- **Why it matters**: A permission error, unusual container, or damaged
  runtime directory silently restores the dual-worker race LIFE-05 was
  built to remove. The person most likely to need the explanation sees a
  normal activation, while the only warning is in a terminal they may not
  have opened.
- **Direction**: Prefer a clear activation refusal with the failed lease
  path and remedy. If availability is judged more important, return an
  explicit `lease_enforced: false` status and persistent UI warning, and
  document that a second launcher is unsafe in that state.
- **Rejected alternative**: Keep a log-only fallback. It preserves startup
  by making the product's resource-safety claim unknowable to the user.
- **Blast radius**: lease acquisition, activation response/model snapshot,
  menu copy, and one pinned policy test.
- **Verification**: Force the lease path to fail. Activation either refuses
  before spawn or every active page visibly reports degraded enforcement;
  no response may imply that the host invariant still holds.
- **Depends on**: none
- **Roadmap impact**: Important for packaged, containerized, or service
  deployments.

### [A2-TRUST-02] Put an explicit budget on save requests

- **Type**: trust
- **Severity**: high
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/server.py:2331-2414`,
  `src/web/server.py:2416-2461`,
  `src/web/server.py:2937-2957`,
  `main.py:13-18`
- **What is true today**: `SaveRunRequest` constrains some scalar values and
  candidate relationships but gives no length or aggregate byte bound to
  prompt text, frames, positions, token arrays, timing arrays, or edit
  lists. Starlette must parse the complete JSON before field validation,
  and append-shaped text is then expanded into per-frame strings. The route
  is unauthenticated when the explicitly supported network bind is used.
- **Why it matters**: One crafted POST can allocate far beyond any model run
  the UI can produce and can write an arbitrary volume under the results
  root. This is a memory and disk exhaustion boundary even on localhost,
  and becomes remotely reachable when a user accepts the network warning.
- **Direction**: Enforce an aggregate request-body ceiling before JSON
  parsing, then enforce domain limits for frame count, generated positions,
  text lengths, and sidecar record counts. Derive those limits from the
  registry's experimental maxima and the existing candidate budget. Return
  413 for bytes and a precise 422 for shape limits.
- **Rejected alternative**: Trust browser controls. The API is public to any
  client that can reach the server, and old or crafted clients bypass those
  controls.
- **Blast radius**: ASGI request handling, Pydantic save models, append
  normalization, and boundary tests.
- **Verification**: Accept the largest legitimate snapshot and append runs;
  reject one byte, frame, position, and candidate record beyond each bound
  without creating a staging directory.
- **Depends on**: A product-level maximum for experimental run length
- **Roadmap impact**: Establishes the artifact budget multimodal generation
  and future uploads will also need.

### Model and XAI correctness

### [A2-XAI-01] Stage DiffusionGemma resume state until a frame lands

- **Type**: defect
- **Severity**: high
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/backends/dgemma_worker.py:340-381`,
  `src/inference/dgemma_sampler.py:567-603`,
  `src/backends/llada_worker.py:72-109`
- **What is true today**: The DiffusionGemma worker builds a candidate
  `resume_frames` list and, after `_forward_resume` returns, always replaces
  retained history with `base_history + kept`. If cancellation is already
  set before the consumer forwards a frame, the sampler drains its producer
  and emits a cancelled terminal frame with an empty checkpoint list.
  DiffusionGemma then commits the empty candidate, truncating history to the
  prefix before the selected frame. LLaDA's equivalent `_commit_resume`
  explicitly refuses an empty candidate for this reason.
- **Why it matters**: A user can stop a resume before its first visible frame
  and leave the browser showing its original run while the worker has
  silently discarded the selected frame and everything after it. The next
  edit, rewind, or probe starts from state the page never saw.
- **Direction**: Give DiffusionGemma the same staged-commit invariant as
  LLaDA: no retained-state mutation until a terminal outcome has reached the
  client and at least one candidate frame exists. Keep cancellation with no
  forwarded frame as a no-op on retained history.
- **Rejected alternative**: Treat immediate cancellation as an instruction
  to truncate. Stop means preserve what the user saw, not adopt invisible
  work.
- **Blast radius**: `DgemmaBackend.handle_resume`, rewind semantics, and new
  backend-level resume tests.
- **Verification**: Cancel before the first frame, during a later frame, and
  after normal completion. Retained history must respectively stay original,
  match exactly what reached the client, and match the completed branch.
- **Depends on**: none
- **Roadmap impact**: A prerequisite for extending resume beyond one canvas.

### [A2-XAI-02] Preserve branch text at a guided DiffusionGemma stop

- **Type**: defect
- **Severity**: medium
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/backends/dgemma_worker.py:400-437`,
  `src/backends/llada_worker.py:500-528`,
  `src/web/static/app.js:2490-2503`,
  `src/web/static/app.js:8089-8155`
- **What is true today**: A guided DiffusionGemma "run to here" drains the
  generator after the visible frame budget, discards its real terminal
  frame, and synthesizes `done` with `final_text: ""`. The browser updates
  `lastFinalText` only for a truthy value, so it retains the original run's
  final text while the visible frame arrays now describe the branch.
  LLaDA decodes the last staged checkpoint for the same guided outcome.
- **Why it matters**: The mismatch is latent during the next edit step, but
  becomes durable if a model switch or another recovery path saves the
  in-progress branch. Metadata can then pair the branch's frames with the
  original run's final output, and any consumer trusting `final_text`
  receives a record that disagrees with itself.
- **Direction**: Track the last forwarded resume frame's text and put it in
  the synthesized terminal frame. Keep draining for GPU ownership, but do
  not throw away the visible branch's textual outcome.
- **Rejected alternative**: Teach the browser that empty final text means
  "derive it yourself" for one backend. Terminal semantics should remain
  model-independent.
- **Blast radius**: DiffusionGemma resume forwarding, guided-edit recovery,
  save fixtures, and terminal-frame parity tests.
- **Verification**: Stop a guided branch at a frame budget and assert that
  terminal text equals the last forwarded frame and the saved run's
  `final_text`.
- **Depends on**: none
- **Roadmap impact**: Keeps guided edits trustworthy before multi-canvas
  resume is attempted.

### [A2-XAI-03] Carry saved signal semantics into the Analytics frames API

- **Type**: regression
- **Severity**: high
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/web/server.py:2733-2743`,
  `src/web/server.py:3351-3395`,
  `src/web/static/analytics.js:3602-3672`,
  `src/web/static/analytics.js:4008-4065`,
  `src/backends/registry.py:30-57`
- **What is true today**: Save writes the worker's signal manifest to
  metadata. The frames endpoint returns token streams, candidates, edit
  markers, canvas indices, and the stop rule, but omits `signals`.
  Analytics fetches metadata and frames separately; the metadata response is
  rendered into rows and never merged into the frames payload consumed by
  `signalChannel()`. The browser therefore takes the legacy no-manifest
  fallback for every real saved run.
- **Why it matters**: Diffusion entropy is declared over
  `frame|position`, while autoregressive entropy is per position. The real
  HTTP path cannot tell them apart, so the entropy chart can read the final
  diffusion frame while the token viewer is scrubbed elsewhere. Unit tests
  pass because they inject `signals` directly into an in-memory frames
  object that the endpoint never produces.
- **Direction**: Return the saved signal manifest from `/frames`, or merge
  the already-fetched metadata into one detail snapshot before overlay
  rendering. Keep the run's own declaration authoritative and preserve the
  legacy fallback only when metadata genuinely predates manifests.
- **Rejected alternative**: Infer axes from model ID or generation shape in
  JavaScript. That recreates the coupling ROADMAP-03 removed and
  misinterprets old or future runs.
- **Blast radius**: `_compute_run_frames`, Analytics detail transport,
  signal-axis tests, and any future manifest-driven overlay.
- **Verification**: Save a diffusion run with provenance signals, fetch its
  frames endpoint, and drive the real response through the VM page while
  scrubbing. Entropy values must follow the selected frame.
- **Depends on**: none
- **Roadmap impact**: Blocks trustworthy saved-run XAI for every
  frame-varying signal.

### [A2-XAI-04] Make entropy bar fills respect the signal's axes

- **Type**: defect
- **Severity**: medium
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/web/static/analytics.js:6347-6366`,
  `src/web/static/analytics.js:6391-6434`,
  `src/backends/registry.py:40-57`
- **What is true today**: `entropyFillColors()` treats bar index as token
  position and `overlayFrameIndex` as the number of positions that exist.
  That equivalence is true for an append-only autoregressive run, where
  frame k introduces position k. The entropy chart is now also offered for
  diffusion runs, where frame index is a denoising step and every bar is a
  position on that frame. The function's comment still says the chart is
  autoregressive-only.
- **Why it matters**: After the manifest transport is repaired, diffusion
  values can come from the correct frame while positions after the frame
  number remain incorrectly dimmed. The chart would combine correct
  numbers with a visual statement that most of those positions do not yet
  exist.
- **Direction**: Pass the entropy channel shape into fill generation. Dim
  future positions only for a `position` channel on an append stream; draw
  every present position normally for `frame|position`.
- **Rejected alternative**: Hide diffusion entropy from Analytics. It is a
  shipped, always-captured signal with a valid per-frame view.
- **Blast radius**: entropy datasets, scrub refresh, original/edited
  crossfade, and axis-aware browser tests.
- **Verification**: Use a diffusion fixture whose frame number is smaller
  than its position count. Scrubbing changes values by frame without
  dimming valid positions; AR behavior remains unchanged.
- **Depends on**: A2-XAI-03 for the real endpoint to supply the axes
- **Roadmap impact**: Completes the axis-aware Analytics contract for
  entropy and sets the pattern for future signals.

### Quality gates and dependency lifecycle

### [A2-QUALITY-01] Test DiffusionGemma resume at the backend boundary

- **Type**: meta
- **Severity**: medium
- **Effort**: M
- **Confidence**: high
- **Evidence**: `tests/inference/test_dgemma_resume.py:1-29`,
  `src/backends/dgemma_worker.py:247-381`,
  `src/web/static/app.js:6127-6137`,
  `tests/backends/test_llada_resume_state.py:1-20`
- **What is true today**: DiffusionGemma has detailed sampler tests, but no
  backend suite drives `_validate_resume`, `_forward_resume`, retained-state
  splicing, guided frame budgets, or rewind through
  `DgemmaBackend.handle_resume`. The multi-canvas backend refusal and the
  generator's `runIsMultiCanvas` Edit Frames gate also have no focused test.
  LLaDA has a backend-level resume-state suite. Both DiffusionGemma defects
  in this report sit above its tested sampler boundary.
- **Why it matters**: The gate can be fully green while a cancelled resume
  corrupts retained history, a guided branch reports stale text, or a
  refactor exposes Edit Frames on a run the backend must reject. This is the
  exact role of QUALITY-01's companion obligation.
- **Direction**: Build a checkpoint-only `DgemmaBackend` harness with a
  recording socket and stubbed `streaming_resume`, parallel to the LLaDA
  suite. Add one browser test for the multi-canvas UI gate. Cover empty,
  cancelled, guided, completed, failed, rewind, and two-canvas paths.
- **Rejected alternative**: Add more sampler-only tests. They cannot observe
  worker history or the terminal frame the backend synthesizes.
- **Blast radius**: New backend and browser fixtures; no production behavior
  until the defects are fixed.
- **Verification**: The current tree's empty-commit and empty-terminal-text
  behavior must fail the new tests, while the existing sampler suite remains
  unchanged.
- **Depends on**: none; land with A2-XAI-01 and A2-XAI-02
- **Roadmap impact**: Pins the single-canvas boundary before multi-canvas
  resume changes it.

### [A2-QUALITY-02] Automate one supervisor-to-worker proxy round trip

- **Type**: meta
- **Severity**: medium
- **Effort**: S
- **Confidence**: high
- **Evidence**: `scripts/ws_smoke_test.py:1-9`,
  `tests/web/test_activation_identity.py:326-369`,
  `tests/web/test_activation_identity.py:378-421`,
  `src/web/server.py:2007-2076`
- **What is true today**: Worker socket dispatch has strong in-process tests,
  and supervisor tests prove the resident handshake precedes stub worker
  traffic. The only test that sends a generation through the actual
  supervisor proxy and receives its frames is a manual script requiring a
  resident model. No automated test sends even a synthetic request in both
  directions through `_pipe`.
- **Why it matters**: A regression in browser-to-worker forwarding, early
  task cancellation, or proxy teardown can pass both halves' unit suites.
  This is the process-isolation seam every generation uses.
- **Direction**: Extend the existing fake worker socket fixture into a
  bidirectional round-trip: browser sends a request, fake worker records it
  and replies, browser receives it, then one side closes and both pipe tasks
  settle. Keep the real-model smoke script as hardware validation.
- **Rejected alternative**: Put a GPU smoke in the default suite. The
  transport contract needs no model and should remain fast and deterministic.
- **Blast radius**: Supervisor WebSocket tests and possibly a small
  injectable proxy helper.
- **Verification**: The test fails when either pipe direction or
  first-completed teardown is removed, and leaves no pending task.
- **Depends on**: none
- **Roadmap impact**: Protects the central process boundary for every model.

### [A2-DEPS-01] Move application lifecycle hooks to lifespan contexts

- **Type**: architecture
- **Severity**: low
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/server.py:1639-1663`,
  `src/backends/worker_base.py:1350-1383`,
  `requirements.txt:191-191`,
  `requirements.txt:1434-1435`
- **What is true today**: Both the supervisor and every worker register
  startup or shutdown work with FastAPI's deprecated `on_event` API. The
  clean suite emits 58 deprecation warnings from these hooks. Exact
  dependency pins make this safe today, but the next deliberate FastAPI
  upgrade must cross a lifecycle boundary that owns orphan sweeping, model
  loading, and worker termination.
- **Why it matters**: Warning volume hides new warnings, and waiting for
  removal turns a controlled migration into an upgrade blocker around the
  most resource-sensitive code in the project.
- **Direction**: Adopt explicit async lifespan context managers in both app
  factories, preserving startup-before-serving and shutdown-after-stop
  ordering. Do it with the worker transport extraction, or before any
  FastAPI upgrade, while focused lifecycle tests still describe the old
  behavior.
- **Rejected alternative**: Filter the warnings. That removes the only early
  signal without reducing future migration risk.
- **Blast radius**: supervisor app construction, `create_worker_app`,
  TestClient fixtures, desktop shutdown, and load task ownership.
- **Verification**: The full suite has no FastAPI lifecycle warnings;
  startup failure, shutdown, and TestClient context tests still prove the
  same order.
- **Depends on**: Coordinate with A2-ORG-01 and A2-ORG-02 to avoid moving the
  same lifecycle code twice.
- **Roadmap impact**: Clears a dependency-upgrade gate; no product feature is
  currently blocked.

### Documentation and routing

### [A2-META-01] Remove shipped work from the public "Next up" list

- **Type**: meta
- **Severity**: low
- **Effort**: S
- **Confidence**: high
- **Evidence**: `README.md:207-212`,
  `docs/HANDOFF.md:180-196`,
  `docs/ROADMAP.md:20-36`,
  `src/backends/dgemma_worker.py:1-8`
- **What is true today**: The README says Mamba-3 and diffusion top-k are
  next, while both are described as shipped in the same README and in the
  current handoff. The DiffusionGemma worker header still says resume is
  unsupported, while the module implements single-canvas resume and the
  GUIDE exposes it. Inventory tests prove that names and pages are present,
  but do not check status prose.
- **Why it matters**: These are the two shortest routing surfaces a new
  contributor reads. Contradictory status sends work toward features that
  already exist and makes live resume code look experimental or dead.
- **Direction**: Replace volatile README "Next up" prose with a pointer to
  the roadmap backlog, and update the worker header to state the exact
  single-canvas boundary. Keep changing current focus in HANDOFF/ROADMAP,
  where the documentation contract already routes it.
- **Rejected alternative**: Add increasingly semantic prose assertions to
  `test_docs_inventory.py`. Presence is mechanically testable; roadmap
  priority is better kept out of duplicated front-page prose.
- **Blast radius**: README and one module docstring only.
- **Verification**: A cold reader sees one current account: shipped models
  in README, current pickup in HANDOFF, rationale/backlog in ROADMAP.
- **Depends on**: none
- **Roadmap impact**: Removes false signals around already-shipped
  directions.

### Code organization

### [A2-ORG-01] Extract model ownership before splitting supervisor routes

- **Type**: organization
- **Severity**: medium
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/server.py:199-719`,
  `src/web/server.py:720-1619`,
  `src/web/server.py:1666-2089`,
  `tests/web/test_worker_lifecycle.py:35-36`
- **What is true today**: `server.py` is 4,399 lines. Lines 199-1,619
  contain hardware probes, artifact/download policy, process monitoring,
  lease ownership, and the 899-line `ModelManager`; model routes and the
  proxy immediately follow. Process spawning and the lease have already
  become modules, and the manager's constructor is injectable, but their
  owner still lives inside the FastAPI entrypoint. Tests import it from
  `server`, tying lifecycle work to a module that also defines every route,
  save schema, Analytics calculation, and page.
- **Why it matters**: Forty-two post-audit commits changed `server.py`.
  Every lifecycle change pays to map persistence and presentation code, and
  every route extraction risks circular imports around the module-global
  manager. The two high-value lease findings in this report sit inside that
  hub.
- **Direction**: First extract `ModelManager`, its lifecycle constants, and
  the probe/artifact helpers it owns into `src/web/model_manager.py`.
  Preserve its injectable spawn/probe contract and temporarily re-export
  `ModelManager` from `server.py` for tests. Keep app construction and the
  singleton in `server.py` until the boundary is stable. Only then move
  model, save, Analytics, collections, and page groups to routers in that
  order.
- **Rejected alternative**: Split routes first. It shortens the file while
  leaving the mutable worker owner and its helpers as a dependency every
  router must reach back into.
- **Blast radius**: supervisor imports, lifecycle tests, app startup, model
  routes, and later router work.
- **Verification**: Existing worker lifecycle, activation identity,
  residency, download, and desktop tests pass without importing FastAPI to
  construct a manager. No torch or transformers import enters the
  supervisor.
- **Depends on**: Land A2-LIFE-01 and A2-TRUST-01 before or as the first
  behavior-pinned step; coordinate with A2-DEPS-01.
- **Roadmap impact**: Makes future artifact/model management changes
  navigable without another supervisor monolith pass.

### [A2-ORG-02] Separate the worker socket shell from backend semantics

- **Type**: organization
- **Severity**: medium
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/backends/worker_base.py:634-1027`,
  `src/backends/worker_base.py:1200-1309`,
  `src/backends/worker_base.py:1318-1653`,
  `tests/backends/test_worker_dispatch.py:1-18`
- **What is true today**: `worker_base.py` is 1,653 lines and owns two
  different abstractions. The first is model-facing: `Backend`, frame
  streaming, provenance, tokenizer/context description, and resource
  measurement. The second is transport-facing: load state, FastAPI app
  construction, per-socket session state, greeting, dispatch, busy
  handling, cancellation, and exclusive/concurrent request policy. Recent
  work split the socket loop into small functions and tests already drive
  it independently, exposing a stable module boundary.
- **Why it matters**: Every model backend imports a file whose final third is
  an HTTP/WebSocket server, and every transport change loads a module full
  of model semantics. Sixteen post-audit commits changed this hub, including
  the most recent structural pass.
- **Direction**: Move `_LoadState`, `_Session`, `create_worker_app`, load
  health, greeting, dispatch, and generation-slot orchestration into
  `worker_app.py` or `worker_socket.py`. Keep `Backend`, `FrameStreamer`,
  provenance, and model-description helpers in `worker_base.py`. Re-export
  `create_worker_app` for one compatibility step.
- **Rejected alternative**: One file per helper class. The useful boundary
  is transport versus backend semantics, not line-count minimization.
- **Blast radius**: `run_worker.py`, worker tests, app lifespan migration,
  and imports in four workers.
- **Verification**: Dispatch/run-identity/resource-pump tests pass against
  the extracted app, worker modules compile in their assigned environments,
  and the supervisor proxy round-trip in A2-QUALITY-02 passes.
- **Depends on**: Coordinate with A2-DEPS-01
- **Roadmap impact**: Gives another worker protocol or model family a small,
  explicit transport surface.

### [A2-ORG-03] Extract a pure generator session snapshot codec

- **Type**: organization
- **Severity**: medium
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/static/app.js:269-449`,
  `src/web/static/app.js:9233-9651`,
  `src/web/static/run_frames.js:30-58`,
  `tests/web/static/snapshot_budget.test.js:1-18`
- **What is true today**: `app.js` is 9,906 lines and remains the owner of
  page DOM, model forms, socket lifecycle, rendering, edits, saving, and
  persistence. Frame, candidate, phase, activation, model, download, and
  error semantics already have tested classic-script seams. Session
  serialization is a cohesive late-file block with stable JSON keys and a
  measured quota test, but it reads and writes dozens of page globals
  directly.
- **Why it matters**: Fifty-four post-audit commits changed `app.js`.
  Persistence defects are hard to isolate from rendering, and the partial
  run fix in this report crosses save, reconnect, and snapshot code. Moving
  functions verbatim into another global script would improve navigation
  but preserve hidden ownership.
- **Direction**: Extract a DOM-free snapshot codec that accepts one explicit
  run-session record and returns full and degraded storage tiers, plus a
  decoder that returns validated state for the parent to apply. Leave
  storage calls and mutation of page globals in `app.js` for the first cut.
  Load it as a classic script before `app.js` and use the existing VM
  harness.
- **Rejected alternative**: Convert the page to native ES modules first.
  The 816-test classic-script harness is valuable, and module conversion
  does not itself choose a state owner.
- **Blast radius**: `app.js`, one new browser script, generator script order,
  session/snapshot tests, and boot restore.
- **Verification**: Snapshot quota, run record, append/snapshot, original
  run, partial run, candidates, and legacy-key round trips all pass through
  the codec; app state changes remain centralized in the caller.
- **Depends on**: A2-LIFE-02 and A2-LIFE-03 should settle the state that is
  serialized before the extraction freezes its shape.
- **Roadmap impact**: Creates a safe first boundary for further generator
  decomposition without a framework or bundler.

### [A2-ORG-04] Canonicalize Analytics wiring before splitting its controller

- **Type**: organization
- **Severity**: medium
- **Effort**: L
- **Confidence**: high
- **Evidence**: `src/web/static/analytics.js:646-2644`,
  `src/web/static/analytics.js:2645-3391`,
  `src/web/static/analytics.js:3392-6657`,
  `src/web/static/analytics.html:713-729`,
  `tests/web/static/analytics_signal_axes.test.js:27-38`
- **What is true today**: `analytics.js` is 7,570 lines with distinct
  catalog/collections, detail transport, Chart.js, token viewer, comparison,
  and deletion responsibilities. Tested transport seams already exist, but
  twelve test files copy their own `ANALYTICS_SCRIPTS` array while the
  generator has one canonical harness list. Adding an extracted classic
  script therefore requires coordinated edits across HTML and many
  fixtures before its behavior is even considered.
- **Why it matters**: Thirty-five post-audit commits changed this file, and
  it changed with `app.js` in twenty. The missing signal manifest crossed
  Python transport, viewer state, and a VM fixture that bypassed the real
  payload. More file splits without canonical wiring would increase this
  class of false-green test.
- **Direction**: First export one Analytics script-order manifest from the
  DOM stub and make every page test consume it. Then extract the DOM-free
  frame-shape/signal-axis adapter, followed by chart and token-viewer
  controllers with explicit namespace objects. Keep catalog selection and
  top-level boot state in `analytics.js` until those contracts settle.
- **Rejected alternative**: Move whole comment sections into more classic
  global files immediately. That improves line count while multiplying
  load-order and shared-global hazards.
- **Blast radius**: Analytics HTML, twelve browser fixtures, detail/viewer
  tests, and later chart modules.
- **Verification**: One harness list exactly matches first-party HTML order;
  a fixture exercises the real frames response shape; each extracted
  controller can be loaded and tested without booting the full page.
- **Depends on**: A2-XAI-03 and A2-XAI-04 should define the frame adapter
  first.
- **Roadmap impact**: Lowers the cost of frame-linked charts and per-run
  detail work.

### [A2-ORG-05] Split visual overlays from browser persistence services

- **Type**: organization
- **Severity**: medium
- **Effort**: M
- **Confidence**: high
- **Evidence**: `src/web/static/overlays.js:1-314`,
  `src/web/static/overlays.js:315-1839`,
  `src/web/static/overlays.js:1840-2095`,
  `src/web/static/overlays.js:2482-2650`
- **What is true today**: `overlays.js` began as shared visual math and DOM
  builders. Its 2,823 lines now also own the server-backed localStorage
  mirror, settings schema, new-run registry, last-run snapshot key, and
  activation progress calculations. Generator, Analytics, menu, settings,
  and Vision all load the file even when they need only one side. The
  responsibilities have different state lifetimes and failure modes.
- **Why it matters**: `overlays.js` changed with `app.js` in 15 commits and
  with `analytics.js` in 16 of its 21 post-audit commits. A change to
  persistence or activation
  ordering enters every visual test's global scope, while a token rendering
  change enters menu/settings pages that draw no tokens.
- **Direction**: First extract durable UI-state and snapshot-key functions
  into a tested `ui_persistence.js`; extract activation progress into its
  own pure helper or beside the activation client. Keep colors, token
  builders, candidate popover, metrics, stopping, revisions, glow settings,
  and drawer behavior in the visual module for now.
- **Rejected alternative**: Duplicate page-specific copies. The shared
  token semantics and settings are intentionally one contract.
- **Blast radius**: script order on all five pages, persistence/settings
  tests, menu and generator boot, and function naming.
- **Verification**: Each page loads only the services it uses; persistence
  failure tests and visual overlay tests run in separate VM contexts; no
  storage access occurs when testing token math.
- **Depends on**: Canonical page script manifests, especially A2-ORG-04
- **Roadmap impact**: Makes shared visualization features cheaper to add
  without growing a cross-page kitchen-sink module.

### [A2-ORG-06] Move the Main Menu tail into a page stylesheet

- **Type**: organization
- **Severity**: low
- **Effort**: S
- **Confidence**: high
- **Evidence**: `src/web/static/style.css:4239-4917`,
  `src/web/static/menu.html:9-12`,
  `src/web/static/index.html:15-17`
- **What is true today**: The final 679 lines of the 4,917-line shared
  stylesheet are explicitly headed "Main Menu" and define landing-page
  video, panel, picker, status, and responsive rules. Every page loads
  `style.css`, while only `menu.html` uses that tail. Other pages already
  layer page-local stylesheets after the shared file.
- **Why it matters**: This is a pure navigation boundary and an inexpensive
  precedent for reducing the CSS map. It also keeps menu-only selectors out
  of every generator, Analytics, Settings, and Vision parse.
- **Direction**: Move the complete Main Menu section to `menu.css`, loaded
  after `style.css` only by `menu.html`. Do not split shared variables,
  header primitives, or model-picker rules until selector ownership has
  been measured page by page.
- **Rejected alternative**: Split CSS by arbitrary size or component names.
  Cascade order and shared selectors would make those cuts behavioral.
- **Blast radius**: one stylesheet link, cache stamping, offline-asset
  checks, and visual verification of the menu.
- **Verification**: Source tests confirm every moved selector is menu-only;
  menu screenshots at desktop and narrow widths are pixel-equivalent; all
  other pages load without `menu.css`.
- **Depends on**: none
- **Roadmap impact**: None; this is the safest navigability-only cut.

## Sequencing

The dependency order is more important than the number of commits. Functional
behavior should be pinned and repaired in the modules that own it today,
then moved. Otherwise an extraction turns an incorrect behavior into a
stable-looking contract.

### Stage 1: close isolated correctness boundaries

1. Land A2-QUALITY-01's backend fixture, then A2-XAI-01 and A2-XAI-02
   as separate reviewed changes. The first test commit should fail only on
   the two named DiffusionGemma outcomes.
2. Add the real `/frames` transport assertion and land A2-XAI-03, then make
   the chart's fill policy shape-aware in A2-XAI-04.
3. Characterize a disconnected run in the VM harness and land A2-LIFE-02.
   Keep terminal Stop behavior unchanged; only the no-terminal disconnect
   path needs provisional text.
4. Land A2-LIFE-04, A2-META-01, and A2-ORG-06 as independent small changes.
5. Decide product limits, then land A2-TRUST-02 with boundary tests before
   multimodal or larger saved artifacts add another unbounded shape.

This stage is safe without GPU inference except for the final manual checks.
It reduces the chance that later file moves carry a known defect.

### Stage 2: make cross-process ownership complete

1. Pin the model-switch interleaving and repair A2-LIFE-01.
2. Settle the availability-versus-safety policy in A2-TRUST-01. If the
   maintainer chooses visible degraded operation rather than refusal, that
   is a user-facing default and needs explicit approval.
3. Add a multiprocessing race fixture and land A2-DATA-01. It should reuse
   the `flock` pattern, not create a second lock abstraction.
4. Add A2-QUALITY-02's proxy round trip, then consume the reconnect epoch in
   A2-LIFE-03.
5. Run the two-supervisor hardware matrix before treating the stage as
   validated. It also covers the old LIFE-05 queue.

### Stage 3: extract owners, not sections

1. Take A2-ORG-01 after the lease and publication fixes. Move the manager
   boundary first; leave route paths and response shapes unchanged.
2. Take A2-ORG-02 and A2-DEPS-01 in a coordinated plan. They touch worker app
   construction once, but should remain reviewable commits: transport move,
   then lifespan migration.
3. Take A2-ORG-03 after interrupted/reconnect state is settled. The first
   cut is a pure codec, not a rewrite of page state.
4. Take A2-ORG-04 after signal transport and fill semantics are green.
   Canonicalize the test script manifest before adding a production script.
5. Take A2-ORG-05 after page manifests are canonical. Split persistence and
   activation services; keep shared visual semantics together.

Route modules, additional CSS cuts, `ar_sampler.py` intervention extraction,
and a reader/compute split in `metrics.py` are later choices. None should be
preloaded into this stage merely because the relevant file is open.

### Combinations to avoid

- Do not combine DiffusionGemma behavior fixes with a sampler or worker
  module move.
- Do not extract `ModelManager` while also changing lease semantics unless a
  prior test-only commit pins the interleaving.
- Do not convert to ES modules while changing XAI, session, or chart
  behavior.
- Do not combine save-input limits with a saved-run schema migration or
  ROADMAP-04 artifact design.
- Do not pair the menu stylesheet split with a visual redesign.
- Do not upgrade FastAPI in the same commit that introduces lifespan
  contexts; migrate against the pinned version first.
- Do not split all supervisor route groups together. Each router should
  follow one settled service owner and preserve exact URLs and JSON shapes.

## Measurements to take on hardware

1. **Lease through a live switch.** Run browser and desktop supervisors,
   load a model in one, start a switch, and attempt activation from the
   other during teardown. Pass means the peer is refused continuously, the
   first switch succeeds, and explicit unload hands ownership over. Repeat
   the old items 310-314 in the same pass.
2. **Lease setup failure.** Point `XDG_RUNTIME_DIR` at an unwritable
   location. Pass means the chosen A2-TRUST-01 policy is visible before any
   worker spawns, not only in terminal logs.
3. **Disconnected partial saves.** For LLaDA, DiffusionGemma, SmolLM3, and
   Mamba-3, interrupt the socket after visible frames, save, and open the run
   in Analytics. Pass means latest text and frames agree, `partial` is shown,
   and no hidden inference remains.
4. **Same-model worker replacement.** Re-activate the same model and device
   while a page retains a completed unsaved run. Pass means the operation
   epoch produces an explicit rescue/freeze/reload and no stale edit control
   can send before that transition.
5. **DiffusionGemma resume edges.** Cancel before its first resumed draft,
   cancel after several drafts, and use Run to Here. Compare the page's
   frames and final text with a later edit. Pass means retained state is
   exactly what the page received and guided terminal text names the visible
   branch.
6. **Saved diffusion entropy.** Save one LLaDA and one DiffusionGemma run,
   open each in Analytics, and scrub early and late frames. Values must
   change with the selected diffusion frame and all present diffusion
   positions must keep normal fill; AR position behavior must not change.
7. **Analytics scaling.** Profile opening and closing detail at roughly 240,
   1,000, and 5,000 synthetic catalog rows. Record table rebuild time, long
   tasks, live Chart instances, and heap after fifty modal cycles. This
   decides whether incremental row updates or pagination merits a future
   finding; static inspection alone did not.
8. **Long live overlays.** Profile a high-step LLaDA run and a multi-canvas
   DiffusionGemma run with revision glow and candidate cycling on and off.
   Record browser long tasks, delivered frame cadence, and heap. Both paths
   are bounded in code, so only measured jank should trigger optimization.
9. **Desktop shutdown.** Close the window during load and during generation.
   Pass means the worker exits and VRAM returns within the documented
   timeout. This covers the daemon-thread risk that static reading could not
   promote to a finding.
10. **Outstanding old queues.** Preserve LIFE-02 items 143-144,
    ROADMAP-03 item 296, META-03 item 326, Mamba cold-download item 330, and
    Vision cold-config item 325 as existing manual debt. Items 217-266 need
    recorded outcomes before this audit treats their UI claims as hardware
    verified.

## Blind spots

- No CUDA device or display was available. Model output, VRAM behavior,
  pywebview, real layout, focus, animation, and browser performance were not
  exercised.
- The Node suite runs shipped scripts in a strong but hand-built DOM stub.
  It does not implement layout, native dialog top-layer behavior, event
  propagation in full, storage quotas outside explicit fixtures, or browser
  process crashes.
- The ignored saved-run corpus was not used as an exhaustive data set.
  Schema readers were tested against fixtures and prior recorded
  measurements; unusual real legacy runs may expose shapes absent there.
- Vendored libraries and quarantined reference implementations were checked
  at integration and differential-test boundaries, not audited internally.
- No fresh machine or empty Hugging Face cache was available, and no
  dependency download or quantization rebuild was attempted.
- This was a correctness and architecture audit, not a penetration test.
  Input and exposure boundaries were read statically, but the server was not
  fuzzed or placed on an adversarial network.
- Operational scripts without focused tests, such as icon rendering,
  desktop-entry installation, and the DiffusionGemma spike, received a
  routing review rather than line-by-line adversarial analysis.
- The 32 pre-existing untracked Cursor plan files were outside the tracked
  product and were deliberately left untouched.
