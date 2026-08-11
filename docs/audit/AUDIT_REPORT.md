# Repository Audit Report

Audit date: 2026-08-10

Scope: static, repository-wide review governed by `AUDIT_BRIEF.md`.
The test suite and permitted static checks supplement source reading.
GPU inference and graphical behavior were not exercised.

## Executive summary

The repository has a sound core direction. Process isolation is the right
answer to incompatible model environments, the supervisor stays out of the
model stacks, the frontend already shares meaningful overlay code, partial
Hub downloads recover, and modern saved runs carry unusually rich XAI data.
The permitted checks support that assessment: all 265 Python tests pass and
all seven JavaScript files parse. This is not a rewrite recommendation.

The urgent problems sit at ownership boundaries rather than in the numerical
cores. A worker retains one unlabelled last run for every browser window;
activation, cancellation, and reconnect are global but uncorrelated; a second
supervisor can create a second "only" worker; and disconnect does not reliably
stop threaded inference. On disk, same-second saves can collide, published
runs are mutated file by file, stale edited-run writers are not rejected, and
save provenance comes from whichever worker happens to be active later.
Analytics then adds two trust failures: stale detail requests can mix runs,
and its convergence/throughput calculations do not consistently count tokens.

The three highest-leverage moves are:

1. **Build a versioned transactional run store** (`ORG-01`, `DATA-01`,
   `DATA-04`, `DATA-05`). Give every bundle a unique ID, revision, immutable
   provenance, complete staged publication, one guarded resolver, and
   versioned readers. This removes several data-loss and Analytics failure
   classes at one boundary.
2. **Make lifecycle operations identifiable and testable** (`LIFE-01` through
   `LIFE-07`, `PROTOCOL-01`, `QUALITY-01`). Run, activation, resident-worker,
   and download identities must survive across supervisor, worker, and
   browser; cancellation and terminal process state need bounded,
   deterministic tests before another environment is added.
3. **Split model semantics before Mamba, without adopting a framework**
   (`ROADMAP-01`, `ROADMAP-03`, `ROADMAP-05`, `ORG-02`). Separate model
   family, stream shape, device support, text adapter, and signal axes; move
   the browser's aligned run arrays and workflow phases into a tested native
   ES-module core.

Several small safety commits should precede those larger seams: bind to
loopback (`TRUST-01`), fence Analytics detail responses (`ANALYTICS-01`),
preserve LLaDA state on failed resume (`LIFE-07`), and stop machine-reading
character length as token convergence (`ANALYTICS-02`). Complete intervention
checkpoint fidelity (`XAI-01`) should follow run ownership rather than waiting
for new XAI features.

The report deliberately does **not** recommend a frontend framework or
bundler, a database as the authoritative run store, one universal sampler,
mass formatting, or OS portability work. Native modules, a rebuildable
summary index if measurements justify one, shared wire builders, and
model-specific numerical loops are the cheaper reversible path.

## Findings index

Effort is directional: **S** is one cohesive commit, **M** is a short
multi-commit change, and **L** is a staged boundary migration.

| ID | Area | Severity | Effort | Title |
|---|---|---|---|---|
| ORG-01 | Organization | medium | M | Extract the run repository before splitting route modules |
| ORG-02 | Organization | medium | L | Extract a tested run-session core before further frontend growth |
| ORG-03 | Organization | medium | M | Consolidate the LLaDA sampling kernel and quarantine its reference program |
| ORG-04 | Organization | medium | S | Share activation orchestration as the first frontend cut |
| LIFE-01 | Lifecycle | high | M | Correlate retained worker state with the originating run |
| LIFE-02 | Lifecycle | high | M | Make worker termination a verified state transition |
| LIFE-03 | Lifecycle | critical | L | Fence activation and reconnection with operation identity |
| LIFE-04 | Lifecycle | high | L | Turn disconnects into bounded interruption, not hidden work |
| LIFE-05 | Lifecycle | high | M | Enforce the one-resident rule across supervisor processes |
| LIFE-06 | Lifecycle | medium | M | Validate a target before evicting the working model |
| LIFE-07 | Lifecycle | high | S | Commit LLaDA retained state only after resume succeeds |
| DATA-01 | Persistence | high | L | Publish saved runs uniquely and transactionally |
| DATA-02 | Persistence | high | L | Make durable UI intent conflict-aware and observable |
| DATA-03 | Persistence | medium | S | Resolve the data directory independently of process CWD |
| DATA-04 | Persistence | high | M | Persist run provenance from the run, not global manager state |
| DATA-05 | Persistence | high | L | Version and validate the saved-run contract |
| TRUST-01 | Trust | high | S | Bind to loopback unless network exposure is explicit |
| TRUST-02 | Trust | high | M | Make every core page work without third-party networks |
| TRUST-03 | Trust | high | L | Resolve and record immutable model artifacts |
| TRUST-04 | Trust | medium | L | Own downloads through cancellation and shutdown |
| PROTOCOL-01 | Protocol | medium | M | Scope every response and error to its operation |
| ANALYTICS-01 | Analytics | high | S | Commit detail responses only to the run that requested them |
| ANALYTICS-02 | Analytics | high | M | Derive token charts from token counts, not text length |
| ANALYTICS-03 | Analytics | medium | L | Page lightweight run summaries instead of loading every bundle |
| ANALYTICS-04 | Analytics | high | M | Make comparison a bounded, coherent run-set transaction |
| RUNTIME-01 | Runtime | medium | L | Bound frame queues and compact append-only streams |
| RUNTIME-02 | Runtime | medium | M | Bound GIF rendering and decouple it from core saves |
| RUNTIME-03 | Runtime | medium | S | Give custom selects keyboard and listener lifecycles |
| XAI-01 | Intervention fidelity | high | M | Retain complete checkpoints for reproducible interventions |
| ROADMAP-01 | Roadmap | high | M | Separate model family, generation shape, and device support |
| ROADMAP-02 | Roadmap | medium | M | Make the registry authoritative for parameter validation |
| ROADMAP-03 | Roadmap | high | L | Describe XAI signals by their axes and capture policy |
| ROADMAP-04 | Roadmap | medium | L | Give multimodal inputs an artifact lifecycle outside WebSocket JSON |
| ROADMAP-05 | Roadmap | high | M | Extract model-specific text semantics before reusing the AR loop |
| QUALITY-01 | Quality | medium | L | Put automated tests around lifecycle and browser contracts |
| QUALITY-02 | Quality | medium | M | Turn the lint baseline into a ratchet, then burn it down |
| META-01 | Meta | medium | M | Reduce HANDOFF to current decisions and move the verification ledger |
| META-02 | Meta | medium | S | Keep the canonical agent contract in tracked files |
| META-03 | Meta | medium | M | Rebuild documentation around one current inventory |
| DEPS-01 | Dependencies | medium | L | Consolidate environment intent before adding `.venv-ssm` |

## Findings in full

### Code organization

### [ORG-01] Extract the run repository before splitting route modules

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/web/server.py:407`,
  `src/web/server.py:937`, `src/web/server.py:1063`,
  `src/web/server.py:1180`, `src/web/server.py:1347`,
  `src/web/server.py:1514`, `src/web/server.py:1703`,
  `src/web/server.py:1874`
- **What is true today**: `server.py` contains hardware probes and
  process ownership, model routes, the WebSocket proxy, save schemas and
  blocking file writes, analytics reads/deletes, UI-state reconciliation,
  and static-page serving. The saved-run functions form the cleanest
  existing seam: they are blocking, already run in threads, and are used
  by save and Analytics, but they currently reach back into global
  manager and registry state.
- **Why it matters**: Data-integrity fixes now have to be reasoned about
  inside a 1,978-line application module, and path validation, format
  compatibility, publication, listing, and deletion have already
  diverged. Splitting routes first would make the same global storage
  logic harder to find without reducing coupling.
- **Direction**: First extract a dependency-light `run_store` with one
  guarded resolver and typed inputs for immutable provenance. It should
  own allocate, stage, validate, publish, list, load, replace, and delete;
  keep FastAPI route declarations in `server.py` until that boundary is
  stable. Next extract `ModelManager` behind process and telemetry
  adapters so lifecycle tests can fake them. Arbitrary per-route routers
  were rejected as the first cut because they add imports while leaving
  state ownership unchanged.
- **Blast radius**: Save and Analytics endpoints, UI-state
  reconciliation, GIF derivation, manager provenance, and tests.
- **Verification**: Existing API payloads must remain compatible while
  run-store tests cover path traversal, concurrent publication,
  replacement, corruption, legacy reads, and injected failures without
  importing FastAPI or model libraries.
- **Blocks**: DATA-01, DATA-04, DATA-05, multimodal artifacts, and a
  maintainable server split.

### [ORG-02] Extract a tested run-session core before further frontend growth

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/web/static/index.html:597`,
  `src/web/static/index.html:600`,
  `src/web/static/app.js:249`, `src/web/static/app.js:261`,
  `src/web/static/app.js:359`, `src/web/static/app.js:400`,
  `src/web/static/app.js:5092`, `src/web/static/app.js:5131`,
  `src/web/static/analytics.html:547`,
  `src/web/static/analytics.html:550`,
  `AUDIT_BRIEF.md:133`
- **What is true today**: The two page entrypoints are classic scripts
  with hundreds of globals and implicit load-order dependencies.
  Generator state is spread across parallel frame arrays, workflow
  booleans, and a string-valued edit phase. Snapshot, restore, truncate,
  save projection, and session restore must each remember every aligned
  array. The comment on `truncateRunArraysAt` records a real timing-chart
  bug caused by one sibling array being omitted. There are no automated
  frontend behavior tests.
- **Why it matters**: New signals and model classes multiply the number
  of arrays and valid phase combinations. The current shape lets a small
  omission produce a saved run whose charts disagree, while async races
  such as ANALYTICS-01 and LIFE-04 have no deterministic test harness.
- **Direction**: Keep the no-build browser architecture, but convert page
  entrypoints incrementally to native ES modules after the narrow
  activation-client cut in `ORG-04`. The first state-bearing module
  should be a pure `RunSession`/workflow reducer owning aligned frames,
  append/truncate/snapshot/restore, terminal provenance, and legal phase
  transitions. Test it with Node's built-in runner, then extract the
  remaining API clients with request epochs. A framework or bundler
  migration was rejected as the first move because it is not needed to
  establish explicit dependencies and test the risky state.
- **Blast radius**: `app.js`, `analytics.js`, shared overlays, script
  tags, session persistence, edit/save flows, and future signal views.
- **Verification**: Reducer tests must cover generate, disconnect,
  partial resume, Retry, Confirm/save failure, model mismatch, and every
  aligned-array invariant. A small browser smoke layer must prove the
  module wiring and modal request cancellation against real DOM events.
- **Blocks**: Reliable Mamba and diffusion-trajectory UI work.

### [ORG-03] Consolidate the LLaDA sampling kernel and quarantine its reference program

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/inference/streaming_sampler.py:5`,
  `src/inference/streaming_sampler.py:23`,
  `src/inference/streaming_sampler.py:154`,
  `src/inference/streaming_sampler.py:363`,
  `src/inference/llada_sampler.py:11`,
  `src/inference/llada_sampler.py:25`,
  `src/inference/llada_sampler.py:46`,
  `src/inference/llada_sampler.py:145`,
  `src/inference/llada_sampler.py:198`
- **What is true today**: Production streaming imports two small math
  helpers from `llada_sampler.py`, then independently implements the same
  CFG, Gumbel, remasking, transfer, and block schedule as that module's
  second generation loop. The old module also carries an unused history
  wrapper, direct model loading, and a standalone program with different
  defaults. Static search finds no caller for those execution paths.
- **Why it matters**: A future LLaDA change has two plausible sampling
  loops to edit, while the dormant one contributes a disproportionate
  share of lint noise and old assumptions. Entropy/top-k or sampling
  fixes can land in one implementation but not the other.
- **Direction**: Extract one synchronous diffusion step/schedule kernel
  with typed inputs, keeping live streaming and any retained offline
  wrapper around it. Quarantine the standalone reference program with
  provenance or remove it once differential tests pass. Reformatting the
  dormant script or adding parallel tests without consolidation was
  rejected because neither establishes one algorithm owner.
- **Blast radius**: LLaDA generation/resume, imports, helper tests,
  reference documentation, and the lint baseline.
- **Verification**: Golden differential tests must match live and
  offline wrappers under identical logits and RNG state for CFG, both
  remasking strategies, and block boundaries; a short LLaDA hardware
  smoke run must remain stable.
- **Blocks**: Low-risk LLaDA sampler evolution and diffusion signal
  capture.

### [ORG-04] Share activation orchestration as the first frontend cut

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/web/static/index.html:597`,
  `src/web/static/index.html:600`,
  `src/web/static/app.js:1392`, `src/web/static/app.js:1451`,
  `src/web/static/menu.js:1294`,
  `src/web/static/menu.js:1346`,
  `src/web/static/overlays.js:1415`,
  `src/web/static/overlays.js:1492`,
  `HANDOFF.md:833`
- **What is true today**: Activation presentation already shares a pure
  progress reducer, but the generator and menu separately own the POST,
  status poll, retry schedule, terminal decision, and cancellation. The
  recorded load-bar correction required coordinated edits in both
  clients, and the two uncorrelated pollers are part of `LIFE-03`.
- **Why it matters**: Activation is the narrowest frontend duplication
  with demonstrated maintenance cost and sits on every model addition.
  Fixing epochs, cancellation, and error handling twice invites the two
  entry paths to diverge again.
- **Direction**: Extract one dependency-light activation client,
  parameterized by progress, ready, error, and cancel callbacks. It can
  begin as one explicitly namespaced classic script and become a native
  module with `ORG-02`; this keeps the first migration reversible and
  costs one extra file jump plus small page adapters. Sharing more
  constants or only the reducer was rejected because that is the current
  boundary and it did not prevent orchestration drift.
- **Blast radius**: Menu activation and generator model switching only.
- **Verification**: Drive both page adapters through the same mocked
  start/loading/ready/error/cancel/stale-epoch sequences and prove their
  transport decisions are identical while navigation remains
  page-specific.
- **Blocks**: The clean implementation of `LIFE-03` and lowers Mamba
  integration risk.

### Process lifecycle and shared state

### [LIFE-01] Correlate retained worker state with the originating run

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/backends/worker_base.py:532`,
  `src/backends/worker_base.py:544`,
  `src/backends/worker_base.py:624`,
  `src/backends/llada_worker.py:299`,
  `src/backends/llada_worker.py:315`,
  `src/backends/dgemma_worker.py:198`,
  `src/backends/smollm3_worker.py:273`
- **What is true today**: One backend instance and one generation lock
  serve every worker WebSocket. Each backend retains exactly one
  `last_run_state`, and resume, substitution, and probe requests identify
  a frame or position but not the run that the browser believes it is
  editing. A generation from a second window therefore replaces the
  state behind the first window's still-visible run. The lock serializes
  this sequence, but does not make the later request belong to the right
  run.
- **Why it matters**: A first window can resume or probe a second
  window's run. If their shapes differ, the user gets a misleading range
  or candidate error. If their shapes happen to agree, the operation can
  succeed against the wrong prompt, which is worse because the result
  looks valid.
- **Direction**: Add a worker-issued run generation token to `done` and
  require it on every stateful follow-up. Reject a stale token before
  reading retained state. That is an incremental safety boundary; a
  later bounded per-session state map can preserve simultaneous editable
  runs if that workflow is worth its memory. A documentation-only
  "single window" restriction is cheaper, but was rejected because the
  server and desktop/browser entry points already allow two windows.
- **Blast radius**: Worker protocol, all three backends, generator edit
  messages, probe/token candidate flows, and protocol tests.
- **Verification**: Open two WebSockets, complete run A then run B, and
  prove that A's resume/substitute/probe is rejected as stale while B's
  succeeds. Repeat with equal frame counts and equal output lengths so
  accidental shape compatibility cannot hide the defect.
- **Blocks**: Reliable SSM state probes and any cross-model or
  multi-window comparison workflow.

### [LIFE-02] Make worker termination a verified state transition

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/server.py:590`,
  `src/web/server.py:597`, `src/web/server.py:620`,
  `src/web/server.py:760`, `src/web/server.py:768`,
  `src/web/server.py:775`, `src/web/server.py:482`,
  `src/web/server.py:1928`
- **What is true today**: The startup monitor records a terminal timeout
  or worker-reported load error and returns without terminating or
  clearing the worker. Separately, the stop path sends `kill()` after a
  wait failure but does not wait for the killed process to exit before
  clearing every manager field. `status()` and the `/generate` gate
  define "active" as an alive process, independently of `load_state`.
- **Why it matters**: Supervisor state can say either "error" while a
  failed worker still owns RAM or VRAM, or "idle" while a killed worker
  has not yet been reaped. A replacement can then be started against
  resources whose release has not been verified. The first condition
  also lets an alive-but-failed worker satisfy the generator page gate.
- **Direction**: Centralize terminal worker finalization around a process
  identity check: stop monitoring, terminate, escalate to kill, await
  process exit, then clear manager state. Startup-monitor failure needs a
  non-self-cancelling variant of that path. Keeping failed workers alive
  for diagnosis was rejected; captured stderr and an explicit debug
  mode are safer than retaining scarce model resources by default.
- **Blast radius**: Model activation, cancellation, switching, shutdown,
  load-progress reporting, and desktop close behavior.
- **Verification**: Use fake subprocesses for startup timeout,
  health-reported error, graceful exit, terminate timeout, and kill
  escalation. Assert no new spawn occurs before the prior process has
  exited and that every terminal manager snapshot agrees with process
  reality.
- **Blocks**: Adding another GPU-only environment, especially Mamba-3,
  without increasing lifecycle failure modes.

### [LIFE-03] Fence activation and reconnection with operation identity

- **Severity**: critical
- **Confidence**: high
- **Evidence**: `src/web/server.py:953`,
  `src/web/server.py:990`, `src/web/server.py:1004`,
  `src/web/static/menu.js:1294`,
  `src/web/static/menu.js:1307`,
  `src/web/static/menu.js:1418`,
  `src/web/static/app.js:13`, `src/web/static/app.js:17`,
  `src/web/static/app.js:1501`,
  `src/backends/worker_base.py:652`
- **What is true today**: Activation is singleton global state with no
  operation ID or owner. The menu polls only for a global terminal
  state and its Cancel button unconditionally stops the current worker.
  A second window can therefore supersede or cancel the first window's
  activation. More seriously, a generator reconnects to whichever
  worker is active, while its cached `activeModel`, device, tokenizer,
  parameter form, and capabilities still describe the worker present at
  page boot. The worker's ready handshake carries no model identity.
- **Why it matters**: Window A can ask for model X, window B can replace
  it with Y, and A can either navigate when Y becomes ready or reconnect
  its X-configured generator to Y. A subsequent Generate can be labeled
  and parameterized as X while Y handles it, often accepting unknown or
  missing fields through defaults rather than failing loudly.
- **Direction**: Give each activation a monotonically increasing
  operation ID plus target model/device, return it from POST, expose it
  in status, and require it for cancellation. Include resident identity
  and epoch in the WebSocket handshake; a generator that sees a mismatch
  must invalidate its run and reload or return to the menu. Merely
  comparing `status.active` in the current poll was rejected because it
  leaves cross-window Cancel and reconnect unsafe.
- **Blast radius**: Model Manager API, menu and generator switch flows,
  WebSocket handshake, cached run/form state, and two-window behavior.
- **Verification**: Interleave X and Y activation, cancellation, failure,
  and readiness from two clients. No client may navigate for another
  operation, cancel another operation, or send an X request to Y after a
  reconnect.
- **Blocks**: Safe addition of every new model and device combination.

### [LIFE-04] Turn disconnects into bounded interruption, not hidden work

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/static/app.js:1524`,
  `src/web/static/app.js:1762`,
  `src/web/static/app.js:1811`,
  `src/web/static/app.js:5689`,
  `src/backends/worker_base.py:655`,
  `src/backends/worker_base.py:663`,
  `src/backends/worker_base.py:698`,
  `src/inference/streaming_sampler.py:377`,
  `src/inference/ar_sampler.py:1391`,
  `src/inference/ar_sampler.py:1398`,
  `src/inference/ar_sampler.py:1419`,
  `src/inference/dgemma_sampler.py:269`,
  `src/inference/dgemma_sampler.py:304`
- **What is true today**: The browser's close handler marks the model
  disconnected but does not leave its generating state; only a `done`
  or `error` frame does that. In the worker, the same receive loop
  awaits a whole generation, so it cannot process a Cancel message or
  notice WebSocket disconnect until the handler returns. AR and
  DiffusionGemma bridge inference through threads and await those threads
  in generator cleanup. DiffusionGemma's `model.generate` receives no
  stopping criterion tied to the connection event. When a cancellation
  event is eventually observed, LLaDA returns without a terminal frame
  while AR can finalize a partial run, so cancellation has no shared
  state contract.
- **Why it matters**: A network drop, navigation, worker switch, or
  browser close during generation can leave the surviving page
  permanently disabled while inference continues unseen. The worker can
  retain the generation lock until a long CPU or experimental-budget run
  finishes, so reconnecting does not restore control.
- **Direction**: Separate socket receiving and generation into supervised
  tasks so Cancel and disconnect set a thread-safe stop signal
  immediately. Give DiffusionGemma an event-backed stopping criterion and
  bound the thread-to-async queue. Emit exactly one scoped `cancelled`
  terminal outcome and define whether retained state is discarded or
  explicitly partial. On the client, transition to an
  explicit interrupted state that preserves partial frames but does not
  present them as complete, with retry/reset behavior. Simply clearing
  `isGenerating` on close was rejected because it would make partial
  state look saveable and still leave backend work running.
- **Blast radius**: Worker WebSocket loop, all sampler cancellation paths,
  supervisor proxy teardown, generator state/status UI, and reconnect.
- **Verification**: Disconnect and reconnect during each model's
  generation on CPU and GPU. Compute and queue growth must stop within a
  measured bound, the generation lock must release, and the page must
  offer an honest recovery without a reload. Every model must emit one
  equivalent terminal cancellation outcome.
- **Blocks**: Long Mamba decode loops and larger multimodal requests,
  where abandoned work is more expensive.

### [LIFE-05] Enforce the one-resident rule across supervisor processes

- **Severity**: high
- **Confidence**: high
- **Evidence**: `desktop.py:60`, `desktop.py:65`,
  `desktop.py:90`, `desktop.py:94`, `desktop.py:101`,
  `main.py:12`, `main.py:18`, `src/web/server.py:407`,
  `src/web/server.py:807`, `src/web/server.py:811`
- **What is true today**: "One worker" is enforced by one in-memory
  `ModelManager`, not by the host. The browser supervisor and desktop
  supervisor deliberately use different ports, and a second desktop
  process deliberately falls back to another port. Each process then
  owns an independent manager and startup sweep. There is no live-owner
  lock or broker between them.
- **Why it matters**: Two supervisors can pass VRAM preflight
  concurrently before either allocation is visible, then launch two
  workers into a device sized for one. Even when the models happen to
  fit together, either UI can switch or shut down only its own worker,
  so the product's central resource invariant no longer matches what the
  machine is doing.
- **Direction**: Add a host-level, stale-safe ownership lease around the
  primary model worker, using an Ubuntu file lock or a small local broker.
  A second desktop launch should focus or navigate to the existing
  instance instead of silently opening a new supervisor. Keep the lease
  scoped to the primary model resource so a future deliberately
  concurrent utility worker can use a separately named budget. Relying
  on the VRAM preflight was rejected because it has a check-then-allocate
  race and cannot govern CPU RAM.
- **Blast radius**: `main.py`, desktop startup, activation, shutdown,
  orphan recovery, and the future utility-worker design.
- **Verification**: Synchronize two supervisors at preflight and request
  activation simultaneously. Exactly one must acquire ownership; the
  other must identify the owner and remain usable without spawning.
  Crash the owner and prove the lease becomes recoverable without
  killing an unrelated process.
- **Blocks**: Safe desktop distribution and any deliberate multi-worker
  roadmap item.

### [LIFE-06] Validate a target before evicting the working model

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/web/server.py:502`,
  `src/web/server.py:505`, `src/web/server.py:512`,
  `src/web/server.py:514`, `src/web/server.py:521`,
  `src/web/static/app.js:1398`,
  `src/web/static/app.js:1404`,
  `src/web/static/menu.js:1353`,
  `README.md:232`
- **What is true today**: Activation stops the resident worker before it
  checks that the target interpreter exists, and before target-specific
  static artifact validation. Both frontend switch paths also delete the
  previous run snapshot before the activation request succeeds. An
  optional DiffusionGemma installation is intentionally listed even when
  activation will fail.
- **Why it matters**: Choosing a missing, partial, or misconfigured
  optional model can evict a healthy resident model and discard an
  unsaved visible run before returning an error that could have been
  known without freeing VRAM. Recovery then requires another slow load,
  and the run snapshot is gone even if the user switches back.
- **Direction**: Split activation into validate, reserve, evict, and
  launch phases. Check interpreter, supported device, local manifest or
  downloadable artifact, and non-destructive resource estimates before
  eviction. Preserve the old run snapshot until the new worker is ready,
  keyed by worker/run epoch so switching back cannot resurrect it
  accidentally. Automatic rollback by loading the old model was rejected
  as the first solution because it doubles failure latency; avoiding
  preventable eviction is cheaper.
- **Blast radius**: Manager activation, menu/generator confirmation,
  session snapshots, load progress phases, and artifact validation.
- **Verification**: Attempt switches to a missing venv, missing and
  partial local checkpoint, offline uncached Hub model, unsupported
  device, insufficient VRAM, and worker load failure. Every
  pre-eviction failure must leave the resident worker and run usable;
  every post-eviction failure must offer an honest recovery path.
- **Blocks**: A trustworthy Mamba row before `.venv-ssm` is installed.

### [LIFE-07] Commit LLaDA retained state only after resume succeeds

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/backends/llada_worker.py:368`,
  `src/backends/llada_worker.py:373`,
  `src/backends/llada_worker.py:374`,
  `src/backends/llada_worker.py:378`,
  `src/backends/llada_worker.py:401`,
  `src/backends/llada_worker.py:419`,
  `src/backends/dgemma_worker.py:287`,
  `src/backends/dgemma_worker.py:323`,
  `src/web/static/app.js:1815`
- **What is true today**: LLaDA truncates its retained
  `tensor_history` before entering the resume `try` block. An inference,
  streaming, cancellation, or socket-send failure therefore leaves the
  worker's only resumable state shortened. The browser error path
  independently restores its full pre-edit snapshot. DiffusionGemma
  already demonstrates the safer shape by building `base_history` plus
  staged resume frames and assigning only after forwarding.
- **Why it matters**: One failed resume can make browser and worker
  disagree about which frames exist. Retry may then fail as out of range
  or branch from a different retained frame while the UI shows the
  restored original.
- **Direction**: Treat resume as a state transaction: leave the current
  state untouched, build a candidate history, and replace the retained
  list only after the accepted terminal outcome. Copying every tensor
  was rejected because a new list of existing prefix references plus new
  resume tensors gives rollback without duplicating tensor storage.
- **Blast radius**: LLaDA resume, guided Retry/Exit, disconnect recovery,
  run-token validation, and worker state tests.
- **Verification**: Inject failure before the first frame, midway,
  during terminal send, and after partial-resume targeting. After every
  failure, the original history and total-step metadata must be
  unchanged and a second resume from any original frame must succeed.
- **Blocks**: Reliable LLaDA intervention and a generalized retained
  state contract.

### Persistence and data integrity

### [DATA-01] Publish saved runs uniquely and transactionally

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/server.py:1226`,
  `src/web/server.py:1229`, `src/web/server.py:1230`,
  `src/web/server.py:1435`, `src/web/server.py:1442`,
  `src/web/server.py:1450`, `src/web/server.py:1485`,
  `src/web/server.py:1214`, `src/web/server.py:1260`,
  `src/web/server.py:1493`, `src/analytics/metrics.py:294`,
  `src/analytics/metrics.py:300`,
  `src/web/static/app.js:6459`,
  `src/web/server.py:1661`, `src/web/server.py:1673`
- **What is true today**: A new run ID has one-second timestamp
  resolution plus the model ID, and its directory is opened with
  `exist_ok=True`. Save requests run in background threads and write the
  published directory file by file, starting with `metadata.json` and
  ending with GIF rendering. There is no save lock, exclusive directory
  creation, staging directory, or atomic publication step. Analytics
  considers a directory visible as soon as it has metadata. For an
  in-place edit, possession of a `run_id` authorizes replacement with no
  expected revision or lineage check, and optional sidecars omitted from
  the new request remain from the old bundle. Deletion recursively
  removes the visible directory in place, so readers and saves can
  observe or race its partial disappearance.
- **Why it matters**: Two windows saving the same model in one second can
  target the same folder and overwrite or interleave into a hybrid run.
  A disk, encoding, or GIF failure after metadata is written leaves a
  visible partial run. In-place edited saves have the sharper version of
  the same problem: failure can partly replace the only good copy. Two
  stale windows can also publish conflicting complete edits, with the
  last writer silently winning, or leave old sidecars advertising
  overlays that the new metadata no longer describes.
- **Direction**: Allocate collision-proof IDs with exclusive creation,
  write a complete bundle into a sibling staging directory, validate it,
  then publish with one atomic rename. For in-place edits, require an
  expected bundle revision/digest, build a complete fresh bundle, and
  compare-and-swap it rather than mutating the live folder. Rename a
  deleted bundle atomically out of the visible namespace before bounded
  cleanup. Adding
  milliseconds or a random suffix alone was rejected because it reduces
  collisions but does not prevent partial publication; a global mutex
  was rejected because serialization cannot identify a stale writer.
- **Blast radius**: Save responses and run IDs, edited-run replacement,
  analytics discovery, new-run cues, collections, and deletion.
- **Verification**: Race many same-model saves under a frozen clock;
  inject a failure at every file write and during GIF rendering; kill the
  process mid-save. Every successful request must produce one internally
  consistent run, every failed request must leave the prior state
  unchanged, and no staging folder may appear in Analytics. Race two
  replacements from one base revision; exactly one may commit, and
  omitted sidecars must be absent from the winner. Race save/read/delete
  and require complete data or a clean not-found result.
- **Blocks**: Trustworthy multimodal saves, which will add larger and more
  failure-prone artifacts, and aggregate analytics.

### [DATA-02] Make durable UI intent conflict-aware and observable

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/ui_state.py:12`,
  `src/web/ui_state.py:18`, `src/web/ui_state.py:58`,
  `src/web/ui_state.py:110`,
  `src/web/static/overlays.js:937`,
  `src/web/static/overlays.js:944`,
  `src/web/static/overlays.js:950`,
  `src/web/static/overlays.js:962`,
  `src/web/static/overlays.js:968`,
  `src/web/static/overlays.js:984`,
  `src/web/static/analytics.js:990`,
  `src/web/server.py:1730`, `src/web/server.py:1737`,
  `src/web/server.py:1837`
- **What is true today**: Atomic file replacement and a process lock
  protect the JSON file from torn writes, but clients replace each key as
  an opaque whole string. Collections, the one value explicitly
  identified as irrecoverable user intent, are read into a window-local
  array and written back wholesale. There is no revision, compare and
  swap, merge, or server-side collection operation. Writes are delayed
  250 ms and failed requests are intentionally swallowed, while the next
  page hydrate overwrites localStorage with the server copy. GET
  reconciliation can also write a value computed from a stale snapshot
  over a concurrent PUT. The `threading.Lock` is process-local, so
  browser and desktop supervisors writing the same file add an
  interprocess lost-update path.
- **Why it matters**: Two windows hydrated from the same value can each
  file a different run; the later PUT silently removes the earlier
  change. A failed PUT still looks saved in the current window's
  localStorage, then disappears when another origin or a later session
  hydrates from the server. Navigating quickly after a change can cancel
  the delayed PUT and immediately hydrate the old value. Atomic bytes are
  therefore being presented as durable intent without conflict or
  failure semantics.
- **Direction**: Make collections server-authoritative through bounded
  semantic operations, or add a revision/ETag and reject stale
  replacements so the client can reload and merge. Pair that semantic
  conflict handling with an interprocess file lock and surface
  persistence failure for this key. `BroadcastChannel` was rejected as
  the primary fix because it does not coordinate the browser and desktop
  origins and does not make disk failure observable.
- **Blast radius**: UI-state API, collections, settings/highlight
  write-through, prompt history, and tests involving two clients.
- **Verification**: Race independent clients adding, removing, renaming,
  and deleting memberships in one and two supervisor processes; prove no
  accepted change is lost. Inject write failures and prove the UI
  distinguishes local-only from durable state.
- **Blocks**: Future user-authored annotations, experiment sets, or other
  durable analytics organization.

### [DATA-03] Resolve the data directory independently of process CWD

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/web/server.py:74`,
  `src/web/server.py:75`, `desktop.py:30`,
  `desktop.py:32`, `desktop.py:36`, `main.py:24`,
  `main.py:26`, `HANDOFF.md:2191`, `HANDOFF.md:2193`,
  `HANDOFF.md:2196`
- **What is true today**: Worker interpreters and static files resolve
  from `REPO_ROOT`, but `RESULTS_DIR` is a relative `Path("results")`.
  The desktop launcher compensates with a process-wide `chdir`; the
  browser launcher does not. The handoff records a real incident in
  which a different working-directory/name situation created two result
  trees and split UI state.
- **Why it matters**: Starting `main.py` by absolute path, from a service,
  or from another directory can make saved runs and durable settings
  appear to vanish because the app silently reads and writes a different
  folder. A user looking in the repository sees no work even though it
  exists elsewhere.
- **Direction**: Resolve one absolute data root at startup, defaulting to
  `REPO_ROOT / "results"` and optionally accepting an explicit
  `--results-dir` or environment setting. Pass it to the run store and
  UI-state store, and display/log the resolved path. Keeping CWD as an
  undocumented configuration mechanism was rejected because it already
  produced split state.
- **Blast radius**: Browser and desktop launchers, run/UI-state stores,
  path display, tests, symlink policy, and setup docs.
- **Verification**: Launch from the repository, `/tmp`, a desktop entry,
  and a service-like working directory. All must see the same default
  runs and state; an explicit alternate data root must be isolated and
  named in the UI/log.
- **Blocks**: Reliable packaging and external-user trust.

### [DATA-04] Persist run provenance from the run, not global manager state

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/server.py:1180`,
  `src/web/server.py:1333`, `src/web/server.py:1341`,
  `src/web/server.py:1369`, `src/web/server.py:1424`,
  `src/web/server.py:1428`, `src/web/server.py:1432`,
  `src/web/server.py:1493`,
  `src/backends/llada_worker.py:77`,
  `src/backends/smollm3_worker.py:80`,
  `src/backends/worker_base.py:587`
- **What is true today**: The save body identifies a model and carries
  generated data, but processor, context limit, library versions, and
  tokenizer identity are read from the supervisor's currently active
  worker while the save executes. The code's soundness argument assumes
  a model switch discards every unsaved run. That assumption is a client
  behavior, not an API invariant, and it does not hold across two
  windows sharing the manager. The supervisor also records the requested
  device, while LLaDA and SmolLM3 may silently fall back from requested
  CUDA to CPU; worker health does not attest the effective device.
- **Why it matters**: Window A can finish a run, window B can switch the
  model or device, and A can then save correct text with B's processor,
  context limit, versions, and tokenizer. The reproducibility block can
  confidently lie, and a CUDA request can be saved as GPU even when the
  worker actually ran on CPU, undermining the main reason it is
  persisted.
- **Direction**: Emit an immutable provenance envelope with the run's
  terminal frame, retain it in the browser run snapshot, and submit that
  exact envelope on save with a worker/run token for validation. The
  worker must attest effective model ID, artifact revision, and device,
  not echo the supervisor's request. A
  save-time check that `body.model == manager.active_id` was rejected as
  the full solution because it only turns silent corruption into a
  failure and still cannot recover the completed run's facts after a
  switch.
- **Blast radius**: Done-frame protocol, all samplers/workers, save
  schema, run snapshots, metadata, and Analytics detail fields.
- **Verification**: Finish a run, switch model and device from another
  client, then save the first run. Its provenance must remain byte-for-byte
  tied to the worker that generated it, or the request must fail before
  writing anything.
- **Blocks**: Credible cross-model comparisons and experiments that mix
  CPU, GPU, SSM, and multimodal workers.

### [DATA-05] Version and validate the saved-run contract

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/server.py:1136`,
  `src/web/server.py:1145`, `src/web/server.py:1180`,
  `src/web/server.py:1445`, `src/analytics/metrics.py:16`,
  `src/analytics/metrics.py:39`, `src/analytics/metrics.py:173`,
  `src/analytics/metrics.py:263`,
  `src/analytics/metrics.py:303`,
  `src/analytics/metrics.py:319`,
  `src/web/static/analytics.js:1755`,
  `src/web/static/analytics.js:1760`,
  `src/web/static/analytics.js:1781`,
  `tests/web/test_save_signals.py:3`
- **What is true today**: Saved runs have no format version. New and
  legacy shapes are inferred from missing files, fields, or token types.
  Pydantic's default extra-field behavior silently drops an unlisted
  token or request signal, a failure mode the source and tests explicitly
  acknowledge. `history.txt` is also parsed by an unescaped sentinel
  line that is written verbatim around model-generated text, so a model
  can emit the same line and create false frame boundaries. Metadata is
  not validated as an object: a valid JSON scalar/list can escape the
  catalog's narrow exception handling. Analytics escapes parameter
  values but inserts parameter keys into `innerHTML` unescaped.
- **Why it matters**: Every new signal needs coordinated edits across
  protocol, save models, files, readers, and two frontends, but an
  incomplete rollout can return HTTP success while discarding data.
  Reader heuristics will become ambiguous when SSM channels, diffusion
  trajectories, and image provenance arrive. A generated frame-header
  line can already corrupt current metrics without corrupt JSON. One
  malformed folder can break or silently disappear from the catalog, and
  a crafted parameter key can execute same-origin markup/script with
  access to local destructive APIs.
- **Direction**: Put an explicit schema version and capture manifest in
  metadata; reject unknown fields at the save boundary; validate each
  bundle before publication; and route reads through version-specific
  adapters into one current in-memory shape. Build metadata DOM with
  `textContent`, never interpolated keys. Treat `history.txt` as a
  human artifact, not a machine protocol, or replace it with JSONL or a
  length-prefixed form. Continuing to infer versions from field presence
  was rejected because combinations of optional signals are not
  versions.
- **Blast radius**: Save models and files, Analytics loaders, legacy
  repair, export/import, tests, and every future XAI signal.
- **Verification**: Golden fixtures for every supported version must
  load into one canonical result. Unknown write fields must fail loudly,
  generated delimiter lines must round-trip, non-object/nested malformed
  metadata must become an explicit invalid-run entry, hostile keys must
  render as text, and unsupported future versions must produce a visible
  compatibility error without changing disk.
- **Blocks**: SSM-native channels, diffusion entropy/top-k trajectories,
  multimodal provenance, and stable external analysis tooling.

### Serving and offline trust

### [TRUST-01] Bind to loopback unless network exposure is explicit

- **Severity**: high
- **Confidence**: high
- **Evidence**: `main.py:9`, `main.py:12`, `main.py:13`,
  `src/web/server.py:953`, `src/web/server.py:1004`,
  `src/web/server.py:1493`, `src/web/server.py:1676`,
  `README.md:263`
- **What is true today**: The browser launcher defaults to
  `0.0.0.0`, while the product is described and opened as a localhost
  tool. The same unauthenticated origin can activate or cancel models,
  submit saves, and permanently delete runs. The desktop launcher,
  by contrast, binds explicitly to `127.0.0.1`.
- **Why it matters**: On a machine or firewall that permits the port,
  the default command exposes expensive model controls and destructive
  data operations to the local network. A future user can reasonably
  read "local" as machine-local and never realize that the default bind
  is broader.
- **Direction**: Default `main.py` to loopback. Keep remote access as an
  explicit `--host 0.0.0.0` choice with a prominent warning and, before
  treating it as supported, an authentication and origin policy. A
  README warning alone was rejected because safe binding is a one-line
  enforceable default.
- **Blast radius**: CLI defaults, setup copy, remote-use expectations,
  WebSocket origin handling, and any future authentication design.
- **Verification**: Prove the default listener is unreachable through a
  non-loopback interface; prove explicit remote mode still works and
  visibly warns before serving destructive endpoints.
- **Blocks**: Trustworthy distribution to users who may run the
  quickstart on shared networks.

### [TRUST-02] Make every core page work without third-party networks

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/static/analytics.html:9`,
  `src/web/static/analytics.html:13`,
  `src/web/static/analytics.html:544`,
  `src/web/static/analytics.html:546`,
  `src/web/static/analytics.js:175`,
  `src/web/static/analytics.js:177`,
  `src/web/static/index.html:7`,
  `src/web/static/menu.html:7`,
  `src/web/static/settings.html:7`
- **What is true today**: Every page contacts Google Fonts, and the
  Analytics page loads Chart.js, Hammer.js, and the zoom plugin from
  jsDelivr without a local fallback. `analytics.js` dereferences
  `Chart` near the top of the file, so a blocked CDN prevents the local
  script from reaching the run browser and not merely from drawing
  charts.
- **Why it matters**: A local-first app loses its saved-run interface
  when offline, behind a restrictive network, or when the CDN fails.
  Third-party code also executes with the app origin's ability to call
  model and deletion APIs. Fixed package versions reduce drift but do
  not provide offline availability or an integrity boundary.
- **Direction**: Vendor the three chart assets and their licenses under
  static assets, and either vendor the font or prefer a system monospace
  stack. Also feature-detect chart support so the table remains usable
  if chart initialization fails. A CDN fallback was rejected as the
  primary path because it still makes offline behavior a secondary,
  less-tested mode.
- **Blast radius**: Static assets, analytics script order, licensing,
  cache stamping, and setup size.
- **Verification**: Block all outbound traffic and open every page from
  a cold browser profile. Generation UI, settings, run table, detail
  metadata, charts, zoom, and deletion must work with zero external
  requests.
- **Blocks**: A trustworthy packaged desktop experience.

### [TRUST-03] Resolve and record immutable model artifacts

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/backends/registry.py:34`,
  `src/backends/registry.py:36`, `src/backends/registry.py:127`,
  `src/backends/llada_worker.py:90`,
  `src/backends/llada_worker.py:95`,
  `src/backends/llada_worker.py:122`,
  `src/inference/hf_download.py:135`,
  `src/inference/hf_download.py:155`,
  `src/inference/hf_download.py:167`,
  `scripts/quantize_diffusiongemma_nf4.py:75`,
  `scripts/quantize_diffusiongemma_nf4.py:80`,
  `scripts/quantize_diffusiongemma_nf4.py:105`,
  `README.md:226`, `README.md:229`
- **What is true today**: Hub checkpoints are registry repo names with no
  revision. Downloads and `from_pretrained` resolve that mutable name,
  and LLaDA executes remote model code. Saved runs retain the name but
  not the resolved Hub commit or weight digest. Hub metadata already
  provides total bytes but no free-space check uses it. The local
  DiffusionGemma artifact is created directly in its final directory,
  including a direct 16 GB `torch.save`; the menu considers any such
  directory downloaded, with no completion manifest or source revision.
  Setup tells the user to download the base and run quantization but does
  not pin or provide the exact base-revision fetch.
- **Why it matters**: The same app commit, parameters, and displayed seed
  can load different weights or remote code after a repository update.
  An interrupted local quantization leaves a directory that looks
  installed but contains a partial state dict. Neither condition gives a
  future user a reliable way to identify or reconstruct the model that
  produced a run.
- **Direction**: Extend the registry artifact descriptor with a pinned
  source revision and load from the resolved snapshot. Persist that
  revision plus relevant digests in run provenance. Preflight cache
  space against remaining bytes with a safety reserve. Quantize into a
  staging directory after the same kind of disk-space check, write a
  manifest naming source revision and quantizer code, validate required
  files, then rename atomically. Pinning only `trust_remote_code` while
  leaving weights mutable was rejected because code and weights are one
  reproducibility unit.
- **Blast radius**: Registry, download/cache probes, both Hub workers,
  DiffusionGemma quantization and loading, metadata, and upgrade docs.
- **Verification**: Run offline from a resolved snapshot, interrupt
  quantization at every phase, and move a Hub branch after caching.
  Activation must use the recorded revision or give an actionable
  missing-artifact error; partial local output must never appear ready.
- **Blocks**: Reproducible external use and additional downloadable model
  families.

### [TRUST-04] Own downloads through cancellation and shutdown

- **Severity**: medium
- **Confidence**: medium
- **Evidence**: `src/web/server.py:660`,
  `src/web/server.py:680`, `src/web/server.py:694`,
  `src/web/server.py:756`, `src/web/server.py:818`,
  `src/inference/hf_download.py:161`,
  `src/inference/hf_download.py:171`,
  `src/inference/hf_download.py:178`,
  `src/inference/hf_download.py:185`,
  `desktop.py:274`, `desktop.py:277`,
  `HANDOFF.md:3202`
- **What is true today**: Prefetch is an asyncio task that delegates to
  a thread, whose download helper starts another thread and joins it
  through completion. `ModelManager.stop()` and the shutdown hook stop
  only the model worker. There is no download cancellation path, and
  cancelling the outer task would not stop either thread. Partial Hub
  blobs are correctly detected and resumable, so cache recovery is
  healthier than task ownership.
- **Why it matters**: Static reasoning, not measured: closing the
  desktop or supervisor during a multi-gigabyte fetch can leave
  network/disk work running or hold process exit past the desktop's
  35-second join. More downloadable models make an operation with no
  owner or shutdown bound increasingly visible.
- **Direction**: Run each download as an identified subprocess or other
  killable owned operation, terminate and await it during cancel and
  shutdown, and preserve the downloader's resumable incomplete files.
  Cancelling only the asyncio task was rejected because `to_thread`
  continues. Deleting the whole model cache was rejected because valid
  snapshots and another process may share it.
- **Blast radius**: Download API/status, cross-page toast, activation
  overlap, Hub cache locking, desktop shutdown, and partial resume.
- **Verification**: Throttle a download, cancel from another page, and
  close the desktop at several phases. All child/thread/network activity
  must stop within a fixed bound, status must become terminal, and the
  next request must resume without deleting valid cached blobs.
- **Blocks**: Additional downloadable models and packaged desktop
  reliability.

### Protocol correlation

### [PROTOCOL-01] Scope every response and error to its operation

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/backends/protocol.py:100`,
  `src/backends/protocol.py:102`,
  `src/backends/worker_base.py:470`,
  `src/backends/worker_base.py:642`,
  `src/backends/worker_base.py:678`,
  `src/backends/smollm3_worker.py:440`,
  `src/backends/smollm3_worker.py:445`,
  `src/backends/smollm3_worker.py:476`,
  `src/web/static/app.js:1564`,
  `src/web/static/app.js:1811`,
  `src/web/static/app.js:3033`,
  `src/web/static/app.js:5232`
- **What is true today**: Successful tokenize, prompt-count, and probe
  responses carry request IDs, but their errors and the global busy
  response carry neither request type nor ID. The frontend sends every
  `error` through one generation-terminal handler that can restore and
  exit the entire edit session. Core frame, done, and error payloads are
  also hand-built dictionaries rather than typed shared envelopes.
- **Why it matters**: A non-terminal probe rejected as busy or stale can
  tear down What If as if generation failed. New Mamba probes and signal
  requests will add more asynchronous operations whose failures cannot
  safely share one unscoped channel.
- **Direction**: Define dependency-light typed wire envelopes and
  builders carrying operation, request ID, run token, stable error code,
  and terminal scope. Route auxiliary failures to their local control;
  reserve connection/model-fatal errors for the session reducer. Parsing
  human error text was rejected as brittle, and Pydantic validation on
  every hot frame was rejected as unnecessary overhead.
- **Blast radius**: Protocol constants/types, worker dispatch and
  backends, frontend message reducer, busy behavior, and contract tests.
- **Verification**: Interleave tokenize, count, probe, generation, and
  busy responses across two sockets. Every result and error must affect
  only its originating operation, while an unscoped fatal error must
  still terminate the session exactly once.
- **Blocks**: Mamba state probes and further asynchronous XAI tools.

### Analytics correctness

### [ANALYTICS-01] Commit detail responses only to the run that requested them

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/static/analytics.js:514`,
  `src/web/static/analytics.js:1710`,
  `src/web/static/analytics.js:1783`,
  `src/web/static/analytics.js:1934`,
  `src/web/static/analytics.js:2551`,
  `src/web/static/analytics.js:2827`,
  `src/web/static/analytics.js:2839`
- **What is true today**: Opening a detail modal starts independent
  metrics and frame fetches. Their callbacks mutate global chart,
  overlay, and modal state without checking that `activeRunId` still
  matches their captured `runId`. Closing sets `activeRunId` to null but
  does not invalidate either request; opening another run starts another
  pair. Charts are destroyed only after a successful metrics response,
  so a new run displays the old run's charts while loading and retains
  them indefinitely on error; fetch wrappers do not centralize
  `response.ok` handling.
- **Why it matters**: Delayed responses can render run A's charts under
  run B's title, combine A's metrics with B's token overlay, or repopulate
  a modal after it was closed. Analytics can therefore present a
  plausible but false comparison, a direct trust failure.
- **Direction**: Assign each modal open a request epoch and use an
  `AbortController` for both fetches. Clear every prior surface
  synchronously, centralize HTTP-status/error rendering, and before every
  state commit require both the epoch and run ID to match. Where
  practical, fetch the pair together and publish one coherent detail
  snapshot. Disabling row clicks while loading was rejected because it
  does not cover closing, retries, or future background refreshes.
- **Blast radius**: Detail-modal loading, chart teardown, overlay
  rendering, error/empty states, and compare navigation.
- **Verification**: Deterministically delay A's metrics and frames on
  opposite sides of B's responses, and close during each phase. At every
  paint, title, metadata, charts, and overlays must share one run ID or
  be empty.
- **Blocks**: Trustworthy aggregate and cross-model comparison views.

### [ANALYTICS-02] Derive token charts from token counts, not text length

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/analytics/metrics.py:59`,
  `src/analytics/metrics.py:67`, `src/analytics/metrics.py:73`,
  `src/analytics/metrics.py:87`,
  `src/web/static/analytics.js:3701`,
  `src/web/static/analytics.js:3713`,
  `src/web/static/analytics.js:3943`,
  `src/web/static/analytics.js:3969`,
  `src/inference/dgemma_sampler.py:153`,
  `src/inference/dgemma_sampler.py:167`,
  `README.md:360`, `README.md:361`
- **What is true today**: Convergence divides mask-glyph count by the
  number of decoded characters in `history.txt`, although the chart is
  labeled as resolved tokens. Replacing one mask with a ten-character
  token therefore advances the curve about ten times as much as
  replacing it with a one-character token. The throughput numerator
  uses the first frame's mask count as one run-wide baseline; when
  DiffusionGemma starts a second canvas and its positions reset, the
  series drops back instead of carrying the committed canvas forward.
- **Why it matters**: Two outputs with the same token-resolution schedule
  can show different convergence solely because their decoded strings
  have different token lengths. Multi-canvas Tokens/s can undercount
  whole committed canvases. These are analytical claims, not cosmetic
  labels, and can reverse a user's model or intervention comparison.
- **Direction**: Persist compact per-frame counts from the protocol:
  canvas token count, unresolved count, newly revealed count, canvas
  index, and cumulative produced count. Derive both charts from those
  values. Modern runs can alternatively be repaired from token records;
  legacy character proxies should be labeled as such or omitted. Using
  `1 - mask_count / first_mask_count` was rejected because it remains
  wrong across canvases, remasks, and adaptive draft churn.
- **Blast radius**: Sampler frame summaries, save metadata, metrics API,
  convergence, Tokens/s, comparison charts, and legacy behavior.
- **Verification**: Use synthetic runs with equal token schedules but
  one- versus multi-character token text, remasks, and two canvases.
  Convergence must be text-invariant, cumulative production monotone,
  and the terminal produced count equal the actual output token count.
- **Blocks**: Credible aggregate statistics and cross-model speed
  comparisons.

### [ANALYTICS-03] Page lightweight run summaries instead of loading every bundle

- **Severity**: medium
- **Confidence**: medium
- **Evidence**: `src/web/server.py:1379`,
  `src/web/server.py:1392`, `src/web/server.py:1393`,
  `src/analytics/metrics.py:282`,
  `src/analytics/metrics.py:294`,
  `src/web/static/analytics.js:583`,
  `src/web/static/analytics.js:647`,
  `src/web/static/analytics.js:918`,
  `src/web/static/analytics.js:1028`,
  `src/web/static/analytics.js:1118`,
  `src/web/static/analytics.js:1517`,
  `src/web/static/analytics.js:1525`,
  `src/web/static/app.js:4021`,
  `src/web/static/app.js:4026`, `HANDOFF.md:2084`
- **What is true today**: Initial Analytics load scans every run
  directory, parses each complete metadata file, returns the full prompt,
  final text, params, and provenance for all runs, and builds every table
  row. The table displays only a 40-character prompt preview. The current
  handoff records 175 runs, and prompt import allows 200,000 characters,
  so valid data can make the list response tens of megabytes before
  token frames are fetched. Collection membership and present counts
  repeatedly scan arrays, with up to 24 collections, before full table
  rebuilds.
- **Why it matters**: Disk reads, response encoding, network transfer,
  browser JSON memory, sorting/grouping, and DOM rebuilds all grow with
  the entire archive. Dense collection membership adds nested scans;
  collection or star changes then rebuild the table. Aggregate analytics
  will add work on top of a list path already doing too much.
- **Direction**: Add a paged summary API containing only table fields and
  a bounded prompt preview, plus a detail-metadata endpoint for the open
  run. Index client membership as sets/maps. Make a rebuildable server
  index or cache only after measuring directory scan cost; the run
  folders remain authoritative. Moving immediately to a database as the
  sole source of truth was rejected because it expands migration and
  recovery work before pagination proves indexing is necessary.
- **Blast radius**: Run list API, table paging/sort/group, collections and
  selection, detail loading, compare, and future aggregate queries.
- **Verification**: Generate 175, 1,000, and 5,000 synthetic summaries
  with maximum-size prompts and 24 dense collections. Measure server
  latency/RSS, response bytes, browser heap, and render/input latency;
  opening a detail must fetch its full metadata lazily and preserve
  current grouping semantics.
- **Blocks**: Aggregate analytics and long-term archive growth.

### [ANALYTICS-04] Make comparison a bounded, coherent run-set transaction

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/server.py:1520`,
  `src/web/server.py:1521`, `src/web/server.py:1585`,
  `src/web/server.py:1587`, `src/web/server.py:1628`,
  `src/web/server.py:1630`, `src/web/server.py:1639`,
  `src/web/static/analytics.js:677`,
  `src/web/static/analytics.js:4785`,
  `src/web/static/analytics.js:4791`,
  `src/web/static/analytics.js:4803`,
  `src/web/static/analytics.js:4868`,
  `src/web/static/analytics.js:4929`
- **What is true today**: Compare accepts an unbounded comma-separated
  ID list. Its metrics path joins IDs directly to `RESULTS_DIR`, unlike
  the guarded frame/delete resolver, so an absolute or parent-relative
  query ID can escape the data root and read any accessible directory
  with the expected metadata/history files. The frontend has no request
  epoch, silently omits errored and autoregressive selections from the
  only comparison chart, and labels survivors with LLaDA-only parameter
  names.
- **Why it matters**: The panel can show fewer runs than the user
  selected without explaining why, repopulate after close or another
  comparison, and label DiffusionGemma values as undefined. The path
  escape is constrained by the expected filenames but still breaks the
  run-store trust boundary, especially under the current network bind.
- **Direction**: Accept a bounded structured compare request, resolve
  every ID through the run store, and return one explicit
  data/unavailable/error record per selection. Publish the response only
  under a matching selection epoch and compare capability intersections
  instead of assuming LLaDA fields. Blocking heterogeneous selections
  was rejected because cross-model comparison is an explicit roadmap
  goal.
- **Blast radius**: Compare API and UI, guarded path resolver,
  capability metadata, request cancellation, labels, and aggregate
  Analytics.
- **Verification**: Test absolute/parent traversal, too many and duplicate
  IDs, deleted/corrupt runs, mixed model families, and delayed A/B
  responses around close/reopen. Every selected run must be represented
  exactly once as data or an explained omission.
- **Blocks**: Cross-model and aggregate comparison.

### Runtime resource use

### [RUNTIME-01] Bound frame queues and compact append-only streams

- **Severity**: medium
- **Confidence**: medium
- **Evidence**: `src/inference/ar_sampler.py:377`,
  `src/inference/ar_sampler.py:393`,
  `src/inference/ar_sampler.py:423`,
  `src/inference/ar_sampler.py:1248`,
  `src/inference/dgemma_sampler.py:356`,
  `src/backends/registry.py:236`,
  `src/backends/registry.py:238`,
  `src/backends/registry.py:239`,
  `src/web/static/app.js:1626`,
  `src/web/static/app.js:7370`
- **What is true today**: Every autoregressive token rebuilds and sends
  the full sequence, and the client retains every full token snapshot
  for scrubbing, saving, and session storage. The producer queues for AR
  and DiffusionGemma have no maximum size. For an N-token AR run this is
  N(N+1)/2 token records: 32,896 at the recommended 256-token ceiling,
  but 2,098,176 at the allowed experimental ceiling of 2,048, before
  repeated text, object, and JSON overhead.
- **Why it matters**: A fast producer or slow browser/network can move
  that quadratic payload into an unbounded worker queue, while the
  browser also stores and later stringifies it. The current recommended
  cap masks the slope; Mamba's long causal decode and future richer
  signals will expose it.
- **Direction**: First cap producer queues and define disconnect-safe
  backpressure. Then add an append frame variant for monotonic AR/SSM
  streams and reconstruct views client-side, using periodic checkpoints
  if random scrubbing needs bounded seek time. Keep full snapshots for
  diffusion, where prior positions genuinely change. Merely lowering
  the experimental cap was rejected because it preserves the
  architectural ceiling for every future append-only model.
- **Blast radius**: AR and SSM frame protocol, worker queues, live span
  sync, scrubber history, session snapshots, save format, and Analytics.
- **Verification**: Measure wire bytes, queue depth, worker RSS, browser
  heap, frame time, and save size at 128, 256, 1,024, and 2,048 tokens
  under an intentionally throttled client. Growth should be linear for
  append-only streams and queues must stay at their configured bound.
- **Blocks**: Mamba-3 scaling and any longer-context autoregressive model.

### [RUNTIME-02] Bound GIF rendering and decouple it from core saves

- **Severity**: medium
- **Confidence**: medium
- **Evidence**: `src/inference/render_gif.py:9`,
  `src/inference/render_gif.py:14`,
  `src/inference/render_gif.py:26`,
  `src/inference/render_gif.py:29`,
  `src/inference/render_gif.py:74`,
  `src/inference/render_gif.py:76`,
  `src/inference/render_gif.py:58`,
  `src/web/server.py:1485`,
  `src/backends/registry.py:50`,
  `src/backends/registry.py:239`
- **What is true today**: Every save renders one 900 by 700 RGB image
  per history frame and retains all images in a Python list before
  encoding the GIF. Raw pixels alone cost about 1.89 MB per frame:
  roughly 242 MB at 128 frames and 3.87 GB at the allowed 2,048-frame
  experimental extreme. GIF creation is inside the save's success path,
  and every GIF is labeled "LLaDA RESPONSE (Diffusion)" regardless of
  model.
- **Why it matters**: A completed long run can fail or kill the
  supervisor while being saved, after inference cost has already been
  paid. The derived visualization can prevent the authoritative text and
  token data from being acknowledged, and its label misidentifies
  DiffusionGemma, autoregressive, and future SSM output.
- **Direction**: Treat the GIF as a bounded, optional derivative after
  the core run is atomically published. Stream frames into the encoder
  where supported, and enforce a documented frame/pixel budget with
  temporal sampling for long runs. Pass a model-neutral or actual model
  label. Merely catching `MemoryError` was rejected because normal
  allocation pressure can destabilize the process before Python raises
  it cleanly.
- **Blast radius**: Save status and result schema, GIF generation, run
  folder completeness rules, Analytics artifact links, and memory tests.
- **Verification**: Save 128, 1,024, and 2,048-frame synthetic histories
  while measuring peak RSS. Core data must remain saved if GIF creation
  fails, peak memory must respect a fixed budget, and every model's label
  must be accurate.
- **Blocks**: Safe long SSM/AR runs and richer multimodal artifacts.

### [RUNTIME-03] Give custom selects keyboard and listener lifecycles

- **Severity**: medium
- **Confidence**: medium
- **Evidence**: `src/web/static/custom_select.js:103`,
  `src/web/static/custom_select.js:107`,
  `src/web/static/custom_select.js:206`,
  `src/web/static/custom_select.js:218`,
  `src/web/static/custom_select.js:249`,
  `src/web/static/app.js:154`,
  `src/web/static/app.js:155`,
  `src/web/static/analytics.js:2827`,
  `src/web/static/analytics.js:3036`,
  `src/web/static/analytics.js:3037`
- **What is true today**: Every `createCustomSelect` installs an
  anonymous document-level click listener and exposes no destroy method.
  Removing the widget leaves that listener and its closed-over DOM tree
  reachable. The control declares `role="listbox"`, but options have no
  option/selected semantics or focus path; keyboard handling only opens
  and closes, while selection requires a pointer. The generator
  explicitly avoids rebuilding its picker to limit the leak, while
  Analytics recreates it for each run detail.
- **Why it matters**: Core parameter, overlay, and Settings choices are
  mouse-only for keyboard users. A long Analytics session also
  accumulates one detached control and document callback per opened run,
  making every later click invoke all of them. The existing workaround
  spreads a hidden lifecycle rule to callers.
- **Direction**: Use one delegated outside-click listener for all open
  selects, or return an idempotent `destroy()` backed by an
  `AbortController` and call it before replacement. Add active-option
  focus, Arrow/Home/End traversal, selection, expanded state, and option
  semantics in the same shared widget. The delegated approach is cheaper
  for listener ownership. Adding arrow handlers alone was rejected
  because it leaves false accessibility semantics and detached
  listeners.
- **Blast radius**: Shared select widget, generator/model/parameter
  pickers, Analytics detail picker, Settings, and frontend tests.
- **Verification**: Keyboard-only tests must open, traverse, select,
  escape, and expose correct state. Create and remove 1,000 controls,
  then prove document listener count and retained detached nodes return
  to baseline and one outside click performs constant work.
- **Blocks**: Nothing directly; it is a bounded long-session cleanup.

### Intervention fidelity

### [XAI-01] Retain complete checkpoints for reproducible interventions

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/inference/streaming_sampler.py:335`,
  `src/inference/streaming_sampler.py:405`,
  `src/inference/streaming_sampler.py:514`,
  `src/inference/streaming_sampler.py:517`,
  `src/backends/llada_worker.py:277`,
  `src/backends/llada_worker.py:299`,
  `src/backends/llada_worker.py:303`,
  `src/backends/llada_worker.py:379`,
  `src/backends/dgemma_worker.py:194`,
  `src/inference/dgemma_sampler.py:420`
- **What is true today**: LLaDA retains token-ID tensors per frame but
  not the reveal-confidence state or frame RNG state. Resume assigns
  confidence `1.0` to every surviving token, and the retained run state
  omits the original seed; the resume path does not restore a selected
  frame's random generators. DiffusionGemma checkpoints similarly retain
  IDs and canvas index while recreating its resume randomness from the
  run seed rather than a frame checkpoint.
- **Why it matters**: An edited LLaDA branch's Heatmap and mean
  confidence contain invented certainty for unchanged tokens. Repeating
  a stochastic branch can also depend on random work performed after the
  selected frame or between attempts, so saved intervention results are
  not fully reproducible from the claimed frame and seed.
- **Direction**: Define a bounded intervention checkpoint containing
  token IDs, reveal/stability confidence state, relevant RNG state,
  canvas provenance, and captured signal state. Keep display history
  compact and retain richer checkpoints only where intervention is
  supported. Reseeding from the original run seed or retaining
  confidence `1.0` was rejected because neither reconstructs the chosen
  frame.
- **Blast radius**: LLaDA and DiffusionGemma retained state, resume
  protocol, Heatmap/mean confidence, run provenance, and checkpoint
  memory budgets.
- **Verification**: Resume the same seeded frame twice after intervening
  random work and compare outputs frame by frame. Unchanged positions
  must preserve their original confidence, and checkpoint storage must
  remain under an explicit bound.
- **Blocks**: Trustworthy edited-run XAI, diffusion trajectories, and
  robust multi-canvas resume.

### Roadmap architecture

### [ROADMAP-01] Separate model family, generation shape, and device support

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/backends/protocol.py:61`,
  `src/backends/protocol.py:68`,
  `src/web/static/menu.js:196`,
  `src/web/static/menu.js:239`,
  `src/web/static/menu.js:282`,
  `src/web/static/menu.js:409`,
  `src/web/server.py:851`, `src/web/server.py:864`,
  `src/web/server.py:933`,
  `src/backends/llada_worker.py:77`,
  `src/web/static/app.js:473`,
  `src/web/static/app.js:603`,
  `src/web/static/app.js:4817`,
  `src/web/static/analytics.js:2532`,
  `ROADMAP.md:823`, `ROADMAP.md:827`, `ROADMAP.md:829`
- **What is true today**: `model_type` has only `diffusion` and
  `autoregressive`, while the frontend uses that one value for family
  icon and label, append-versus-denoise UI, Analytics chart gating, and
  CPU capability. The accepted Mamba-3 direction is a distinct SSM,
  generates append-only like AR, and is explicitly GPU-only. Today an
  unreadable or absent GPU produces no headroom value, which the server
  treats as "fits"; a diffusion row can then be selected and LLaDA
  silently falls back to CPU. Free RAM is displayed but never enforced.
- **Why it matters**: Registering Mamba as `autoregressive` incorrectly
  offers CPU fallback and erases its family. Adding `state_space` makes
  existing exact comparisons treat it as diffusion and expose
  denoising-only behavior. Current GPU-only models can already attempt an
  unintended, potentially host-OOMing CPU load. Special-casing Mamba's ID
  would make every later model repeat the same fight.
- **Direction**: Migrate to orthogonal data: a display/architecture
  family, a generation shape such as `append_only` versus
  `iterative_canvas`, explicit supported/default devices with per-device
  memory requirements, named intervention/signal capabilities, and an
  explicit frame coordinate/resume scope instead of inferring
  single-canvas support from observed frames. Enforce support in the
  supervisor, not only the menu. Preserve
  `model_type` when reading legacy run metadata. A one-off `mamba`
  condition was rejected because it converts the next model into
  permanent branching debt.
- **Blast radius**: Registry and protocol, menu glyph/device controls,
  generator gating, Analytics chart selection, saved metadata, and docs.
- **Verification**: Schema-driven tests should cover combinations that
  do not exist today: GPU-only append-only, CPU-capable append-only,
  diffusion with and without resume, and a new family with a unique
  signal. Include global/canvas-local frame coordinates and resume scope;
  no UI decision may inspect a model ID.
- **Blocks**: Mamba-3 directly, multi-canvas DiffusionGemma resume, plus
  Phi/Gemma AR additions and any model with nonstandard device support.

### [ROADMAP-02] Make the registry authoritative for parameter validation

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `src/backends/registry.py:54`,
  `src/backends/registry.py:57`, `src/backends/registry.py:64`,
  `src/backends/registry.py:67`,
  `src/backends/llada_worker.py:45`,
  `src/backends/llada_worker.py:150`,
  `src/backends/llada_worker.py:171`,
  `src/backends/llada_worker.py:174`,
  `src/backends/dgemma_worker.py:37`,
  `src/backends/smollm3_worker.py:155`,
  `src/backends/smollm3_worker.py:159`,
  `src/backends/worker_base.py:187`,
  `src/backends/worker_base.py:207`,
  `src/backends/worker_base.py:210`,
  `src/web/static/app.js:4309`,
  `src/web/static/app.js:4317`
- **What is true today**: The registry drives controls, defaults, and
  bounds, but each worker separately coerces and clamps request fields.
  LLaDA's omitted-field defaults already disagree: the registry says
  generation and block lengths are 160, while the worker falls back to
  128 and 32. SmolLM3 recently added registry-backed default lookup
  specifically after this duplication caused a drift risk; the other
  workers still duplicate it. WebSocket requests have no discriminated
  input models, the measured context ceiling produces only a frontend
  warning, and the maximum 200,000-character prompt count tokenizes
  synchronously on the worker event loop.
- **Why it matters**: The browser happens to send every field, masking
  the disagreement. API clients, a partially upgraded frontend, or the
  next model can receive behavior different from the schema the app
  displays. Each parameter addition currently requires synchronized
  edits in at least registry, frontend assumptions, and a bespoke worker
  parser. Direct or stale clients can send malformed or over-context
  work into model code, and a maximum-size count can delay frame/control
  delivery; the latter magnitude is unmeasured.
- **Direction**: Add one dependency-light validator that resolves
  device overrides, defaults, primitive types, options, and
  recommended/experimental bounds from `ParamSpec`, reached through
  discriminated request models at the worker boundary. Enforce the
  loaded model's context budget before inference and offload bounded
  prompt tokenization. Keep
  model-specific relational checks, such as LLaDA divisibility, in the
  worker. Copying SmolLM3's lookup into the other two workers was
  rejected because it preserves three coercion implementations.
- **Blast radius**: Protocol types, registry invariants, all workers,
  parameter UI error reporting, and request tests.
- **Verification**: For every registered model and device, omit each
  field in turn and prove the worker result equals the advertised
  default; test boundary, invalid type, invalid option, experimental,
  context overflow, malformed request IDs, and relational-error cases
  from one generated matrix. Keep an event-loop heartbeat alive during
  the maximum accepted prompt count.
- **Blocks**: Low-friction Mamba parameters and any dynamic registry.

### [ROADMAP-03] Describe XAI signals by their axes and capture policy

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/web/server.py:1136`,
  `src/web/server.py:1152`, `src/web/server.py:1194`,
  `src/web/static/app.js:289`,
  `src/web/static/app.js:1634`,
  `src/web/static/analytics.js:2651`,
  `src/web/static/analytics.js:4404`,
  `src/backends/registry.py:193`,
  `src/inference/dgemma_sampler.py:94`,
  `src/inference/dgemma_sampler.py:107`,
  `src/inference/dgemma_sampler.py:112`,
  `HANDOFF.md:3173`, `HANDOFF.md:3182`,
  `HANDOFF.md:3188`
- **What is true today**: Signal shape is implicit in storage location.
  Confidence and entropy are scalar fields on each token record, while
  alternatives are one position-indexed list assumed immutable after
  sampling. Analytics detects entropy on the final frame and turns that
  into one value per position. Mamba's state norms and write intensity
  add new axes, while diffusion entropy and candidates vary by both
  position and denoising frame. DiffusionGemma's current
  `entropy_signal` actually materializes a full float32 softmax and emits
  argmax confidence, not entropy; for 256 positions by roughly 262K
  vocabulary entries, that probability tensor alone is about 256 MiB.
- **Why it matters**: Adding another short token key works only for
  another scalar with existing semantics. Reusing the AR alternatives
  file for diffusion would discard history; repeating bespoke sidecars
  and gates for each SSM signal would couple capture, storage, and views
  again. Reusing the existing toggle would also conflate confidence with
  entropy while paying a large temporary-memory cost. The next two
  accepted roadmap items both hit this boundary.
- **Direction**: Add a versioned signal manifest that names each channel,
  unit, axes such as frame/position/canvas, capture mode, and data
  location. Specify capture budgets and use numerically stable reductions
  over logits for max probability, entropy, and optional top-k without
  retaining a full probability tensor longer than required. Preserve hot
  scalar fields for compatibility, but normalize new captures through
  that description and let views declare which shapes they support. A
  generic untyped `signals` dictionary on every token was rejected
  because it moves ambiguity instead of modeling it.
- **Blast radius**: Frame/done contract, save version, capture toggles,
  token sidecars, overlay availability, Analytics charts, and exports.
- **Verification**: Fixtures must cover one scalar per position, a
  position-by-frame trajectory, one value per frame, and an absent
  opt-in channel. Hand-computable logits must distinguish entropy from
  max confidence; hardware must measure peak VRAM and latency.
  Unsupported shapes must produce an explicit unavailable view without
  losing the underlying data.
- **Blocks**: Mamba-3 phase-2 overlays and diffusion entropy/top-k.

### [ROADMAP-04] Give multimodal inputs an artifact lifecycle outside WebSocket JSON

- **Severity**: medium
- **Confidence**: medium
- **Evidence**: `src/backends/dgemma_worker.py:111`,
  `src/backends/dgemma_worker.py:115`,
  `src/backends/dgemma_worker.py:163`,
  `src/web/server.py:1180`, `src/web/server.py:1182`,
  `src/web/static/app.js:6211`,
  `scripts/quantize_diffusiongemma_nf4.py:32`,
  `scripts/quantize_diffusiongemma_nf4.py:103`,
  `scripts/spike_diffusiongemma.py:122`,
  `ROADMAP.md:995`, `ROADMAP.md:1002`,
  `ROADMAP.md:1010`
- **What is true today**: Generation and saving have a text prompt but no
  concept of an input artifact. The shared WebSocket message is JSON,
  DiffusionGemma validates a string prompt, and saved runs cannot retain
  an image, digest, preprocessing identity, or source dimensions. Phase
  3 currently says to extend the generate message and add upload UI but
  does not establish ownership or cleanup. The quantizer appears to save
  the complete state and copy processor configuration, but the current
  spike/worker use only text and the retained vision path and VRAM budget
  have not been demonstrated.
- **Why it matters**: Putting base64 image bytes directly in generation
  JSON would duplicate a large object through browser, supervisor, and
  worker, interact badly with reconnect/switch, and leave no durable
  identity for the saved result. Passing a client filesystem path would
  be unsafe and unavailable to remote browser clients.
- **Direction**: First run a bounded hardware spike that inventories the
  NF4 artifact, loads `AutoProcessor`, performs one image-conditioned
  generation, measures host/VRAM peaks, and settles persistence or
  hashing. Only then add a bounded multipart upload that validates type,
  decoded dimensions, and size, stores a content-addressed temporary
  artifact, and gives generation an opaque reference. On a successful
  transactional save, copy or link the input plus processor/preprocessing
  provenance into the run; expire unclaimed artifacts. Base64 in the
  WebSocket was rejected except as a small prototype because it has no
  lifecycle or backpressure boundary.
- **Blast radius**: Frontend input state, supervisor upload endpoints,
  worker request contract, DiffusionGemma processor path, run store,
  cleanup, and provenance.
- **Verification**: Test malformed files, deceptive MIME/extension,
  dimension and byte limits, abandoned uploads, model switches,
  reconnect, save failure, and text-only parity. Hardware must confirm
  the NF4 artifact accepts image inputs and peak host/VRAM use with the
  vision tower before upload UI implementation.
- **Blocks**: Phase 3 multimodal image input.

### [ROADMAP-05] Extract model-specific text semantics before reusing the AR loop

- **Severity**: high
- **Confidence**: high
- **Evidence**: `src/backends/worker_base.py:228`,
  `src/backends/worker_base.py:252`,
  `src/backends/worker_base.py:259`,
  `src/inference/ar_sampler.py:51`,
  `src/inference/ar_sampler.py:78`,
  `src/inference/ar_sampler.py:84`,
  `src/inference/ar_sampler.py:107`,
  `src/inference/ar_sampler.py:122`,
  `src/inference/ar_sampler.py:138`,
  `ROADMAP.md:823`, `ROADMAP.md:826`,
  `ROADMAP.md:829`, `HANDOFF.md:3164`,
  `HANDOFF.md:3169`
- **What is true today**: The reusable AR sampler contains SmolLM-specific
  chat templating, ChatML turn termination, control-token stripping, and
  `<think>` channel parsing. The base prompt counter makes the same
  `apply_chat_template(... enable_thinking=...)` assumption. Proposed
  Mamba checkpoints are base completion models and may have no chat
  template; future Phi and Gemma models have different templates and
  control tokens.
- **Why it matters**: Reusing the numeric loop can silently change prompt
  semantics, stop on the wrong token, fail during the automatic prompt
  count, or misclassify reasoning and answer text. Labeling a base model
  as if it were instruction-tuned would make the UI easy to run and hard
  to interpret.
- **Direction**: Extract a typed per-model text adapter owning input mode
  (`chat` or raw completion), input construction, prompt counting, stop
  IDs, display sanitization, and output-channel splitting. Keep sampling
  math and cache mechanics shared. Model-ID conditions inside
  `ar_sampler.py` were rejected because every new tokenizer would extend
  a central branch.
- **Blast radius**: Registry capabilities, worker base, AR/SSM workers,
  prompt UI copy, done payload/provenance, and tokenizer tests.
- **Verification**: Test a tokenizer with no chat template, at least two
  different chat templates and turn terminators, thinking on/off, and
  raw completion. The saved prompt count must exactly match the input
  used by generation in every case.
- **Blocks**: Mamba-3 directly, then Phi and Gemma AR additions.

### Quality gates and tests

### [QUALITY-01] Put automated tests around lifecycle and browser contracts

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `AUDIT_BRIEF.md:133`,
  `AUDIT_BRIEF.md:239`, `src/web/server.py:407`,
  `src/web/server.py:953`, `src/web/server.py:1063`,
  `src/backends/worker_base.py:532`,
  `tests/backends/test_load_status.py:1`,
  `tests/web/test_ui_state.py:1`,
  `tests/inference/test_ar_sampler.py:1`
- **What is true today**: The 265-test Python suite passes and has strong
  focused coverage for AR numeric behavior, load progress, tokenizer
  identity, saved signals, and UI-state helpers. It does not exercise
  `ModelManager`, activation/eviction, process escalation, the
  supervisor proxy, multiple worker WebSockets, save publication, or
  JavaScript behavior. The 24,900-line frontend's automated gate is
  syntax checking.
- **Why it matters**: The highest-consequence findings in this report
  live exactly in the untested seams: cross-window ownership,
  disconnect, stale async responses, and file publication. Refactoring
  those seams before Mamba without deterministic tests trades visible
  debt for invisible regressions.
- **Direction**: Add fake-process and fake-health unit tests for
  `ModelManager`, in-process WebSocket contract tests for the worker and
  proxy, failure-injected run-store tests, and pure Node tests for
  frontend reducers/API clients. Add only a thin browser integration
  layer for real DOM and navigation behavior. More GPU golden tests were
  rejected as the first priority because these control-plane failures
  are reproducible without a model.
- **Blast radius**: Test layout and fixtures, extracted seams, CI time,
  and dependency policy for the optional browser runner.
- **Verification**: Each LIFE, DATA, and ANALYTICS high-severity finding
  must have a deterministic regression test, and the core suite must
  remain GPU-free and complete in bounded time.
- **Blocks**: Safely implementing nearly every high-leverage change in
  the sequencing section.

### [QUALITY-02] Turn the lint baseline into a ratchet, then burn it down

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `pyproject.toml:13`, `pyproject.toml:18`,
  `pyproject.toml:29`, `pyproject.toml:32`,
  `README.md:454`, `HANDOFF.md:79`,
  `HANDOFF.md:198`, `AGENTS.md:64`
- **What is true today**: Ruff is configured for the repository's stated
  complexity and nesting limits, but `ruff check src tests` exits with
  156 findings: 129 line-length, 8 nesting, 3 complexity, and 16 other
  findings. The documented original baseline is 159 while current
  session notes say 156. There is no checked baseline artifact or
  zero-exit gate, so a new violation can hide behind an unrelated
  removal.
- **Why it matters**: The three complexity failures include
  `create_worker_app`, `_save_run_blocking`, and the dormant LLaDA loop,
  directly overlapping risky audit findings. A remembered count does not
  tell a future agent whether its change made a specific file safer.
- **Direction**: First remove the non-style findings and the dormant
  reference noise. Until zero is practical, store a path/code-specific
  baseline and fail on additions; require clean Ruff on changed files.
  Then reduce E501 in touched modules rather than mass-reflowing the
  repository. Disabling configured rules was rejected because the
  complexity signals are already identifying real seams.
- **Blast radius**: Verification scripts or CI, contributor workflow,
  touched-file policy, and a finite set of cleanup commits.
- **Verification**: The gate must exit zero on the accepted baseline,
  fail when one synthetic violation is added, and never pass because a
  different finding disappeared.
- **Blocks**: A dependable quality signal during the audit remediation.

### Documentation, dependencies, and agent routing

### [META-01] Reduce HANDOFF to current decisions and move the verification ledger

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `AGENTS.md:3`, `AGENTS.md:4`,
  `HANDOFF.md:74`, `HANDOFF.md:2031`,
  `HANDOFF.md:2061`, `HANDOFF.md:2188`,
  `HANDOFF.md:2202`, `HANDOFF.md:3114`,
  `HANDOFF.md:3214`, `src/backends/registry.py:312`,
  `src/backends/registry.py:323`
- **What is true today**: `AGENTS.md` routes every cold session through
  `HANDOFF.md`, which is 3,223 lines. Almost two thousand lines are
  session shipment narrative, followed by a 132-item manual validation
  ledger and two different backlog entries numbered zero. The ledger
  still calls Alternatives off "the default", while the registry now
  defaults it on. The audit needed a special brief telling agents to
  read only the first 73 lines.
- **Why it matters**: Every future session pays context and attention for
  historical implementation detail before reaching current work. Stale
  "this session" language and obsolete test premises can make a correct
  hardware result look like a regression. Partially shipped backlog
  entries also make it easy to revive completed work or miss the one
  unvalidated item that matters.
- **Direction**: Keep HANDOFF as a bounded cold-start page: orientation,
  current validated state, at most a few next candidates, settled
  decisions, and links. Move reusable hardware checks into a dedicated
  manual-verification document and rely on git history or an archived
  changelog for shipment narrative. Merely adding a table of contents
  was rejected because it improves navigation but not context cost or
  stale duplication.
- **Blast radius**: Agent startup instructions, session-end habit,
  manual QA workflow, and references from README/ROADMAP.
- **Verification**: A cold agent should identify architecture,
  constraints, unvalidated hardware work, and the next decision by
  reading under roughly 200 lines, with no duplicate item numbers or
  "this session" history; every manual scenario's defaults must match
  the registry.
- **Blocks**: Efficient remediation sessions after this report.

### [META-02] Keep the canonical agent contract in tracked files

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `.gitignore:1`, `AGENTS.md:51`,
  `AGENTS.md:17`, `AGENTS.md:19`, `AGENTS.md:28`,
  `AGENTS.md:53`, `AGENTS.md:55`,
  `.cursor/rules/python-venv.mdc:6`,
  `.cursor/rules/python-venv.mdc:7`,
  `.cursor/rules/python-venv.mdc:9`,
  `ROADMAP.md:1134`, `ROADMAP.md:1137`
- **What is true today**: Tracked guidance tells agents to follow
  `.cursor/rules/` and points to `.cursor/plans/`, while `.cursor` is
  ignored. The detailed rules and historical plans therefore depend on
  one maintainer's local checkout and do not travel with a clone. The
  local Python rule also says every Python command must use `.venv`,
  while tracked guidance requires `.venv-dgemma` and `.venv-ar` for
  their incompatible workers.
- **Why it matters**: A contributor or fresh agent can obey every tracked
  instruction and still miss mandatory environment, model, and coding
  constraints, while a current Cursor session can follow the local rule
  and run a model command against the wrong Transformers environment.
  References to absent plans also look like missing project evidence
  instead of optional local history.
- **Direction**: Put canonical durable rules in tracked `AGENTS.md` or
  linked tracked documents. Keep local Cursor rules as thin adapters to
  those sources, explicitly routing each model command to its
  environment, and either track decision records worth preserving or
  remove the plans reference. Tracking all transient editor state was
  rejected; only the contract and durable rationale need portability.
- **Blast radius**: Contributor onboarding, agent configuration, plan
  references, and duplication between local and repository rules.
- **Verification**: Clone into a clean directory with no local Cursor
  state; every mandatory command, environment boundary, model
  constraint, and workflow rule must still be discoverable from tracked
  files with no dead links. In a configured checkout, no local rule may
  contradict that interpreter matrix.
- **Blocks**: Consistent execution of every later plan.

### [META-03] Rebuild documentation around one current inventory

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `ROADMAP.md:20`, `ROADMAP.md:22`,
  `ROADMAP.md:1096`, `ROADMAP.md:1099`,
  `ROADMAP.md:1106`, `README.md:207`,
  `README.md:241`, `README.md:393`,
  `README.md:435`, `README.md:454`
- **What is true today**: The roadmap's current status still opens with
  "both models", and its quick map omits SmolLM3, the AR environment and
  sampler, several frontend modules, settings, and analytics CSS. README
  both requires a CUDA GPU and explains that a GPU-less host can run
  SmolLM3; its Implementation Status both places Highlight tokens in
  Settings and later says it moved to the Overlay drawer; its Ruff count
  is stale.
- **Why it matters**: These are routing and setup claims, not harmless
  prose. A new user can stop at a false GPU requirement, while an agent
  can inspect the wrong files or preserve a UI location that no longer
  exists.
- **Direction**: Define ownership: README for current user-visible truth
  and setup, ROADMAP for future decisions and dependencies, HANDOFF for
  immediate state. Generate or mechanically check the file/environment
  inventory and a small set of duplicated claims. Adding another shipped
  bullet to every document was rejected because that process caused the
  drift.
- **Blast radius**: README status, roadmap orientation/quick map,
  HANDOFF conventions, Help/About synchronization, and documentation
  checks.
- **Verification**: A path/link checker plus assertions for model count,
  environment files, device support, and selected UI ownership claims
  must pass; a manual README walkthrough must work on both GPU and
  CPU-only setup paths.
- **Blocks**: Clear planning for Mamba and future contributors.

### [DEPS-01] Consolidate environment intent before adding `.venv-ssm`

- **Severity**: medium
- **Confidence**: high
- **Evidence**: `AGENTS.md:17`, `AGENTS.md:28`,
  `pyproject.toml:1`, `pyproject.toml:15`,
  `requirements-dgemma.txt:9`,
  `requirements-dgemma.txt:29`,
  `requirements-dgemma.txt:35`,
  `requirements-dgemma.txt:50`,
  `requirements-desktop.txt:13`, `README.md:207`,
  `ROADMAP.md:1115`, `ROADMAP.md:1122`,
  `ROADMAP.md:1123`, `ROADMAP.md:1127`,
  `HANDOFF.md:3167`
- **What is true today**: Four requirements files and three virtual
  environments encode mutually incompatible model stacks. The roadmap
  explicitly named a new incompatible environment as a consolidation
  trigger, and that trigger fired with `.venv-ar`; Mamba now proposes
  `.venv-ssm`. `pyproject.toml` remains tool configuration only, while
  the model registry separately hardcodes each interpreter path. The
  DiffusionGemma freeze contains both CUDA 12 and CUDA 13 package
  families without recorded resolver inputs, and advertised Python
  support starts at 3.10 while Ruff targets 3.11.
- **Why it matters**: A fifth environment adds another place to update
  setup, rules, registry, docs, install commands, and transitive freezes.
  The current files are reproducible, but there is no single
  authoritative map explaining which direct requirements produce each
  lock, interpreter, wheel index, or mixed CUDA resolution.
- **Direction**: Record direct dependency groups and environment
  metadata in one tracked project manifest, then generate or lock each
  incompatible environment separately with hashes, Python version, and
  wheel-index metadata. Regenerate DiffusionGemma from a clean
  environment and document whether both CUDA families are required.
  Preserve exact per-environment resolutions and CUDA source choices; do
  not force incompatible extras into one install. Keeping the flat
  freezes as generated lock outputs is a cheaper reversible migration
  than replacing all install tooling at once.
- **Blast radius**: Setup, registry launch paths, four requirements
  files, future SSM native builds, CI/cache keys, and dependency update
  procedure.
- **Verification**: Build every environment from a clean machine using
  only tracked manifests, reproduce pinned versions, and prove changing
  one group's direct dependency cannot silently alter another group's
  lock. Each environment must pass `pip check` and report its disk
  footprint and intended Torch/CUDA build.
- **Blocks**: Mamba-3's `.venv-ssm` and scalable addition of more model
  families.

## Sequencing

The dependency order below is intentionally not a single mega-plan. Each
number is a boundary to validate before the next one takes a dependency on
it.

1. **Land isolated safety fixes first.**
   - The best first commit is `LIFE-07`: add failure-injected LLaDA resume
     tests, stage candidate history, and prove a failed resume leaves the
     original state untouched. It is high severity, small, and changes no
     protocol or architecture.
   - Follow as separate commits with loopback defaulting (`TRUST-01`),
     synchronous detail clearing plus request epochs (`ANALYTICS-01`), and
     an absolute/configurable data root (`DATA-03`).
   - Vendor Analytics dependencies (`TRUST-02`) independently; it should not
     wait for frontend modularization.

2. **Establish gates before moving boundaries.**
   - Add the lifecycle/run-store test fixtures from `QUALITY-01` as each seam
     is extracted; do not create a large test-only prelude with no production
     owner.
   - Install the path/code-specific Ruff ratchet (`QUALITY-02`) now, then
     burn down findings only in files touched by later work.
   - Shorten and make the agent contract portable (`META-01`, `META-02`) so
     every remediation session starts from the same rules. `META-03` should
     be refreshed at milestone boundaries rather than after every internal
     commit.

3. **Build the run-store boundary.**
   - Extract behavior-preserving storage operations (`ORG-01`) after
     `DATA-03` fixes the root.
   - Add unique staged publication, complete replacement, revisions, and
     compare-and-swap (`DATA-01`).
   - Add versioned validation/read adapters (`DATA-05`), then thread immutable
     worker provenance through the terminal run contract (`DATA-04`).
   - Make GIFs bounded non-authoritative derivatives only after publication
     semantics exist (`RUNTIME-02`).
   - This sequence unlocks exact token summaries (`ANALYTICS-02`), lightweight
     pagination (`ANALYTICS-03`), the guarded compare boundary
     (`ANALYTICS-04`), and later multimodal artifacts (`ROADMAP-04`).

4. **Make process and socket ownership explicit.**
   - Extract/test the manager process adapter, then make termination and
     pre-eviction validation reliable (`LIFE-02`, `LIFE-06`).
   - Share activation orchestration first (`ORG-04`), then add
     activation/resident epochs and run ownership
     (`LIFE-03`, `LIFE-01`) together with operation-scoped envelopes
     (`PROTOCOL-01`).
   - Once run ownership is explicit, preserve complete intervention
     checkpoints (`XAI-01`) rather than only token IDs.
   - Propagate disconnect/cancel through inference and bounded queues
     (`LIFE-04`); the first bounded-queue step of `RUNTIME-01` belongs here.
   - Add host-level ownership after one manager's lifecycle is trustworthy
     (`LIFE-05`). Move downloads into the same owned-operation model
     (`TRUST-04`).
   - `DATA-02` can proceed in parallel once its semantic collection API or
     revision policy is settled; it should not reuse model-operation state.

5. **Extract frontend state around the settled protocol.**
   - Move aligned frame operations and legal workflow phases into the tested
     native-module core (`ORG-02`), then move model/download API clients.
   - Fix the select lifecycle (`RUNTIME-03`) while shared controls gain
     explicit module ownership.
   - Compact append-only streams only after the reducer can reconstruct them
     and the run-store version can distinguish them (`RUNTIME-01`).

6. **Prepare existing models for Mamba before adding Mamba.**
   - Split family, stream shape, device support, and resource requirements
     (`ROADMAP-01`); extract model-specific text adapters
     (`ROADMAP-05`); and centralize registry-driven parameter validation
     (`ROADMAP-02`). Migrate and test the three existing models first.
   - Pin/attest artifacts (`TRUST-03`) and consolidate environment intent
     (`DEPS-01`) before creating `.venv-ssm`.
   - Add the Mamba baseline only after those commits are validated. Add the
     axis-aware signal manifest (`ROADMAP-03`) before its native XAI phase and
     before diffusion entropy/top-k, not necessarily before the baseline
     decode.
   - Consolidate the LLaDA sampling kernel (`ORG-03`) before adding
     diffusion entropy/top-k to it.

7. **Take deferred cleanup last.**
   - Continue the Ruff burn-down and remaining documentation cleanup
     without mixing formatting churn into behavioral reviews.

Changes that should **not** be combined:

- Do not add Mamba while refactoring process ownership or dependency tooling.
- Do not combine run-directory relocation, format versioning, and existing
  data migration in one irreversible step; land the explicit root first and
  keep readers backward compatible.
- Do not combine native-module conversion with new XAI behavior. Extract and
  test current semantics, then add features.
- Do not introduce append/delta frames at the same time as diffusion signal
  trajectories. One changes transport shape; the other changes analytical
  axes.
- Do not mix mass lint/format cleanup with lifecycle, persistence, or sampler
  changes.

## Measurements to take on hardware

These are measurements, not substitutes for the deterministic tests named in
the findings.

1. **Disconnect and cancellation latency**
   - For LLaDA, DiffusionGemma, SmolLM3 GPU, and SmolLM3 CPU, disconnect near
     start/middle/end and record time until GPU/CPU utilization stops,
     generation lock release, process RSS stabilization, and UI recovery.
   - Repeat DiffusionGemma guided "Run to Here"; measure requested frames
     against frames actually computed and silently drained.

2. **Streaming resource curve**
   - At 128, 256, 1,024, and 2,048 append-only tokens, capture serialized wire
     bytes, largest frame, producer queue depth, worker RSS/VRAM, browser heap,
     main-thread long tasks, and live-render frame time.
   - Repeat with an intentionally throttled browser/WebSocket consumer. This
     confirms the magnitude behind `RUNTIME-01`, not just its quadratic
     formula.

3. **Long-session browser retention**
   - Open/close at least 500 Analytics details and compare heap snapshots,
     detached nodes, document listeners, chart instances, and click-handler
     time before/after (`RUNTIME-03`, `ANALYTICS-01`).
   - Run at least 20 generate/scrub/edit/retry cycles and look for monotone
     browser heap or worker VRAM growth, including the retained AR KV cache.

4. **Save and GIF peak cost**
   - Save synthetic 128, 1,024, and 2,048-frame histories while recording
     supervisor peak RSS, CPU time, event-loop responsiveness, output bytes,
     and failure behavior. Confirm the current raw-image estimate behind
     `RUNTIME-02`.
   - Interrupt saves at representative phases on a disposable data root and
     inspect what Analytics can see.

5. **Analytics scale**
   - With 175, 1,000, and 5,000 synthetic runs, including maximum-length
     prompts, measure list scan/encode latency, response bytes, server RSS,
     browser heap, first usable paint, sort/group latency, and star/collection
     repaint latency.
   - Open maximum-length AR and multi-canvas DiffusionGemma runs and record
     detail JSON size, chart construction time, hover/pan frame time, and
     tooltip collision-plugin cost.

6. **Worker switch and host ownership**
   - Measure terminate-to-VRAM-release time for every model, graceful and
     forced termination, and whether the current eight-second settle window
     is sufficient.
   - Barrier-synchronize browser and desktop supervisors at activation to
     reproduce the cross-process preflight race. Also launch a second desktop
     instance and close one during load.

7. **Download and artifact lifecycle**
   - Measure cold, warm, fully offline cached, partial-cache resume,
     insufficient-disk, cancel, and desktop-close paths for both Hub models.
     Observe child/thread lifetime and bytes written after the UI exits.
   - Interrupt DiffusionGemma quantization during state-dict write and each
     metadata copy on a disposable output path; record the current menu/load
     behavior before implementing a manifest.

8. **Model-specific expensive signals**
   - Compare DiffusionGemma entropy-signal off/on for milliseconds per
     denoising frame, peak VRAM, host transfer, queue depth, and payload size.
   - For SmolLM3, time a late-position probe and substitution with retained
     cache versus forced fresh prefill, and record retained-cache VRAM across
     repeated runs.
   - Resume the same LLaDA frame twice after unrelated RNG use and compare
     branch frames, unchanged-token confidence, Heatmap, and mean confidence;
     measure the memory cost of retaining complete intervention checkpoints.

9. **Visual smoothness**
   - Under the worst supported birth-glow settings, record browser frame time
     and dropped frames for fast AR generation and LLaDA/DiffusionGemma
     updates.
   - Repeat detail crossfade, entropy hover, and 2,000-point chart zoom under
     both QtWebEngine and the GTK fallback. These observations are the only
     basis for prioritizing visual micro-optimization beyond the bounded
     structural findings above.

10. **Future hardware gates**
    - Before Mamba lands, measure native-kernel install/load failure output,
      effective device attestation, VRAM, tokens/s, and raw-completion prompt
      behavior.
    - Before multimodal work lands, measure vision-tower VRAM/host-memory
      headroom and upload/decode peak memory at the proposed image limits.
