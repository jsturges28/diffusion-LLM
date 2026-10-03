# AUDIT_BRIEF_2026-10: a second full sweep of diffusion-LLM

This brief governs the repository audit begun on 2026-10-02. It reads the
whole current repository and produces `docs/audit/AUDIT_REPORT_2026-10.md`.
It changes nothing else after this brief is established.

The first audit, dated 2026-08-10, found forty issues and led to a sustained
remediation campaign. That campaign rebuilt important ownership boundaries,
added four model families, and greatly expanded automated coverage. Its
analysis remains useful, but its inventory and several of its assumptions are
now historical. This is a fresh audit of the current tree, not an addendum
that assumes only new files can contain new problems.

The extra emphasis in this pass is code organization. The application has
continued to grow after the first audit's extractions. Large files are not
findings by themselves, but they are expensive maps for a maintainer or agent
to keep rebuilding. This audit must identify cohesive, reversible boundaries
where a split would reduce that cost without replacing local complexity with
global indirection.

## The contract

1. **After this brief, the only path the audit may create or modify is
   `docs/audit/AUDIT_REPORT_2026-10.md`.** Do not edit source, tests,
   configuration, dependencies, ordinary documentation, the plan, or any
   artifact from the first audit.
2. **Do not fix anything.** A defect, stale claim, missing test, or easy
   extraction is evidence for the report, not permission to make a drive-by
   change.
3. **The 2026-08 report is immutable.** Do not revise
   `docs/audit/AUDIT_REPORT.md` to make old analysis agree with the current
   tree. The implementation ledger records what happened after it.
4. **Subagents obey this same contract.** Restate the read-only rule, the one
   writable report, and the ban on source fixes in every audit prompt.
5. **Every finding cites exact `path:line` evidence.** Git history, test
   output, and measurements may supplement a finding, but do not replace a
   current-tree citation for its mechanism.
6. **Mark confidence and distinguish observation from inference.** Runtime
   behavior that was not measured is a hypothesis. A low-confidence concern
   can be useful; an overstated concern is not.
7. **Write confirmed findings into the report as the sweep proceeds.** Long
   audits are vulnerable to context compaction. The report is the durable
   record.
8. **Run verification, but never repair it during the audit.** A failing test
   or lint gate is a finding or a stated baseline limitation.
9. **No commits, staging, branches, or repository configuration changes.**
10. **Stop at analysis.** Do not create a second implementation brief or
    ledger. The maintainer will triage the report before choosing normal
    roadmap work or another campaign.

The sandbox has no GPU and no display. Static analysis, CPU-safe tests, git
history, and the existing saved artifacts are available. Claims about real
model inference, CUDA memory, pywebview, or browser layout must be labelled
and routed to a bounded hardware measurement.

## What is different from the first audit

The first audit's method remains sound: a strict read-only sweep, parallel
tracks, evidence-backed findings, a capped report, explicit sequencing, and a
hardware measurement section. Its routing and seeds do not carry forward
unchanged.

- `docs/HANDOFF.md` is now a bounded cold-start page rather than a 3,223-line
  session log.
- Browser behavior now has a substantial Node `vm` test suite rather than
  syntax checks alone.
- Static assets are vendored, dependency intent is consolidated, saved runs
  are versioned and transactional, and lifecycle operations carry identity.
- Mamba-3, Vision, diffusion candidates, candidate flicker, revision
  signalling, and the adaptive-stopping readout all postdate the old audit.
- `src/web/server.py`, `src/web/static/app.js`, and
  `src/web/static/analytics.js` are materially larger even after useful
  seams were extracted.
- Native ES modules remain deliberately deferred because the shipped classic
  scripts are exercised directly by the current `vm` harness.

Completed old findings are therefore regression contracts, not seeds to
rediscover. A regression may become a new finding only when current evidence
shows that the invariant no longer holds.

## What is being audited

Eight concerns overlap across six exploration tracks.

1. **Code organization and navigation.** Map responsibilities, state owners,
   dependency direction, and test seams. Identify the cheapest reversible
   cuts in monoliths and the files that should remain whole.
2. **Lifecycle, correlation, and persistence.** Recheck the one-resident
   rule, process ownership, WebSocket operation identity, run identity,
   cancellation, publication, UI intent, and multi-window behavior.
3. **Runtime resources and responsiveness.** Inspect queue bounds, frame
   representation, DOM and chart work, saved-run scaling, blocking work on
   async loops, cache growth, and model or download resource release.
4. **Correctness of model and XAI semantics.** Follow confidence, entropy,
   candidates, revisions, forgetting, adaptive stopping, edits, and
   substitutions from model output through wire format, browser state, disk,
   and Analytics.
5. **Robustness, security, and local-first trust.** Inspect input boundaries,
   paths, archives and images, network exposure, artifact pinning, offline
   behavior, partial failure, error ownership, and destructive operations.
6. **Duplication with a receipt.** Prefer duplicated behavior that has already
   drifted or repeatedly required parallel edits over merely similar-looking
   code.
7. **Roadmap positioning.** Test the current architecture against
   multi-canvas resume, multimodal generation, additional model families,
   frame-linked analysis, and other accepted or plausible directions.
8. **Quality and routing.** Check tests, lint and dependency gates,
   documentation claims, manual verification debt, and the cost of finding
   the authoritative answer.

Ubuntu-only support remains accepted. Portability is not a finding unless the
current documentation or code claims another platform.

## The organization rubric

Do not recommend splitting a file because it crossed an arbitrary line count.
For each serious decomposition candidate, record:

- the responsibilities and mutable state it owns today;
- its incoming and outgoing dependencies;
- which responsibilities change together in git history;
- the existing tests that constrain the boundary;
- the first behavior-preserving extraction;
- the stable contract across that extraction;
- the migration order and rollback point;
- the verification needed before the next cut;
- the indirection, load-order, or ownership cost the split introduces; and
- a credible alternative that leaves the file intact.

Classify the proposed boundary:

- **ownership boundary**, where state or a resource gains one clear owner;
- **semantic boundary**, where model or product behavior becomes explicit;
- **transport boundary**, where a stable request or frame contract exists;
- **presentation boundary**, where rendering can be tested independently; or
- **navigation-only split**, where the benefit is discoverability.

Navigation-only splits are legitimate, especially for CSS, but they should
not outrank correctness or ownership work merely because they are easy.

The old audit already extracted `run_store.py`, `worker_process.py`,
`model_lease.py`, `collections.py`, `run_frames.js`, `run_phases.js`,
activation and model clients, text adapters, append-only AR handling, and the
LLaDA kernel. Confirm those seams before proposing work around them. Do not
recommend them again under a new title.

## How to route the audit

Read these orientation sources before drawing conclusions:

1. `AGENTS.md`, all of it, for the tracked contract.
2. `docs/HANDOFF.md`, all of it, for the bounded current architecture.
3. `README.md`, all of it, as the public claims surface.
4. `docs/ROADMAP.md`: orientation, accepted directions, deliberate stopping
   points, experimental backlog, Vision measurements, quick map, dependency
   decision, and documentation routing.
5. `docs/audit/IMPLEMENTATION_LEDGER.md`: opening status, finding table, and
   the entries or deviations for any old invariant under review.
6. `docs/audit/AUDIT_REPORT.md`: executive summary, index, sequencing, and
   only the full old findings needed to understand a regression contract.
7. `docs/MANUAL_VERIFICATION.md`: ledger rules, current checked-state
   summary, old unchecked items, and scenarios relevant to a candidate
   finding.
8. `docs/GUIDE.md` when checking user-visible behavior or parameters.
9. `docs/TIGERSTYLE.md`, `pyproject.toml`, and the environment lock tooling
   when checking code or dependency gates.

Use git history from 2026-08-10 onward to distinguish accumulated churn from a
large but stable implementation. Inspect co-change and blame only to explain
a current boundary; history is not evidence that current code is wrong.

## Current large-file routing

These counts are navigation aids measured at the start of the audit. Vendored
assets are excluded.

| Lines | Path | Initial routing question |
|---:|---|---|
| 9,906 | `src/web/static/app.js` | Which run state, rendering, edit, and persistence responsibilities still share necessary state? |
| 7,570 | `src/web/static/analytics.js` | Where do catalog, detail, charts, token viewer, and comparison form stable boundaries? |
| 4,917 | `src/web/static/style.css` | Which page or component sections can split without cascade-order surprises? |
| 4,399 | `src/web/server.py` | Which ownership and route groups can leave the application root? |
| 2,823 | `src/web/static/overlays.js` | Is this still one visual toolkit, or also an unrelated browser platform layer? |
| 1,846 | `src/web/static/menu.js` | Which activation and download behavior already belongs to tested clients? |
| 1,669 | `src/inference/ar_sampler.py` | Are decode, substitute, forced, and probe loops meaningfully separate? |
| 1,653 | `src/backends/worker_base.py` | Can protocol dispatch separate from backend and provenance ownership? |
| 1,572 | `src/web/static/analytics.css` | Is page-local splitting useful after shared styles are accounted for? |
| 1,148 | `src/analytics/metrics.py` | Do catalog, schema reading, and metric computation have distinct consumers? |
| 860 | `src/inference/dgemma_sampler.py` | Has signal capture or streaming created a reusable boundary? |
| 789 | `src/inference/streaming_sampler.py` | Does LLaDA orchestration remain cohesive around the shared kernel? |
| 746 | `src/web/run_store.py` | Has the extracted store retained a narrow persistence role? |

Line count is only a prompt to inspect. A report finding needs a consequence,
a boundary, and verification.

## Establishing the baseline

Before the deep tracks:

- inventory tracked first-party files and current line counts;
- collect functions and classes in the largest files;
- inspect imports and classic-script order;
- inspect commits and co-change since 2026-08-10;
- run `.venv/bin/python -m pytest`;
- run `.venv/bin/python scripts/lint_ratchet.py`;
- run `node --check` on first-party JavaScript; and
- run `node --test tests/web/static/*.test.js`.

Use only the interpreter assigned by `AGENTS.md`. Do not invoke system or user
Python. Do not import worker code under the wrong environment merely to
inspect it.

Record exact command outcomes and test counts in the report. A test pass is
evidence for the behavior it actually covers, not a general clean bill.

## Regression matrix

Group the old findings into contracts and report one status for each:

- run publication, schema, provenance, and explicit data root;
- process termination, activation identity, run identity, host lease, and
  pre-eviction validation;
- disconnect, cancellation, queue, and append-frame behavior;
- response scoping and Analytics request coherence;
- offline assets, artifact pinning, and owned downloads;
- model axes, registry parameters, text semantics, signal manifests, and
  dependency intent;
- intervention checkpoint fidelity;
- test and lint gates; and
- bounded documentation and inventory routing.

Use **holds**, **regressed**, **not fully verified**, or **known remainder**.
Do not call a hardware-dependent contract regressed merely because this audit
cannot exercise it.

Known remainder at the start includes hardware items for `LIFE-02`,
`LIFE-05`, `ROADMAP-03`, and `META-03`; deferred native-module work under
`ORG-02`; and untaken multimodal artifact lifecycle work under `ROADMAP-04`.
Confirm their current ledger state before reporting them.

## Six exploration tracks

### Track 1: supervisor and ownership

Read `src/web/server.py`, `src/web/run_store.py`,
`src/web/worker_process.py`, `src/web/model_lease.py`,
`src/web/ui_state.py`, `src/web/collections.py`, `main.py`, `desktop.py`, and
their tests. Cover process lifecycle, concurrency, persistence, routes,
shutdown, destructive operations, and the `ModelManager` boundary.

### Track 2: workers, protocol, and inference

Read `src/backends/`, `src/inference/`, model probes and quantization scripts,
and their tests. Cover protocol dispatch, run ownership, cancellation, queue
bounds, sampler semantics, candidate and signal capture, context limits,
device behavior, and the organization of `worker_base.py` and samplers.

### Track 3: generator frontend

Read generator HTML, `app.js`, generator and shared CSS, run-state modules,
activation/model/download clients, candidate and phase modules, and browser
tests. Cover global state, script ordering, hydration, WebSocket handling,
rendering, scrub/edit workflows, save semantics, keyboard behavior, and
session persistence.

### Track 4: Analytics and shared frontend

Read Analytics HTML, `analytics.js`, `analytics.css`, `overlays.js`, detail
and collections clients, vendored chart integration, saved-run readers, and
browser tests. Cover request correlation, table scale, charts, comparison,
shared token rendering, signal declarations, export, deletion, and
generator-to-Analytics drift.

### Track 5: post-audit architecture and roadmap fit

Trace Mamba-3, Vision, top-k candidates, flicker, revision signals, adaptive
stopping, and DiffusionGemma resume across registry, backend, browser, disk,
Analytics, tests, GUIDE, and ROADMAP. Check that newer work respects the
model-axis, signal-axis, provenance, artifact, and one-resident contracts.

### Track 6: quality, trust, and routing

Read tests, scripts, manifests and locks, tracked contracts, public and in-app
documentation, manual verification, and audit records. Cover boundary
coverage, lint and dependency enforcement, offline and network claims,
security-sensitive inputs, stale documentation, and cold-start cost.

Each track returns evidence-backed candidates, explicit clean areas, coverage
limits, and decomposition seams. Cross-cutting synthesis owns duplication,
severity, sequencing, and the final finding set.

## The report

Write one report, `docs/audit/AUDIT_REPORT_2026-10.md`, with:

1. **Executive summary**, one screen: what is healthy, what is urgent, the
   highest-leverage moves, and what is deliberately not recommended.
2. **Coverage and regression matrix**: what was read and run, which old
   contracts hold, and what could not be verified.
3. **Findings index**: ID, type, area, severity, effort, and title.
4. **Monolith and decomposition map**: responsibility maps, recommended first
   cuts, leave-intact decisions, and ordering hazards.
5. **Findings in full**, grouped by area.
6. **Sequencing**: dependencies, reversible first commits, combinations to
   avoid, and work that should wait for hardware evidence.
7. **Measurements to take on hardware**: bounded scenarios with an observable
   pass condition.
8. **Blind spots**: unread or unmeasured areas, if any.

Cap the report at roughly 30 to 40 findings. Fewer is better when the omitted
items would not change a decision. Coverage does not require manufacturing a
finding in every track.

### Finding schema

Use new IDs such as `A2-ORG-01`, `A2-LIFE-01`, or `A2-RUNTIME-01` so they
cannot be confused with the first audit.

```
### [A2-AREA-01] Imperative title

- **Type**: defect | regression | architecture | organization | trust | meta
- **Severity**: critical | high | medium | low
- **Effort**: S | M | L
- **Confidence**: high | medium | low
- **Evidence**: `path:line`, `path:line`
- **What is true today**: the current mechanism.
- **Why it matters**: the concrete consequence.
- **Direction**: the shape of a fix and its trade-off.
- **Rejected alternative**: the plausible option not recommended.
- **Blast radius**: contracts and files that move.
- **Verification**: what proves the change worked.
- **Depends on**: other findings or evidence, if any.
- **Roadmap impact**: what this blocks or enables, if anything.
```

A style preference with no consequence is not a finding. A decomposition
finding that cannot name a stable contract is not ready.

## Non-goals

- Fixing findings during the audit.
- Re-grading or rewriting the first report.
- Treating all old known remainder as new debt.
- Choosing implementation details that require maintainer judgement.
- Introducing a frontend framework, bundler, database, universal sampler, or
  mass formatting campaign without evidence that the cheaper path fails.
- OS portability.
- Replacing model-specific numerical loops merely because their outer shapes
  resemble one another.
- Creating a second remediation campaign before triage.

## When the audit is done

Verify every citation and report cross-reference. Confirm that no source,
test, configuration, dependency, ordinary documentation, legacy audit file,
or plan changed. Hand back with the report path, the three highest-leverage
findings, verification outcomes, hardware limits, and any area that did not
receive full coverage.

The report is the handoff. Do not update `HANDOFF.md`, `README.md`,
`ROADMAP.md`, in-app copy, or the old implementation ledger during this
session, even when the audit finds them stale.
