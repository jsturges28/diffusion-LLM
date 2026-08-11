# IMPLEMENTATION_LEDGER: state of the audit remediation

State for the 40 findings in `AUDIT_REPORT.md`. The report is the immutable
analysis; this file is the moving part. Read `IMPLEMENTATION_BRIEF.md` for how
to work a finding, and update this file in the same commit as the change it
describes.

Stage 1 is in progress. The stage map at the bottom is the agreed plan;
statuses below move as each finding lands.

## How to read this

**Statuses.** `ready` means every blocker is done and it can be started.
`blocked` means it cannot. `in progress` means started but its Verification
clause does not yet pass end to end, which is the normal state of an **L**
finding across several commits. `needs hardware` means the automated half
passes and the maintainer's confirmation on real hardware is outstanding.
`done` means both halves pass. `deferred` and `superseded` need a line in
Deviations explaining why.

**Blocked by** names only dependencies the report states explicitly. Where the
report implies an order without naming an edge, the cell reads `stage N order`,
which means the stage map below is the reason. This column is a mechanical
restatement of `AUDIT_REPORT.md:1829-1925`; where the two disagree, the report
wins and this file should be corrected.

Severity and effort are copied from the report's index. **S** is one cohesive
commit, **M** a short multi-commit change, **L** a staged boundary migration.

## Ready now

Seven findings have no unmet blockers. The first four are the remaining
isolated safety fixes; the last three are the gates the report wants
installed before any boundary moves.

- **TRUST-01** (high, S): bind to loopback unless network exposure is
  explicit.
- **ANALYTICS-01** (high, S): a stale detail response can populate the panel
  of a different run.
- **DATA-03** (medium, S): resolve the data directory independently of the
  process working directory. This one also unblocks the whole run-store stage.
- **TRUST-02** (high, M): vendor the Analytics chart dependencies so the page
  works without third-party networks. Independent of frontend modularization.
- **QUALITY-02** (medium, M): install the lint ratchet now, burn down later
  and only in files that later work touches.
- **META-01** (medium, M): reduce `HANDOFF.md` to a cold-start page and move
  the 132-item verification ledger out of it.
- **META-02** (medium, S): move the canonical agent contract into tracked
  files, since `.cursor/` is gitignored and its Python rule contradicts the
  three-environment matrix.

`QUALITY-01` is not on this list because it is not a standalone task. The
report asks for its lifecycle and browser-contract fixtures to land with each
seam as that seam is extracted, rather than as a test-only prelude with no
production owner. Track it as an obligation attached to other findings.

## Needs a maintainer decision before it can start

- **DATA-02** offers two paths and the sequencing says it should not begin
  until one is chosen: make collections server-authoritative through bounded
  semantic operations, or add a revision and ETag scheme that rejects stale
  replacements and lets the client reload and merge. Either way it is paired
  with an interprocess file lock and a visible persistence failure. The choice
  affects the UI-state API shape, so it wants deciding before code.
- **ANALYTICS-02 has an unresolved position in the order.** The executive
  summary lists it among the small safety commits that should precede the
  larger seams (`AUDIT_REPORT.md:46-51`), while the sequencing places it as
  something the run-store stage unlocks (`AUDIT_REPORT.md:1866-1868`). Both
  readings are defensible: the finding's Direction persists new per-frame
  counts, which touches the saved-run contract and therefore wants `DATA-05`
  first, but it also says modern runs can be repaired from existing token
  records, which needs no format change. It is parked at `blocked` on the
  sequencing's reading. Reopening it early is reasonable if the fix derives
  from records already saved and adds no unversioned fields.

## Hardware validation queue

Findings land here when their automated verification passes but the
maintainer's confirmation on real hardware is outstanding. The report's
standing measurement programme is separate and lives at
`AUDIT_REPORT.md:1927-2011`.

- **LIFE-07**: the finding's own Verification clause is met in full by
  `tests/backends/test_llada_resume_state.py`, which injects failure at all
  four named points and proves the retained history and step count survive
  byte for byte. Outstanding is one smoke run on real hardware, because no
  part of the resume path can execute here: load LLaDA, generate, remask and
  run to a frame, then edit again from an earlier original frame. Nothing
  depends on this finding, so the queue entry blocks nothing.

## Status table

| ID | Sev | Eff | Status | Blocked by | Commits |
|---|---|---|---|---|---|
| LIFE-07 | high | S | needs hardware | none | Commit LLaDA resume state only after the run lands |
| TRUST-01 | high | S | ready | none | |
| ANALYTICS-01 | high | S | ready | none | |
| DATA-03 | medium | S | ready | none | |
| TRUST-02 | high | M | ready | none | |
| QUALITY-02 | medium | M | ready | none | |
| META-01 | medium | M | ready | none | |
| META-02 | medium | S | ready | none | |
| QUALITY-01 | medium | L | companion | lands with each seam | |
| ORG-01 | medium | M | blocked | DATA-03 | |
| DATA-01 | high | L | blocked | ORG-01 | |
| DATA-05 | high | L | blocked | DATA-01 | |
| DATA-04 | high | M | blocked | DATA-05 | |
| RUNTIME-02 | medium | M | blocked | DATA-01 | |
| ANALYTICS-02 | high | M | blocked | DATA-05, see decision above | |
| ANALYTICS-03 | medium | L | blocked | DATA-01, DATA-05 | |
| ANALYTICS-04 | high | M | blocked | DATA-01 | |
| LIFE-02 | high | M | blocked | stage 4 order | |
| LIFE-06 | medium | M | blocked | LIFE-02 | |
| ORG-04 | medium | S | blocked | stage 4 order | |
| LIFE-03 | critical | L | blocked | ORG-04 | |
| LIFE-01 | high | M | blocked | LIFE-03 | |
| PROTOCOL-01 | medium | M | blocked | LIFE-03 | |
| XAI-01 | high | M | blocked | LIFE-01 | |
| LIFE-04 | high | L | blocked | LIFE-03 | |
| LIFE-05 | high | M | blocked | LIFE-02 | |
| TRUST-04 | medium | L | blocked | LIFE-04 | |
| DATA-02 | high | L | blocked | maintainer decision | |
| RUNTIME-01 | medium | L | blocked | LIFE-04, then ORG-02 + DATA-05 | |
| ORG-02 | medium | L | blocked | PROTOCOL-01 | |
| RUNTIME-03 | medium | S | blocked | ORG-02, paired | |
| ROADMAP-01 | high | M | blocked | stage 6 order | |
| ROADMAP-05 | high | M | blocked | stage 6 order | |
| ROADMAP-02 | medium | M | blocked | stage 6 order | |
| TRUST-03 | high | L | blocked | stage 6 order | |
| DEPS-01 | medium | L | blocked | stage 6 order | |
| ROADMAP-03 | high | L | blocked | stage 6 order | |
| ORG-03 | medium | M | blocked | stage 6 order | |
| ROADMAP-04 | medium | L | blocked | DATA-01, stage 6 order | |
| META-03 | medium | M | deferred | milestone boundaries | |

## Stage map

Derived from `AUDIT_REPORT.md:1829-1925`. Each stage is a boundary to validate
before the next takes a dependency on it.

**Stage 1, isolated safety fixes.** `LIFE-07` first, then loopback binding
(`TRUST-01`), synchronous detail clearing with request epochs
(`ANALYTICS-01`), and an absolute configurable data root (`DATA-03`), each as
its own commit. Vendor the Analytics dependencies (`TRUST-02`) independently.

**Stage 2, gates before boundaries move.** Install the path-specific and
code-specific Ruff ratchet (`QUALITY-02`). Shorten `HANDOFF.md` and make the
agent contract portable (`META-01`, `META-02`) so every later session starts
from the same rules. `QUALITY-01` fixtures attach to seams as they are cut.
`META-03` refreshes at milestone boundaries, not after every commit.

**Stage 3, the run-store boundary.** Extract behavior-preserving storage
operations (`ORG-01`) once the root is explicit, then unique staged
publication with complete replacement, revisions, and compare-and-swap
(`DATA-01`), then versioned validation and read adapters (`DATA-05`), then
immutable worker provenance threaded through the terminal run contract
(`DATA-04`). Make GIFs bounded non-authoritative derivatives (`RUNTIME-02`)
only after publication semantics exist. This stage unlocks `ANALYTICS-02`,
`ANALYTICS-03`, `ANALYTICS-04`, and later `ROADMAP-04`.

**Stage 4, explicit process and socket ownership.** Extract and test the
manager process adapter, then make termination and pre-eviction validation
reliable (`LIFE-02`, `LIFE-06`). Share activation orchestration (`ORG-04`)
before adding activation and resident epochs with run ownership (`LIFE-03`,
`LIFE-01`) and operation-scoped envelopes (`PROTOCOL-01`), which move
together. With run ownership explicit, preserve complete intervention
checkpoints (`XAI-01`). Propagate disconnect and cancel through inference and
bounded queues (`LIFE-04`), which carries the first bounded-queue step of
`RUNTIME-01`. Add host-level ownership (`LIFE-05`) once one manager's
lifecycle is trustworthy, and move downloads into the same owned-operation
model (`TRUST-04`). `DATA-02` can run in parallel once its fork is settled,
and must not reuse model-operation state.

**Stage 5, frontend state around the settled protocol.** Move aligned frame
operations and legal workflow phases into a tested native-module core
(`ORG-02`), then the model and download API clients. Fix the select lifecycle
(`RUNTIME-03`) as shared controls gain module ownership. Compact append-only
streams (`RUNTIME-01`) only once the reducer can reconstruct them and the
run-store version can distinguish them.

**Stage 6, prepare the existing models before adding Mamba.** Split family,
stream shape, device support, and resource requirements (`ROADMAP-01`),
extract model-specific text adapters (`ROADMAP-05`), and centralize
registry-driven parameter validation (`ROADMAP-02`), migrating and testing the
three existing models first. Pin and attest artifacts (`TRUST-03`) and
consolidate environment intent (`DEPS-01`) before `.venv-ssm` exists. The
Mamba baseline comes only after those are validated. The axis-aware signal
manifest (`ROADMAP-03`) precedes its native XAI phase and diffusion entropy
and top-k, though not necessarily the baseline decode. Consolidate the LLaDA
sampling kernel (`ORG-03`) before adding diffusion entropy and top-k to it.

**Stage 7, deferred cleanup.** Continue the Ruff burn-down and the remaining
documentation cleanup, without mixing formatting churn into behavioral
reviews.

## Deviations and corrections

When implementation shows a finding is mistaken, incomplete, or that its
Direction does not survive contact with the code, add an entry here under the
finding's ID with what was learned and what was done instead. Do not edit
`AUDIT_REPORT.md`; it is the record of what was believed on 2026-08-10, and
the difference between that and what turned out to be true is worth keeping.

No finding has been contradicted so far. The entries below are things learned
while working one finding that belong to another, recorded rather than
opportunistically fixed.

### LIFE-07

The Direction held exactly. `streaming_resume` builds its own sequence with
`torch.cat` and never writes through the base tensor
(`src/inference/streaming_sampler.py:490-498`), so staging really did cost one
list of existing references and no tensor storage. DiffusionGemma was not
merely the safer shape but the precise target: `dgemma_worker.py:312-323`
already sends its terminal frame before assigning the spliced history, and
LLaDA now matches it.

Two clarifications the finding left implicit, both settled with the
maintainer and pinned by tests:

- **Cancellation commits.** `streaming_resume` returns without a terminal
  frame when its cancel event is set, and the browser keeps the frames it
  received, so discarding the partial history would recreate the same
  worker/browser disagreement in the other direction. "Commit only after
  resume succeeds" means after an accepted *terminal outcome*, and a cancel
  is one.
- **The terminal send is part of the transaction.** On the guided
  "run to here" path the terminal frame is the worker's own, so the commit
  moved after that `send_json`, matching the completed path where the
  sampler's terminal frame has already gone out through the streamer.

**Noted, not fixed (adjacent).** On a guided partial resume, `tensor_history`
changes while `total_steps` deliberately keeps describing the run it branched
from, so `_validate_resume` bounds-checks the frame index against one and
computes remaining steps from the other. That is pre-existing and does not
prevent LIFE-07's Verification clause from passing, so it was left alone. It
belongs with `LIFE-01`'s retained-state contract.
