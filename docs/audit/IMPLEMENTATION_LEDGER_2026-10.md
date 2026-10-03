# IMPLEMENTATION_LEDGER_2026-10: state of the second audit's remediation

State for the 21 findings in `docs/audit/AUDIT_REPORT_2026-10.md`, raised by
the read-only audit that `docs/audit/AUDIT_BRIEF_2026-10.md` governed on
2026-10-01. The report is the immutable analysis and this file is the moving
part: update it in the same commit as the change it describes. The first
campaign's `docs/audit/AUDIT_REPORT.md` and `docs/audit/IMPLEMENTATION_LEDGER.md`
are records of 2026-08 and are not edited for these findings.

Short on purpose, and held there by `tests/test_audit_ledger_2026_10.py`. The
first ledger reached 3,899 lines because every session appended its story to
it. Here a finding gets one row and a decision one line, with any longer
reasoning in `docs/ROADMAP.md`, and git history keeps the rest.

**As of 2026-10-02**, Stage 1 is done but for `A2-TRUST-02`, which waits on
the maintainer's limits for one run, and every other Stage 1 fix has cleared
hardware. Stage 2, cross-supervisor ownership, is next; the report validates
it as a whole with a two-supervisor hardware run. The report's Sequencing
section orders the rest, and its Combinations to avoid bind every pass.

## How to work a finding

1. Read its entry in the report, then check its premise against today's code
   before planning. One has already turned out half wrong (`A2-XAI-01`).
2. Deliberate, plan and implement as `AGENTS.md` describes. A regression test
   is shown failing first, each cohesive fix is its own commit, proposed and
   created only on the maintainer's greenlight, and nothing is pushed.
3. Hardware scenarios go in `docs/MANUAL_VERIFICATION.md`, and their numbers
   in the finding's row.

## Statuses

`ready` has no unmet dependency. `blocked` waits on what its row names: a
finding, a decision, or `stage N order` where the report orders work without
naming an edge. `needs hardware` passes its automated half and waits on the
maintainer's confirmation. `done` passes both. `deferred` needs a line under
Decisions saying why.

## Findings

| ID | Stage | Status | Commits | Manual items | Waits on |
|---|---|---|---|---|---|
| A2-QUALITY-01 | 1 | done | `891bc02` | 380 to 384 | |
| A2-XAI-01 | 1 | done | `4a09032` | 380 to 384 | |
| A2-XAI-02 | 1 | done | `3aeae87` | 380 to 384 | |
| A2-XAI-03 | 1 | done | `0242f01` | 385 | |
| A2-XAI-04 | 1 | done | `b24eeb4` | 386 | |
| A2-LIFE-02 | 1 | done | `ef3eb08`, `fa08716` | 389 to 392 | |
| A2-LIFE-04 | 1 | done | `bf9b7de` | | |
| A2-META-01 | 1 | done | `4a09032`, `4cc3b7d` | | |
| A2-ORG-06 | 1 | done | `e5bbc86` | 393 | |
| A2-TRUST-02 | 1 | blocked | | | the maintainer's limits for one run |
| A2-LIFE-01 | 2 | ready | | | |
| A2-TRUST-01 | 2 | blocked | | | the maintainer's policy without a lease |
| A2-DATA-01 | 2 | ready | | | |
| A2-QUALITY-02 | 2 | ready | | | |
| A2-LIFE-03 | 2 | blocked | | | `A2-QUALITY-02` |
| A2-ORG-01 | 3 | blocked | | | `A2-LIFE-01`, `A2-TRUST-01`, `A2-DATA-01` |
| A2-ORG-02 | 3 | blocked | | | stage 3 order, in one plan with `A2-DEPS-01` |
| A2-DEPS-01 | 3 | blocked | | | stage 3 order, in one plan with `A2-ORG-02` |
| A2-ORG-03 | 3 | blocked | | | `A2-LIFE-03` |
| A2-ORG-04 | 3 | blocked | | | stage 3 order |
| A2-ORG-05 | 3 | blocked | | | `A2-ORG-04` |

## Decisions

Each settled with the maintainer during remediation.

- `A2-XAI-01`'s premise was half right: the page cuts its own frames when it
  sends a resume, so page and worker agreed. The fix still commits exactly
  the frames the worker forwarded.
- A resume stopped before its first frame keeps the worker's history.
- A guided edit ends as cancelled only when it stopped short of its budget.
- The DiffusionGemma worker is tested under `.venv` through a stand-in for
  `dgemma_nf4`, which imports `bitsandbytes`.
- A run saved before signal manifests existed is read by its stream's shape.
- A DiffusionGemma commit frame reads its canvas's last draft for entropy and
  says *as of step N*.
- A run's opening frame names its worker while the run token stays on its
  terminal frame; the reasoning is in `docs/ROADMAP.md`'s settled decisions.
- An interrupted diffusion run's saved text keeps its mask glyphs.
- `A2-ORG-06` also dropped four menu rules that matched nothing.

## Open decisions

- `A2-TRUST-02`: the limits for one run. The framing was agreed on
  2026-10-02: limits are per run, which is per turn once chat exists; the
  prompt's limit comes from the context window, with an explicit number for
  Mamba-3, which has none; and a conversation stays a set of linked runs
  rather than one growing run.
- `A2-TRUST-01`: refuse to run, or run visibly degraded, when the residency
  lease cannot be taken.

## Raised during remediation

Not findings, and each its own slice when taken.

| What | State | Commits | Manual items |
|---|---|---|---|
| DiffusionGemma's entropy at commit frames, on both pages | done | `d0ecae0`, `067038d` | 387, 388 |
| The entropy row held for every model that records entropy | done | `11affa1` | 394 |
| DiffusionGemma's stopped text holds only its last canvas and can run ahead of the page | open | | |
| Edit Frames and What If? on an interrupted run are refused as if it had been replaced | open | | |
| A resume stopped before its first frame could restore the frames the page cut | open | | |
| An interrupted save carries no run token, so a retried save can duplicate | accepted | | |
| The Help "signals" panel is at its 2,200-word budget | open | | |

## Baselines

At the audit on 2026-10-01: 2,456 Python tests, 816 browser tests, and no
Ruff findings. On 2026-10-02, with this ledger: 2,520 Python tests passing
and 6 skipped, 876 browser tests, and still no Ruff findings.
