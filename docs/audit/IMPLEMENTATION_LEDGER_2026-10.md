# IMPLEMENTATION_LEDGER_2026-10: state of the second audit's remediation

State for the 21 findings in `docs/audit/AUDIT_REPORT_2026-10.md`, raised by
the read-only audit that `docs/audit/AUDIT_BRIEF_2026-10.md` governed on
2026-10-01. The report is the immutable analysis and this file is the moving
part: each slice's closing docs commit moves its rows and names the commits
that landed them, since a commit cannot name its own hash. The first
campaign's `docs/audit/AUDIT_REPORT.md` and `docs/audit/IMPLEMENTATION_LEDGER.md`
are records of 2026-08 and are not edited for these findings.

Short on purpose, and held there by `tests/test_audit_ledger_2026_10.py`. The
first ledger reached 3,899 lines because every session appended its story to
it. Here a finding gets one row and a decision one line, with any longer
reasoning in `docs/ROADMAP.md`, and git history keeps the rest.

**As of 2026-10-03**, Stages 1 and 2 are done. Stage 2, cross-supervisor
ownership, was validated as a whole by the two-supervisor passes confirmed
for items 310 to 314, 389 to 396, 400 to 404 and 405. Stage 3, extracting
owners, is under way: `A2-ORG-01`, `A2-ORG-02`, `A2-DEPS-01` and
`A2-ORG-03` are done, and `A2-ORG-04`'s first slice is confirmed on
hardware with item 408. The report's Sequencing section orders the rest,
and its Combinations to avoid bind every pass.

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
| A2-TRUST-02 | 1 | done | `8be0e91`, `c0cbbde`, `7f6198d` | 397 to 399 | |
| A2-LIFE-01 | 2 | done | `2363eca` | 395 | |
| A2-TRUST-01 | 2 | done | `fc8374f` | 396 | |
| A2-DATA-01 | 2 | done | `4027c22`, `1b63867`, `a1eac0d` | | |
| A2-QUALITY-02 | 2 | done | `1a4fb18` | | |
| A2-LIFE-03 | 2 | done | `5f42ac3`, `749ecc6` | 400 to 404 | |
| A2-ORG-01 | 3 | done | `ac7f0b7`, `0355477` | | |
| A2-ORG-02 | 3 | done | `c3c007f` | | |
| A2-DEPS-01 | 3 | done | `2a6e87d` | 406 | |
| A2-ORG-03 | 3 | done | `a4ff5e5` | 407 | |
| A2-ORG-04 | 3 | ready | `64281e2`, `548f022` | 408 | |
| A2-ORG-05 | 3 | ready | | | |

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
- `A2-LIFE-01`'s window is narrower than the report says: the release came
  after the old worker had exited, with nothing awaited before the re-claim,
  so the gap was an instant rather than the wait. Its test pauses a switch
  inside that instant, since one process cannot meet it by chance.
- When the lease's lock file cannot be created, activation is refused with
  the path and the fix rather than going ahead without a lease
  (`A2-TRUST-01`).
- `A2-TRUST-02`'s limits are per run, which is per turn once chat exists,
  and a conversation stays a set of linked runs. Today's experimental
  slider tops are the limit for one run, under a 256 MiB ceiling on the
  save body. One 1,000,000-character prompt cap holds saves and Generate
  alike, and is the only one Mamba-3, with no window, has.
- Deleting a run takes the publication lock too, and a save whose run has
  vanished becomes a new run, decided inside that lock (`A2-DATA-01`). Its
  verification is the suite's forked-process races; the Stage 2
  two-supervisor run covers it on hardware.
- The resident worker is named by a value drawn when the supervisor starts
  plus its activation number, which alone starts again after a restart
  (`A2-LIFE-03`).
- A run whose worker is gone locks its edit tools in place, still savable,
  rather than reloading the page, and so does a run that lost its
  connection, with its own reason. An open edit session closes as Exit does
  unless it holds a branch the page can still save.
- `A2-ORG-01` moved the manager verbatim. The routes reach its shared
  helpers through the module, so one patch reaches both callers.
- `A2-ORG-02` moved the worker's socket shell into `worker_app.py` with no
  re-export. The resource pump stays in `worker_base.py`, where the tests'
  patches reach it.
- `A2-ORG-03`'s codec is `run_snapshot.js`. Storage and every write to page
  state stay in `app.js`, and the stored keys are unchanged.
- `A2-ORG-04` is taken in slices. The first gave every page one script list
  in the DOM stub, and moved Analytics' frame and signal reads into
  `overlay_series.js`; the chart and token-viewer controllers remain.

## Open decisions

None.

## Raised during remediation

Not findings, and each its own slice when taken.

| What | State | Commits | Manual items |
|---|---|---|---|
| DiffusionGemma's entropy at commit frames, on both pages | done | `d0ecae0`, `067038d` | 387, 388 |
| The entropy row held for every model that records entropy | done | `11affa1` | 394 |
| DiffusionGemma's stopped text holds only its last canvas and can run ahead of the page | open | | |
| Edit Frames and What If? on an interrupted run are refused as if it had been replaced | done | `cc47981` | 402 |
| A resume stopped before its first frame could restore the frames the page cut | open | | |
| An interrupted save carries no run token, so a retried save can duplicate | accepted | | |
| The Help "signals" panel is at its 2,200-word budget | open | | |
| A long run's save waits seconds on drawing its GIF preview | open | | |
| The GIF preview is drawn outside the publication lock, so two near-simultaneous replacements can leave the earlier one's | open | | |
| The process-race tests fork a multi-threaded process, which Python warns about 31 times a run (16 from `DATA-02`'s tests, 15 from `A2-DATA-01`'s) | open | | |
| The supervisor's own INFO logs reach no handler, so its results-directory line at startup has never been shown | open | | |

## Baselines

At the audit on 2026-10-01: 2,456 Python tests, 816 browser tests, and no
Ruff findings. On 2026-10-03, after `A2-ORG-04`'s first slice: 2,611 Python
tests passing and 6 skipped, 940 browser tests, no Ruff findings, and 34
warnings, down from 92 before the lifespan move. One of the 34 is torch
failing to start CUDA in the agent sandbox, which a machine with a working
GPU does not raise.
