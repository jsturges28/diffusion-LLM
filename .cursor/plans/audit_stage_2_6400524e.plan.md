---
name: Audit stage 2
overview: Install the lint ratchet, make the agent contract portable and the cold-start page bounded, and group the documentation under docs/, so every later remediation session starts from the same rules and pays a fraction of the context cost.
todos:
  - id: quality-02
    content: "QUALITY-02: add scripts/lint_ratchet.py with a tracked per-file-per-rule lint_baseline.json, fix the 16 mechanical findings (reading the three SIM105 sites rather than converting blindly), leave the 11 complexity findings to their owners, add tests/test_lint_ratchet.py and the AGENTS.md verification line"
    status: completed
  - id: meta-02
    content: "META-02: add tracked TIGERSTYLE.md referencing pyproject.toml for its numbers, repoint AGENTS.md and HANDOFF.md at it, un-ignore .cursor/plans/ so the three ROADMAP references become true, rewrite the two local .mdc files as thin pointers that fix the venv contradiction, and add tests/test_docs_links.py"
    status: in_progress
  - id: meta-01-ledger
    content: "META-01 part one: extract the 132-item ledger and the activation-failure runbook into MANUAL_VERIFICATION.md, preserving the 1-101 validated / 102-126 outstanding / 127-132 confirmed state and correcting the stale Alternatives default against registry.py"
    status: pending
  - id: meta-01-roadmap
    content: "META-01 part two: fold the still-live decisions from HANDOFF lines 2094-2196 and 3124-3160 into ROADMAP.md, dropping the ones the audit superseded"
    status: pending
  - id: meta-01-cut
    content: "META-01 part three: cut HANDOFF.md to a cold-start page under 200 lines, fix the Alternatives wording at line 30, update AGENTS.md's session-end habit, and add tests/test_handoff_bounded.py"
    status: pending
  - id: docs-layout
    content: "Documentation layout: git mv the reference and campaign documents into docs/ and docs/audit/, leaving README.md, AGENTS.md and LICENSE at root, then rewrite every cross-reference and let tests/test_docs_links.py prove none broke"
    status: pending
  - id: stage2-handback
    content: "Stage boundary: full verification, finalize the ledger deviations, and hand back with the clone test as the one manual item"
    status: pending
isProject: false
---

# Audit remediation, stage 2: gates before boundaries move

Governed by [IMPLEMENTATION_BRIEF.md](IMPLEMENTATION_BRIEF.md), state in [IMPLEMENTATION_LEDGER.md](IMPLEMENTATION_LEDGER.md). Order: `QUALITY-02`, `META-02`, `META-01`, then a documentation layout move. The middle two are swapped from the ledger's listing so the contract's home is settled before the cold-start page is rewritten to point at it, and the move comes last so it is a pure rename over content that has already stopped changing.

Decisions already settled: fix the 16 mechanical lint findings and leave the 11 complexity ones to their owners; ratchet per file-and-rule cell rather than clean-on-touch; canonical rules in tracked docs with `.cursor/rules/` left ignored; track `.cursor/plans/`; delete the shipment narrative; fold live decisions into `ROADMAP.md`; and group the documentation under `docs/`.

## 1. QUALITY-02: the ratchet, then the reachable half of the burn-down

New `scripts/lint_ratchet.py` plus a tracked `lint_baseline.json`. Ruff's JSON output, aggregated to counts per (file, rule). Fails when any cell grows or appears, reports cells that shrank, and takes `--update` to ratchet down. The per-cell granularity is forced by the Verification's third clause: a total count would pass when one finding replaces another.

Fix the 16 mechanical findings, which is what the Direction's "first remove the non-style findings" can actually reach:

- three unused imports in [src/inference/llada_sampler.py](src/inference/llada_sampler.py) and [tests/inference/test_load_progress.py](tests/inference/test_load_progress.py)
- four whitespace and newline nits, all in `llada_sampler.py`
- three `pathlib` swaps in [src/inference/load_progress.py](src/inference/load_progress.py) and [src/web/ui_state.py](src/web/ui_state.py)
- two nested-`with` merges in the load-progress test
- three `SIM105` suppressible-exception sites at [src/web/server.py](src/web/server.py) lines 772 and 1140, and `ui_state.py:135`

The `SIM105` three get read rather than converted mechanically. `contextlib.suppress` satisfies the linter without improving anything, and TigerStyle forbids silent suppression, so any site that is hiding a real error stays as it is and stays in the baseline, with a note saying why.

The 11 remaining complexity and nesting findings are **not** touched. Each sits inside a function a blocked finding owns: `create_worker_app` and four nesting hits in [src/backends/worker_base.py](src/backends/worker_base.py) belong to stage 4, `_save_run_blocking` in `server.py` to stage 3, `generate` plus two nesting hits in `llada_sampler.py` to `ORG-03` in stage 6, and the `render_gif.py` and `server.py:597` hits to `RUNTIME-02` and lifecycle. Doing them here is exactly the mixing the brief forbids. Recorded as a deviation.

Worth recording alongside it: `llada_sampler.py` carries 56 of the 156 findings, 36% of the baseline, and it is not dead code, since [src/inference/streaming_sampler.py](src/inference/streaming_sampler.py) line 23 imports live helpers from it. Most of the lint debt therefore discharges as a consequence of `ORG-03`, not as stage 7 grind.

Tests in `tests/test_lint_ratchet.py` drive the comparator with synthetic data: zero exit on the accepted baseline, failure on one added violation, and failure on a swap where one finding is added while another disappears. Expected landing: 156 down to roughly 140. Add the command to [AGENTS.md](AGENTS.md)'s verification list.

## 2. META-02: make the contract survive a clone

New tracked `TIGERSTYLE.md` at the repository root, matching the existing flat layout. This is the real gap: the standard [AGENTS.md](AGENTS.md) line 56 calls "the repo's TigerStyle rules" is not in `.cursor/rules/` at all, it is in the maintainer's personal Cursor user settings, so it travels to no clone and to no other machine. The document carries the standard and **references** [pyproject.toml](pyproject.toml) for the numbers (70 columns, complexity 10, nesting 3) rather than restating them where they could drift.

Repoint [AGENTS.md](AGENTS.md) line 58 and `HANDOFF.md` line 2055 at it, and confirm AGENTS.md's environment section already covers everything in the local `model-constraints.mdc` (VRAM figures, one resident model, no GPU or display in the sandbox).

Track the plans. `.gitignore` line 1 currently ignores all of `.cursor`; it becomes an ignore of `.cursor/*` with an exception for `.cursor/plans/`, which makes the three [ROADMAP.md](ROADMAP.md) references at lines 6, 814, and 1137 true instead of pointing at 39 files no clone has. `.cursor/rules/` stays ignored.

Rewrite the two local `.mdc` files as thin pointers to the tracked documents. They are gitignored so they will not appear in the commit, but `python-venv.mdc` currently tells every session to use `.venv` for all Python, which contradicts both AGENTS.md and the sibling `model-constraints.mdc`, and a current session obeying it would run a worker command against the wrong transformers.

New `tests/test_docs_links.py` implements the "no dead links" half of the Verification directly: every repository-relative path referenced from a tracked markdown file must exist and be tracked by git. That test is what would have caught this finding.

## 3. META-01: a cold-start page under 200 lines

[HANDOFF.md](HANDOFF.md) is 3,233 lines, of which roughly 95% is shipment log and verification ledger. Three commits:

**Extract the ledger.** New tracked `MANUAL_VERIFICATION.md` taking lines 2212 to 3122 (the 132 items) plus the activation-failure runbook at 2647 to 2670, which is a maintainer runbook that exists nowhere else and that items 57 and 73 depend on. The validation state at lines 2087 to 2089 must survive the move intact: items 1 to 101 validated, 127 to 132 confirmed, and **102 to 126 not validated**, which is 25 items of real outstanding debt. Correct the stale premise on the way: item 1 asserts "Alternatives off (the default)" while [src/backends/registry.py](src/backends/registry.py) line 312 now sets `default=True`, and line 30 and lines 1642 to 1643 say the same thing.

**Fold decisions into ROADMAP.** The still-live entries from lines 2094 to 2196 and 3124 to 3160 (collections without eviction, the typed-token multi-token cap, the unmeasured KV cache, the modality-aware confidence default) move into [ROADMAP.md](ROADMAP.md). Drop the ones the audit superseded: collections eviction is now `DATA-02`, and the load-bar measurement belongs to the report's measurement programme.

**Cut the page.** Keep lines 1 to 73 (orientation, with line 30's Alternatives wording fixed), the Conventions block at 2050 to 2060, and a short pointer to the campaign. Delete the narrative at 74 to 2030, "Previously shipped" at 2031 to 2049, and "North star and backlog" at 3224 to 3233, which duplicates ROADMAP. Both entries numbered `0.` disappear with their sections, resolving the duplicate numbering. Also delete the `results/` rename note at 2201 to 2209: its stated cause is that `RESULTS_DIR` is relative to the process working directory, which `DATA-03` eliminated, so it is now a historical incident already cited in the audit report.

Update [AGENTS.md](AGENTS.md)'s session-end habit at lines 100 to 121, which currently describes maintaining a file that is about to stop existing in that form.

A small `tests/test_handoff_bounded.py` asserts the page stays under 200 lines. The failure mode this finding exists to fix is growth by accretion, and a bound is the only thing that prevents it recurring.

## 4. Documentation layout

Last, as its own commit, because it is a pure move and reads best reviewed as one. The root currently holds nine markdown files and this stage would otherwise add two more.

- Root keeps `README.md`, `AGENTS.md`, and `LICENSE`. All three are convention-bound: GitHub renders the first, agent tooling looks for the second at the root, and the third is expected there.
- `docs/` takes `ROADMAP.md`, `HANDOFF.md`, `TIGERSTYLE.md`, and `MANUAL_VERIFICATION.md`.
- `docs/audit/` takes the whole campaign: `AUDIT_BRIEF.md`, `AUDIT_REPORT.md`, `IMPLEMENTATION_BRIEF.md`, and `IMPLEMENTATION_LEDGER.md`. They are one self-contained body of work with a beginning and an end, and grouping them says so.

Moving `HANDOFF.md` off the root is the one judgement call here. It is the file `AGENTS.md` sends you to second, so it stays discoverable through an explicit link, and putting it beside the roadmap it refers to is tidier than a crowded root. Easy to reverse if it reads wrong in practice.

Three things make this safe rather than risky:

- Every move is `git mv`, so history follows the file. That matters because META-01 deletes two thousand lines on the promise that git remembers them.
- `tests/test_docs_links.py` from META-02 already exists by this point and fails on any reference that no longer resolves, which is the whole risk of the move.
- Nothing outside markdown depends on these paths. The only reference is a comment at `tests/inference/test_ar_sampler.py:954`, and `pyproject.toml` mentions `README.md`, which is not moving.

`AUDIT_REPORT.md` is declared immutable by the brief. Moving it changes no byte of its content and no line number, so the `AUDIT_REPORT.md:1829-1925` style citations throughout the ledger keep pointing at the same text; only the path prefix updates.

`.cursor/plans/` stays where it is rather than moving to `docs/plans/`. Cursor writes plan files there itself, so relocating them would mean copying by hand after every planning session. Tracking it in place, as META-02 does, is enough to make the ROADMAP references true.

## Verification

Per commit: `.venv/bin/python -m pytest` (346 passing now), `.venv/bin/python -m ruff check src tests` against the ratchet rather than the bare count, `node --check` and `node --test tests/web/static/*.test.js` if any JS changes, and ReadLints.

The one thing needing you is META-02's clone test, and it now covers the layout move too: `git clone` this repository into an empty directory with no Cursor state, then confirm every mandatory command, environment boundary, and model constraint is discoverable from tracked files and that the build plans are actually present. `tests/test_docs_links.py` automates the dead-link half, but "can a cold contributor work from this" is a judgement call.

## Ledger

Updated in the same commit as each change, as always. Deviations to record: QUALITY-02's untouched complexity findings and why, the 36% concentration in `llada_sampler.py`, META-02's discovery that TigerStyle was never in `.cursor/rules/` at all, and META-01's finding that the `results/` trap was already fixed by DATA-03. The layout move is not a finding, so it gets a line in the stage summary rather than a deviation entry.

After this stage, stage 3 opens with `ORG-01`, which `DATA-03` already unblocked.