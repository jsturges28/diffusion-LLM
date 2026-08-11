# IMPLEMENTATION_BRIEF: working the audit findings

This brief governs every session that implements `docs/audit/AUDIT_REPORT.md`, not just
the first. The report calls them remediation sessions. There are 40 findings
across seven sequenced stages, so this is a campaign rather than a task, and
the thing that makes a campaign work is that each session starts from an
accurate picture and leaves one behind.

Three files carry the work, and they have different jobs:

- **`docs/audit/AUDIT_REPORT.md`** is the analysis of record, dated 2026-08-10. Treat it
  as immutable. Do not edit it to reflect progress or to correct it.
- **`docs/audit/IMPLEMENTATION_LEDGER.md`** is the state: what is done, what is ready,
  what is blocked, what needs hardware, and where the report has since been
  found wrong. Every session updates it.
- **This brief** is the rhythm: how to pick work, how to work it, and what
  outranks your judgement.

## Starting a session

1. Read this brief and the ledger's **Ready now** section. Do not read the
   whole report; it is 2,011 lines and most of it is not your problem today.
2. Read in full only the findings you intend to take. Each one carries its own
   Direction, Blast radius, and Verification, which is what you implement
   against.
3. For orientation, read `docs/HANDOFF.md`. `META-01` has since reduced it to a
   bounded cold-start page, so the caveat that used to sit here, telling you
   to read its first 73 lines and skip the rest, no longer applies.
4. Tell the maintainer what you propose to take and why, and deliberate before
   planning. This is the repo's standing cadence and it matters more here than
   usual, because several findings have a fork in them that is the
   maintainer's to settle.

## The approval rhythm

The maintainer greenlights a **stage**, not a commit. Deliberate the stage's
finding set and its internal order, plan it, and once that is approved,
implement, verify, and commit each finding in it without stopping for
per-commit approval. Report back at the stage boundary with what landed, what
is waiting on hardware, and what the next stage now looks like.

Three things still interrupt a greenlit stage, and none of them are
negotiable:

- Anything on the **Stop and ask** list below. Approving a stage approves the
  plan; it does not license you to decide the questions the plan deferred.
- **A finding that turns out to be wrong**, or whose Direction does not
  survive contact with the code. Record it under Deviations in the ledger and
  raise it, rather than reinterpreting it alone.
- **A Verification clause you cannot satisfy.** Do not lower the bar to keep
  moving. Hand back with what is blocking it.

Stages 3 and 4 are long and carry several **L** findings. If your budget thins
mid-stage, stop at a finding boundary with the tree working and the ledger
accurate, rather than leaving a boundary migration half-applied.

## Working one finding

The report's **Direction** is a direction, not a design. It names the shape of
a fix and the alternative that was rejected, which is enough to plan from and
not enough to skip planning. The deliberate, Plan, Agent cadence from
`AGENTS.md` applies at the stage level, so a stage plan should be concrete
enough that each finding inside it does not need a fresh round of approval.

The finding's **Verification** clause is the acceptance criterion. Most of them
describe tests that do not exist yet, and writing those tests is part of the
finding, not a follow-up. Per `QUALITY-01`, land test fixtures with the seam
they cover rather than building a test-only prelude that no production code
owns yet.

Commit granularity stays fine even though approval is coarse: one finding per
commit where the finding allows it, so a greenlit stage reads back as a
reviewable sequence rather than one opaque lump. The findings marked **L** are
staged boundary migrations and will need several commits; make each one a step
that leaves the tree working, and keep the finding **in progress** in the
ledger until its Verification clause passes end to end.

**Update the ledger in the same commit as the change it describes.** If the
two are split, the state and the tree can disagree, which is the failure mode
this whole campaign exists to remove.

## The rules that outrank your judgement

These come from the report's sequencing section (`AUDIT_REPORT.md:1829-1925`),
which is the dependency order the audit derived. The ledger's Blocked-by
column is a mechanical restatement of it; where they disagree, the report
wins and you should fix the ledger.

Do not start a finding whose blockers are not **done**. Do not opportunistically
fix an adjacent finding because you are already in the file; note it in the
ledger and leave it. And do not combine these, quoting the report directly:

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

## Stop and ask

Bring these to the maintainer rather than choosing:

- **Anything that changes the on-disk format of saved runs**, relocates the
  data root, or migrates existing runs. Users have real runs on disk.
- **Any change to a default a user can see**, including model parameters,
  since a changed default silently reinterprets old comparisons.
- **The forks the report leaves open.** `DATA-02` explicitly offers two paths,
  server-authoritative semantic operations or a revision and ETag scheme with
  stale rejection, and the sequencing says it should not start until one is
  settled. Treat any finding whose Direction contains "or" the same way.
- **Anything that would make a rollback hard.** Prefer the reversible half
  first, always.

## When the report is wrong

The findings are analysis, not scripture, and they were produced without
running the app. If implementation shows a finding is mistaken, incomplete, or
that its Direction does not survive contact with the code, do not quietly
deviate and do not edit the report. Record what you learned in the ledger's
**Deviations** section under that finding's ID, and raise it with the
maintainer. A finding that turns out to be wrong is a useful result; a finding
that was silently reinterpreted is a lie in the record.

## Verification before handing back

- `.venv/bin/python -m pytest`, currently 265 passing.
- `.venv/bin/python -m ruff check src tests`, currently 156 known findings.
  Once `QUALITY-02` installs the ratchet, that number may only go down.
- `node --check` on every changed `.js` file, and ReadLints on everything you
  touched.
- The finding's own Verification clause.

**Hardware cannot be exercised here**: no GPU, no display. When a finding's
verification needs real inference or a real window, the automated half still
has to pass, and the hardware half goes into the ledger's hardware queue along
with a manual checklist in your handback. A finding sits at **needs hardware**
until the maintainer clears it. That status does not block unrelated work, but
it does block anything the ledger lists as depending on it.

The report's own measurement programme is at `AUDIT_REPORT.md:1927-2011`. Those
are for the maintainer, and several of them are worth running before the
findings they inform, particularly the streaming resource curve behind
`RUNTIME-01` and the worker switch timings behind `LIFE-02` and `LIFE-05`.

## Documentation, deliberately restrained

`AGENTS.md` asks each session to update `docs/HANDOFF.md`, `README.md`, and
`docs/ROADMAP.md` as work lands. For this campaign the ledger carries the state
instead, and the standing habit is narrowed to three cases:

- Update `docs/HANDOFF.md`'s orientation only when the architecture it describes
  actually changes.
- Update `README.md` or `docs/ROADMAP.md` only when a finding changes something
  user-visible or a decision they record.
- Update the in-app About and Help copy when a user would notice the change.

This restraint is itself a finding. `META-03` observes that adding a shipped
bullet to every document after every commit is the process that produced the
drift the audit found, so forty findings must not become forty rounds of
documentation churn.

## Out of scope

The report's non-recommendations stand, and re-litigating them costs a session
(`AUDIT_REPORT.md:53-57`): no frontend framework, no bundler, no database as
the authoritative run store, no universal sampler, no mass formatting, and no
OS portability work. Native ES modules, a rebuildable summary index if the
measurements justify one, shared wire builders, and model-specific numerical
loops are the cheaper reversible path.

New features wait for their gating findings. Mamba, multimodal input, and
diffusion entropy and top-k are all stage 6 or later, and every one of them
has prerequisites in the ledger.

## Ending a session

Update the ledger, then hand back with four things: what landed and under
which commits, what became ready as a result, what is waiting on hardware, and
where you stopped. If you left a finding half-migrated, say exactly which step
is next, because the next session will be cold.
