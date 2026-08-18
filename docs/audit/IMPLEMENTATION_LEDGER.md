# IMPLEMENTATION_LEDGER: state of the audit remediation

State for the 40 findings in `docs/audit/AUDIT_REPORT.md`. The report is the
immutable analysis; this file is the moving part. Read
`docs/audit/IMPLEMENTATION_BRIEF.md` for how to work a finding, and update
this file in the same commit as the change it describes.

**Stage 1 is complete and verified on hardware.** All five isolated safety
fixes landed; the maintainer cleared the whole validation queue on 2026-08-11.
That pass also turned up an unrelated offline model-loading gap, recorded
under `TRUST-03`, whose availability half was pulled forward as its own
commit.

**Stage 2 is complete.** `QUALITY-02`, `META-02`, and `META-01`, in that
order, plus a documentation layout move that was not a finding: the reference
documents now live under `docs/` and this campaign's four files under
`docs/audit/`, leaving only `README.md`, `AGENTS.md`, and `LICENSE` at the
root. Every move was a `git mv`, and `AUDIT_REPORT.md` moved byte for byte, so
its line citations still resolve and its immutability holds.

**Stage 3 is complete.** Pass one landed `ORG-01` and `DATA-01`: the run
store is its own dependency-light module and a saved run now publishes whole
or not at all. The parameter-key XSS was pulled forward from `DATA-05` as a
standalone commit. Pass two landed `DATA-05`, `DATA-04`, and `RUNTIME-02`,
whose decisions were settled with the maintainer before pass one began: no
migration of the existing corpus, `history.txt` demoted to a human artifact
with `frames.jsonl` as the machine format, and `DATA-04`'s provenance
envelope without the validation token that `LIFE-03` now owns. The three
analytics findings it unlocked are unstarted and belong to no stage of their
own; see Ready now.

**Stage 4's three passes have all landed and cleared hardware.** Pass three
took `LIFE-01` and `PROTOCOL-01` together, in three commits, because the
envelope one defines is what carries the token the other issues. It also
pulled forward the half of `DATA-02` that needs no fork settled, and closed
the `QUALITY-01` gap those two findings sat on: the worker's message loop had
no tests at all. What remains of stage 4 is `XAI-01` and `LIFE-04`, both now
ready, plus `TRUST-04` behind `LIFE-04`.

**Stage 5 has started, and its first half is verified.** Clearing the
hardware queue on 2026-08-17 released `ORG-02`, whose state core landed in
four commits, the aligned frame family, the pre-edit baseline, the phase
table and the model API client, and cleared hardware on 2026-08-18. Three
pieces of that finding remain: the download API client, the native ES
module conversion its Direction asks for, and the server-rendered boot
state that would retire the loading overlay.

Testing it turned up a save bug the audit had missed, fixed in three
commits and recorded under Deviations. One item, 148, was reclassified as
unreachable on this hardware rather than left pending, because its scenario
needs two models resident at once on a card that cannot hold both.

Baselines: 793 tests passing (from 265 at the campaign's start), 145
browser tests under `node --test`, and Ruff at 128 in `src tests`, gated per
file and per rule by `scripts/lint_ratchet.py` rather than remembered.

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

Five findings have no unmet blockers.

**The rest of stage 4**, two of them.

- **XAI-01** (high, M), released by `LIFE-01`: preserve complete
  intervention checkpoints rather than only token IDs. Run ownership being
  explicit is the condition the report attaches to it.
- **TRUST-04** (medium, L), released by `LIFE-04`: move downloads into the
  same owned-operation model as activation, now that a long-running
  operation can be cancelled and its disconnect is bounded.

`LIFE-04` was on this list and is done; see its entry below. `ORG-02` was
on it too and is now in the hardware queue instead: its state core is
written and tested, but none of its callers has been run.

**The analytics trio**, unlocked by stage 3 and still unstarted. None of it
touches the worker protocol, so it is the group that cannot collide with
anything above.

- **ANALYTICS-02** (high, M): exact token summaries. See the decision note
  below; its position in the order is the open question, not its readiness.
- **ANALYTICS-03** (medium, L): lightweight pagination.
- **ANALYTICS-04** (high, M): the guarded compare boundary.

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

All four stage 1 entries were cleared on 2026-08-11, as was `DATA-01`: the
maintainer confirmed a fresh save reaching Analytics with its GIF and a guided
edit through Confirm exercising the compare-and-swap replacement. What each
showed is recorded under Deviations. `DATA-04`, `DATA-05` and `RUNTIME-02`
were cleared on 2026-08-11 in the same sitting: the maintainer confirmed items
133 to 141 of `docs/MANUAL_VERIFICATION.md`, including the two the sandbox
could say least about, the amber invalid row's alignment against its
neighbours and the two-window model switch.

**As of 2026-08-18 two entries remain**, and neither blocks anything.
The stage 4 findings cleared on 2026-08-17, which released `XAI-01`,
`LIFE-04` and `ORG-02`; `ORG-02`'s own state core and the save work that
came out of testing it cleared on 2026-08-18, items 162 to 166. What is
left is `TRUST-03`'s offline retest and `LIFE-02`'s two staged-failure
items, 143 and 144, which are awkward to arrange rather than pending.

`ORG-02`'s browser-smoke clause is recorded as unmet rather than queued:
satisfying it in the sandbox would mean taking jsdom as the project's
first JavaScript dependency, and items 162 and 163 cover the same ground
on real hardware.

- **LIFE-02 and LIFE-06**: partly cleared on 2026-08-14. The maintainer
  confirmed items 142 and 145 of `docs/MANUAL_VERIFICATION.md`, which between
  them carry both findings' user-facing claim: a refused switch leaves the
  loaded model and its run alone, and cancelling a load frees the worker.
  Items 143 and 144 (a worker that fails *after* spawning is terminated, and
  the redirect that follows) are open, and are awkward for a reason worth
  recording: `LIFE-06` now catches every cheap way of breaking a model before
  a worker is ever spawned, so the failure mode `LIFE-02` fixes has to be
  staged deliberately. The lever is in item 143. Item 146 was reclassified as
  a log-watch note, since a process wedged in the kernel is not something to
  arrange on purpose.
- **LIFE-03**: cleared, with one item unreachable rather than pending. Items
  147 and 150 were confirmed on 2026-08-14 and 149 on 2026-08-17, which
  between them carry the finding's user-facing claim: one window does not
  navigate for another's load, a window whose model is taken saves its run
  and reloads, and the ordinary path is untouched by the handshake.
  Item 148 cannot be staged on this hardware and is recorded that way rather
  than left open. It needs two models loading at once, and `LIFE-06`'s
  pre-eviction check correctly refuses the second on a card that cannot hold
  both; LLaDA and SmolLM3 together come to roughly the whole 23.49 GiB. The
  same fence is covered automatically in
  `tests/web/test_activation_identity.py`, where a cancel naming another
  operation is refused, and the maintainer met the behaviour by accident
  during the `LIFE-05` incident. A larger card makes the item live again.
- **LIFE-05 (single-instance slice)**: the probe and the launch decision are
  tested against real HTTP servers on ephemeral ports, and mutation-checked
  (removing the guard fails three cases in under three seconds rather than
  hanging, which the first version of those tests did). Item 153, the one
  that matters because it is the accident that produced the OOM, was
  confirmed on 2026-08-15: a second launch from the icon opens no second
  window. Item 154, the fallback when an unrelated process holds 8760, was
  confirmed on 2026-08-17. The slice is cleared; the host-level lease it
  deliberately left out is still deferred, and is described under Deviations.
- **LIFE-01, PROTOCOL-01 and the DATA-02 slice**: cleared on 2026-08-17,
  items 157 to 161. The ordinary single-window path is undisturbed, a run
  survives a reload and stays editable, a second window's generation refuses
  the first window's resume, a refused probe leaves What If open, and a
  collection filed just before a navigation is still there afterwards.
  Two of those took three attempts, and both were the items' fault rather
  than the code's. The reasons are now written into 159 and 160, because
  each would otherwise cost the next session the same hour: a second window
  comes from a browser rather than a second launch, which item 153 makes
  counter-intuitive, and a typed token that matches a captured candidate is
  answered from the run's own record without a probe ever being sent, so the
  refusal under test never happens.
- **TRUST-03 (offline slice)**: the automated half asserts that both Hub
  workers pin every `from_pretrained` call to local files, and that being
  offline with nothing cached now reports what happened instead of a urllib3
  retry dump. Outstanding is the case that found it: turn networking off and
  activate LLaDA and SmolLM3, both of which are already downloaded. Both
  should load at their usual speed with no hang and no error.
  DiffusionGemma needs no retest; it never touched the Hub.

## Status table

| ID | Sev | Eff | Status | Blocked by | Commits |
|---|---|---|---|---|---|
| LIFE-07 | high | S | done | none | Commit LLaDA resume state only after the run lands |
| TRUST-01 | high | S | done | none | Bind to loopback unless exposure is asked for |
| ANALYTICS-01 | high | S | done | none | Fence detail responses to the run that asked |
| DATA-03 | medium | S | done | none | Resolve the data root without asking the cwd |
| TRUST-02 | high | M | done | none | Serve every page without a third-party origin |
| QUALITY-02 | medium | M | done | none | Ratchet the lint baseline instead of remembering it |
| META-01 | medium | M | done | none | Three commits: checklist out, decisions to ROADMAP, page cut |
| META-02 | medium | S | done | none | Put the agent contract where a clone can read it |
| QUALITY-01 | medium | L | companion | lands with each seam | |
| ORG-01 | medium | M | done | none | Extract the run store out of the supervisor |
| DATA-01 | high | L | done | none | Publish saved runs whole or not at all |
| DATA-05 | high | L | done | none | Three commits: strict boundary, version and frame stream, invalid runs |
| DATA-04 | high | M | done | none | Persist run provenance from the run, not manager state |
| RUNTIME-02 | medium | M | done | none | Bound the GIF and label the model that produced it |
| ANALYTICS-02 | high | M | ready | DATA-05 (done), see decision above | |
| ANALYTICS-03 | medium | L | ready | DATA-01, DATA-05 (both done) | Gained evidence: 211 dirs scanned per load, plus ~10s to paint one long run |
| ANALYTICS-04 | high | M | ready | DATA-01 (done) | |
| LIFE-02 | high | M | needs hardware | none | Two commits: the process seam, then verified termination |
| LIFE-06 | medium | M | needs hardware | none | Validate a switch target before evicting the working model |
| ORG-04 | medium | S | done | none | Two commits: the shared activation client, then the menu |
| LIFE-03 | critical | L | done | none | Two commits: operation identity, then the resident mismatch |
| LIFE-01 | high | M | done | none | Name every run and refuse a stateful request that means another |
| PROTOCOL-01 | medium | M | done | none | Two commits: scoped error envelopes, then the client routing |
| XAI-01 | high | M | ready | LIFE-01 (done) | |
| LIFE-04 | high | L | done | LIFE-03 (done) | Carried RUNTIME-01's queue bound, as its own Direction asks |
| LIFE-05 | high | M | partial | none | Single-instance the desktop launcher; host lease deferred, see Deviations |
| TRUST-04 | medium | L | ready | LIFE-04 (done) | |
| DATA-02 | high | L | partial | maintainer decision | Lost-update slice only; conflict semantics still forked |
| RUNTIME-01 | medium | L | partial | ORG-02 + DATA-05 | Queue bound landed with LIFE-04; append-only frames remain |
| ORG-02 | medium | L | partial | none | State core verified; download client, ES modules and server-rendered boot remain |
| RUNTIME-03 | medium | S | blocked | ORG-02, paired | |
| ROADMAP-01 | high | M | blocked | stage 6 order | |
| ROADMAP-05 | high | M | blocked | stage 6 order | |
| ROADMAP-02 | medium | M | blocked | stage 6 order | |
| TRUST-03 | high | L | blocked | stage 6 order | Offline slice only: Load cached weights without asking the Hub |
| DEPS-01 | medium | L | blocked | stage 6 order | |
| ROADMAP-03 | high | L | blocked | stage 6 order | |
| ORG-03 | medium | M | blocked | stage 6 order | |
| ROADMAP-04 | medium | L | blocked | DATA-01, stage 6 order | |
| META-03 | medium | M | deferred | milestone boundaries | |

## Stage map

Derived from `AUDIT_REPORT.md:1829-1925`. Each stage is a boundary to validate
before the next takes a dependency on it.

**Stage 1, isolated safety fixes. Done.** `LIFE-07` first, then loopback
binding (`TRUST-01`), an absolute configurable data root (`DATA-03`),
synchronous detail clearing with request epochs (`ANALYTICS-01`), and the
vendored Analytics dependencies (`TRUST-02`), each as its own commit.
`DATA-03` was moved ahead of `ANALYTICS-01` against the report's listing
order, deliberately, so the run-store stage would unblock early if the session
ran short. Three new test files, one new browser test harness, and a
`node --test` line in `AGENTS.md` came with it.

**Stage 2, gates before boundaries move. Done.** The path-specific and
code-specific Ruff ratchet (`QUALITY-02`), then the portable agent contract
(`META-02`), then the bounded cold-start page (`META-01`). The last two ran in
the opposite order to this map on purpose, so the contract's home was settled
before the page that points at it was rewritten. A documentation layout move
closed the stage. `QUALITY-01` fixtures still attach to seams as they are cut,
and `META-03` refreshes at milestone boundaries rather than after every
commit.

**Stage 3, the run-store boundary. Pass one done.** Behavior-preserving
extraction of the storage operations (`ORG-01`) and then unique staged
publication with complete replacement, revisions, and compare-and-swap
(`DATA-01`) have landed. Pass two has landed versioned validation and read
adapters (`DATA-05`, three commits) and immutable worker provenance threaded
through the terminal run contract (`DATA-04`), leaving GIFs as bounded
non-authoritative derivatives (`RUNTIME-02`), which publication semantics now
allow. `ANALYTICS-02`, `ANALYTICS-03` and `ANALYTICS-04` are unblocked;
`ROADMAP-04` still waits on its own stage.

**Stage 4, explicit process and socket ownership. Passes one and two done.**
Pass one extracted and tested the manager process adapter, made termination a
verified transition and put validation before eviction (`LIFE-02`, `LIFE-06`),
in three commits. Pass two shared activation orchestration behind one client
(`ORG-04`) and gave every activation an operation identity, with a resident
handshake on the socket (`LIFE-03`, the stage's only critical), in four. It is
much the largest stage, ten findings plus `DATA-02` in parallel, so it is being
worked in passes the way stage 3 was.

`LIFE-01` and `PROTOCOL-01` were deliberately held back from pass two. The
report says the three move together, but `LIFE-03` alone was four commits, and
both of the others build on the operation identity it establishes rather than
being needed by it. They are pass three. With run ownership explicit, preserve
complete intervention
checkpoints (`XAI-01`). Propagate disconnect and cancel through inference and
bounded queues (`LIFE-04`), which carries the first bounded-queue step of
`RUNTIME-01`. Add host-level ownership (`LIFE-05`) once one manager's
lifecycle is trustworthy, and move downloads into the same owned-operation
model (`TRUST-04`). `DATA-02` can run in parallel once its fork is settled,
and must not reuse model-operation state.

**Stage 5, frontend state around the settled protocol. Started.** The
aligned frame operations, the pre-edit baseline, the legal workflow phases
and the model API client are extracted and tested (`ORG-02`, four commits),
awaiting hardware. Three pieces of that finding remain: the download API
client, the native ES module conversion the Direction asks for, and the
server-rendered boot state that would retire the loading overlay. Then fix
the select lifecycle (`RUNTIME-03`) as shared controls gain module
ownership, and compact append-only streams (`RUNTIME-01`) only once the
reducer can reconstruct them and the run-store version can distinguish
them.

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
`docs/audit/AUDIT_REPORT.md`; it is the record of what was believed on 2026-08-10, and
the difference between that and what turned out to be true is worth keeping.

No finding has been contradicted so far. The entries below are things learned
while working one finding that belong to another, recorded rather than
opportunistically fixed.

### Stage 1 hardware pass, 2026-08-11

All four queued findings were confirmed by the maintainer in one sitting and
the queue is empty. Two results are worth keeping rather than just ticking:

- **LIFE-07 was stress tested past its clause.** The maintainer ran four
  consecutive Retry cycles on one run. Each Retry resumes from frame 0 against
  a history that the previous resume already replaced, so a commit that leaked
  a truncated list would have surfaced by the fourth pass as an out-of-range
  rejection or a branch from the wrong frame. It did not.
- **TRUST-02 passed, and found something else.** Every page rendered with the
  network physically off, webfont included. Model *activation* did not, and
  that turned out to be an unrelated pre-existing gap; see the TRUST-03 entry
  below, which is where the discovery is recorded.

`DATA-03` and `ANALYTICS-01` were confirmed exactly as written: four launch
contexts resolving to one directory with no stray `/tmp/results`, the delete
dialog naming `/tmp/isolated` under an alternate root, and the detail panel
holding one run's identity through throttled interleaving and closes.

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

### TRUST-01

**Deliberately narrowed, with the maintainer's agreement.** The Direction ends
with "and, before treating it as supported, an authentication and origin
policy". That is a design project, not a one-line default, and holding a high
severity one-line fix behind it would have been the wrong trade. What landed
is the safe default, an explicit `--host` that warns before serving, and
README copy that calls remote use a trusted-network convenience rather than a
supported deployment. Authentication and WebSocket origin policy remain
unaddressed and unclaimed; if remote access is ever to be supported, they need
a finding of their own.

**The Verification clause turned out to be fully automatable**, which the
report did not assume. `tests/test_main.py` binds real sockets and reaches for
them across this host's own network address, proving a loopback listener is
unreachable there while a `0.0.0.0` listener is reachable. The pair matters:
the unreachable half alone would also pass behind a firewall. Both skip on a
host with no non-loopback address, which includes the agent sandbox, so they
are worth re-running once on the maintainer's machine, where they were
confirmed to run and pass during this session.

### DATA-03

The Direction's "explicit `--results-dir` or environment setting" was settled
as both, with the flag winning, since the two serve different callers: the
flag is for a person, the variable is what reaches a desktop entry or a bare
`uvicorn src.web.server:app`.

**The resolver lives in its own module**, `src/web/data_root.py`, for an
ordering reason worth remembering. The supervisor resolves its root at import
time, so `main.py` has to write `--results-dir` into the environment *before*
importing the server; importing the server to learn the variable's name would
already be too late. A module that pulls in nothing breaks that cycle, and it
gives `ORG-01` somewhere obvious to grow the run store.

`desktop.py` keeps its `os.chdir(REPO_ROOT)`. The data root no longer needs
it, but worker subprocesses are still spawned relative to the process working
directory, so removing it would have been a lifecycle change smuggled into a
persistence fix. Its comment was corrected to say which of the two reasons
still applies.

**One UI correction came with it.** The Analytics delete confirmation built
its label from a hardcoded `"results/"`, which stops being true the moment the
root is configurable, and a dialog about permanent deletion is the worst place
to name the wrong directory. The resolved root now rides along on
`/api/analytics/system`, which that page already fetches.

**Noted, not fixed (adjacent).** `_compute_run_metrics` joins
`RESULTS_DIR / run_id` with no containment guard, unlike `_existing_run_dir`,
`_compute_run_frames`, and `_delete_run_blocking`, which all resolve and check
the parent. That is a pre-existing traversal gap and belongs to the run-store
stage rather than to making the root absolute.

### ANALYTICS-01

**Deliberately narrowed, with the maintainer's agreement.** The Direction adds
"where practical, fetch the pair together and publish one coherent detail
snapshot". A combined endpoint would touch the run-store read path that
`ORG-01` and `DATA-05` are about to take ownership of, so it was left out; the
epoch makes the two responses behave as one commit without changing the wire.
Worth reconsidering alongside `ANALYTICS-03`, which is already going to rework
how this page reads runs.

**The seam and the JS test harness are new ground**, so the conventions they
set are worth naming. `src/web/static/detail_requests.js` is a classic global
script following `overlays.js`, not an ES module: `analytics.js` is 5,586
lines of `var` globals loaded as a classic script, and converting it is
`ORG-02`'s job in stage 5. The test loads the shipped file into a `vm`
context, which gives top-level `var` the same become-a-global behavior a
browser gives it, so the production file needs no `module.exports` tail that
only tests would use. The same harness will work for `overlays.js` when
`QUALITY-01` reaches it. The command is
`node --test tests/web/static/*.test.js`, now recorded in `AGENTS.md`; a bare
directory argument does not work, since Node reads it as a module path.

**Noted, not fixed (adjacent).** The compare panel has the same race and none
of the fix: `fetchCompare` has no abort and no epoch, and `showComparison`
destroys `chartCompareConv` inside its `.then` rather than before the fetch.
It shares no state with the detail panel, so the fence did not need to cover
it, and comparison as a bounded coherent transaction is exactly what
`ANALYTICS-04` is for. What did land here is the other direction: leaving the
detail view for compare now retires the detail requests, which it did not
before.

**Also noted.** `fetchRuns`, `fetchCompare`, and `fetchSystemInfo` still skip
`response.ok` and still have no rejection handler; only the two detail fetches
were centralized, because those are the ones this finding is about.

### TRUST-02

**The assets were fetchable after all.** The plan assumed they would not be,
since the npm registry fails at the agent sandbox's proxy, and budgeted for
handing the maintainer a list of `curl` commands. jsDelivr, Google Fonts, and
raw.githubusercontent are reachable with network permission, so everything is
vendored and committed. Worth knowing for `TRUST-03`, which has to fetch far
larger artifacts: the blocked path is npm specifically, not the network.

`scripts/vendor_assets.py` is how they got there and how they get bumped. It
writes `vendor/README.md` with each file's source URL, byte count, and
SHA-256, so a version bump reviews as a diff rather than as an act of faith.
That manifest is deliberately the shape `TRUST-03` will want for model
artifacts.

**The font is one file per subset, not per weight.** JetBrains Mono is a
variable font, so Google returns the same woff2 for weights 300, 400, and 500
and varies only the descriptor. Downloading per weight wrote three
byte-identical copies of each subset, 253 KiB where 86 KiB does the same job.
All six subsets are kept, including Cyrillic and Greek, because the models can
emit them and `unicode-range` means a page still loads only what it needs. The
`--font-mono` variable in `style.css` was left alone; only the `@font-face`
source changed.

**The feature detection was the substantive code change.** `analytics.js`
dereferenced `Chart` at the top level in three places, so a missing library
did not degrade the charts, it stopped the file parsing and took the run
table, metadata, overlays, and deletion with it. Those are guarded now and the
chart renderers return early, so the page survives a library that fails to
load even though vendoring should mean it never does.

**Noted, not fixed (adjacent).** The vendored scripts carry no Subresource
Integrity attributes. They are same-origin now, so SRI would guard against
nothing an attacker who can write to the repository could not also change; the
manifest hashes are the meaningful record. Revisit only if these ever move
back to a remote origin.

### TRUST-03

**Found early, by TRUST-02's offline test rather than by working this
finding.** With the network physically off, every page rendered but two of the
three models would not activate. SmolLM3 failed loudly with a
`NameResolutionError` for `huggingface.co` while fetching
`additional_chat_templates`; LLaDA hung at "loading weights 0%" for over two
minutes instead of erroring. DiffusionGemma loaded normally.

The asymmetry explains the cause exactly. DiffusionGemma's checkpoint is a
local directory (`registry.py:127`), so it never contacts the Hub. The other
two are Hub repo ids, and while `download_with_progress` already answers the
"is it cached" question correctly and returns a local snapshot path with no
network (`hf_download.py:152-153`), both workers then discard that path for
loading and hand `from_pretrained` the *repo id* with no `local_files_only`.
Transformers therefore goes back to the Hub to revalidate a model that is
already fully on disk. The two fail differently only because of their
environments: 4.53+ in `.venv-ar` makes an API call that fails fast, while
4.38.2 plus `trust_remote_code=True` retries with long timeouts.

This is squarely inside this finding's Direction, which already says to "load
from the resolved snapshot" (`AUDIT_REPORT.md:896-898`). But the finding is
stage 6 and is mostly about reproducibility, while what the offline test
exposed is availability, so the maintainer chose to pull the availability half
forward as its own commit. **TRUST-03 stays blocked** for everything else it
asks for: pinned source revisions, persisted weight digests, cache-space
preflight, and a completion manifest for the local quantized artifact.

**Why the early slice used `local_files_only=True` rather than the snapshot
path.** Passing the resolved snapshot path is the more thorough fix and is
what this finding will eventually want, because the path names the exact
commit. It also changes `tokenizer.name_or_path`, which
`describe_tokenizer` writes into saved-run provenance and the Analytics detail
panel, so runs saved after it would display a long cache path where earlier
runs show `GSAI-ML/LLaDA-8B-Instruct`. That is a user-visible saved-run change
and belongs with the versioning this finding will do properly. The flag buys
the offline behavior with none of it.

**What the slice landed** (commit "Load cached weights without asking the
Hub"): `local_files_only=True` on all four `from_pretrained` calls across
`llada_worker` and `smollm3_worker`, placed after `download_with_progress`
returns, since a successful return already means every file is present. Plus
`WeightsUnavailableError` in `hf_download`, raised only when the cause chain
actually shows a connectivity failure, so a 403 or a full disk keeps its own
message rather than being relabelled as an offline problem.

**Still open for this finding**, and unchanged by the slice: pinned source
revisions in the registry, resolved revisions and weight digests in run
provenance, cache-space preflight against remaining bytes, and a completion
manifest for the locally quantized DiffusionGemma artifact. The slice deals
with availability only.

### LIFE-06

**The VRAM check cannot move, so it was split rather than moved.**
The finding asks for validation before eviction, and three of the
four checks move cleanly. The fourth cannot: the real reading is only
meaningful once the resident worker's memory has come back, which is
what `_preflight_vram`'s settle loop waits for. So there are now two,
answering different questions. Before eviction, a non-destructive
estimate counting the resident model's VRAM as reclaimable, which
refuses only the case that was already hopeless. After eviction, the
existing check, unchanged and still authoritative.

**Device support needed a registry field, which overlaps
`ROADMAP-01`.** DiffusionGemma has always refused a non-CUDA device,
but from inside `load()`, after the previous model was evicted for
it. The supervisor had no way to know. `ModelCapabilities` gains
`supported_devices`, defaulting to both, and the maintainer agreed to
the minimal version rather than deferring the check: `LIFE-06`'s
Verification names "unsupported device" as a case that must leave the
resident worker usable, so skipping it would have meant a Verification
clause knowingly unmet. `ROADMAP-01` will fold it into a proper
device-support axis.

**A refusal is not a fault.** With validation in front, a switch to
an impossible target returns an ordinary answer, but the route's
generic handler turned it into a 500 with a stack trace in the log.
`ActivationRefused` (a `RuntimeError`, so existing handlers still
catch it) gets a 409 and a single log line, which keeps real faults
findable.

**The client half was half the bug.** Both entry paths discarded the
run before the POST, so a refusal the server now makes cheaply would
still have cost the user the run on screen. The discard moved into
the branch where a worker is actually ready. The menu's
re-select-the-resident-model case had different semantics already (it
spawns nothing, so its run must survive) and that is carried on the
selection rather than recomputed.

**The honest `/generate` gate needed somewhere to land.** `LIFE-02`
made a failed load stop satisfying the gate, which means a redirect
to a menu that never read activation state and would have bounced the
user with no explanation. The menu now reads it once on arrival, and
stays quiet during a selection it is already reporting on.

### ORG-02

**The generator paints twice, and the loading overlay is a curtain
over the second paint.** Found by removing the curtain and watching
what it had been hiding, which is a more concrete account of the
problem than the finding currently carries, so it is worth writing
down before stage 5 opens.

Every navigation is a full document load, and the served HTML knows
nothing about the session. So the browser paints a skeleton, and
then `boot()` fetches `/api/models` and rebuilds the page into its
real shape. Concretely:

- `buildParamPanel` starts with `paramFields.innerHTML = ""` and
  fills the whole hyperparameter column from `param_specs`, after the
  fetch resolves. That is the largest movement on the page.
- `index.html` carries 45 `hidden` occurrences, around twenty of them
  elements whose visibility is decided by runtime state: the
  scrubber, the overlay picker, the thinking panel, the edit
  controls, the entropy row.
- `restoreSessionState` reads `sessionStorage`, which is synchronous,
  but runs inside the same `.then()` because it needs `activeModelId`
  from the fetch.

None of this is slow, and none of it is untidy code. It is what
happens when the page's opening state lives behind a request instead
of in the document.

**The fix that removes it rather than hides it** is rendering that
opening state into the HTML at serve time, so first paint is already
correct. No framework and no bundler, which the report rules out
anyway, and the machinery exists: `_serve_stamped_page` already
rewrites the HTML on its way out to stamp asset versions. That is
`ORG-02`'s work, because the run-session core it describes is exactly
what would own the state being rendered.

**Recorded because I got it wrong once.** The overlay was removed on
the argument that `LIFE-02`'s `/generate` gate proves a model is
already serving, so nothing is waiting for a load. True, and it
answered only the job the overlay is named after. The maintainer saw
the page visibly assemble and asked for the curtain back, which is
where it now is. A future session should not repeat the removal
without doing the server-render first; the comment in `index.html`
next to the overlay says so too.

**Four things were kept from the detour**, all the same shape:
removing a source of movement rather than covering it, which is what
the server-render should eventually do wholesale.

- `#scrubber-section` holds its height when idle, so a finished run
  no longer resizes the canvas above it.
- `#prompt-context` holds its line before the first token count
  arrives. Only a ready worker can produce that count, so the whole
  column below the prompt box used to drop a step on every load.
- `.analytics-new-dot` keeps a two-digit slot on both pages that
  carry it. It sits inside a header link with more links to its
  right, so a badge appearing from nothing slid all of them across.

The fourth is the interesting one, because it says where this
technique stops. `#entropy-profile-row` is reserved **only when the
resident model reports entropy**, which today means the
autoregressive one. Holding it unconditionally would put a permanent
empty strip under every diffusion run: a cost paid on every frame to
avoid a shift paid once. So the rule is not "reserve everything that
can appear" but "reserve what will appear for this model", and the
markup starts the row absent because before `/api/models` answers
there is no model to ask. `setEntropyProfileVisible` owns all three
states and `boot()` settles it once the model is known.

That distinction outlives the curtain. When boot state is rendered
into the HTML at serve time, the server will know which model is
resident, so the conditional part becomes a question the template
can answer directly rather than one the client re-answers after a
fetch. `ROADMAP-03` would bring entropy to the diffusion models and
make the predicate itself wrong; it is a one-line change in
`setEntropyProfileVisible`, flagged in the comment there.

### LIFE-05

**Confirmed on hardware, by accident, on 2026-08-14.** The
maintainer opened two windows from the desktop launcher to test
`LIFE-03` and got a CUDA out-of-memory instead: one worker holding
15.31 GiB while a second tried to take 6.17 more from a 23.49 GiB
card. That is the finding reproduced, and more convincingly than the
synchronised-preflight experiment the report proposes, because it
came from doing something ordinary.

The mechanism is `desktop.py:_resolve_port`. A second launch finds
port 8760 taken and deliberately falls back to an ephemeral one, so
there are two supervisors, two `ModelManager`s, and two enforcements
of "one resident model" over a GPU neither knows it shares.
`LIFE-03` cannot help, and was never meant to: it fences operations
inside one supervisor. It also explains why the maintainer's Cancel
worked when item 148 expected a refusal, since that window's
supervisor genuinely owned that load.

**Only the cheap half was taken, deliberately.** A second desktop
launch now identifies the running instance and stands down instead of
starting a rival supervisor, which removes the accident. The rest of
the finding, a host-level lease covering the browser supervisor and a
manually launched `main.py` too, is deferred.

The reasoning, which is a partial disagreement with the report's
severity. `LIFE-05` is rated high partly on invariant purity: "the
product's central resource invariant no longer matches what the
machine is doing." For a single-user local tool the residual failure
after this change is loud, self-inflicted and recoverable by closing
a window, which is a much weaker case than a silent wrong answer. It
is worth contrasting with `LIFE-03`, rated critical and taken in
full, whose failure is a page generating against a model it was not
configured for and producing overlays that look right. In an
explainability tool, quiet and plausible beats loud and obvious as a
thing to spend effort on.

What remains open, for whoever picks it up: browser and desktop
running together, and a deliberately launched `main.py` alongside
either. Both are two-step acts rather than a slip, which is why they
wait.

**Raising the existing window is best-effort and the code says so.**
Activating another process's window is the window manager's to allow
and generally refused under Wayland, so `focus_running_window` tries
`wmctrl` then `xdotool` and shrugs. The guarantee is that no second
supervisor starts; the printed message is the part that always works.

### DATA-02

**The half that needed no decision was taken first.** The finding's
Direction is a fork, server-authoritative collection operations or a
revision and ETag scheme, and the sequencing says it should not start
until one is chosen. But three of the things it describes are lost
updates that neither branch would settle differently, so they were
pulled forward the way the parameter-key XSS was pulled out of
`DATA-05`. The fork itself is untouched and still the maintainer's.

**Two supervisors could overwrite each other, and now cannot.** The
lock in `ui_state.py` was a `threading.Lock`, which is process-local
while the browser entry point and the desktop app are two processes
pointed at one results directory. It is now paired with an `flock`.

The lock is taken on a sidecar, `ui_state.lock`, not on the state
file, and that detail is the whole thing: writes go through
`os.replace`, which swaps in a new inode, so a lock held on the file
being replaced stops excluding anyone the moment the first writer
finishes. The sidecar is only ever opened.

**A read-modify-write was holding the lock for half of itself.**
`GET /api/ui-state` loaded a snapshot, then both reconcilers computed
a pruned value from it and called `set_ui_state_key`, which takes the
lock only for the write. A PUT landing in that gap was overwritten by
a value computed before it existed. `mutate_ui_state_key` now runs
the read, the transform and the write under one hold, and both
reconcilers use it.

**Tested by racing, not by inspection.** A lock taken over the wrong
span looks identical from the outside, so the tests run sixteen
threads and eight forked processes appending to one key and assert
nothing is lost. Both mutants confirm it: reverting the helper to
read-then-set fails three cases, and removing the `flock` fails
exactly the process one and nothing else.

**Adjacent, noted, not fixed.** `run_store.py:353-356` has the
identical process-local lock with a comment deferring the same
problem. It was left alone; one supervisor still owns the run
directory in practice, and widening this commit into the run store
would have mixed two boundaries.

**The client half needed no second window either.** The report frames
`DATA-02` around two clients racing, but the write path lost changes
with one. `persistSet` debounced every PUT by 250 ms and nothing
flushed on the way out, so filing a run and navigating inside that
window discarded the timer with the document, and the next page's
hydrate wrote the older server copy back over it.

Collections now skip the debounce entirely rather than being flushed
more carefully. The debounce exists to coalesce streams of writes,
and collections do not arrive in streams: a star click, a rename, a
delete, a checkbox. The keys that genuinely do stream keep it, and
gain a flush on `visibilitychange` and `pagehide`, with `keepalive`
set only when the body fits the small shared budget that flag draws
on. Two of these keys may reach 262,144 characters, and a keepalive
request over the budget is refused outright rather than slowed.

**A failed write used to look like a successful one.** `persistPutKey`
caught rejections into an empty block and never read the status, so a
4xx resolved as success. Since `persistSet` writes localStorage
first, the tab kept showing the change as saved and the value only
disappeared later, in another window, with nothing connecting the two
events. Failures now reach a per-key handler; collections is the only
key that registers one, because it is the only one whose loss the
user cannot shrug off, and the message says the change is local to
that window rather than guessing at a cause.

Mutation-checked the same way as the store half: putting collections
back on the debounce, removing the flush, ignoring the status, and
setting `keepalive` unconditionally each fail the suite.

### QUALITY-01

**The message loop had no tests at all**, which is worth stating
plainly because it was not obvious: `tests/backends/` looks
well-covered, but every file in it calls a `handle_*` method
directly. Nothing reached `create_worker_app`'s dispatch, so the ready
handshake, the generation lock, the busy refusals, cancel and the
unknown-type reply were all unexercised.

That is where `LIFE-01` and `PROTOCOL-01` actually live. Both clauses
are about interleaving requests across two sockets, and both turn on
decisions the loop makes rather than a handler: whether a second
socket is refused as busy, and which run a stateful request may
reach. `tests/backends/test_worker_dispatch.py` covers it with a real
worker app and FastAPI's test client, which is the `QUALITY-01`
obligation attaching to this seam rather than a test-only prelude.

**Two things the tests found, both about the loop rather than the
findings.**

`cancel` cannot interrupt a generation. The loop awaits the handler
inline, so while a generation runs, the loop that would read a cancel
from that socket is reading nothing, and a cancel on a second socket
sets a different connection's event. The frontend never sends one
today, so nothing is broken in practice, but the message type reads
as though it works. That is `LIFE-04`'s ("propagate disconnect and
cancel through inference"), and the dispatch test that covers the
branch says so where the next reader will see it.

Parking a generation from a test needs `call_soon_threadsafe`. An
`asyncio.Event` set from the test thread does not reliably wake a
waiter in the loop's thread, and the failure is a hang rather than an
assertion. The first version of these tests hung the suite. They now
carry a five-second bound on every park, so a regression of that kind
fails in seconds, which is the same lesson the `LIFE-05` tests
recorded and the second time it has been learned here.

### ORG-02

**Two forks were settled before any code moved**, both against the
report's literal wording, both with the maintainer.

The Direction asks for native ES modules, and `ORG-04` says its
activation client "can begin as one explicitly namespaced classic
script and become a native module with `ORG-02`". The conversion is
deferred anyway. The pattern it would replace is not a stopgap: three
extracted modules carry 91 passing browser tests through a `vm`
harness that `AGENTS.md` documents, and converting means changing
load semantics on four pages plus the harness itself. The tested
state the finding is actually about is separable from the dependency
hygiene modules would buy, so it is being done first.

The Verification clause ends "a small browser smoke layer must prove
the module wiring and modal request cancellation against real DOM
events". That cannot run here: no display, and the repo has **zero
JavaScript dependencies**, no `package.json` and no `node_modules`.
Satisfying it means jsdom, which would be the project's first JS
dependency, or a browser binary. It is handed to the maintainer as
manual items instead, the same way every GPU and display clause in
this campaign has been.

**The first module is the frame family.** Six arrays indexed by frame
were declared separately and enumerated by hand at nine sites:
appended, frozen into the original-run copy, snapshotted, restored,
truncated, cleared, projected into the save payload, serialised, and
read back. `run_frames.js` owns all nine, and a source-inspection
test asserts that no bare `frameHistory` and friends survive, because
a family one call site can still take apart is not a family.

**The invariant is not "all six are equal", which is what makes it
worth writing down.** A session snapshot deliberately drops three of
them when localStorage refuses the full payload, so what holds is
that each array is either empty or exactly as long as `history`. A
degraded restore is then repaired by the first truncate, which sets
every array to the same length whether or not it had one. That
behaviour was load-bearing and documented only in a comment on the
restore path; it is now a test.

**Checked on the way in, not just on the way out.** Every operation
writes all six, so a postcondition can only catch a bug inside the
module. The precondition catches a caller that reached past it and
changed one array alone, which is what every call site used to do,
and without it the next truncate would paper over the damage by
squaring all six up to one length. The run would then carry frames
whose token detail belonged to different frames.

**One behaviour change, in a commit that is otherwise a move.** The
append site pushed to `perFrameElapsed` only when the worker sent an
`elapsed`, while its five siblings grew unconditionally, so a frame
without one desynchronised the family. It cannot arrive from a real
worker, because `FrameStreamer` stamps `elapsed` on everything it
forwards, so the fix carries the previous cumulative value and is
inert in practice. Recorded because "extract and test current
semantics" is the report's own instruction, and this is the one place
the extraction did not.

**Seven mutants, all caught**, including both historical bugs: an
append that skips one array, and a truncate that skips
`perFrameElapsed`, which is the one that knocked the Timing chart's x
axis out of step with every other chart.

**The baseline is its own family, deliberately.** The run as it was
before the first edit keeps four of the six arrays plus the candidate
sets, and not canvas index or reveal counts, because nothing compares
those and the session snapshot is already large enough to be refused
by the storage quota. Folding it into the live family would either
copy two arrays nobody reads or hide which ones are actually there.
Two rules the call sites used to carry moved with it: capturing is a
no-op once a baseline exists, so a branch cannot overwrite the run it
branched from, and a snapshot predating the frame count falls back to
the live run's length rather than reading as no baseline at all.

**The phase table is described, not reimplemented, and that was the
whole design decision.** Eight editing phases lived in a string that
ten call sites assigned directly, with nothing saying which were
reachable from where; the workflow existed only as the union of
whichever buttons happened to be enabled. `run_phases.js` owns the
transition table and refuses anything not in it.

What it deliberately does **not** own is the clearing. Each site
still resets its own companions exactly as it did, and the module
checks on arrival that it did. The tidier design would have moved
that too, and it would have been the wrong change here: the guided
edit flow needs a GPU, so none of this can be exercised in the
sandbox, and describing existing behaviour while refusing anything
that does not match it is the version whose failure mode is a loud
throw rather than a quietly different workflow.

One ordering did have to change. The site that lands back in `edit`
after a partial resume used to set the phase and then clear the
resume it had just finished; since consistency is checked on arrival,
the clear now comes first. Both are plain assignments with nothing
between them.

**The table was read off the call sites**, not imagined, and the four
workflows in `tests/web/static/run_phases.test.js` are transcriptions
of them: a diffusion edit to review, an edit-another loop, What If,
and Retry. Six mutants on the table, all caught, including making
Confirm reachable from a locked edit that was never generated.

**The model client turned out to be about agreement, not transport.**
Four pages fetched `/api/models` and each then decided for itself
what the answer meant: whether a model counts as resident, what an
absent device defaults to, whether a missing `gpu_name` is false or
undefined. Analytics and Settings implement the same rule for the
same link, and the second said so in a comment rather than in code. A
reading that drifts is worse than a fetch written twice, because the
pages disagree about one response while both look right.

**No request epoch, against the Direction's wording.** It asks to
"extract the remaining API clients with request epochs", but every
page calls this once at boot, so nothing can be in flight twice and a
fence would be machinery with no caller, which is what
`QUALITY-01`'s brief warns against. The pattern already exists in
`detail_requests.js` for the page that genuinely needs it, and the
comment in `model_client.js` says where to add it if a page ever
reloads its model list.

**Two of the accessors are load-bearing rather than tidy.** An absent
device reads as null instead of defaulting to `cuda`, because a guess
there labels a run with hardware it never touched, which is the
failure `DATA-04` exists to prevent. And an `active` id naming a
model the list does not carry reads as none rather than falling back
to the first entry, for the same reason.

### PROTOCOL-01

**Landed in two commits with `LIFE-01` between them**, which is not
the order the report implies but is the one that made both smaller.
The envelopes went in first, worker-side only, so behaviour was
unchanged and the taxonomy was provable before anything depended on
it. `LIFE-01`'s refusal then had a builder to use instead of a
twenty-second hand-built error dict. The frontend came last.

**Scope is a property of the operation, not of the site.** Twenty-one
sites built errors by hand, and the one that made this concrete is
`_send_busy`: it serves generation and probe both, and the same
refusal must end a run in the first case and change nothing in the
second. It now takes the request it is refusing. That is the whole
finding in one function.

**Three scopes, not two.** Fatal, run, and request. The report's
Direction says "route auxiliary failures to their local control;
reserve connection/model-fatal errors for the session reducer", which
reads as two, but a failed *generation* is neither: the socket is
fine, so it is not fatal, and it is not auxiliary either, because the
client truncates the run optimistically before the worker answers and
something has to roll that back.

**Plain dicts, not pydantic models.** The Direction says "typed
shared envelopes", and the report separately rejects validating hot
frames. These are cold, but they live beside the frame path, and
`protocol.py` is imported by three venvs holding deliberately
incompatible dependencies. Keeping it importable was worth more than
types the callers already have.

**The classifier is its own file so it could be tested.**
`wire_errors.js`, following the `activation_client.js` precedent from
`ORG-04`: a classic global script touching no DOM, driven directly in
a `vm`. It decides what a frame means and `app.js` decides what to do,
which is the split that lets the meaning be checked without a browser.
A source-inspection test covers the part a unit test cannot, that the
page actually consults it, and the mutants confirm the pair is needed:
breaking the classifier fails only the Node tests, ignoring it in
`app.js` fails only the Python one.

**An unrecognised scope is read as fatal**, which is the behaviour
every error had before this existed. A newer worker inventing a fourth
scope therefore makes an older page over-react rather than go silent.
That is the failure worth having: too much cleanup is recoverable, and
a half-applied edit left on screen because a frame was not understood
is not.

**Not done here.** The report also asks for typed envelopes on the
`frame` and `done` paths. Those already leave through one place,
`FrameStreamer`, and rebuilding them would have been churn on the hot
path in a commit about errors. The unscoped-`error` problem was the
part with a user-visible failure attached.

### LIFE-01

**The Direction held, and the token needed one thing it did not
mention.** The report asks for a worker-issued run generation token on
`done`, required on every stateful follow-up. A monotonic counter
would have been the obvious shape and would have been wrong: a fresh
worker restarts it at one, and a page is not always reloaded when its
worker is replaced. `handleResident` forces a reload only on a model
or device change (`app.js`), so activating the *same* model again
leaves a browser holding a token the new worker would hand out again,
and the check would pass on a run that never existed. The token is
therefore a per-backend nonce plus the counter.

Per-backend rather than per-process, which is stronger than the
deployment needs: one worker process hosts one backend today, and the
first version keyed the nonce to the process. A test constructing two
backends caught it. A scheme that quietly depended on that arrangement
would be a trap for whoever changes it.

**Stamped in one place.** The report's evidence lists five sites that
build a terminal frame, but `FrameStreamer` is already the single
*exit* for all of them, which its docstring has said since `DATA-04`.
So the token is stamped beside the provenance envelope, and a backend
cannot finish a run without naming it.

**Only generation advances the token, and that is deliberate.** A
resume continues the run it branched from and keeps its identity. The
alternative was considered and rejected twice over: advancing at the
start of a resume refuses the very window that asked if the resume
then fails, and advancing at the commit is too late, because
`LIFE-07` moved the commit after the terminal frame has already gone
out. The residual case, two windows editing one run they both watched
complete, is not the finding's scenario; each generation makes a new
run, so two windows cannot both hold the same one without one of them
having watched the other's.

**The state clearing came with it, at the maintainer's call.**
SmolLM3 already discarded its trace at the top of a generation; the
two diffusion workers did not, so a failed generate left the previous
run resumable behind a token the browser still believed in. `begin_run`
does both halves, because a token outliving its state names a run the
worker cannot answer for, and state outliving its token is reachable
by a request naming the run it replaced.

**The client had to change in the same commit**, or the tree would
not have worked between them: the page adopts the token from every
terminal frame and returns it on resume, substitution and probe. It
also rides in the session snapshot, without which reloading the page
and then editing the restored run would be refused as stale even
though the worker still holds exactly that run. If the worker was
replaced in the meantime the nonce differs and the refusal is
correct, which is the point of carrying a token rather than a flag.

**Six mutants, one of which was informative.** Removing the
"no token sent" branch changed nothing, because the general mismatch
check catches `None` anyway. The branch stayed, since a client that
never learned the run is a different situation to report than an
ordinary two-window race, but the test now pins that the two
*messages differ* rather than pinning either string.

### LIFE-04

**Done on 2026-08-18**, carrying `RUNTIME-01`'s bounded-queue step,
which this finding's own Direction asks for as well.

Four things shipped, in the order they had to.

**The stop signal became a `threading.Event`.** Its readers are model
threads: the autoregressive decode loop and DiffusionGemma's streamer
both check it from inside the thread running the forward pass. Only
the event loop sets it.

**Producer queues are bounded** at 32 frames, in the new
`src/inference/frame_queue.py`, with a put that is bounded in time as
well as depth and re-reads the stop event on every wait. Bounding
alone would have introduced the deadlock it invites, a producer
waiting for space while the consumer waits for the producer, so the
consumer now drains while it waits for the thread to finish. That
drain is what makes the bound safe, and it is tested directly.

**The socket loop reads while a generation runs.** The generation is
an `asyncio.Task`; the loop stays on `receive_json`. `gen_lock` is
gone, replaced by a worker-scoped in-flight task reference, which is
what makes the busy check synchronous instead of a wait. Two windows
still contend for one model because that reference is worker-scoped,
while each socket settles only the generation it started. Removing
the two `async with` blocks dropped the Ruff ratchet from 128 to 126.

**Every model ends a stopped run the same way**, with `done` carrying
`cancelled: true` through `FrameStreamer`, so it keeps its elapsed,
provenance and run token and stays savable. LLaDA previously sent no
terminal frame at all on a cancelled generate, which left the page
waiting on a run that had already stopped. The retained state is
explicitly partial, matching what `LIFE-07` settled for a cancelled
resume, and the save records `partial: true` so a truncated run
cannot read as complete in Analytics later.

**Deviation, and the reason.** The plan said to give DiffusionGemma an
event-backed `StoppingCriteria`. Reading the installed model code
showed that would not work: `generate` consults an externally supplied
criterion once per canvas, in `_finalize_canvas`, so it cannot
interrupt a single-canvas run at all. Its streamer's `put_draft` is
called on every denoising step, so the streamer raises instead. That
is our own code, testable without a GPU, and finer-grained than the
mechanism the plan named. The criterion was not added, because an
untestable second path whose return-type contract we would be guessing
at is worse than none.

**What a user will notice.** Generate becomes Stop while a run is in
flight, and leaving the page now ends the run rather than leaving the
model computing for a page that is gone. The frames survive either
way. Both are in the Help copy.

**Hardware still owed**: the manual items below. Nothing here proves a
GPU actually stops, only that the signal reaches the code that would
stop it.

**Left open: a stopped run can survive unlabelled as somebody else's
baseline.** Found by the maintainer on hardware while working item
168, by taking the unorthodox path through it: stop a run, open Edit
Frames or What If, and resume to the end.

The run that gets saved is correctly *not* marked partial, and that is
worth stating because it looks like a miss. A stopped run keeps its
configured schedule rather than its achieved length (`total_steps`
comes from `params["steps"]`, `src/backends/llada_worker.py`), so a
resume computes `remaining = total_steps - frame_index` and genuinely
runs the schedule out; SmolLM3's substitution likewise regenerates to
the full `max_new_tokens`. The branch really did finish.

What is not recorded is that the *pre-edit* half did not. An edited
save bundles its original as `original_frame_tokens`, and here that
baseline is the truncated run. So the Original/Edited crossfade
compares a run that was cut short against one that ran to completion,
and nothing says so: `Diverged N/total` and the timing comparison read
as though the intervention lengthened the run, when the baseline was
simply stopped early.

Not fixed, deliberately. It needs an `original_partial` field and a
label on the Original side of the crossfade, which is another touch of
the save format for a case that is rare and currently harmless in
every way except interpretation. Whoever takes `ANALYTICS-04`, the
guarded compare boundary, is already in this code and should fold it
in there.

### RUNTIME-01

**Measured, by accident, on 2026-08-18.** The maintainer ran a
1234-token SmolLM3 generation to exercise the light session snapshot
and found the restored run had no hover, no candidates and no entropy
profile, and that saving it wrote a run with no token overlay.

The cause is sharper than "long run". `frame_tokens` stores the whole
token array for *every* frame, and for an autoregressive model the
frame count is the token count, so the record count is quadratic:
1234 tokens is 761,995 records, roughly 34 MB of JSON against a
sessionStorage budget nearer 5. For SmolLM3 the light fallback is not
an unlucky corner past a few hundred tokens; it is guaranteed.
Diffusion escapes it only because its frames are a fixed `gen_length`
wide.

The shape is confirmed in the sampler rather than inferred:
`_build_frame` is documented as building "one full-snapshot protocol
frame for the growing sequence" (`src/inference/ar_sampler.py`), so
frame *n* carries all *n* records.

**Then measured properly, on a 2047-token run, on 2026-08-18.** That
is 2,096,128 token records, which corroborates the report's own
projection of 2,098,176 at the 2,048 ceiling (`AUDIT_REPORT.md:1191`).
The arithmetic was never in doubt; the timings are the useful part,
because they cost more than the payload does:

- Saving the edited run took 30 to 45 seconds, with visible animation
  stutter throughout.
- Opening it in Analytics took around 10 seconds before the canvas
  painted.
- Scrubbing it afterwards was, in the maintainer's words, like moving
  through molasses.

**The correctness half held, which narrows the finding usefully.**
That same 2047-token run round-trips through save and Analytics with
everything intact: hover, candidates, the entropy profile, and the
Original/Edited crossfade carrying both elapsed series. So the wall is
specifically the sessionStorage quota, not the on-disk format and not
the Analytics reader, both of which handle two million token records
correctly if slowly. Whoever takes this finding is fixing cost and one
storage ceiling, not a broken format.

That distinction cost an hour of confusion first, and it is written
into manual item 166 as well: a saved run opened in Analytics reads
from disk, which has no quota, so it looks perfect at any length and
is a convincing false pass for the session-snapshot behaviour.

Stopgapped rather than fixed: `saveRun` now refuses a run that came
back without its detail instead of writing the hollowed-out version
silently. The refusal goes away by itself once the payload is linear.

**The first step landed with `LIFE-04` on 2026-08-18.** Producer
queues are bounded at 32 frames with a stop-aware timed put
(`src/inference/frame_queue.py`), which is where this finding and
`LIFE-04`'s Direction meet: both ask for it, so it was done once. A
slow reader can no longer turn the quadratic payload into unbounded
worker memory, and a producer whose consumer has gone stops rather
than parking forever.

What remains is the finding proper, and the bound does not touch it:
the append-only frame variant for monotonic AR and SSM streams, with
client-side reconstruction and periodic checkpoints if random
scrubbing needs bounded seek time. Diffusion keeps full snapshots,
where prior positions genuinely change. Until then the payload is
still quadratic; it is merely quadratic in a bounded pipe.

### Found while verifying and fixed: saving one run twice

**Not an audit finding.** The maintainer found it during `ORG-02`'s
hardware pass on 2026-08-18, and it predates this campaign.

Save a run, navigate before the reply arrives, come back, press Edit
Frames: two rows in Analytics for one generation. The server
published the run, and the client's success handler belonged to a
document that was gone, so `runSaved` stayed false and the session
snapshot never learned otherwise. Edit Frames' `if (!runSaved)` guard
then saved it again.

**The same shape as the `DATA-02` slice from two sessions earlier**, a
request in flight during a navigation where the server does the work
and the client loses its record of it, and I did not think to look for
siblings after fixing that one. The flush that worked there does not
work here: a collections PUT is the whole operation, but a save needs
the *response*, because that is where the run id comes from. No amount
of delivery guarantees a reply to a page that no longer exists.

**So the fix is identity.** Only the create path was unguarded;
replace has had compare-and-swap since `DATA-01`. `LIFE-01`'s run
token already names the generation, and because `begin_run` advances
only on generate, one token maps to at most one created run: an edit
of that run is a replace. The store resolves the token first and the
client's memory becomes the fallback, so a save for a generation
already published lands on the run it already made.

Two details the tests found rather than the design. The store stamps
the token itself, beside the revision, rather than accepting it inside
a bundle a caller composed, because otherwise resolution and
persistence are two callers' business and can disagree. And a
replacement carrying no token keeps whatever identity the run already
has, or it becomes findable only by an id, which is the thing the next
lost reply forgets.

**In-process only, like the revision check it sits beside.**
`_PUBLISH_LOCK` makes resolution and publication one step within one
supervisor. A second supervisor breaks the token guarantee exactly as
it breaks the revision one; that is `LIFE-05`'s deferred half, and
`ui_state.py` now carries the interprocess pattern to copy.

### Found while verifying and fixed: saving on a panel opening

The setting for the bug above, and worth removing on its own terms.
Entering Edit Frames or What If wrote a full save. The archaeology
says it was deliberate rather than a relic: it arrived in the same
commit as the bundled pre-edit copy (`3729bb2`, 2026-07-19), as
insurance for a run about to be branched.

It bought less than it cost. The edited save already carries the
original with it, so the data never needed the pre-save; the session
snapshot already survives a navigation, so the protection was only
against closing the tab; and the write fired on *opening a panel*,
before anything destructive, which on a long autoregressive run means
posting tens of megabytes to look at candidates.

Three things start a save now, each either a button the user pressed
or a run about to be lost: Save, Confirm, and the model-mismatch
rescue. This had to follow the token, not precede it: without it, a
Confirm with no remembered run id would create a second row rather
than replace, which is what the entry save existed to prevent.

The maintainer's reasoning for removing rather than deferring it is
worth keeping. A multi-turn interface makes the unit of work a
conversation and makes editing a fork, at which point create-then-
replace stops fitting at all. The token survives that redesign,
because every turn still has a generation underneath it;
`lastSavedRunId` and `expected_revision` as a way of saying *which
run* do not.

### Found while verifying, not yet fixed

Two gaps surfaced by the maintainer's hardware pass on 2026-08-14,
both real, neither a regression from the pass that found them. Left
alone deliberately: one wants a measurement first, and the other
belongs to a finding nobody has opened yet.

**The Main Menu's VRAM numbers are a boot snapshot.** `loadModels()`
runs once at page load (`src/web/static/menu.js`) and nothing
re-reads `/api/models` afterwards, so every free-VRAM figure and
every green headroom pill is as old as the page. The maintainer hit
the visible consequence: the LLaDA row said `+5.2 GiB` and the header
said 22.2 GiB free, and activating it was refused because only 15.3
GiB would actually have been free. Both numbers were honest when
written; only one of them was current.

`LIFE-06` made this more visible rather than causing it. Before, the
stale row led to an eviction and a failure; now it leads to an
accurate refusal that contradicts the screen it was clicked from. The
cheap fix is to re-read `/api/models` after a refused activation, so
the row that was just proved wrong stops claiming otherwise. It is
not `ANALYTICS-03`'s pagination work and it is not `ORG-04`, which is
why it is recorded here rather than folded into either.

**The pre-eviction headroom check does not wait for VRAM to
settle.** `_validate_headroom` takes one instantaneous
`_free_vram_gib()` reading, while its post-eviction counterpart
`_preflight_vram` polls for up to `VRAM_SETTLE_TIMEOUT_S` waiting for
a stopped worker's memory to come back. So a switch attempted soon
after another switch can be refused on a reading that would have
cleared a second later. The maintainer's numbers fit that shape: the
shortfall was roughly one small worker's worth of memory against a
menu snapshot taken before anything was loaded.

Deliberately not fixed by guessing a second settle window. The
report's own measurement programme has the terminate-to-VRAM-release
timings for every model, and that measurement is what should decide
whether this reading needs a wait, whether the existing eight seconds
is right, and whether the two checks should share one number.

### ORG-04

**The duplication was worse than the finding counted, and pass one
had just added to it.** The finding names two clients. There were
four readers of the activation endpoint, three of them polling loops:
the generator's boot watch, its switch poll, the menu's selection
poll, and the one-shot read `LIFE-06` added a pass earlier. The
finding's argument for extracting was a load-bar fix that needed
coordinated edits in both clients; pass one repeated the mistake,
putting the ready-branch discard in both files and the failure
surfacing in only one.

**Injecting the scheduler is what made the loop testable.** Injecting
`fetch` alone would have tested one turn of the poll. With the
scheduler injected too, a test holds the pending callback and decides
when the next tick happens, so "does this stop when it should" is
answerable, which is most of what a poll loop can get wrong.

**Four pass-one source-inspection tests broke and were re-anchored,
not relaxed.** They asserted properties (the run is discarded only in
the ready path, re-selecting the resident model keeps its run) whose
anchors were function names this finding deleted. Worth noting as the
predictable cost of testing a classic script by inspection, which
`ORG-02` is meant to end.

### LIFE-03

**The operation id has to outlive the worker.** The obvious reading
is that finalization should clear it along with everything else
describing a dead activation. That breaks the case it exists for: a
client polls for the outcome of the load *it* started, so if the id
is cleared when the load fails, the client sees an id that is not its
own, concludes the activation was superseded, and stops without
reporting the failure to anyone. The number now survives exactly the
way the error message `LIFE-02` retained does.

**An absent operation on cancel is refused, not waved through.** It
was tempting to treat a body-less cancel as legacy and allow it,
since that is what every caller sent before this change. But not
naming an activation is not the same as owning one, and allowing it
would have left the cross-window Cancel exactly as it was. Cancelling
when *nothing* is loading is still a no-op: there is nothing to
protect, and a stale window tidying up after itself should not be
told off for it.

**The handshake is supervisor-side, with a worker-side cross-check.**
The finding says to include resident identity in the WebSocket
handshake, and the obvious place is the worker's own `model_status`.
The supervisor sends it instead, as a `resident` frame before any
worker traffic, because the supervisor is what owns the operation id
and because it needs no change in three virtualenvs. The worker's
`model_status` gained its model id anyway: it is two lines, and a
disagreement between the two is the only thing that could catch a
proxy pointed at the wrong worker, which the supervisor's own
statement by definition cannot.

**Saving outliving its worker is what made the rescue possible**, and
that was an accident of `DATA-04` rather than a design for this. A
save no longer reads the resident worker for provenance, and
`ws.onclose` never disabled the Save button, so an unsaved run can
still be written down after the worker that produced it is gone.
Continuing it cannot: resume, What If and probe all read retained
state that died with the process. That asymmetry is the whole
argument for saving the run and then letting it go, rather than
keeping it on screen where it would look actionable.

### LIFE-02

**The seam had to come before the fix.** Nothing in the suite touched
`ModelManager`, and the finding's Verification asks for five process
scenarios, so the first commit of this pass buys testability and
changes no behavior: `src/web/worker_process.py` owns spawning and
the handle, and the manager takes the spawn function, the health
probe and its four timeouts as constructor arguments defaulting to
today's values.

**The gates were the bug, not `status()`.** The finding reads as
though "active" is defined wrong, and the fix looks like redefining
it. But `status()` has a second caller: `_models_snapshot` uses it for
the menu's residency label and for `resident_reclaimable_gib`, and a
worker halfway through loading really is holding that VRAM, so making
`status()` mean "ready" would have quietly changed VRAM accounting
during every load. The two questions are different, so there are now
two predicates: `status()` still answers "does a process exist", and
`is_serving()` answers "can this take a request". Only the `/generate`
and `/ws` gates moved.

**The failure has to outlive the process.** Finalization clears the
manager's fields, which is the point, but it was also clearing the
reason, and with the `/generate` gate now honest a failed load
redirects to a menu that had nothing to say. `_finalize` keeps
`load_state` and `load_error` when it is given a reason; the next
activation clears them. A deliberate stop passes no reason and lands
on idle as before.

**Mutation-checked, since these tests are new and the code they
cover had none.** Restoring the old monitor behavior (record the
error, return with the worker alive) fails six of them; removing the
wait after SIGKILL fails two more.

Two things fell out of it. Moving the health read behind an injected
probe flattened the monitor's loop enough to clear a nesting finding,
taking the lint ratchet to 131. And an agent sandbox will not let
this process signal a child in its own session, which is exactly what
`spawn_worker` creates, so `tests/web/test_worker_process.py` checks
terminate and kill by delegation against a stand-in and keeps its
real subprocesses to ones that exit on their own. That is not a
weaker test than it looks: whether SIGTERM ends a process is
CPython's business, and whether terminate-then-kill is the right
sequence is the manager's, tested against a fake.

### RUNTIME-02

**Streaming frames into the encoder bounds nothing on its own.** The
Direction says to stream where supported, and Pillow does accept a
generator, so the obvious reading is that streaming is the fix and the
budget is a refinement. It is the other way round. Pillow's GIF writer
collects a paletted copy of every frame before it writes any of them,
so peak memory still grows with frame count no matter how the frames
arrive. Streaming is worth having, because it stops the full-size RGB
images from piling up alongside those copies, but the frame budget is
the load-bearing change.

Measured on this host, peak RSS above baseline for one render:

| frames | before | after |
|---|---|---|
| 128 | 383 MB | 36 MB |
| 400 | 1,252 MB | 167 MB |
| 1,024 | 3,185 MB | 148 MB |

The last row is the point: past the budget the cost stops tracking the
run. 1,024 frames now costs less than 400 did.

**The budget is 300, chosen against the corpus rather than picked.**
179 of the 182 saved runs are at or under it, so in practice almost
nothing is sampled. The frame counts cluster hard: 67 runs exceed 240
but only 3 exceed 300, so 240 would have resampled a third of the
corpus for no memory benefit worth the loss.

**Pillow merges consecutive identical frames**, which surfaced while
making the tests fast: a GIF drawn on a canvas too small to show the
frame's text came back with 112 frames instead of 300. Harmless for
real runs, whose frames differ, but it means a GIF's frame count is a
ceiling rather than an exact count, and a test that shrinks the canvas
to save time has to keep the content distinct.

### DATA-04

**The device nobody was recording.** The finding named the two-window
model switch, and that is real, but the quieter half was that no part
of the system knew where a model had actually loaded. LLaDA and
SmolLM3 fall back to CPU when CUDA is requested and unavailable, and
that fallback was a local variable inside `load`: not stored, not in
`/health`, not anywhere. The supervisor could only ever report what
it asked for. So on a GPU-less host every run was saved as a GPU run,
with no switch and no second window involved. Backends now set
`effective_device`, `/health` reports it, and the envelope carries it.

**Every terminal frame goes out through one object now.** Provenance
has to ride the done frame, and done frames were being assembled at
four sampler sites plus three worker sites that each remembered a
different set of fields: LLaDA's synthetic done had no `elapsed`,
DiffusionGemma's had one, and DiffusionGemma's resume bypassed
`FrameStreamer` entirely. Adding `FrameStreamer.send_done` gave the
worker-built ones the same treatment as the sampler-built ones, which
is what makes "every run carries provenance" a property rather than a
list of call sites to keep in step.

**`RunProvenance` is the one save-boundary model that is not
strict**, which is a deliberate exception to what `DATA-05` just
established. The other four describe what the browser sends and
should refuse a field the server does not know. This one is a worker
payload echoed back, and workers are the part of the system most
likely to gain a field first; a save must not start failing because
one of them learned to attest something new.

**A model mismatch is logged, not refused.** The audit allowed either
"provenance stays tied to the generating worker" or "the request
fails". Tied is implemented, so the failure path is not needed: the
attested model wins over the claimed one and a disagreement is a
warning. Refusing would discard a real, complete run over a
disagreement about its label.

### DATA-05

**The catalog's silent skip was hiding a crash, not preventing one.**
`list_runs` caught `JSONDecodeError` and `AssertionError` and dropped
the run. Two things were wrong with that. A run that vanishes reads as
a deleted run, and the natural response is to save it again. And the
catch did not cover the failure that actually mattered: a
`metadata.json` holding a list or a string got past `json.loads` and
died on `data["run_id"] = ...` with a bare `TypeError` about item
assignment, which nothing caught, so one bad file returned a 500 for
every run. Both are now entries carrying `invalid` and a reason.

No test covered the skip, which is why the neighboring crash could sit
next to it.

**A future version and a damaged run are told apart deliberately.** A
forward version is almost always this build being old, not the run
being broken, so it gets "saved by a newer version, update to open it"
rather than "unreadable". The wording is the point: calling a good run
corrupt invites deleting it. Nothing else is read from such a run, so
no field written by an unknown build is interpreted.

**The transcript adds a trailing newline that was never in the
frame.** Found by the golden fixtures, not suspected beforehand. The
v0 writer puts a newline after each frame body and the next delimiter
begins with another, so `parse_history` returns every frame except
the last with a trailing newline. `frames.jsonl` is exact, which
means the two eras return subtly different strings for identical
runs.

It changes no number the app shows, because `compute_convergence`
strips each frame before counting mask characters, and
`tests/analytics/test_run_schema.py` asserts the two eras produce
identical convergence output. Left in place rather than fixed: v0 runs
are not being rewritten, so "fixed" would mean the reader silently
editing what it read.

**A missing frame file is now a 404 by contract rather than by
accident.** `parse_history` asserted the file's existence, and the
metrics route turns `FileNotFoundError` into a 404, so under
`python -O` the assertion would vanish and a damaged run would 500
instead. `read_frame_texts` raises explicitly. A damaged run is an
operating error, not a programmer error.

**The version and the capture manifest are stamped by the store, not
the caller.** They describe the bytes on disk, so `_stage_and_publish`
writes them the same way it writes the revision. A caller cannot
forget them, and `validate_staged` cross-checks the manifest against
the directory rather than against the bundle that produced it, so a
writer that drops a file is caught by its own manifest.

**Unknown fields at the save boundary now fail loudly.** All four models
carry `extra="forbid"`, so a signal added to the client without the server
gets a 422 naming the key instead of a run saved without it and an HTTP 200
saying otherwise. Verified that this rejects nothing the browser actually
sends: every key in `app.js`'s save payload was compared against the model's
declared fields, and the two that looked undeclared (`type`, `experimental`)
turned out to belong to the generate WebSocket message, a different `payload`
variable in the same file.

`params` stays open on purpose. Its keys come from each model's own registry,
so forbidding unknown ones there would mean editing the save model to add a
hyperparameter. The strictness is about the envelope, not its cargo.

### DATA-01

**Publication happens on the metadata rename, not a directory rename.** The
Direction says "publish with one atomic rename", meaning the staging
directory. That is not available: the destination already exists as the
reservation from `allocate`, renaming onto a non-empty directory fails, and
removing the reservation first opens a window where the name is free for
another caller to take. Moving the files in and `metadata.json` last has no
such window, and it is still a single atomic rename that publishes. It works
because every reader in the app already decides a directory is a run by
looking for that file, so a reserved-but-unpublished directory is invisible
without anything being taught to ignore it.

**The race test found a real bug in the first design.** Staging was keyed on
the run id, so two callers replacing one run wrote into the same scratch
directory and wiped each other mid-write. Staging is now private per attempt.
The same test then found the deeper half: two callers can both read revision
N, both pass the check, and both publish, which is the last-writer-wins the
check exists to prevent. Read-and-publish is now serialized under one lock.
One lock for all replacements rather than one per run, because a replacement
is a user pressing Confirm and there is no contention worth a more
complicated structure. In-process only, which is the right scope while one
supervisor owns the data root; `LIFE-05` is where a second becomes possible.

**"Kill the process mid-save" is the one Verification clause not met as
written**, because it cannot be staged inside pytest. What is proved instead
is the property that matters after such a kill: a staged bundle and a
reserved empty directory are both invisible to `list_runs` and both refused
by the resolver, so a killed save leaves nothing that looks like a run.

**The GIF moved after publication.** Not `RUNTIME-02`, which still owns
bounding its memory and fixing its label, but the save no longer fails
because a derivative failed. A render error is logged and the run stays
saved, which is what the finding means by the core data being acknowledged.

**Two working directories now exist in the data root**, `.staging` and
`.trash`, and they persist empty between saves. Both are dot-prefixed and
carry no metadata, so no reader counts them; `list_runs` skips dot-prefixed
entries by name now rather than relying on the accident that they contain no
metadata.

**Revisions are additive.** The save response gained `run_id` and `revision`,
the request gained `expected_revision`, and a stale replacement answers 409
rather than winning. All 180 existing runs have no revision and read as 0, so
they remain editable without being rewritten, which is the no-migration
decision holding.

### ORG-01

**The traversal guard that was missing is now impossible to miss.**
`_compute_run_metrics` joined `RESULTS_DIR / run_id` with no containment
check, while its three siblings all resolved and compared the parent. Every
run-id call site now goes through `run_store.resolve_run_dir`, so there is one
answer to "is this a run I am allowed to touch". The gap was recorded under
`DATA-03` as belonging to this stage; it is closed.

**The named exceptions subclass the builtins on purpose.**
`RunNotFoundError` extends `FileNotFoundError` and `InvalidRunIdError` extends
`ValueError`, so every route that already turned those into a 404 or a 400
keeps working without an edit. That is what let the extraction stay
behavior-preserving across five endpoints.

One follow-on: adding the guard to the metrics path meant a malformed id in a
`/compare` request started raising `ValueError` where it used to fall through
as not-found, which would have turned one bad row into a 500 for the whole
comparison. That route now treats both as a per-run error entry.

**One deliberate behavior change, small and worth naming.** Unifying "what is
a run" on `run_store.is_run_dir` made it metadata-based rather than
directory-based, and `_existing_run_ids` inherited that. A folder with no
`metadata.json` is a half-written save that Analytics cannot open, so keeping
a "new run" cue or a collection entry alive for one was never right. Six
reconciliation fixtures created bare directories as shorthand for "a run
exists" and now write metadata, which makes them describe a real run rather
than the minimum that used to pass.

**Verified byte-identical.** The extracted writer was diffed against the
pre-extraction inline writer over the same bundle: same file set, same bytes
in all four files, including the two-space metadata indent and the compact
sidecar encoding. That mattered because 180 saved runs are read by parsers
tuned to exactly this output.

`_save_run_blocking`'s complexity 22 finding is gone rather than relocated:
the optional-metadata chain became a table and the device branch became a
helper, so the whole file dropped a `C901` and the ratchet moved 140 to 139.

### META-01

**3,233 lines to 114.** Three commits, each a move rather than a copy so the
tree never held the same content twice: the 132-item checklist and its
activation-failure runbook to `docs/MANUAL_VERIFICATION.md`, the settled decisions
to `docs/ROADMAP.md`, then the narrative deleted and the page rewritten.

**The verification ledger's state was the thing to be careful with.** It
records validation per range rather than per item, and the ranges say that
**items 102 to 126 have never been validated**. That is 25 scenarios of real
outstanding debt, predating the audit, and it was one careless paste away from
being lost in a file move. It is now stated at the top of the new document and
in the handoff.

**Item 1 was inverted, exactly as the report said.** It asked the tester to
confirm behavior with "Alternatives off (the default)" while
`registry.py:319-327` has defaulted that capture on. A checklist item whose
premise is backwards is worse than a missing one, because a correct result
reads as a regression; the report is right that this class of staleness is the
real cost of the file's size.

**One thing turned out not to need a home.** The `results/` rename trap read
as load-bearing operational knowledge, and the audit's own evidence cites it.
But its stated cause is that `RESULTS_DIR` was relative to the process working
directory, which `DATA-03` removed in stage 1. What was left is a historical
incident already recorded in `docs/audit/AUDIT_REPORT.md`, so it was deleted rather than
relocated.

**Both entries numbered `0.` disappeared with their sections**, which resolves
the duplicate numbering without a renumbering pass. `tests/test_handoff_bounded.py`
now fails on duplicate top-level list numbers, on the phrases that grew the
old file, and on any page over 200 lines. The bound is the point: this file
did not get long through one bad decision, it got long through a hundred
reasonable appends, and only a number that fails stops that.

**`AGENTS.md`'s session-end habit was the upstream cause** and was rewritten
to say which document takes what, rather than to ask for an append to the
handoff. `docs/audit/IMPLEMENTATION_BRIEF.md`'s instruction to read only the first 73
lines was removed, which the brief itself had predicted would happen.

### META-02

**The finding understated the gap, and the missing piece was the whole
standard.** The report cites `.cursor/rules/python-venv.mdc` contradicting the
three-environment matrix, which was true and is fixed. But TigerStyle, which
`AGENTS.md` called "the repo's" coding standard, was not in `.cursor/rules/`
at all. It lived in one maintainer's *user-level* Cursor settings, so it
travelled to no clone and to no second machine. Around 250 lines of mandatory
standards existed nowhere in the repository. `docs/TIGERSTYLE.md` now carries them,
and defers to `pyproject.toml` for line length, complexity, and nesting so the
prose and the linter cannot drift apart.

**`.cursor/plans/` turned out to hold 39 real build plans**, not an empty
path, and `docs/ROADMAP.md` cites them in three places as the canonical history for
named milestones. The maintainer chose to track them, so `.gitignore` now
ignores `.cursor/*` with an exception for `plans/`. `.cursor/rules/` stays
ignored on purpose: the tracked documents are canonical and the `.mdc` files
were rewritten as thin pointers that say so, which also removes the venv
contradiction for current sessions even though they are not committed.

**Cleared on 2026-08-11.** The maintainer cloned into an empty directory with
no Cursor state and read their way in. The mechanical half stays automated by
`tests/test_docs_links.py`, and `git ls-tree` confirms the shape a clone gets:
`README.md`, `AGENTS.md` and `LICENSE` at the root, four documents under
`docs/`, four under `docs/audit/`, 41 build plans under `.cursor/plans/`, and
nothing under `.cursor/rules/`.

**The link test needed two modes, and the reason is worth keeping.** A first
attempt treated every backticked `foo/bar` as a path claim and produced a wall
of false positives, because this prose is full of fragments that look like
paths and are not: `vendor/README.md` written relative to the directory under
discussion, `backends/` naming a subdirectory in passing, `Results/` recalling
a folder that no longer exists. So markdown links are checked strictly, since
somebody wrote those expecting them to resolve, while backtick spans are
checked only when their first segment is a tracked top-level entry or the path
exists on disk. That second clause is what detects this finding's real shape,
present locally and absent from every clone, and a first draft that lacked it
would have skipped the exact reference that was broken.

### QUALITY-02

**The Direction's first instruction does not survive contact.** It says to
"first remove the non-style findings", of which there were 27 against 129
line-length. Sixteen came out cleanly and did. The other eleven are all
`C901` and `PLR1702`, and every one of them sits inside a function that a
currently blocked finding owns:

- `create_worker_app` at complexity 21, plus four nesting hits, in
  `worker_base.py`, which is `LIFE-02`, `LIFE-03` and `PROTOCOL-01` in stage 4
- `_save_run_blocking` at complexity 22 in `server.py`, which is `ORG-01` and
  `DATA-01` in stage 3
- `generate` at complexity 14, plus two nesting hits, in `llada_sampler.py`,
  which is `ORG-03` in stage 6
- nesting in `render_gif.py` (`RUNTIME-02`) and at `server.py:597` (lifecycle)

Refactoring them here would be doing stage 3, 4 and 6 work early under a lint
banner, which the brief forbids in two separate places. They stay in the
baseline and come out when their owners arrive.

**One of the three `SIM105` sites was left deliberately.** In
`ModelManager._stop_locked`, `await self._monitor_task` is wrapped in
`except (asyncio.CancelledError, Exception): pass`, which swallows a monitor
that died of a real fault with nothing logged. `contextlib.suppress` would
have satisfied the linter while preserving exactly that, so the site keeps
its finding and gained a comment saying the shape is the problem and
`LIFE-02` owns it. The other two are genuine best-effort cleanups whose real
errors are reraised or already logged, and they were converted.

**The burn-down is mostly a stage 6 consequence, not a stage 7 slog.**
`llada_sampler.py` carries 56 of the original 156 findings, 36% of the whole
baseline, almost all line-length from upstream reference code. It is not dead
code: `streaming_sampler.py:23` imports live helpers from it while its
`generate` is the dormant reference program, which is precisely the split
`ORG-03` exists to make. Expect the number to fall sharply there and to move
very little before then.

**The gate caught its author first.** The very next commit, `META-02`, added a
test file with two long lines and a collapsible `if`, and the ratchet refused
it. Fixed rather than absorbed into the ceiling, which is the behavior the
whole mechanism exists to produce.

**Where the gate ended up.** 156 to 140. `scripts/lint_ratchet.py` compares
per (file, rule) against `lint_baseline.json`, because the Verification's
"never pass because a different finding disappeared" rules out comparing
totals; `tests/test_lint_ratchet.py` proves that with a swap case that holds
the total constant while moving a finding between files. A file with no
recorded debt starts at a ceiling of zero, so new code has to be clean, which
is the half of the policy that stops the baseline growing with the
repository.

