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

**Stage 3 is in progress.** Pass one landed `ORG-01` and `DATA-01`: the run
store is its own dependency-light module and a saved run now publishes whole
or not at all. The parameter-key XSS was pulled forward from `DATA-05` as a
standalone commit. Pass two is `DATA-05`, `DATA-04`, and `RUNTIME-02`, whose
decisions were settled with the maintainer before pass one began: no
migration of the existing corpus, `history.txt` demoted to a human artifact
with `frames.jsonl` as the machine format, and `DATA-04`'s provenance
envelope without the validation token that `LIFE-03` will own.

Baselines: 422 tests passing (from 265 at the campaign's start), 12 browser
tests under `node --test`, and Ruff at 137 in `src tests`, gated per file and
per rule by `scripts/lint_ratchet.py` rather than remembered.

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

One finding has no unmet blockers, and it is stage 3's first step, which
`DATA-03` unblocked.

- **ORG-01** (medium, M): extract behavior-preserving storage operations out
  of `server.py`. Became ready when `DATA-03` made the root explicit, and it
  is the first step of the run-store stage rather than of stage 1.

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
neighbours and the two-window model switch. Three entries are open:

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
- **LIFE-03**: everything the sandbox can reach passes. The supervisor's half
  is tested by execution against the endpoints, including the interleaved
  two-caller case, and mutation-checked (removing the ownership test on cancel
  fails four cases); the shared client's operation filtering is tested under
  `node --test` with an injected scheduler. What needs two real windows is the
  half a user would notice: `docs/MANUAL_VERIFICATION.md` items 147 to 150.
  Item 149 is the one to do carefully, since the rescue save is the part that
  can lose work if it is wrong.
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
| ANALYTICS-03 | medium | L | ready | DATA-01, DATA-05 (both done) | |
| ANALYTICS-04 | high | M | blocked | DATA-01 | |
| LIFE-02 | high | M | needs hardware | none | Two commits: the process seam, then verified termination |
| LIFE-06 | medium | M | needs hardware | none | Validate a switch target before evicting the working model |
| ORG-04 | medium | S | done | none | Two commits: the shared activation client, then the menu |
| LIFE-03 | critical | L | needs hardware | none | Two commits: operation identity, then the resident mismatch |
| LIFE-01 | high | M | ready | LIFE-03 (done) | |
| PROTOCOL-01 | medium | M | ready | LIFE-03 (done) | |
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

