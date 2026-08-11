# IMPLEMENTATION_LEDGER: state of the audit remediation

State for the 40 findings in `AUDIT_REPORT.md`. The report is the immutable
analysis; this file is the moving part. Read `IMPLEMENTATION_BRIEF.md` for how
to work a finding, and update this file in the same commit as the change it
describes.

**Stage 1 is complete and verified on hardware.** All five isolated safety
fixes landed and all five are `done`; the maintainer cleared the whole
validation queue on 2026-08-11. That pass also turned up an unrelated offline
model-loading gap, which is recorded under `TRUST-03` and whose availability
half was pulled forward as its own commit.

Stage 2 is next: `QUALITY-02`, `META-01`, and `META-02`. `ORG-01` is also
unblocked but belongs to stage 3.

Baselines: 329 tests passing (from 265), 12 browser tests under `node --test`,
and Ruff at exactly 156 in `src tests`.

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

Three findings have no unmet blockers: the two remaining gates the report
wants installed before any boundary moves, and stage 3's first step, which
`DATA-03` unblocked.

- **META-01** (medium, M): reduce `HANDOFF.md` to a cold-start page and move
  the 132-item verification ledger out of it.
- **META-02** (medium, S): move the canonical agent contract into tracked
  files, since `.cursor/` is gitignored and its Python rule contradicts the
  three-environment matrix.
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

All four stage 1 entries were cleared on 2026-08-11; what each of them showed
is recorded under Deviations. One new entry arrived from that same pass:

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
| META-01 | medium | M | ready | none | |
| META-02 | medium | S | ready | none | |
| QUALITY-01 | medium | L | companion | lands with each seam | |
| ORG-01 | medium | M | ready | DATA-03 (done) | |
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

**Where the gate ended up.** 156 to 140. `scripts/lint_ratchet.py` compares
per (file, rule) against `lint_baseline.json`, because the Verification's
"never pass because a different finding disappeared" rules out comparing
totals; `tests/test_lint_ratchet.py` proves that with a swap case that holds
the total constant while moving a finding between files. A file with no
recorded debt starts at a ceiling of zero, so new code has to be clean, which
is the half of the policy that stops the baseline growing with the
repository.

