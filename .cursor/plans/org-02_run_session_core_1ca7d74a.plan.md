---
name: org-02 run session core
overview: "Extract the generator's run state into tested, dependency-light modules: first the aligned frame family that nine separate sites currently enumerate by hand, then the frozen original-run family, then the workflow phase transitions, then the duplicated model API client."
todos:
  - id: token-reset
    content: Clear activeRunToken in resetRunState alongside its siblings, with a test
    status: completed
  - id: run-frames
    content: Extract run_frames.js owning the six aligned arrays, adopt it in app.js, and test the invariant and the historical bug
    status: completed
  - id: original-frames
    content: Move the frozen original-run family into the same core
    status: completed
  - id: run-phases
    content: Extract run_phases.js owning legal phase transitions and what each clears, leaving rendering in app.js
    status: in_progress
  - id: model-client
    content: Replace the four independent /api/models fetches with one client plus page adapters
    status: pending
  - id: records
    content: Record the deviations, add the two manual items, and verify
    status: pending
isProject: false
---

# ORG-02: a tested run-session core

## The finding, as a number

`frameMeanConf` is one of six arrays that must stay the same length. Beyond its
declaration it appears at exactly nine sites, and eight of them are places where
the whole family is written out by hand:

```1699:1699:src/web/static/app.js
  frameMeanConf.push(
```

append (1699), freeze into `original*` (1849), snapshot (5206), restore (5225),
truncate (5254), reset (6228), save projection (6541), session serialize (7525),
session deserialize (7598).

Adding a seventh array means getting all nine right. The report cites the
timing-chart bug caused by missing one; scoping this session turned up a second
instance, below. That is what the first module removes: nine enumerations become
one.

Two facts make this cheaper than the line count suggests. The frame arrays are
referenced **only in `app.js`**, so no cross-file coordination is needed. And
`captureEditSnapshot` and `restoreEditSnapshot` already treat the family as a
unit, just by hand.

## Settled before planning

- **The reducer comes first; the ES module conversion is a separate later
  step.** `ORG-04` sanctions starting as a namespaced classic script. This keeps
  the `vm` test pattern that currently carries 66 passing browser tests.
- **The browser smoke layer becomes manual items.** The clause wants real DOM
  events; there is no display here and the repo has zero JavaScript
  dependencies, no `package.json`. Taking jsdom to satisfy it would be the
  project's first JS dependency. Recorded as a deviation.

## Commits

### A. Clear the run token with its siblings

`resetRunState` clears `lastFinalText` (6235) and `lastRunProvenance` (6237) but
not `activeRunToken`, which I added two commits ago among exactly those
siblings. Benign today, since a failed generate leaves no frames to edit, but it
is this finding's own failure mode arriving within days of the finding. One line
and a test, before anything moves.

### B. `run_frames.js`: the six aligned arrays

A namespaced classic script beside
[activation_client.js](src/web/static/activation_client.js), same shape, no DOM:

- `runFramesCreate()` returns one object with `history`, `tokens`,
  `canvasIndex`, `meanConf`, `elapsed`, `revealed`.
- `runFramesAppend`, `runFramesTruncate`, `runFramesClear`,
  `runFramesSnapshot`, `runFramesRestore`, `runFramesLength`.
- Every mutation asserts all six lengths agree afterwards and throws if not. A
  violated invariant means a run whose charts disagree, which is worth failing
  loudly at the point of corruption rather than at save time.

**Restore mutates in place** rather than reassigning, so `app.js` holds a single
`var runFrames` that is never reassigned. That is what keeps the diff mechanical:
roughly 154 references become `runFrames.history` and friends.

`runFramesToJson` / `runFramesFromJson` keep the existing sessionStorage keys
(`frameHistory`, `frameTokens`, ...) so a snapshot written by the current build
still restores. The `/api/save` payload keys (`mean_conf` and the rest) are a
server contract and do not change.

### C. The frozen second family

`originalFrameHistory`, `originalFrameTokens`, `originalPerFrameElapsed`,
`originalMeanConf`, `originalPositionAlts`: about 64 references, captured once at
the first `handleDone` (1846-1850) and never appended to. Same treatment, kept
separate from B because the lifecycle differs and it halves each diff.

**A clean stopping point.** If budget thins, stop here: the aligned-array half of
the Verification clause is met and the tree is coherent.

### D. `run_phases.js`: which transitions are legal

`remaskMode` is a string with eight values touched 37 times, with companion flags
(`scrubberActive`, `substitutionMode`, `guidedResumeAction`, `guidedTargetFrame`,
`remaskModeEdits`, `isResuming`) that no single place keeps consistent.

The module owns the phase set, the legal-transition table, and what each
transition clears. `app.js` keeps the rendering: `updateGuidedUI` (5489-5625) is
almost entirely DOM and stays where it is, reading the phase rather than owning
it. This is the split the report asks for and the "push ifs up" rule already in
`TIGERSTYLE.md`.

This commit is what completes the Verification clause: generate, disconnect,
partial resume, Retry, Confirm and save failure, model mismatch.

### E. `model_client.js`

`GET /api/models` is fetched independently in four files:
[app.js](src/web/static/app.js) 592, [menu.js](src/web/static/menu.js) 1447,
[analytics.js](src/web/static/analytics.js) 5772,
[settings.js](src/web/static/settings.js) 660. One client, four adapters, with
the request-epoch pattern from `detail_requests.js`.

## Verification

- `node --test tests/web/static/` for each new module: the invariant, the
  historical missed-sibling bug, append and truncate and snapshot round-trips,
  and for D every transition in and out of each phase plus the illegal ones.
- Full `pytest`, the ratchet, `node --check` on changed files, ReadLints.
- Mutation checks on each module, as with the last three.
- Two manual items for the half that needs a browser: the generator's full edit
  workflow still behaves after the wiring changes, and the Analytics detail
  modal still cancels in flight.