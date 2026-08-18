---
name: stage 4 pass three
overview: Finish stage 4 by giving every worker run an identity that a stale request cannot forge (LIFE-01) and every error a scope that stops auxiliary failures tearing down the edit session (PROTOCOL-01), preceded by the half of DATA-02 that needs no fork decision.
todos:
  - id: uistate-lock
    content: Add an interprocess flock to ui_state.py and move both GET reconcilers onto a mutate-under-lock helper so a stale snapshot cannot overwrite a concurrent PUT
    status: completed
  - id: persist-flush
    content: Write collections without debounce, flush remaining keys on visibilitychange and pagehide, and surface a failed PUT via showToast
    status: completed
  - id: error-envelopes
    content: Add dependency-light error builders to protocol.py with stable codes and fatal/run/request scopes, and route every hand-built error dict through them
    status: completed
  - id: run-token
    content: Stamp a nonce-plus-counter run token in FrameStreamer, require it on resume, substitute and probe, and clear retained state at generate start on all three backends
    status: completed
  - id: error-routing
    content: Extract wire_errors.js as a testable classifier and route fatal, run and request errors to their owners in app.js
    status: completed
  - id: dispatch-tests
    content: Add the first create_worker_app message-loop tests, including the two-socket interleaving both findings' clauses require
    status: completed
  - id: records
    content: Update the ledger and manual checklist, and hand back with the hardware items
    status: completed
isProject: false
---

# Stage 4 pass three, plus the DATA-02 slice

Five commits in two independent groups. The `DATA-02` slice touches persistence and
the frontend; pass three touches the worker protocol. They share no files, so the
order is a convenience: small and certain first, so a thin budget still lands the
data-loss fix.

## Group one: the DATA-02 half that needs no fork

### 1. Close the lost-update paths in the store

`set_ui_state_key` serializes with a process-local `threading.Lock`
([src/web/ui_state.py](src/web/ui_state.py):59-61), so the browser supervisor and
the desktop supervisor can each read, modify, and write the same file. Add an
`fcntl.flock` on the state file around the read-modify-write. No precedent exists
in the repo to copy; `run_store.py:353-356` carries the identical gap and an
explicit comment deferring it, so this establishes the pattern and the ledger notes
the other site rather than fixing it.

Second path, same commit: `GET /api/ui-state` loads a snapshot, then
`_reconcile_new_runs` and `_reconcile_collections` compute a pruned value from that
snapshot and write it back ([src/web/server.py](src/web/server.py):2430-2433). A PUT
landing in between is overwritten by a value computed before it existed. Fix by
adding `mutate_ui_state_key(results_dir, key, fn)` to `ui_state.py`, which loads,
applies, and writes inside one held lock, and moving both reconcilers onto it.

### 2. Stop losing a collection change to navigation

`persistSet` writes localStorage immediately and debounces the PUT by 250 ms
([src/web/static/overlays.js](src/web/static/overlays.js):944-966). Nothing flushes
on unload, so filing a run and navigating within 250 ms drops the PUT, and the next
page's `persistHydrate` overwrites localStorage from the server's older copy. One
window, no race.

- **Write `diffusion_collections` immediately, with no debounce.** The debounce
  exists to coalesce streams of writes; collections change on discrete user actions
  (star, rename, delete, chooser checkbox). Making the debounce per-key removes the
  window entirely for the one key the report calls irrecoverable intent.
- **Flush the remaining debounced keys on `visibilitychange` (hidden) and
  `pagehide`.** `keepalive: true` only below a safe body size, since the keepalive
  cap is well under the 262,144-character key limit.
- **Surface the failure.** `persistPutKey` swallows both rejections and non-`ok`
  responses (overlays.js:968-979). Give it an optional failure callback; the
  collections save path passes one that calls `showToast`
  ([src/web/static/analytics.js](src/web/static/analytics.js):5355), which is
  available exactly where collections are edited.

## Group two: the worker protocol

### 3. Give every error a code and a scope (PROTOCOL-01, worker half)

`protocol.py` holds message-type constants and no envelopes
([src/backends/protocol.py](src/backends/protocol.py):110-137). Every error is a
hand-built `{"type": "error", "message": ...}`, in `_send_busy`, `_send_load_error`,
the unknown-type branch, and each backend's exception handlers.

Add plain-dict builders to `protocol.py`, not pydantic: the report rejects
validation on hot frames, and `src/inference/` does not import from `src/backends/`
today, an edge worth not adding. Each error gains a stable `code` and a `scope` from
three values:

- `fatal`: the connection or model is gone. Terminates the session, as today.
- `run`: the stateful operation failed. Roll back the edit snapshot; the socket is fine.
- `request`: one auxiliary request failed. Tell only its caller.

`_send_busy` ([src/backends/worker_base.py](src/backends/worker_base.py):582-591)
serves both generation and probe, so it must echo the rejected request's type and id
to pick a scope. No frontend change in this commit, so behaviour is unchanged and the
taxonomy lands provably before anything depends on it.

### 4. Number every run and reject a stale one (LIFE-01)

Better than the report's five-site estimate: `FrameStreamer` is already the single
exit for every terminal frame (worker_base.py:67-137, "Every `done` leaves through
`run` or `send_done`"), so the token is stamped in exactly one place.

- **Shape**: a random per-incarnation nonce chosen at worker startup, plus a run
  counter, e.g. `"a3f9c1:4"`. A bare counter is unsafe, because `handleResident`
  reloads only on a model or device mismatch
  ([src/web/static/app.js](src/web/static/app.js):1593-1618), so reloading the same
  model gives a fresh worker whose first token equals the one the browser still holds.
- **Validation**: `resume` (LLaDA, DiffusionGemma), `substitute` and `probe`
  (SmolLM3) require the token and reject a mismatch before reading `last_run_state`,
  with a `run`-scoped error from commit 3.
- **State clears at generate start on all three.** SmolLM3 already does
  ([src/backends/smollm3_worker.py](src/backends/smollm3_worker.py):249); the two
  diffusion workers do not, so a failed generate leaves the previous run resumable
  behind a token the browser still thinks is current.

### 5. Route errors to their owners (PROTOCOL-01, frontend half)

`handleError` restores the edit snapshot and exits guided mode on *any* error while
`remaskMode !== null` (app.js:1851-1873), which includes a probe rejected as busy
during What If.

Extract the classification into `src/web/static/wire_errors.js`, a small classic
script with no export tail, following the `activation_client.js` precedent from
`ORG-04` so it is testable under `node --test`. `app.js` consults it and dispatches:
`fatal` to today's path, `run` to snapshot rollback, `request` to the handler owning
that `request_id`.

## Verification

- **New**: `tests/backends/test_worker_dispatch.py`, the first tests for
  `create_worker_app`'s message loop, `gen_lock` and busy handling, which currently
  have none. This is the `QUALITY-01` obligation attaching to this seam.
- Both findings' clauses need two interleaved sockets; `TestClient` against a fake
  backend covers it without a GPU. `LIFE-01` explicitly demands a repeat with equal
  frame counts and output lengths so shape compatibility cannot mask the defect.
- `node --test` for `wire_errors.js`; full pytest, ratchet, ReadLints.
- Hardware items for the two-window cases go to `docs/MANUAL_VERIFICATION.md`.

## Recorded, not fixed

`run_store.py`'s identical process-local lock, and `DATA-02`'s conflict semantics,
which stay parked until the fork is settled.