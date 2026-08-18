---
name: stage 4 pass one
overview: Give `ModelManager` a testable process seam, then make worker termination a verified state transition (LIFE-02) and validate a switch target before evicting the working model (LIFE-06).
todos:
  - id: process-seam
    content: Extract src/web/worker_process.py (WorkerHandle protocol, SubprocessHandle, spawn_worker) and inject spawn, health probe and timeouts into ModelManager with today's values as defaults. No behaviour change. Land with the import-purity test.
    status: completed
  - id: life-02
    content: "LIFE-02: add _end_process (terminate, wait, kill, wait again) and _finalize with the process identity check. Route the three monitor failure exits through it without self-cancelling or taking the lock. Retain the failure message after finalization; clear it on the next activate. Add is_serving() and use it at the /generate and /ws gates only, leaving status() semantics alone."
    status: completed
  - id: life-02-tests
    content: "Write tests/web/test_worker_lifecycle.py with a scriptable fake handle and probe: startup timeout, health-reported error, graceful exit, terminate timeout, kill escalation, and a superseded activation. Assert no spawn precedes the prior exit and that terminal snapshots match process reality."
    status: completed
  - id: life-06-server
    content: "LIFE-06 server half: reorder activate into validate, evict, launch. Check interpreter, supported device (new supported_devices on ModelCapabilities, CUDA-only for DiffusionGemma), local checkpoint presence, and a non-destructive headroom estimate, all before _stop_locked. Keep _preflight_vram as the post-eviction authority."
    status: completed
  - id: life-06-tests
    content: "Write tests/web/test_activation_validation.py: missing interpreter, unsupported device, missing checkpoint and insufficient headroom each leave the resident worker alive and manager state unchanged."
    status: completed
  - id: life-06-client
    content: Move clearSessionState() in app.js and overlaysClearLastRun() in menu.js into their ready branches so a failed switch keeps the run. Add a menu boot read of /api/models/activation that surfaces a retained load error through showError. Source-inspection tests for both.
    status: completed
  - id: pass-boundary
    content: Full verification, ledger updated in the same commits, new MANUAL_VERIFICATION items for the failed-switch and failed-load-redirect scenarios, and a handback naming what stage 4 pass two (ORG-04 then LIFE-03) now looks like.
    status: completed
isProject: false
---

# Stage 4, pass one: make the supervisor's lifecycle verifiable

Stage 4 is "explicit process and socket ownership": ten findings plus `DATA-02`
in parallel. This pass takes the supervisor's own lifecycle, which is
self-contained, changes no wire protocol, and is the only part of stage 4 whose
Verification clause the sandbox can genuinely satisfy.

## The defect, confirmed

Two exits in `_monitor_startup` set `load_state = "error"` and return while the
worker is **still running**:

```601:617:src/web/server.py
            while True:
                if proc.poll() is not None:
                    self.load_state = "error"
                    ...
                    return
                if (
                    not responded
                    and time.monotonic() > startup_deadline
                ):
                    self.load_state = "error"
```

The `_apply_health` error branch (`src/web/server.py:635`) is a third. Meanwhile
`status()` calls any live PID "active":

```488:497:src/web/server.py
    def _alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
        )

    def status(self, model_id: str) -> str:
        if self.active_id == model_id and self._alive():
            return "active"
        return "inactive"
```

So a worker that hangs past the 180 s deadline, or reports a load error and
stays up, keeps its VRAM and still satisfies the `/generate` gate
(`src/web/server.py:2134`) and the `/ws` gate (`:1123`). Separately
`_stop_locked` escalates to `kill()` and clears every field without waiting for
death (`:787-794`), which is why `_preflight_vram`'s eight-second settle window
exists: it stands in for a wait we never do.

## Why a seam comes first

No test anywhere touches `ModelManager`, activation, termination, or the `/ws`
proxy. `LIFE-02`'s Verification asks for fake subprocesses across startup
timeout, health-reported error, graceful exit, terminate timeout and kill
escalation. That is unreachable against `subprocess.Popen`, a live `httpx`
probe, and 180-second wall-clock constants. Three narrow injection points fix
that without extracting the whole manager:

- **Process**: new `src/web/worker_process.py` with a `WorkerHandle` protocol
  (`poll`, `terminate`, `kill`, `wait`, `pid`), a `SubprocessHandle` wrapping
  `Popen`, and `spawn_worker(...)` holding today's `_worker_popen_kwargs()`
  behaviour. Stdlib only, so it gets the same import-purity test
  `tests/web/test_run_store.py` already applies to `run_store`.
- **Health probe**: an injectable async callable, defaulting to the current
  `httpx` GET.
- **Timeouts**: constructor arguments defaulting to the existing module
  constants, so a test can set them to milliseconds. No clock abstraction.

## Commit sequence

### 1. Extract the process seam (no behaviour change)

`ModelManager.__init__` gains `spawn`, `probe` and the timeouts as keyword
arguments with today's values as defaults. `activate` calls `self._spawn(...)`
instead of `subprocess.Popen(...)`. Nothing else moves. Lands with the
import-purity test and a test proving the real handle wraps a `Popen` faithfully.

### 2. LIFE-02: one verified terminal path

Add `_end_process(handle)`: terminate, wait, escalate to kill, **wait again**,
and only then treat it as gone; log loudly if it survives both.

Add `_finalize(handle, *, error)` built on the process identity check the
finding asks for:

```
await self._end_process(handle)
if self._proc is not handle:
    return          # superseded by a newer activation; leave its state alone
... clear identity, port, versions, tokenizer, context ...
```

- `_monitor_startup`'s three failure exits call `_finalize(proc, error=msg)`.
  They must **not** cancel the monitor task, because that is themselves, and
  they must not take `self._lock`, because `_stop_locked` awaits the monitor
  while holding it. The identity check is what makes lock-free safe.
- `_stop_locked` cancels the monitor first, then calls
  `_finalize(self._proc, error=None)`.
- `error=None` means "idle, nothing to report" (cancel, switch, shutdown).
  `error=<message>` keeps `load_state = "error"` and `load_error` after the
  process is gone, so the menu can still say why. The next `activate()` clears
  it.

**Gate honesty, without moving residency accounting.** Add
`is_serving(model_id)`, which is alive *and* `load_state == "ready"`, and use it
at the two gates only (`/generate` at `:2134`, `/ws` at `:1123`). `status()`
keeps its current meaning, because `_models_snapshot` uses it for
`resident_reclaimable_gib` and the menu's "Resident" label
(`src/web/server.py:911,924`, `src/web/static/menu.js:279,660`), and a loading
worker really does hold that VRAM. The audit's objection is that the *gates*
ignore `load_state`; this fixes exactly that.

Per the decision taken: a failed load now redirects `/generate` to the Main
Menu rather than serving a page backed by a dead worker.

### 3. LIFE-06: validate before evicting

`activate` currently stops the resident worker at `:524` and only then checks
the interpreter at `:526`. Reorder into validate, evict, launch, with every
check raising before `_stop_locked()`:

- interpreter `venv_python` exists (move the existing check up),
- device supported: add `supported_devices: Tuple[str, ...] = ("cuda", "cpu")`
  to `ModelCapabilities` in [src/backends/protocol.py](src/backends/protocol.py),
  set CUDA-only on the DiffusionGemma entry
  ([src/backends/registry.py](src/backends/registry.py):128). Ledger note: this
  is the minimal version of what `ROADMAP-01` will subsume.
- local checkpoint directory present for non-Hub checkpoints, which is
  DiffusionGemma's real failure mode and the one the README calls out as
  deliberately listed even when activation will fail,
- a non-destructive VRAM estimate, reusing the existing `_model_headroom_gib`
  (free plus reclaimable, minus required) so an impossible switch is refused
  without freeing anything. The existing post-eviction settle check at
  `_preflight_vram` stays as the authoritative one.

### 4. LIFE-06 client half, and telling the user why

Both frontends discard the run before knowing the switch will succeed:

- [src/web/static/app.js](src/web/static/app.js):1417 calls `clearSessionState()`
  before the POST. Move it into `pollSwitch`'s ready branch, immediately before
  `location.reload()`, which preserves the comment's intent (the reload is
  known to be a switch) while keeping the run through a failure.
- [src/web/static/menu.js](src/web/static/menu.js):1359 calls
  `overlaysClearLastRun()` before the POST. Move it into the ready branch
  before navigating.

The menu never reads `/api/models/activation` on boot, so after commit 2 a
redirect would land silently. Add a boot read: when the state is `error`, show
the retained message through the existing `showError`
([src/web/static/menu.js](src/web/static/menu.js):178), which uses
`textContent` and so needs no escaping work.

## Tests

New `tests/web/test_worker_lifecycle.py`, driving the manager with a scriptable
fake handle and probe, in the repo's existing `asyncio.run(...)` style (there is
no `pytest-asyncio`). One case per `LIFE-02` scenario, plus the two assertions
the finding names: no spawn happens before the prior process has exited, and
every terminal manager snapshot agrees with process reality. Also the
superseded-activation case, where a late monitor failure must not clear a newer
worker's state.

New `tests/web/test_activation_validation.py` for `LIFE-06`: missing
interpreter, unsupported device, missing local checkpoint, and insufficient
headroom must each leave the resident worker alive and the manager unchanged.

JS changes get source-inspection tests, matching
`tests/web/test_analytics_invalid_runs.py`.

## Hardware queue

New `docs/MANUAL_VERIFICATION.md` items: a real failed switch leaves the
previous model and its run usable; a cancelled load frees VRAM within the
measured window; `/generate` after a failed load lands on the menu with the
reason shown. Worth pairing with the report's own measurement item, the
terminate-to-VRAM-release timings, which decide whether the eight-second settle
window survives as a fallback.