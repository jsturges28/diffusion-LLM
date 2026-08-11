---
name: resident nav and bar fixes
overview: "Four fixes: make re-selecting the resident model a navigation instead of a load, flip the overlay picker upward when the output border would clip it, reserve a tail on DiffusionGemma's load bar for the GPU copy, and make the bar climb visibly and finish rather than vanishing partway."
todos:
  - id: menu-resident
    content: "Menu resident shortcut: capture active_device in loadModels, add isResidentSelection, reword the confirm to \"Go back to the Generation page?\", skip overlaysClearLastRun and the loading UI on that path, and branch on the activate response's state to navigate immediately or fall through to pollActivation."
    status: completed
  - id: select-flip
    content: Add a bounded clipping-ancestor walk and a pure customSelectShouldDropUp helper to custom_select.js, toggle a drop-up class in open() after unhiding, and add the .custom-select.drop-up rule (top to bottom, inverted shadow) to style.css.
    status: completed
  - id: load-ceiling
    content: "load_progress.py: add host_stage_ceiling (default 1.0, degenerating to today's formula), thread the monotonic floor as peak_fraction, baseline the CUDA counter, simplify the stage rule to on_device > 0, fix the terminal sample's hardcoded device stage, and pass HOST_STAGE_CEILING_NF4 from dgemma_worker."
    status: completed
  - id: load-tests
    content: "Extend tests/inference/test_load_progress.py: ceiling 1.0 equals today's fraction, read phase caps at the ceiling, copy phase scales into the remaining band, monotonic across the handoff, CUDA baseline subtraction, and the stage rule at on_device == 0."
    status: completed
  - id: bar-cadence
    content: Tighten the activation polls to 250 ms (server.py only once responded, menu.js, and both app.js pollers), leaving the 800 ms error retries as backoff.
    status: completed
  - id: bar-complete
    content: Add the ready case and OVERLAYS_LOAD_COMPLETE_HOLD_MS to overlaysActivationProgress, then hold at 100 percent before reload / overlay hide / menu navigation, gated on whether a bar was on screen so indeterminate loads never flash one.
    status: completed
  - id: docs-handoff
    content: Update the Help and About copy (resident carve-out, picker flip), rewrite HANDOFF items 51/53/55, retire the verified ones, keep 57, add items for the new behaviors, and touch README/ROADMAP where descriptions drift.
    status: completed
  - id: verify
    content: Run pytest, node --check, the disposable harness for the pure JS helpers, ReadLints, and the 70-column audit; then propose the two commits.
    status: completed
isProject: false
---

# Resident navigation, picker flip, and load-bar corrections

Four independent fixes, landing as two commits: the two menu/UI behaviors, then the two progress-bar corrections (which share [src/inference/load_progress.py](src/inference/load_progress.py) and the poll cadence).

## 1. Re-selecting the resident model is navigation, not a load

The backend already no-ops this ([src/web/server.py](src/web/server.py) lines 484-490), so no weights are re-read. The lost output is the unconditional clear at [src/web/static/menu.js](src/web/static/menu.js):1292. Hyperparameters survive because they are keyed per model, which stays as-is.

In [src/web/static/menu.js](src/web/static/menu.js):

- Capture `info.active_device` in `loadModels()` (line 1377) into a module-level `activeDevice`; it is currently discarded.
- Add `isResidentSelection(model, li)`: `model.status === "active"`, and for AR rows only, `li._getDevice() === activeDevice`. Diffusion rows offer one placement, so `status === "active"` already implies the device agrees. This deliberately avoids mirroring the server's `_resolve_device` null-request default.
- `beginConfirm` (line 732): message becomes `"Go back to the Generation page?"` and the check button's aria-label drops "and load".
- `selectModel` (line 1283): hoist the loading UI (`setRowLoading` plus unhiding `activationBox`) into a small helper. On the resident path, skip `overlaysClearLastRun()` and skip that helper, then branch on the POST response, which already returns the discriminator:

```924:930:src/web/server.py
    return JSONResponse(
        {
            "ok": True,
            "active": manager.active_id,
            "state": manager.load_state,
        }
    )
```

`state === "ready"` means the no-op fired, so navigate straight to `/generate`. Anything else means the worker had died and the POST respawned it, so start the loading UI and fall through to `pollActivation()` as today.

The header selector already early-returns on matching model and device, so this brings the two activation paths into agreement.

## 2. Overlay picker flips up when it would be clipped

The occluded rows in the screenshot are the picker's option list, not the drawer. Two things combine: `#output-section` sets `overflow: hidden` ([style.css](src/web/static/style.css):861), and the list is pinned downward at `top: calc(100% + 4px)` ([style.css](src/web/static/style.css):2111). `#overlay-output-wrap` on Analytics has the same `overflow: hidden` ([analytics.css](src/web/static/analytics.css):770), and Analytics loads `style.css`, so one shared change fixes both pages.

In [src/web/static/custom_select.js](src/web/static/custom_select.js), `open()` currently just unhides the list. Add:

- A pure `customSelectShouldDropUp({ boundTop, boundBottom, wrapTop, wrapBottom, listHeight, gap })` so the decision is unit-testable without a DOM. It flips only when the list does not fit below **and** there is more room above, so it never flips into a worse position.
- A bounded ancestor walk (stop at `document.body`, hard iteration cap) finding the nearest ancestor whose `overflowY` is not `visible`, falling back to the viewport.
- `open()` measures after unhiding, since a hidden list has no height, and toggles a `drop-up` class on the wrap.

In [style.css](src/web/static/style.css), a `.custom-select.drop-up .custom-select-list` rule swapping `top` for `bottom: calc(100% + 4px)` and inverting the shadow offset.

Every dropdown in the app is built by this factory, so the header model picker, param selects, and Group By all inherit it; none of them sit low enough to trigger it today.

## 3. DiffusionGemma's bar reserves a tail for the GPU copy

DiffusionGemma is the only worker that materializes the whole checkpoint as anonymous RAM before any transfer:

```395:397:src/inference/dgemma_nf4.py
    state_dict = torch.load(
        str(nf4_path / STATE_DICT_NAME), map_location="cpu"
    )
```

For a `.pt` the target is the file size, so RSS reaches the target exactly before `_move_preserving_sharing` copies a byte, and the reading is a max over both counters ([load_progress.py](src/inference/load_progress.py):307).

Capping the fraction alone does not work: the monotonic floor would then jump the bar from the cap to 100 percent the instant the copy starts. Reserving a tail means compressing the read into `[0, ceiling]` and the copy into `[ceiling, 1]`. To keep SmolLM3 and LLaDA provably untouched, the ceiling is an opt-in parameter defaulting to `1.0`, at which the formula degenerates to today's exactly:

```python
if on_device > 0 and ceiling < 1.0:
    fraction = ceiling + (1 - ceiling) * min(1, on_device / target)
else:
    fraction = min(1, max(resident, on_device) / target)
    if ceiling < 1.0:
        fraction = min(fraction, ceiling)
fraction = max(fraction, peak_fraction)
```

In [src/inference/load_progress.py](src/inference/load_progress.py):

- `sample_load_progress` and `progress_sample` take `host_stage_ceiling` (module constant `HOST_STAGE_CEILING_NF4 = 0.9` documents the one value in use).
- Thread the monotonic floor back as `peak_fraction` rather than `peak_bytes`. With a fixed target these are the same monotone quantity, but the fraction is what the two-segment formula needs and what the user actually sees. `loaded_bytes` stays in the payload as diagnostics only; the UI reads `fraction`, `total_bytes`, and `stage`.
- Baseline `cuda_allocated_bytes()` the way RSS is baselined, and subtract it.
- Replace the stage rule with `"device" if on_device > 0 else "weights"`. The current `on_device >= resident` comparison ([load_progress.py](src/inference/load_progress.py):315-319) is why "Moving to GPU" only flashed at the end on DiffusionGemma. This is exact for all three models and is a **user-visible improvement on SmolLM3 too**, where the label will now appear when the copy starts rather than once VRAM overtakes RSS.
- The terminal sample in the `finally` block hardcodes `stage: "device"`, which is wrong on a CPU load. Emit the stage that matches.

[src/backends/dgemma_worker.py](src/backends/dgemma_worker.py):83 passes the ceiling. LLaDA and SmolLM3 pass nothing and their arithmetic is unchanged.

## 4. The bar climbs visibly and finishes

The bar stopping near 7 percent is not a fast load. The sampler's terminal 100 percent is written to the worker attribute, but `_apply_health` nulls progress on the ready transition, so the client never sees it and the page navigates off whatever it last caught. Three stacked polls compound it: the sampler writes every 250 ms, `_monitor_startup` reads `/health` every 500 ms ([server.py](src/web/server.py):591), and the client polls every 500 ms, so the browser is up to a second stale at a 500 ms refresh.

Cadence, to 250 ms: [server.py](src/web/server.py):591 (only once `responded` is true, so the pre-response phase does not double its failed connects during the worker's torch import), [menu.js](src/web/static/menu.js):1273, [app.js](src/web/static/app.js):1349, and [app.js](src/web/static/app.js):462. The 800 ms error retries stay as backoff.

Completion, in [src/web/static/overlays.js](src/web/static/overlays.js):

- `overlaysActivationProgress` gains a `state === "ready"` case returning `{ determinate: true, percent: 100, label: "Ready" }`, plus a shared `OVERLAYS_LOAD_COMPLETE_HOLD_MS = 180`.

Both callers gate the hold on whether a bar was actually on screen, read from the DOM (`container.hidden` / `activationProgress.hidden`) rather than a parallel flag, so an unmeasurable checkpoint that ran indeterminate never flashes a bar for 180 ms at the end:

- [app.js](src/web/static/app.js) `pollSwitch` ready branch (line 1335) holds before `location.reload()`.
- [app.js](src/web/static/app.js) `handleModelStatus` ready branch (line 1465) holds before hiding the overlay. `modelReady` and `updateGenerateButton()` still fire immediately; only the overlay hide waits.
- [menu.js](src/web/static/menu.js) `pollActivation` ready branch (line 1257) holds before navigating. The resident shortcut from fix 1 bypasses this entirely.

## Verification

- `.venv/bin/python -m pytest`, with new cases in [tests/inference/test_load_progress.py](tests/inference/test_load_progress.py): ceiling `1.0` reproduces today's fraction exactly, the read phase caps at the ceiling, the copy phase scales into the remaining band, no regression across the handoff, CUDA baseline subtraction, and the stage rule at `on_device == 0`.
- `node --check` on each changed `.js`, plus a disposable harness (the pattern prior sessions used) for `customSelectShouldDropUp` and the reducer's ready case, deleted before handback.
- ReadLints on everything touched; 70-column audit on new lines.
- GPU and display cannot be exercised here, so all four fixes hand back with manual checklist items.

## Docs

- [index.html](src/web/static/index.html):319 ("Switching models starts you on a clean slate") needs the resident carve-out. Line 429's drawer paragraph gets a short clause about the picker opening upward near the bottom. Line 318's load-bar paragraph stays accurate as written. Check the About modal in the same pass.
- [HANDOFF.md](HANDOFF.md):1648-1698: retire verified items 48, 49, 50, 52, 54, 56; rewrite 51, 53, 55 against the new behavior; **keep 57**, which was never reported on; add items for the resident shortcut, the picker flip, and the SmolLM3 label change.
- [README.md](README.md) and [ROADMAP.md](ROADMAP.md) touched only where the descriptions drift.