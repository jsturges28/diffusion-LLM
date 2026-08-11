---
name: Confirm click, glyphs, Xet, ticker
overview: "A small polish pass: make only the check/X clickable in the menu confirm state, redraw the two family glyphs, actually disable Xet (set the flag before huggingface_hub imports) for a smooth download bar, fade only the ticker text, fix the dropdown hover popover so each row shows its own readout, and trim DiffusionGemma's name."
todos:
  - id: confirm-clickable
    content: "style.css: .menu-model-row.is-confirmed { pointer-events: none } + .menu-model-confirm { pointer-events: auto } so only the check/X are clickable in the confirm state."
    status: completed
  - id: family-glyphs
    content: "menu.js: redraw the diffusion glyph as a standalone reversed epsilon (optional faint D behind) and the AR glyph as an @/a with a connecting loop + R-leg; keep the <title> tooltips. Iterate on render."
    status: completed
  - id: xet-progress
    content: Set HF_HUB_DISABLE_XET before huggingface_hub imports (top of server.py and run_worker.py); drop the redundant late env set in hf_download.py; keep the byte-unit tqdm filter so the download bar fills smoothly.
    status: completed
  - id: ticker-text-fade
    content: "app.js/style.css: fade only the ticker's inner text span (not the whole pill), keeping the device-pill border static."
    status: completed
  - id: hover-closure
    content: "app.js renderModelSelector: fix the var-in-loop closure (IIFE) so each dropdown option's hover shows its own VRAM side-popup, not always SmolLM3's."
    status: completed
  - id: dgemma-name
    content: "registry.py: drop '(NF4)' from DGEMMA.display_name so the dropdown row aligns; NF4 stays in the description/About."
    status: completed
  - id: verify
    content: py_compile, node --check, pytest, ReadLints; hand back a manual GPU/model checklist (confirm-state clicks, glyphs, smooth Xet-off download, text-only ticker fade, per-row hover popover, trimmed dgemma name).
    status: completed
isProject: false
---

# Confirm Clickability, Glyphs, Xet Progress, Ticker

Small follow-ups from the latest testing round. Independent, low-risk fixes.

## 1. Confirm state: only check/X clickable

The menu confirm's non-button area still shows a pointer cursor (clicks there are already no-ops via `beginConfirm`'s `confirming` guard). In [style.css](src/web/static/style.css): add `.menu-model-row.is-confirmed { pointer-events: none; }` and `.menu-model-confirm { pointer-events: auto; }` (it already has `cursor: default`; the buttons keep `cursor: pointer`). Result: the name/tag area is inert (default cursor), only the check/X are live.

## 2. Family glyphs (iterate visually)

Redraw both in [menu.js](src/web/static/menu.js) (`_AR_ICON` / `_DIFFUSION_ICON`), keeping the `<title>` tooltip:
- **Diffusion**: replace the D+F overlay (it reads as "Ð", the stems collide at x=7) with a standalone **reversed-epsilon** glyph (two left-opening stacked arcs), optionally a very faint "D" bowl behind it for the superposition flavor.
- **Autoregressive**: an "a"/"@" bowl + an enclosing ring that opens bottom-right + a tail that arcs over the top back to the start (the loop) + a short downward-left diagonal (the R-leg).

Prototype 2-3 candidates and confirm on render; exact `d` paths are cheap to tweak.

## 3. Download bar: actually disable Xet

Root cause: `huggingface_hub` caches the flag at import (`constants.HF_HUB_DISABLE_XET = _is_true(os.environ.get(...))`, [constants.py:294](.venv/lib/python3.12/site-packages/huggingface_hub/constants.py)), and the downloader checks that constant ([file_download.py:1735](.venv/lib/python3.12/site-packages/huggingface_hub/file_download.py)). Our `os.environ[...] = "1"` inside `download_with_progress` ([hf_download.py](src/inference/hf_download.py)) runs after `huggingface_hub` is already imported, so it is a no-op and Xet keeps bypassing our `tqdm_class` (0%/8% then jump).

Fix: set the flag **before the first `huggingface_hub` import**:
- Top of [server.py](src/web/server.py) (before its imports; none of them load `huggingface_hub`, which is imported lazily in `_is_downloaded` / the download task) - `os.environ.setdefault("HF_HUB_DISABLE_XET", "1")`.
- Top of [run_worker.py](src/backends/run_worker.py) (before `importlib.import_module(info.worker_module)`, which transitively imports transformers -> hub) so activation-time downloads are smooth too.
- Remove the now-redundant late `os.environ[...]` set in `download_with_progress`; keep the byte-unit tqdm filter.

Then the classic downloader's `tqdm(unit="B")` flows through our hook and the bar fills smoothly. Verify Xet-off download speed is acceptable (else fall back to a disk-size poller).

## 4. Ticker fades text only (keep border)

`.device-pill-collapsed` opacity transition fades the whole pill (border + bg + text). In [app.js](src/web/static/app.js) `setModelSelectValue` / `startCollapsedTicker`, put the ticker text in an inner `<span class="ticker-text">` and toggle `ticker-fade` on that span; in [style.css](src/web/static/style.css) move the opacity transition to `.ticker-text` and leave the pill border static (the pill's `min-width: 34px; text-align: center` already fits the shorter `+Z`/`-Z`).

## 5. Dropdown hover popover shows the wrong row

Closure bug: in [app.js:594-619](src/web/static/app.js) `var info` is function-scoped, so every option's `mouseenter`/`mouseleave` closes over the final `info` (SmolLM3). Fix: capture per iteration via an IIFE (as `buildOptionDevice` already does for its buttons), e.g. `(function (info) { li.addEventListener("mouseenter", ...); li.addEventListener("mouseleave", ...); })(info);`. Then each row toggles its own popover.

## 6. Trim DiffusionGemma's name

Drop "(NF4)" from `DGEMMA.display_name` in [registry.py](src/backends/registry.py) ("DiffusionGemma-26B-A4B (NF4)" -> "DiffusionGemma-26B-A4B") so the dropdown stays aligned; NF4 remains in the model description and About/Help. (Ellipsis + `title` on `.model-select-name` stays as the fallback for future long names.)

## 7. Verify

`.venv/bin/python -m py_compile` changed modules, `node --check` changed JS, `.venv/bin/python -m pytest`, ReadLints. Manual (needs GPU/model): confirm-state name/tag area is inert while check/X work; the two glyphs render as intended; a fresh download fills the bar smoothly (Xet off) end to end; the ticker fades only the text with a steady border; each dropdown row's hover shows its own VRAM readout; DiffusionGemma's row fits without "(NF4)".