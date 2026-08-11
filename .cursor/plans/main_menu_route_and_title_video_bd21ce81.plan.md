---
name: Main Menu route and title video
overview: Add a dedicated Main Menu landing page at `/` with a looping title-screen video backdrop and a GPU/VRAM-aware model-selection modal, move the generator to `/generate`, and (if time) port the layered "Diff vs Original" sliders into the Analytics run modal.
todos:
  - id: server-routing
    content: Add menu route at /, move generator to /generate (+ /index.html redirect) in src/web/server.py, keeping cache-busting intact
    status: completed
  - id: api-models-vram
    content: Extend /api/models to return gpu_name, free_vram_gib, and per-model effective-VRAM fits bool
    status: completed
  - id: menu-page
    content: Create menu.html + menu.js with looping title-screen video backdrop and green model-selection modal
    status: completed
  - id: menu-fits-select
    content: Render GPU/free-VRAM state, grey out non-fitting rows, and on select activate the model then navigate to /generate
    status: completed
  - id: menu-styles
    content: Add menu styles (video backdrop, modal, rows, checking/loading states) to style.css
    status: completed
  - id: generator-menu-link
    content: Add a back-to-Menu link to the generator header in index.html; leave boot/switchModel unchanged
    status: completed
  - id: desktop-comment
    content: Refresh desktop.py docstring/comment to reflect that / is now the menu (no functional change)
    status: completed
  - id: verify-shell
    content: node --check, py_compile, pytest, ReadLints; write the manual GUI/GPU verification checklist
    status: completed
  - id: analytics-diff-sliders
    content: (If time) Extract layered diff into overlays.js, refactor app.js, and add Original/Edited opacity + difference-blend sliders to the Analytics modal
    status: completed
isProject: false
---

## Main Menu route (#2) + title video (#3), then Analytics diff sliders (#1) if time

Settled decisions: generator moves to `/generate`; the menu's "fits" check counts a resident model's VRAM as reclaimable (switching frees it); the title video is wired title-less first.

### 1. Server routing + system info (`[src/web/server.py](src/web/server.py)`)
- Add `@app.get("/")` -> `_serve_stamped_page("menu.html")` (replacing the current `serve_index`), add `@app.get("/generate")` -> `_serve_stamped_page("index.html")`, and make `@app.get("/index.html")` a `RedirectResponse("/generate")` for back-compat. `analytics.html` and the `_NoCacheStaticFiles` catch-all mount are unchanged, so the auto-stamp cache-busting rule is preserved.
- Extend `list_models()` (`/api/models`, line 343) to also return top-level `gpu_name` and `free_vram_gib` (reuse `_gpu_name()` / `_free_vram_gib()`), and a per-model `fits` bool. Effective-VRAM logic mirroring `_preflight_vram`:
  - `active` model -> `fits = True`.
  - else `fits = (free is None) or (free + resident_reclaimable) >= info.min_vram_gib`, where `resident_reclaimable = REGISTRY[active_id].min_vram_gib if active_id else 0`.
  - `free is None` (unreadable) -> `fits = True`, matching the pre-flight's skip-on-unreadable behavior.

### 2. Menu page (new `[src/web/static/menu.html](src/web/static/menu.html)` + `[src/web/static/menu.js](src/web/static/menu.js)`, styles appended to `[src/web/static/style.css](src/web/static/style.css)`)
- Full-bleed `<video autoplay loop muted playsinline>` from `/assets/title-screen.mp4` (confirmed present, 1.7 MB, tracked directly in git, no LFS). Include the existing `#bg-grid` / `#bg-floaters` behind it; on video `error`, hide the video so the animated grid shows through (graceful fallback, no `ascii_scene.js` coupling). Title-less for now.
- Centered model-selection modal in the green terminal palette. On load: fetch `/api/models`, show a transient "Checking for GPU and availability..." state, then render detected GPU + free VRAM and one row per model. Rows with `fits === false` are greyed out and non-clickable with a tooltip (e.g. "Needs ~18 GiB; not enough free VRAM").
- Selecting a fitting model: POST `/api/models/{id}/activate` (the existing blocking activate), show a "Loading <model>..." state on the row/modal, and on `{ok:true}` `window.location.assign("/generate")`. On failure, surface the returned message inline and re-enable the rows.

### 3. Generator page (`[src/web/static/index.html](src/web/static/index.html)`, served at `/generate`)
- Add a small "< Menu" link in `#header-nav` back to `/`. Keep the existing header Model selector and boot flow (`[src/web/static/app.js](src/web/static/app.js)` `boot()` line 3524, `switchModel()` line 948) unchanged; generation behavior is untouched.

### 4. Desktop (`[desktop.py](desktop.py)`)
- No functional change (it already opens `/`, now the menu). Refresh the stale docstring/comment that implies `/` is the app.

### 5. (#1, only if time) Analytics layered diff sliders
- Extract the layered render into `[src/web/static/overlays.js](src/web/static/overlays.js)` as a pure DOM helper `overlaysBuildDiffLayers(origTokens, editedTokens, diffData, {origOpacity, editedOpacity, blend}, maskChar)` returning the two stacked `.diff-layer` nodes.
- Refactor `app.js` `renderDiffOverlay` / `buildDiffLayerSpans` / `diffLayerColor` (lines 1269-1335) to call the shared helper (keeping page state local).
- In the Analytics modal, replace the single-layer static `renderDiffOverlay()` (`[src/web/static/analytics.js](src/web/static/analytics.js)` line 944) with the layered helper and add an Original/Edited opacity + Difference-blend control row (`[src/web/static/analytics.html](src/web/static/analytics.html)`, styles in `[src/web/static/analytics.css](src/web/static/analytics.css)`), shown only in `diff` mode. Default: final frame; a per-frame scrubber is a deferred stretch.

### Verification (per AGENTS.md)
- `node --check` on each changed/new `.js`; `.venv/bin/python -m py_compile src/web/server.py`; `.venv/bin/python -m pytest`; ReadLints on changed files.
- GUI/GPU cannot run in-sandbox, so hand back a manual checklist: (a) `/` loads the menu, video loops muted; (b) GPU name + free VRAM show, and fits/greyout is correct with a model resident vs none; (c) selecting a fitting model loads it and lands on `/generate`; (d) `/index.html` redirects to `/generate`; (e) `desktop.py` window opens on the menu; (f) if #1 lands, analytics diff sliders blend Original/Edited with difference blend.

### Scope notes
- One cohesive commit for #2 + #3 (the shell); #1 as a separate commit if it lands this session.
- No em-dashes in any copy; follow TigerStyle and existing file conventions.
