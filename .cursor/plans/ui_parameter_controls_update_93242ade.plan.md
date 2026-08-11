---
name: UI parameter controls update
overview: Add remasking selector, tighten default parameter bounds, add client-side divisibility validation, and implement an Experimental Mode toggle that lifts the recommended bounds.
todos:
  - id: remasking-select
    content: Add remasking dropdown to HTML, wire it through JS payload and server validation, pass to streaming_generate
    status: completed
  - id: default-bounds
    content: Update HTML min/max attrs and server PARAM_LIMITS to new recommended bounds (steps 8-150, gen_length 16-160, etc.)
    status: completed
  - id: client-validation
    content: Add pre-generate divisibility validation in JS with inline visual hints on gen_length/block_length inputs
    status: completed
  - id: experimental-mode
    content: Add toggle switch + tooltip in HTML, two-tier limits in JS and server, mode-aware clamping
    status: completed
isProject: false
---

# UI Parameter Controls Update

## Summary

Four changes across the same four files: add a remasking dropdown, enforce new default bounds with pre-generate validation, and add an Experimental Mode toggle that swaps between recommended and absolute limits.

## 1. Add Remasking parameter

- **[index.html](src/web/static/index.html):** Add a `<select id="param-remasking">` with two options: `low_confidence` (label "Low Confidence", selected by default) and `random` (label "Random"). Place it between the CFG Scale input and the button group in `#param-row`.
- **[app.js](src/web/static/app.js):** Grab the new `paramRemasking` ref. Include `remasking: paramRemasking.value` in the `startGeneration()` WebSocket payload. Add it to the `setGenerating()` disable/enable list.
- **[server.py](src/web/server.py):** In `_validate_params()`, extract and validate `remasking` (must be `"low_confidence"` or `"random"`, default `"low_confidence"`). Pass it through to `streaming_generate()`.
- **[style.css](src/web/static/style.css):** Add a `.param-group select` rule matching the existing `input[type="number"]` style (same background, border, font, padding).

## 2. Update default parameter bounds

Replace the current wide-open limits in **[server.py](src/web/server.py)** `PARAM_LIMITS` and the HTML `min`/`max` attributes with two tiers:

**Recommended bounds** (default mode):


| Param        | Min | Max |
| ------------ | --- | --- |
| steps        | 8   | 150 |
| gen_length   | 16  | 160 |
| block_length | 8   | 160 |
| temperature  | 0.0 | 1.0 |
| cfg_scale    | 0.0 | 2.0 |


**Absolute bounds** (Experimental Mode):


| Param        | Min | Max  |
| ------------ | --- | ---- |
| steps        | 1   | 1024 |
| gen_length   | 1   | 1024 |
| block_length | 1   | 1024 |
| temperature  | 0.0 | 10.0 |
| cfg_scale    | 0.0 | 20.0 |


The server always enforces the absolute bounds (clamping). The client enforces either recommended or absolute bounds depending on mode.

## 3. Client-side divisibility validation

In **[app.js](src/web/static/app.js)**, add a `validateParams()` function called from `startGeneration()` before sending the WebSocket message. It checks:

- `gen_length % block_length === 0` -- if not, show a message in `#status-message` like "Gen Length (128) must be divisible by Block Length (24)" and abort.
- `steps % (gen_length / block_length) === 0` -- if not, show a similar message about steps and num_blocks.

This gives instant feedback without waiting for the server round-trip. The server-side validation remains as a backstop.

Additionally, add `input` event listeners on `paramGenLength` and `paramBlockLength` that run a lightweight check and toggle a visual warning (e.g., border turns `var(--danger)` color and a small `#validation-hint` span appears below `#param-row`) if the divisibility constraint is currently violated. The Generate button is not disabled by this -- just a visual cue -- since the user might still be mid-edit.

## 4. Experimental Mode toggle

- **[index.html](src/web/static/index.html):** Add a new row between `#prompt-row` and `#param-row` containing:
  - A toggle switch (styled checkbox): `<input type="checkbox" id="toggle-experimental">` with label "Experimental"
  - An info icon (the `?` character or unicode `\u24d8`) that, on hover, shows a tooltip with the disclaimer text: *"Removes recommended parameter bounds. Values outside the default range may produce erratic or unstable results."*
- **[app.js](src/web/static/app.js):**
  - Define two objects: `LIMITS_RECOMMENDED` and `LIMITS_EXPERIMENTAL` containing the bounds from the table above.
  - On toggle change, call `applyLimits(isExperimental)` which iterates over the param inputs and updates their HTML `min`/`max` attributes. If the current value exceeds the new bounds, clamp it.
  - `validateParams()` and `startGeneration()` use whichever limits are active.
  - Send `experimental: true/false` in the WebSocket payload so the server knows which bounds to apply.
- **[server.py](src/web/server.py):**
  - Define `PARAM_LIMITS_RECOMMENDED` and `PARAM_LIMITS_EXPERIMENTAL` (the two tiers).
  - In `_validate_params()`, check `data.get("experimental", False)` and select the appropriate limits dict. Always clamp to the chosen tier.
- **[style.css](src/web/static/style.css):** Style the toggle switch, tooltip, and the `#mode-row` container. The toggle uses the existing accent green for the "on" state. The tooltip uses a small absolute-positioned box on hover.

## Files touched

- [src/web/static/index.html](src/web/static/index.html) -- remasking select, experimental toggle row, validation hint span
- [src/web/static/app.js](src/web/static/app.js) -- limits objects, validation logic, experimental toggle handler, remasking in payload
- [src/web/server.py](src/web/server.py) -- two-tier PARAM_LIMITS, remasking validation, experimental flag
- [src/web/static/style.css](src/web/static/style.css) -- select styling, toggle switch, tooltip, validation hint

