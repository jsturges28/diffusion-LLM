---
name: glow settings and load sweep
overview: Make the token birth glow tunable per model class behind a class dropdown, restyle sub-settings as indented and dimmed rather than hidden, replace the dead time before the load bar with a labeled indeterminate sweep, and fix the leaked separator above the Analytics crossfade plus the inverted pager arrow colors.
todos:
  - id: subrows
    content: "Add .settings-row-parent and .settings-row-sub styling (indent, no separator between a parent and its children), extend .settings-row-disabled to cover range inputs and the custom select mount, and convert #diffusion-mode-row from hidden to disabled in settings.html and both toggle sites in settings.js."
    status: completed
  - id: glow-schema
    content: Add the four integer glow keys (brightness percent 50-200 default 100, fade ms 200-2000 default 500, per diffusion and autoregressive) to SETTINGS_DEFAULTS, parseSettings with clamping, and settingsEqual in overlays.js, plus cloneSettings in settings.js.
    status: completed
  - id: glow-ui
    content: "Build the three glow sub-rows in settings.html and settings.js: a model-class custom select defaulted from the active model, stacked brightness and fade sliders with numeric readouts that re-sync on class change, and a sample token that replays the flash on drag and on click. Update the row description that hardcodes half a second."
    status: completed
  - id: glow-apply
    content: "Parameterize @keyframes token-birth on --token-birth-shadow, --token-birth-shadow-off, and --token-birth-duration; build the shadow strings from the brightness multiplier and write them on #output-area from the if (activeModel) block in boot(); derive TOKEN_BIRTH_MAX_CONCURRENT from the fade as clamp(round(fadeSeconds * 96), 48, 192)."
    status: completed
  - id: load-sweep
    content: Change overlaysActivationProgress to return { mode, percent, label } with hidden/sweep/fill, add the starting branch labeled "Starting worker", update both call sites in app.js and menu.js to render the sweep and omit the percent, add the is-sweep CSS with a reduced-motion fallback, and rewrite the finishLoadingProgress and finishActivationProgress comments for the new always-visible track.
    status: completed
  - id: css-fixes
    content: "Zero margin-top, padding-top, and border-top on #run-blend-row in analytics.css only, and invert .alt-pager-btn colors so rest is --text-primary, hover brightens the wash, and disabled is --text-dim."
    status: completed
  - id: verify-docs
    content: Run pytest, node --check, ReadLints, and the 70-column audit; update README, ROADMAP, HANDOFF, and the About and Help modals; hand back the manual checklist centered on SmolLM3 GPU at 2x brightness with a 2000ms fade and LLaDA at steps=8.
    status: completed
isProject: false
---

# Glow sub-settings, sub-row grouping, and the load sweep

Five items, all frontend. No Python changes, so `pytest` is a regression check rather than the point.

## 1. Sub-row grouping (do first)

The separator comes from the shared row rule, so grouping means dropping it on the *parent*:

```920:923:src/web/static/style.css
.modal-body .settings-row {
  padding: 16px 0;
  border-bottom: 1px solid var(--border);
}
```

Add a `.settings-row-parent` (no `border-bottom`, reduced `padding-bottom`) and a `.settings-row-sub` (left inset around 24px, reduced `padding-top`, no border except on the last sub-row of a group). Reuse the existing dim treatment for the inactive state:

```927:933:src/web/static/style.css
.settings-row-disabled {
  opacity: 0.4;
}

.settings-row-disabled .toggle-switch {
  pointer-events: none;
}
```

Extend `.settings-row-disabled` to also neutralize `input[type=range]` and the custom select mount, since today it only covers `.toggle-switch`.

Unify Mode to match: `#diffusion-mode-row` currently toggles `hidden` in two places (`syncControls` at [settings.js:66-68](src/web/static/settings.js) and the change handler at [settings.js:146-148](src/web/static/settings.js)). Both become a `settings-row-disabled` class toggle plus `disabled` on the control, and the `hidden` attribute comes off the element in [settings.html:68](src/web/static/settings.html).

## 2. Glow settings schema

Four new integer keys in `SETTINGS_DEFAULTS` ([overlays.js:624-632](src/web/static/overlays.js)), integers so `settingsEqual` stays an exact comparison:

- `glowBrightnessDiffusion`, `glowBrightnessAutoregressive`: percent, 50 to 200, default 100
- `glowFadeMsDiffusion`, `glowFadeMsAutoregressive`: 200 to 2000, step 50, default 500

Defaults reproduce today's look exactly for both classes. `parseSettings` clamps each to its range and falls back to the default on a non-integer, matching how it already treats an absent `tokenBirthGlow` as on. Thread through `cloneSettings`, `settingsEqual`, and `resetStaged` in [settings.js](src/web/static/settings.js); no carve-out needed in `resetStaged` since these controls live on this page.

## 3. Settings page controls

Under the existing Token birth glow row ([settings.html:75-84](src/web/static/settings.html)), three sub-rows:

- **Model class**: a `createCustomSelect` with Diffusion and Autoregressive. Not persisted (it selects what you are editing, not a preference). Default it from the active model's `capabilities.model_type`, which is already reachable from the `/api/models` fetch that `revealGenerationLink` makes at [settings.js:223](src/web/static/settings.js); fall back to Diffusion.
- **Brightness** and **Fade**: stacked `input[type=range]` rows with a tabular numeric readout ("100%", "500 ms"). Changing the class re-syncs both sliders from the staged values for that class.
- **Preview**: a sample token that replays the flash on every slider `input` event and on click, driven by the same custom properties as the real thing so it cannot drift from it.

Update the row description, which currently hardcodes the old timing: "fading over half a second".

## 4. Applying the glow

Deliver via custom properties on `#output-area` rather than four separate values, so the keyframe stays a plain substitution with no `var()` nested inside `rgba()`:

```css
@keyframes token-birth {
  from { text-shadow: var(--token-birth-shadow); }
  to   { text-shadow: var(--token-birth-shadow-off); }
}
```

with `animation-duration: var(--token-birth-duration)` replacing the literal `0.5s` at [style.css:1963-1965](src/web/static/style.css). JS builds both strings from the brightness multiplier `m`, scaling the current values: inner `6px * m` at `min(0.9 * m, 1)` alpha, outer `12px * m` at `min(0.5 * m, 1)`, and the "off" string is the same radii at zero alpha. Scaling the radius as well as the alpha is what gives real headroom, since alpha alone only has 11 percent left above the current 0.9.

Apply once per page load inside the `if (activeModel)` block of `boot()` ([app.js:6308](src/web/static/app.js)), which is safe because a model switch ends in `location.reload()` ([app.js:1368](src/web/static/app.js)) so `activeModel` is resolved exactly once. Note that `applySettings()` at [app.js:3056](src/web/static/app.js) is currently dead code and is not the hook.

**The concurrency cap has to follow the fade** or the FIFO, not the timer, will end the longest glows:

```1786:1786:src/web/static/app.js
var TOKEN_BIRTH_MAX_CONCURRENT = 48;
```

Trail length is roughly `rate * fade_seconds`, so a 2s fade at GPU autoregressive speeds blows past 48 and the tail gets chopped rather than faded. Derive it as `clamp(round(fadeSeconds * 96), 48, 192)`. The 96 tokens/sec ceiling is chosen so the default 0.5s fade yields exactly 48, leaving today's behavior bit-identical.

## 5. Load sweep

`starting` already reaches the browser: set at [server.py:551](src/web/server.py) and returned as `state` by `/api/models/activation` at [server.py:954](src/web/server.py). Frontend-only change.

`overlaysActivationProgress` ([overlays.js:821](src/web/static/overlays.js)) returns a boolean `determinate` today, but there are now three outcomes. Change it to `{ mode, percent, label }` with `mode` one of `hidden`, `sweep`, `fill`:

- `ready`: fill, 100, "Ready"
- `starting`: sweep, "Starting worker"
- `downloading` with a usable total: fill, "Downloading"; without: sweep, same label
- `loading` with a usable total: fill, "Loading weights" or "Moving to GPU" per `progress.stage`; without: sweep, "Loading"
- anything else: sweep, "Loading" (the pollers only run during an activation, so `idle` should not arrive here)

Both call sites branch on `determinate` today and both need updating: [app.js:425-437](src/web/static/app.js) and [menu.js:1268-1281](src/web/static/menu.js). In sweep mode they show the track and omit the percent from the label. This is a real gain on the menu, which currently hides the whole `#menu-activation-progress` block and so shows nothing at all during the gap.

CSS: an `is-sweep` class on `#load-progress-fill` and `.menu-activation-fill` setting a partial width and animating `transform: translateX(...)` on a loop, with the existing `transition: width` suppressed while sweeping so the two cannot fight. Under `prefers-reduced-motion` the sweep is a static dim fill.

**Behavior change to call out:** both `finishLoadingProgress` ([app.js:446-455](src/web/static/app.js)) and `finishActivationProgress` ([menu.js:1289-1296](src/web/static/menu.js)) read `container.hidden` to decide whether a bar was ever on screen, so an unmeasurable checkpoint currently finishes with no bar at all. With a sweep always present, every activation now ends on a brief 100 percent fill. That reads better than a sweep vanishing, but it contradicts the comments in both functions, which need rewriting rather than left stale.

## 6. The two CSS fixes

**Analytics separator.** `#run-blend-row` is styled in both sheets, and Analytics loads `style.css` first. The Analytics block at [analytics.css:895-903](src/web/static/analytics.css) redeclares layout but never sheds `margin-top: 8px`, `padding-top: 8px`, and `border-top: 1px solid var(--border)` from [style.css:2628-2638](src/web/static/style.css). Zero those three in the Analytics block. Do not touch `style.css`: the generator still stacks the row on its own line and wants the separator.

**Pager arrows.** Invert the signal in [style.css:2793-2814](src/web/static/style.css) with no new colors: rest becomes `--text-primary`, hover keeps `--text-primary` and brightens the wash to `rgba(255, 255, 255, 0.12)`, and `:disabled` becomes `--text-dim`. Rewrite the "reads as a position indicator" comment, since the page is always named elsewhere (the chart title, or the Original/Edited header in the What If popup).

## 7. Verification and docs

`.venv/bin/python -m pytest`, `node --check` on each changed JS file, ReadLints, and the 70-column audit. Update README, ROADMAP, HANDOFF, and the About and Help modals in `index.html` for the glow controls and the new "Starting worker" phase.

Manual checklist for the maintainer, since none of this can be exercised without a display or GPU:

- SmolLM3 on GPU at 2x brightness and a 2000ms fade, which is the worst case for paint cost and the one that exercises the raised concurrency cap
- LLaDA at `steps=8`, where a burst of roughly 20 simultaneous reveals hits the glow at once
- Defaults on both classes look identical to today
- The sweep during a cold first load and during a swap, on both the menu and the generator, including the handoff from sweep to fill and the closing 100 percent
- Reduced-motion: no sweep animation, no glow
