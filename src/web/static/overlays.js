// Shared, pure overlay primitives used by both the live generator
// page (app.js) and the Analytics Suite (analytics.js). Loaded as a
// classic global script before those files (same pattern as
// custom_select.js), so it must not depend on either page's state.
//
// Everything here is a pure function of its arguments: coloring
// scales, per-position commit steps, and the counterfactual diff.
// The stateful wrappers, memoization, and rendering stay in the page
// scripts.

"use strict";

// Unresolved-token glyph. Named distinctly so it never collides with
// each page's own MASK_CHAR global (classic scripts share scope).
var OVERLAYS_MASK_CHAR = "\u2591";

// Map confidence in [0,1] to a green intensity for the heatmap.
function heatColor(c) {
  var clamped = Math.max(0, Math.min(1, c));
  var sat = Math.round(35 + 55 * clamped);
  var light = Math.round(32 + 30 * clamped);
  return "hsl(135, " + sat + "%, " + light + "%)";
}

// Map a commit step to an early->late hue: early settles read light
// green, late settles read red-orange. maxStep normalizes to the run
// length so the scale means "early vs late in the run".
function commitColor(step, maxStep) {
  var frac = maxStep > 0 ? step / maxStep : 0;
  frac = Math.max(0, Math.min(1, frac));
  var hue = Math.round(130 - 115 * frac);
  var sat = Math.round(60 + 22 * frac);
  var light = Math.round(62 - 10 * frac);
  return "hsl(" + hue + ", " + sat + "%, " + light + "%)";
}

// Divergence coloring: changed tokens glow magenta, unchanged tokens
// fade to a dim neutral so an intervention's footprint stands out.
function diffColor(changed) {
  if (changed) {
    return "hsl(320, 80%, 66%)";
  }
  return "hsl(0, 0%, 45%)";
}

// Reference maximum for normalizing per-token entropy (nats) into a
// display fraction. Entropy arrives raw from the sampler because
// normalizing by log(vocab) over a ~128k vocabulary would squash
// every realistic value into the bottom of a [0,1] scale. 5 nats is
// roughly a uniform choice among ~150 tokens, about as torn as a
// language model gets in practice.
var OVERLAYS_ENTROPY_REF_NATS = 5.0;

// Normalize raw entropy (nats) into [0,1] against the reference max.
function overlaysEntropyFraction(e) {
  if (typeof e !== "number" || !isFinite(e) || e < 0) {
    return 0;
  }
  return Math.min(1, e / OVERLAYS_ENTROPY_REF_NATS);
}

// The one place the entropy ramp is defined: a decisive distribution
// reads cool blue, a torn one reads hot amber. Kept off the green
// confidence heatmap and the magenta diff so no two overlays read as
// each other.
function overlaysEntropyHue(e) {
  return Math.round(205 - 160 * overlaysEntropyFraction(e));
}

// Map per-token entropy onto that ramp.
function entropyColor(e) {
  var frac = overlaysEntropyFraction(e);
  var sat = Math.round(55 + 30 * frac);
  var light = Math.round(52 + 8 * frac);
  return "hsl(" + overlaysEntropyHue(e) + ", " + sat + "%, "
    + light + "%)";
}

// Brighter twin of entropyColor, for the hovered column of the
// entropy profile. Same hue so the ramp still reads; lifted
// saturation and lightness so a column a few pixels wide stands out
// from its neighbors.
function entropyGlowColor(e) {
  return "hsl(" + overlaysEntropyHue(e) + ", 100%, 74%)";
}

// Place the candidate popover horizontally: aligned to the token's
// left edge, pulled back inside the viewport when the token sits near
// the right margin. Both arguments are viewport-space rects (the
// popover is fixed at body level).
function overlaysPopoverLeft(tokenRect, popoverBox) {
  return Math.min(
    Math.max(8, tokenRect.left),
    Math.max(8, window.innerWidth - popoverBox.width - 8)
  );
}

// Place it vertically, preferring above the token. The browser draws
// the native title tooltip below the cursor and we cannot move that,
// so a popover below would sit underneath the tooltip. Falls back to
// below when the token is too close to the top of the viewport.
function overlaysPopoverTop(tokenRect, popoverBox) {
  var above = tokenRect.top - popoverBox.height - 6;
  if (above >= 8) {
    return above;
  }
  var below = Math.min(
    tokenRect.bottom + 6,
    window.innerHeight - popoverBox.height - 8
  );
  return Math.max(8, below);
}

// ---- Candidate popover chrome ----
//
// The popover pages between the two runs' candidate sets, which only
// exist together from the divergence point rightward: left of it a
// branch copies its prefix verbatim, so there is one set and nothing
// to page through.

var OVERLAYS_ALT_PAGES = ["original", "edited"];

function overlaysAltPageLabel(page) {
  return page === "original" ? "Original" : "Edited";
}

// The popover's heading. ``page`` is null for a single candidate set,
// which renders the plain title and no pager. ``onPage`` is called
// with the page an arrow moves to.
function overlaysBuildAltHeading(pos, page, onPage) {
  var heading = document.createElement("div");
  heading.className = "alt-heading";
  var title = document.createElement("span");
  title.textContent = "Position " + (pos + 1) + ": "
    + (page === null
      ? "candidates"
      : overlaysAltPageLabel(page));
  heading.appendChild(title);
  if (page === null) {
    return heading;
  }
  var pager = document.createElement("span");
  pager.className = "alt-pager";
  for (var i = 0; i < OVERLAYS_ALT_PAGES.length; i++) {
    pager.appendChild(
      overlaysBuildAltPager(
        OVERLAYS_ALT_PAGES[i], page, onPage
      )
    );
  }
  heading.appendChild(pager);
  return heading;
}

function overlaysBuildAltPager(target, page, onPage) {
  var label = overlaysAltPageLabel(target) + " run";
  var button = document.createElement("button");
  button.type = "button";
  button.className = "alt-pager-btn";
  button.textContent =
    target === "original" ? "\u2039" : "\u203A";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.disabled = target === page;
  button.addEventListener("click", function (event) {
    // The popover sits over the token view on both pages, whose own
    // handlers would otherwise treat this as a token interaction.
    event.stopPropagation();
    onPage(target);
  });
  return button;
}

// Render a candidate token's raw text readably. Alternatives keep
// control tokens and whitespace intact (the sampler deliberately
// does not sanitize them), so make the invisible ones visible rather
// than showing a blank row.
function overlaysAltDisplay(text) {
  if (typeof text !== "string" || text.length === 0) {
    return "\u2205";
  }
  return text
    .replace(/\n/g, "\u21B5")
    .replace(/\t/g, "\u21E5")
    .replace(/ /g, "\u00B7");
}

// Per-position commit step for a run: the step after which a position
// last changed to its final value. Derived purely from the frame
// token stream (the final frame is ground truth), so it is exact for
// LLaDA (resolved tokens are frozen) and a "settle" proxy for
// DiffusionGemma. Positions still unresolved at the last frame get
// -1 (left uncolored). ``frames`` is an array of per-frame token
// arrays; each token is ``{t, m, id, c?}`` or a masked placeholder.
function overlaysComputeCommitSteps(frames) {
  var frameCount = frames.length;
  if (frameCount === 0) {
    return [];
  }
  var finalTokens = frames[frameCount - 1];
  if (!finalTokens) {
    return [];
  }
  var width = finalTokens.length;
  var steps = new Array(width);
  for (var i = 0; i < width; i++) {
    var finalTok = finalTokens[i];
    if (!finalTok || finalTok.m) {
      steps[i] = -1;
      continue;
    }
    var finalId = finalTok.id;
    var settle = 0;
    for (var f = 0; f < frameCount; f++) {
      var ft = frames[f];
      if (!ft || i >= ft.length) {
        continue;
      }
      var tk = ft[i];
      if (!tk || tk.id !== finalId) {
        settle = f + 1;
      }
    }
    steps[i] = settle;
  }
  return steps;
}

// Per-token color for one layer of the counterfactual diff overlay.
// The original layer reads cyan in ghost mode (blend off); with the
// difference blend on it adopts the edited layer's diff colors so
// matching tokens cancel to black. Remask origins glow orange and
// divergences magenta. ``diff`` is an overlaysComputeDiff() result.
function overlaysDiffLayerColor(diff, index, isOriginal, blend) {
  if (isOriginal && !blend) {
    return "#2dd4ff";
  }
  if (diff && diff.origins[index]) {
    return "#ff8a3d";
  }
  if (diff && diff.changed[index]) {
    return "hsl(320, 80%, 66%)";
  }
  return "#e6e6e6";
}

// A colorFor callback over one diff layer. Masked positions return
// null so the .token-mask class colors them, keeping the mask glyph
// identical to the single-layer paths.
function overlaysDiffColorFor(diff, isOriginal, blend) {
  return function (index, tok) {
    if (!tok || tok.m) {
      return null;
    }
    return overlaysDiffLayerColor(
      diff, index, isOriginal, blend
    );
  };
}

// Which of two exactly overlapping layers receives pointer events:
// the more opaque one, ties going to the edited run. The layers share
// a grid cell, so without an explicit choice the later sibling wins
// every hit test even when faded to nothing, leaving the layer the
// user is actually reading inert.
function overlaysEditedOwnsPointer(origOpacity, editedOpacity) {
  return editedOpacity >= origOpacity;
}

// Re-apply that choice to layers already in the DOM. The generator
// updates layer opacity inline while a slider drags rather than
// re-rendering, so ownership has to follow the same path or the
// pointer would stay with whichever layer happened to win at build
// time.
function overlaysApplyLayerPointers(
  root, origOpacity, editedOpacity
) {
  var editedTakes = overlaysEditedOwnsPointer(
    origOpacity, editedOpacity
  );
  var orig = root.querySelector(".token-layer-original");
  var edited = root.querySelector(".token-layer-edited");
  if (orig) {
    orig.style.pointerEvents = editedTakes ? "none" : "auto";
  }
  if (edited) {
    edited.style.pointerEvents = editedTakes ? "auto" : "none";
  }
}

// One token span. Carries ``token-span`` and ``data-pos`` because
// every interaction on both pages keys off exactly those two: the
// hover highlight, the candidate popover, the entropy
// cross-highlight, and the generator's remask click. A layer built
// without them looks right and does nothing.
//
// Beyond the required colorFor, ``opts`` takes four optional
// callbacks, all defaulting to the plain Analytics behavior so that
// page passes none of them:
//
//   titleFor(index, tok)            -> hover tooltip
//   maskedFor(index, tok)           -> mask a resolved token
//   classFor(index, tok, masked)    -> extra classes
//   opacityFor(index, tok, masked)  -> inline opacity
//
// The generator needs all four. It draws the mask glyph over
// positions the user selected for remasking even though their tokens
// are resolved (hence maskedFor), marks those and its clickable and
// substitutable positions with their own classes (classFor), and
// grades a mask's opacity by the model's live predicted confidence
// (opacityFor). Analytics has no live run, so its masks are flat.
function overlaysBuildTokenSpan(index, tok, mask, opts) {
  var span = document.createElement("span");
  overlaysSyncTokenSpan(span, index, tok, mask, opts);
  return span;
}

// Apply a position's appearance to a span that may already be on the
// page. Split out of the builder so the live generation path can keep
// one node per position and update it in place: rebuilding the whole
// output every frame meant laying out one inline box per *character*,
// several hundred of them, on every step.
//
// Every write is guarded by a read, because the guard is the point.
// An unconditional textContent assignment relayouts the block even
// when the text is identical, which is the cost this exists to avoid.
//
// This owns the span's class attribute outright and will overwrite
// anything else written there, so transient decoration a caller wants
// to survive a resync belongs on its own attribute (the birth glow
// uses data-born) rather than on the class list.
function overlaysSyncTokenSpan(span, index, tok, mask, opts) {
  // A missing token is a hole in the canvas, drawn as the mask glyph
  // rather than skipped: two layers only line up if both emit a span
  // per position.
  var masked = !tok || !!tok.m;
  // Consulted only for a token that is really there, so the hook can
  // add masking but never strip it off a hole and leave tok.t to be
  // read from null below.
  if (!masked && opts.maskedFor) {
    masked = !!opts.maskedFor(index, tok);
  }
  var pos = String(index);
  if (span.getAttribute("data-pos") !== pos) {
    span.setAttribute("data-pos", pos);
  }
  var className = "token-span "
    + (masked ? "token-mask" : "token-resolved");
  var extraClass = opts.classFor
    ? opts.classFor(index, tok, masked)
    : "";
  if (extraClass) {
    className += " " + extraClass;
  }
  if (span.className !== className) {
    span.className = className;
  }
  var text = masked ? mask : tok.t;
  if (span.textContent !== text) {
    span.textContent = text;
  }
  // Cleared rather than skipped when absent: on a reused span the
  // previous frame's value would otherwise stick.
  var color = opts.colorFor ? opts.colorFor(index, tok) : null;
  var nextColor = color ? color : "";
  if (span.style.color !== nextColor) {
    span.style.color = nextColor;
  }
  var opacity = opts.opacityFor
    ? opts.opacityFor(index, tok, masked)
    : null;
  var nextOpacity = opacity !== null ? String(opacity) : "";
  if (span.style.opacity !== nextOpacity) {
    span.style.opacity = nextOpacity;
  }
  var title = opts.titleFor ? opts.titleFor(index, tok) : "";
  if (span.title !== title) {
    span.title = title;
  }
}

// Build one stacked layer of token spans. ``opts`` carries the layer
// class, its opacity in [0,1], an ``interactive`` flag deciding which
// layer takes the pointer, and is passed through to
// overlaysBuildTokenSpan for the per-token callbacks. Pure: the
// caller owns the container and must give it the stacking mode.
function overlaysBuildTokenLayer(tokens, opts) {
  var mask = opts.maskChar || OVERLAYS_MASK_CHAR;
  var layer = document.createElement("div");
  layer.className = "token-layer " + opts.layerClass;
  layer.style.opacity = String(opts.opacity);
  layer.style.pointerEvents =
    opts.interactive ? "auto" : "none";
  for (var i = 0; i < tokens.length; i++) {
    layer.appendChild(
      overlaysBuildTokenSpan(i, tokens[i], mask, opts)
    );
  }
  return layer;
}

// Build the two stacked layers for the "Diff vs Original" overlay:
// the original and edited runs drawn on top of each other with
// independent opacity and an optional difference blend. Pure: returns
// a DocumentFragment of two ``.token-layer`` nodes; the caller owns
// the container (and must give it the stacking mode). ``diff`` is an
// overlaysComputeDiff() result; ``opts`` carries opacities in [0,100]
// (originalOpacity / editedOpacity), a ``blend`` flag, and an
// optional titleFor(index, token).
function overlaysBuildDiffLayers(
  origTokens, editedTokens, diff, opts, maskChar
) {
  var options = opts || {};
  var origOpacity =
    typeof options.originalOpacity === "number"
      ? options.originalOpacity : 50;
  var editedOpacity =
    typeof options.editedOpacity === "number"
      ? options.editedOpacity : 100;
  var blend = !!options.blend;
  var editedTakes = overlaysEditedOwnsPointer(
    origOpacity, editedOpacity
  );

  var origLayer = overlaysBuildTokenLayer(origTokens || [], {
    layerClass: "token-layer-original",
    opacity: origOpacity / 100,
    interactive: !editedTakes,
    maskChar: maskChar,
    colorFor: overlaysDiffColorFor(diff, true, blend),
    titleFor: options.titleFor,
  });

  var editLayer = overlaysBuildTokenLayer(editedTokens || [], {
    layerClass: "token-layer-edited",
    opacity: editedOpacity / 100,
    interactive: editedTakes,
    maskChar: maskChar,
    colorFor: overlaysDiffColorFor(diff, false, blend),
    titleFor: options.titleFor,
  });
  if (blend) {
    editLayer.style.mixBlendMode = "difference";
  }

  var frag = document.createDocumentFragment();
  frag.appendChild(origLayer);
  frag.appendChild(editLayer);
  return frag;
}

// Compare a run's final frame against a retained original run's final
// frame, position-aligned on the shared canvas. Returns per-position
// change flags, the original display text (for tooltips), the
// remask-origin positions, and a divergence summary. ``cur`` and
// ``orig`` are final-frame token arrays; ``remaskEdits`` is the list
// of ``{frame_index, token_positions}`` edits (may be empty/absent).
function overlaysComputeDiff(cur, orig, remaskEdits) {
  var result = {
    changed: [],
    origText: [],
    origins: {},
    changedCount: 0,
    totalCount: 0,
  };
  if (!cur || !orig) {
    return result;
  }
  var edits = remaskEdits || [];
  for (var e = 0; e < edits.length; e++) {
    var positions = edits[e].token_positions || [];
    for (var p = 0; p < positions.length; p++) {
      result.origins[positions[p]] = true;
    }
  }
  var width = Math.min(cur.length, orig.length);
  for (var i = 0; i < width; i++) {
    var c = cur[i];
    var o = orig[i];
    var cResolved = !!c && !c.m;
    var oResolved = !!o && !o.m;
    var changed = false;
    if (cResolved && oResolved) {
      result.totalCount++;
      changed = c.id !== o.id;
    } else if (cResolved !== oResolved) {
      result.totalCount++;
      changed = true;
    }
    if (changed) {
      result.changedCount++;
    }
    result.changed[i] = changed;
    result.origText[i] =
      o ? (o.m ? OVERLAYS_MASK_CHAR : o.t) : "";
  }
  return result;
}

// ---- Durable UI state (server-backed localStorage mirror) ----
//
// The desktop app's window origin (scheme://host:port) can change
// between launches because the launcher's port varies, which partitions
// localStorage and made Settings, prompt history, the analytics "new
// run" cue, and the generate teaser reset across restarts. The server
// persists these keys in results/ui_state.json (see src/web/ui_state.py).
// We hydrate localStorage from it once on boot and write through on
// change, so the fast synchronous localStorage reads elsewhere keep
// working unchanged.

var PERSIST_KEYS = [
  "diffusion_settings",
  "diffusion_new_runs",
  "diffusion_prompt_history",
  "diffusion_generate_teased",
  "diffusion_download_toast_corner",
  // One per page: the two drawers sit in containers of different
  // heights, so a shared offset would land sensibly on at most one.
  "diffusion_overlay_drawer_top_generator",
  "diffusion_overlay_drawer_top_analytics",
];

// Debounce PUTs per key so rapid writes (e.g. successive settings
// toggles) coalesce into one network call.
var PERSIST_PUT_DEBOUNCE_MS = 250;
var persistPutTimers = {};

// Write `value` to localStorage immediately (so the many synchronous
// reads see it at once) and debounce a write-through PUT to the server.
// Unknown keys are stored locally only.
function persistSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_e) {
    // Non-fatal: fall through to the server write regardless.
  }
  if (PERSIST_KEYS.indexOf(key) === -1) {
    return;
  }
  if (persistPutTimers[key]) {
    clearTimeout(persistPutTimers[key]);
  }
  persistPutTimers[key] = setTimeout(function () {
    persistPutTimers[key] = null;
    persistPutKey(key, value);
  }, PERSIST_PUT_DEBOUNCE_MS);
}

function persistPutKey(key, value) {
  try {
    fetch("/api/ui-state/" + encodeURIComponent(key), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: value }),
    }).catch(function () {
      // Non-fatal: the in-session localStorage copy still holds.
    });
  } catch (_e) {
    // Ignore: server persistence is best-effort.
  }
}

// Fetch server state, mirror it into localStorage, then run `onReady`.
// Always calls `onReady` exactly once (even on failure) so a page never
// hangs on a persistence hiccup. Server values overwrite any stale
// local copy left by a previous window origin.
function persistHydrate(onReady) {
  var done = false;
  function finish() {
    if (done) {
      return;
    }
    done = true;
    onReady();
  }
  try {
    fetch("/api/ui-state")
      .then(function (response) {
        return response.json();
      })
      .then(function (state) {
        persistApplyHydrated(state);
        finish();
      })
      .catch(finish);
  } catch (_e) {
    finish();
  }
}

function persistApplyHydrated(state) {
  if (!state || typeof state !== "object") {
    return;
  }
  for (var i = 0; i < PERSIST_KEYS.length; i++) {
    var key = PERSIST_KEYS[i];
    if (typeof state[key] !== "string") {
      continue;
    }
    try {
      localStorage.setItem(key, state[key]);
    } catch (_e) {
      // Non-fatal: this key just will not hydrate this session.
    }
  }
}

// ---- Shared settings model (generator + settings page) ----
//
// Durable user preferences, persisted under SETTINGS_KEY (see
// persistSet / PERSIST_KEYS). Both pages source their defaults and
// parsing here so the schema lives in exactly one place.
//
// Two fields are edited outside the Settings page. Commit Order left
// long ago: it is a per-view overlay option, not a preference.
// highlightTokens followed it, and is now a checkbox in each page's
// overlay drawer, next to the tokens it affects. It stays in this
// blob so one preference still governs both pages, which means
// settings.js has to keep round-tripping a field it no longer shows.

var SETTINGS_KEY = "diffusion_settings";

// ---- Token birth glow tuning ----
//
// Brightness is a percentage multiplier on the flash, fade is its
// duration. Both are per model class, because the rate the tokens
// arrive at decides what reads well: the trail an eye can follow is
// roughly rate times fade, so a 40 token/second autoregressive run
// needs a longer, brighter flash than a diffusion step does to leave
// any trail at all. The defaults reproduce the single fixed look that
// these replaced, so an existing profile sees no change.
var GLOW_BRIGHTNESS_MIN = 50;
var GLOW_BRIGHTNESS_MAX = 200;
var GLOW_BRIGHTNESS_DEFAULT = 100;
var GLOW_FADE_MS_MIN = 200;
var GLOW_FADE_MS_MAX = 2000;
var GLOW_FADE_MS_DEFAULT = 500;
var GLOW_FADE_MS_STEP = 50;

// The flash at 100% brightness: two blurred copies of the text.
var GLOW_INNER_BLUR_PX = 6;
var GLOW_OUTER_BLUR_PX = 12;
var GLOW_INNER_ALPHA = 0.9;
var GLOW_OUTER_ALPHA = 0.5;

// The settings keys each model class reads, keyed on the model_type
// from ModelCapabilities. Written out rather than derived from the
// class name so every key is greppable as a literal; a new class
// (state space is on the roadmap) is one entry here plus an option
// in the Settings picker.
var GLOW_KEYS = {
  diffusion: {
    brightness: "glowBrightnessDiffusion",
    fadeMs: "glowFadeMsDiffusion",
  },
  autoregressive: {
    brightness: "glowBrightnessAutoregressive",
    fadeMs: "glowFadeMsAutoregressive",
  },
};

var GLOW_CLASS_OPTIONS = [
  { value: "diffusion", label: "Diffusion" },
  { value: "autoregressive", label: "Autoregressive" },
];

var SETTINGS_DEFAULTS = {
  highlightTokens: true,
  diffusionText: false,
  diffusionTextMode: "default",
  gpuTicker: true,
  tokenBirthGlow: true,
  glowBrightnessDiffusion: GLOW_BRIGHTNESS_DEFAULT,
  glowFadeMsDiffusion: GLOW_FADE_MS_DEFAULT,
  glowBrightnessAutoregressive: GLOW_BRIGHTNESS_DEFAULT,
  glowFadeMsAutoregressive: GLOW_FADE_MS_DEFAULT,
  // "total" is the run average, "last" the most recent step. Lives
  // here rather than on the Settings page because its control is the
  // footer readout itself, like highlightTokens and the drawers.
  tpsMode: "total",
};

// Parse a stored settings JSON string into a complete settings object,
// falling back to the defaults for any missing or invalid field. Never
// throws; corrupt storage yields the defaults.
function parseSettings(raw) {
  var settings = {
    highlightTokens: SETTINGS_DEFAULTS.highlightTokens,
    diffusionText: SETTINGS_DEFAULTS.diffusionText,
    diffusionTextMode: SETTINGS_DEFAULTS.diffusionTextMode,
    gpuTicker: SETTINGS_DEFAULTS.gpuTicker,
    tokenBirthGlow: SETTINGS_DEFAULTS.tokenBirthGlow,
    glowBrightnessDiffusion:
      SETTINGS_DEFAULTS.glowBrightnessDiffusion,
    glowFadeMsDiffusion: SETTINGS_DEFAULTS.glowFadeMsDiffusion,
    glowBrightnessAutoregressive:
      SETTINGS_DEFAULTS.glowBrightnessAutoregressive,
    glowFadeMsAutoregressive:
      SETTINGS_DEFAULTS.glowFadeMsAutoregressive,
    tpsMode: SETTINGS_DEFAULTS.tpsMode,
  };
  if (!raw) {
    return settings;
  }
  try {
    var parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      // Default on when the key is absent (older saved state), so a
      // fresh profile meets the highlight rather than having to find
      // a control for it. An explicit false is still honored.
      settings.highlightTokens = parsed.highlightTokens !== false;
      settings.diffusionText = !!parsed.diffusionText;
      settings.diffusionTextMode =
        parsed.diffusionTextMode === "cycle" ? "cycle" : "default";
      settings.gpuTicker = parsed.gpuTicker !== false;
      // Default on when absent, like highlightTokens: a profile
      // saved before this setting existed should still meet the
      // effect rather than having it silently off forever.
      settings.tokenBirthGlow = parsed.tokenBirthGlow !== false;
      parseGlowInto(settings, parsed);
      settings.tpsMode =
        parsed.tpsMode === "last" ? "last" : "total";
    }
  } catch (_e) {
    // Corrupt storage: keep the defaults.
  }
  return settings;
}

// Fold the four per-class glow values out of stored state, clamped to
// their ranges. Clamped rather than rejected because the bounds can
// tighten later and a value saved under the old ones is still a
// coherent intent; only a non-number falls back to the default.
function parseGlowInto(settings, parsed) {
  var classes = Object.keys(GLOW_KEYS);
  for (var i = 0; i < classes.length; i++) {
    var keys = GLOW_KEYS[classes[i]];
    settings[keys.brightness] = clampGlowValue(
      parsed[keys.brightness],
      GLOW_BRIGHTNESS_MIN,
      GLOW_BRIGHTNESS_MAX,
      GLOW_BRIGHTNESS_DEFAULT
    );
    settings[keys.fadeMs] = clampGlowValue(
      parsed[keys.fadeMs],
      GLOW_FADE_MS_MIN,
      GLOW_FADE_MS_MAX,
      GLOW_FADE_MS_DEFAULT
    );
  }
}

// A stored glow value as a whole number inside [min, max].
function clampGlowValue(value, min, max, fallback) {
  if (typeof value !== "number" || !isFinite(value)) {
    return fallback;
  }
  var rounded = Math.round(value);
  if (rounded < min) {
    return min;
  }
  if (rounded > max) {
    return max;
  }
  return rounded;
}

// The glow's two shadow layers at full strength, plus the same layers
// at zero alpha for the animation to land on.
//
// Brightness scales the blur radii as well as the alphas: alpha alone
// tops out barely above the default 0.9, which is nowhere near enough
// headroom to make a fast autoregressive run legible.
//
// Both endpoints come back together because the "off" string has to
// carry the peak's radii. Letting the radius differ between them
// would have the browser interpolate the blur size and re-rasterize a
// different-sized shadow on every tick, which is the expensive shape
// this animation has always avoided.
function overlaysGlowShadow(brightnessPercent) {
  var scale = clampGlowValue(
    brightnessPercent,
    GLOW_BRIGHTNESS_MIN,
    GLOW_BRIGHTNESS_MAX,
    GLOW_BRIGHTNESS_DEFAULT
  ) / 100;
  var inner = (GLOW_INNER_BLUR_PX * scale).toFixed(1);
  var outer = (GLOW_OUTER_BLUR_PX * scale).toFixed(1);
  var innerAlpha = Math.min(GLOW_INNER_ALPHA * scale, 1);
  var outerAlpha = Math.min(GLOW_OUTER_ALPHA * scale, 1);
  return {
    peak:
      glowShadowLayer(inner, innerAlpha.toFixed(3))
      + ", " + glowShadowLayer(outer, outerAlpha.toFixed(3)),
    off:
      glowShadowLayer(inner, "0")
      + ", " + glowShadowLayer(outer, "0"),
  };
}

function glowShadowLayer(blurPx, alpha) {
  return (
    "0 0 " + blurPx + "px rgba(255, 255, 255, " + alpha + ")"
  );
}

// Write the glow's custom properties onto `el`, which is where the
// keyframes read them from. Shared so the Settings page preview and
// the live canvas cannot drift apart.
function overlaysApplyGlowVars(el, brightnessPercent, fadeMs) {
  if (!el) {
    return;
  }
  var shadow = overlaysGlowShadow(brightnessPercent);
  var duration = clampGlowValue(
    fadeMs,
    GLOW_FADE_MS_MIN,
    GLOW_FADE_MS_MAX,
    GLOW_FADE_MS_DEFAULT
  );
  el.style.setProperty("--token-birth-shadow", shadow.peak);
  el.style.setProperty("--token-birth-shadow-off", shadow.off);
  el.style.setProperty(
    "--token-birth-duration", duration + "ms"
  );
}

// Every page that animates anything has to ask this, so it lives
// here rather than being reimplemented per page. Unprefixed because
// it is a plain predicate about the environment, not part of the
// overlay model. Total: an environment without matchMedia is treated
// as having no preference, which is the same answer a browser that
// does not know the query gives.
function prefersReducedMotion() {
  try {
    return window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
  } catch (_e) {
    return false;
  }
}

// The glow pair a model class reads, falling back to the diffusion
// pair for a class that has no entry yet.
function overlaysGlowFor(settings, modelType) {
  var keys = GLOW_KEYS[modelType] || GLOW_KEYS.diffusion;
  return {
    brightness: settings[keys.brightness],
    fadeMs: settings[keys.fadeMs],
  };
}

// The stored settings, or the defaults when storage is unavailable
// or empty. parseSettings never throws, so this is total.
function overlaysLoadSettings() {
  var raw = null;
  try {
    raw = localStorage.getItem(SETTINGS_KEY);
  } catch (_e) {
    // Storage unavailable: parseSettings(null) yields the defaults.
  }
  return parseSettings(raw);
}

// Read the one preference that lives outside the Settings page, so
// both drawers agree without either of them owning the storage.
function overlaysReadHighlightTokens() {
  return overlaysLoadSettings().highlightTokens;
}

// Write it back through the whole blob, since the Settings page saves
// the same key wholesale and a partial write would drop its fields.
function overlaysWriteSetting(key, value) {
  var settings = overlaysLoadSettings();
  settings[key] = value;
  persistSet(SETTINGS_KEY, JSON.stringify(settings));
}

function overlaysWriteHighlightTokens(on) {
  overlaysWriteSetting("highlightTokens", !!on);
}

function overlaysWriteTpsMode(mode) {
  overlaysWriteSetting(
    "tpsMode", mode === "last" ? "last" : "total"
  );
}

// Field-wise equality, driving the Settings page Save/Reset enablement.
function settingsEqual(a, b) {
  return (
    a.highlightTokens === b.highlightTokens
    && a.diffusionText === b.diffusionText
    && a.diffusionTextMode === b.diffusionTextMode
    && a.gpuTicker === b.gpuTicker
    && a.tokenBirthGlow === b.tokenBirthGlow
    && a.glowBrightnessDiffusion === b.glowBrightnessDiffusion
    && a.glowFadeMsDiffusion === b.glowFadeMsDiffusion
    && a.glowBrightnessAutoregressive
      === b.glowBrightnessAutoregressive
    && a.glowFadeMsAutoregressive === b.glowFadeMsAutoregressive
    && a.tpsMode === b.tpsMode
  );
}

// ---- "New runs" registry (shared across generator + analytics) ----
//
// Run IDs saved since the user last viewed them in Analytics.
// Persisted server-side (via persistSet) so the generator's
// Analytics-link count and the analytics table's per-row dots agree
// across page navigations and survive restarts; a run is cleared
// individually when its detail is opened or when the run is deleted.

var OVERLAYS_NEW_RUNS_KEY = "diffusion_new_runs";

function overlaysReadNewRuns() {
  try {
    var raw = localStorage.getItem(OVERLAYS_NEW_RUNS_KEY);
    if (!raw) {
      return [];
    }
    var parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_e) {
    return [];
  }
}

function overlaysWriteNewRuns(ids) {
  // Write-through to the server so the cue survives restarts and stays
  // consistent across the generator, menu, and analytics pages.
  persistSet(OVERLAYS_NEW_RUNS_KEY, JSON.stringify(ids));
}

// Returns true if the run was newly added (was not already tracked),
// so callers can flash the "+1" cue only for genuinely new runs.
function overlaysAddNewRun(runId) {
  if (!runId) {
    return false;
  }
  var ids = overlaysReadNewRuns();
  if (ids.indexOf(runId) === -1) {
    ids.push(runId);
    overlaysWriteNewRuns(ids);
    return true;
  }
  return false;
}

function overlaysClearNewRun(runId) {
  var ids = overlaysReadNewRuns();
  var idx = ids.indexOf(runId);
  if (idx !== -1) {
    ids.splice(idx, 1);
    overlaysWriteNewRuns(ids);
  }
}

function overlaysNewRunCount() {
  return overlaysReadNewRuns().length;
}

function overlaysIsNewRun(runId) {
  return overlaysReadNewRuns().indexOf(runId) !== -1;
}

// ---- Last-run snapshot (generator, session-scoped) ----
//
// The generator's completed-run snapshot, written and read by app.js
// so a trip to Analytics and back restores the output. Only the key
// and the clear live here, because two pages have to drop it:
// activating a model ends in a location.reload() on the generator and
// on the menu alike, and by the time the generator boots it can no
// longer tell that reload from a navigation. The page that *starts*
// the switch can, so each clears on its way out and neither needs to
// know the other's storage.

var OVERLAYS_LAST_RUN_KEY = "diffusion_last_run";

function overlaysClearLastRun() {
  try {
    sessionStorage.removeItem(OVERLAYS_LAST_RUN_KEY);
  } catch (_e) {
    // Storage unavailable: there is nothing to clear.
  }
}

// ---- Activation progress (generator overlay + menu bar) ----
//
// How long a completed bar is left at 100% before the page moves on.
// Long enough to read as finishing rather than vanishing, and short
// enough to disappear into a load measured in seconds.
var OVERLAYS_LOAD_COMPLETE_HOLD_MS = 180;
//
// One poll of /api/models/activation, reduced to what the loading UI
// should show. Pure and shared because two pages render the same
// moment: the generator's full-screen overlay during a switch, and
// the menu's inline bar during a first activation. They looked
// different only because each had written its own wording.
//
// `mode` is one of:
//   "fill"  a real measurement: draw the track filled to `percent`.
//   "sweep" a phase with no measurable target: draw the track with an
//           indeterminate sweep and no number.
//   "hidden" nothing to show.
//
// The sweep is why there are three modes rather than a determinate
// flag. An activation opens with several seconds of work that cannot
// be measured at all: a worker process spawning, importing torch and
// transformers in its own virtualenv, and only then reading the
// checkpoint headers that give the bar its target. Parking a real bar
// at 0% through that reads as hung, and showing nothing (which is
// what the menu used to do) reads as nothing happening. A sweep is
// honest about having no number while still saying work is underway.
//
// A "loading" state with no usable progress is the same situation:
// see load_progress.load_target_bytes, which returns a zero total
// rather than guess at an unfamiliar checkpoint layout.
function overlaysActivationProgress(state, progress) {
  // The worker reaches ready in the same breath as its last progress
  // sample, and the supervisor drops progress on that transition, so
  // the closing 100% never survives the trip to the browser. Naming
  // the completed state here is what lets the bar finish instead of
  // disappearing at whatever the last poll happened to catch.
  if (state === "ready") {
    return { mode: "fill", percent: 100, label: "Ready" };
  }
  // Named rather than folded into "Loading" because it is a different
  // wait with a different cause, and saying which one is running is
  // the difference between a slow start and an apparently hung one.
  if (state === "starting") {
    return {
      mode: "sweep", percent: 0, label: "Starting worker",
    };
  }
  var out = { mode: "sweep", percent: 0, label: "Loading" };
  if (state === "downloading") {
    out.label = "Downloading";
  } else if (state === "loading") {
    // The sampler names the counter it is reporting, so this tracks
    // the weights whether they route through RAM first or stream
    // straight to the GPU. Before the first sample there is no stage
    // to name yet, and the generic label stands in.
    if (progress) {
      out.label =
        progress.stage === "device"
          ? "Moving to GPU"
          : "Loading weights";
    }
  } else {
    // idle, error, or a state this build does not know about: no
    // activation is in flight to draw one way or the other.
    return { mode: "hidden", percent: 0, label: "Loading" };
  }
  if (!progress || typeof progress.fraction !== "number") {
    return out;
  }
  // A zero total is the "could not measure this checkpoint" signal.
  // Keep the label, keep sweeping.
  if (!(progress.total_bytes > 0)) {
    return out;
  }
  var fraction = progress.fraction;
  if (fraction < 0) {
    fraction = 0;
  } else if (fraction > 1) {
    fraction = 1;
  }
  out.mode = "fill";
  out.percent = Math.round(fraction * 100);
  return out;
}


// ---- Draggable overlay drawer (generator + analytics) ----
//
// The drawer tucks against its container's top-right corner and
// slides out when its handle is clicked. Where it sits vertically is
// a matter of taste, since it can come to rest over whatever the run
// happened to draw there, so the collapsed drawer can be dragged up
// and down that edge. Open, it is a row of controls, and a stray drag
// on the way to a checkbox would only be a nuisance, so dragging is
// collapsed-only.
//
// Two things shape the implementation. The group already animates
// `transform` for its slide, so the drag moves `top` instead: on an
// absolutely positioned box that is a plain layout move, and the two
// can never contend for the same property. And this owns the handle's
// click as well as its drag, because only one listener can reliably
// decide whether a release was a click or the end of a drag; split
// across two, the answer would depend on registration order.

var OVERLAYS_DRAG_THRESHOLD_PX = 5;

// Largest `top` that still leaves the drawer fully inside its
// container. Zero when the container is shorter than the drawer,
// which pins it flush with the top rather than letting it hang out
// the bottom where the handle would be unreachable.
function overlaysDrawerMaxTop(group, container) {
  var room = container.clientHeight - group.offsetHeight;
  return room > 0 ? room : 0;
}

function overlaysClampDrawerTop(group, container, top) {
  if (top < 0) {
    return 0;
  }
  var max = overlaysDrawerMaxTop(group, container);
  return top > max ? max : top;
}

function overlaysReadDrawerTop(storageKey) {
  try {
    var raw = localStorage.getItem(storageKey);
    if (raw === null) {
      return null;
    }
    var value = parseFloat(raw);
    return isFinite(value) ? value : null;
  } catch (_e) {
    return null;
  }
}

// Wire one drawer. `onToggle` receives the open state the click asked
// for, so each page keeps its own open/close behavior while the
// click-versus-drag question is answered here, once.
function overlaysMakeDrawerDraggable(options) {
  var group = options.group;
  var handle = options.handle;
  var container = options.container;
  var storageKey = options.storageKey;
  var onToggle = options.onToggle;
  if (!group || !handle || !container) {
    return;
  }

  var pointerDown = false;
  var dragging = false;
  var justDragged = false;
  var startY = 0;
  var grabY = 0;
  // null means "wherever the stylesheet puts it". Kept here rather
  // than re-read from style.top so the saved value is a number the
  // whole time and never round-trips through a "123px" string.
  var currentTop = overlaysReadDrawerTop(storageKey);

  // Applied unclamped at startup on purpose: the group is `hidden` at
  // that point, so every box metric reads 0 and a clamp would flatten
  // any saved offset to zero. It was clamped when saved, and the
  // resize handler re-clamps once the box is real.
  if (currentTop !== null) {
    group.style.top = currentTop + "px";
  }

  function isOpen() {
    return group.classList.contains("open");
  }

  function setTop(top) {
    currentTop = overlaysClampDrawerTop(group, container, top);
    group.style.top = currentTop + "px";
  }

  handle.addEventListener("click", function () {
    // A drag ends with a click on release. Swallow it, or moving the
    // drawer would also open it.
    if (justDragged) {
      return;
    }
    if (typeof onToggle === "function") {
      onToggle(!isOpen());
    }
  });

  handle.addEventListener("pointerdown", function (event) {
    if (event.button !== undefined && event.button !== 0) {
      return;
    }
    if (isOpen()) {
      return;
    }
    pointerDown = true;
    dragging = false;
    startY = event.clientY;
    grabY = event.clientY - group.getBoundingClientRect().top;
    try {
      handle.setPointerCapture(event.pointerId);
    } catch (_e) {
      // Capture is best-effort; the move handler still tracks.
    }
  });

  handle.addEventListener("pointermove", function (event) {
    if (!pointerDown) {
      return;
    }
    var moved = Math.abs(event.clientY - startY);
    if (!dragging && moved > OVERLAYS_DRAG_THRESHOLD_PX) {
      dragging = true;
      group.classList.add("is-dragging");
    }
    if (!dragging) {
      return;
    }
    var top =
      event.clientY
      - container.getBoundingClientRect().top
      - grabY;
    setTop(top);
    event.preventDefault();
  });

  function endDrag(event) {
    if (!pointerDown) {
      return;
    }
    pointerDown = false;
    try {
      handle.releasePointerCapture(event.pointerId);
    } catch (_e) {
      // Capture may not be held; nothing to release.
    }
    if (!dragging) {
      return;
    }
    dragging = false;
    group.classList.remove("is-dragging");
    persistSet(storageKey, String(currentTop));
    justDragged = true;
    setTimeout(function () {
      justDragged = false;
    }, 0);
  }

  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);

  // A shrinking viewport can strand the drawer past its container's
  // bottom edge, where the handle cannot be reached to drag it back.
  window.addEventListener("resize", function () {
    if (currentTop === null || !group.offsetHeight) {
      return;
    }
    setTop(currentTop);
  });
}
