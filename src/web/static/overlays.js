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
function overlaysBuildTokenSpan(index, tok, mask, opts) {
  // A missing token is a hole in the canvas, drawn as the mask glyph
  // rather than skipped: two layers only line up if both emit a span
  // per position.
  var masked = !tok || !!tok.m;
  var span = document.createElement("span");
  span.setAttribute("data-pos", String(index));
  span.className = "token-span "
    + (masked ? "token-mask" : "token-resolved");
  span.textContent = masked ? mask : tok.t;
  var color = opts.colorFor(index, tok);
  if (color) {
    span.style.color = color;
  }
  if (opts.titleFor) {
    span.title = opts.titleFor(index, tok);
  }
  return span;
}

// Build one stacked layer of token spans. ``opts`` carries the layer
// class, its opacity in [0,1], an ``interactive`` flag deciding which
// layer takes the pointer, a colorFor(index, token), and an optional
// titleFor(index, token). Pure: the caller owns the container and
// must give it the stacking mode.
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
// persistSet / PERSIST_KEYS). app.js applies the live effects; the
// Settings page (settings.js) edits them. Both source their defaults
// and parsing here so the schema lives in exactly one place. Commit
// Order is intentionally not here: it is a per-view overlay option now,
// not a saved preference.

var SETTINGS_KEY = "diffusion_settings";

var SETTINGS_DEFAULTS = {
  highlightTokens: false,
  diffusionText: false,
  diffusionTextMode: "default",
  gpuTicker: true,
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
  };
  if (!raw) {
    return settings;
  }
  try {
    var parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      settings.highlightTokens = !!parsed.highlightTokens;
      settings.diffusionText = !!parsed.diffusionText;
      settings.diffusionTextMode =
        parsed.diffusionTextMode === "cycle" ? "cycle" : "default";
      // Default on when the key is absent (older saved state).
      settings.gpuTicker = parsed.gpuTicker !== false;
    }
  } catch (_e) {
    // Corrupt storage: keep the defaults.
  }
  return settings;
}

// Field-wise equality, driving the Settings page Save/Reset enablement.
function settingsEqual(a, b) {
  return (
    a.highlightTokens === b.highlightTokens
    && a.diffusionText === b.diffusionText
    && a.diffusionTextMode === b.diffusionTextMode
    && a.gpuTicker === b.gpuTicker
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
