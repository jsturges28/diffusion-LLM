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

function overlaysBuildDiffLayerSpans(
  tokens, diff, isOriginal, blend, maskChar
) {
  var mask = maskChar || OVERLAYS_MASK_CHAR;
  var frag = document.createDocumentFragment();
  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    if (!tok) {
      continue;
    }
    var span = document.createElement("span");
    if (tok.m) {
      span.textContent = mask;
      span.style.color = "var(--mask-color)";
    } else {
      span.textContent = tok.t;
      span.style.color =
        overlaysDiffLayerColor(diff, i, isOriginal, blend);
    }
    frag.appendChild(span);
  }
  return frag;
}

// Build the two stacked layers for the "Diff vs Original" overlay:
// the original and edited runs drawn on top of each other with
// independent opacity and an optional difference blend. Pure: returns
// a DocumentFragment of two ``.diff-layer`` nodes; the caller owns the
// container (and must give it the stacking mode). ``diff`` is an
// overlaysComputeDiff() result; ``opts`` carries opacities in [0,100]
// (originalOpacity / editedOpacity) plus a ``blend`` flag.
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

  var origLayer = document.createElement("div");
  origLayer.className = "diff-layer diff-layer-original";
  origLayer.style.opacity = String(origOpacity / 100);
  origLayer.appendChild(
    overlaysBuildDiffLayerSpans(
      origTokens || [], diff, true, blend, maskChar
    )
  );

  var editLayer = document.createElement("div");
  editLayer.className = "diff-layer diff-layer-edited";
  editLayer.style.opacity = String(editedOpacity / 100);
  if (blend) {
    editLayer.style.mixBlendMode = "difference";
  }
  editLayer.appendChild(
    overlaysBuildDiffLayerSpans(
      editedTokens || [], diff, false, blend, maskChar
    )
  );

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
