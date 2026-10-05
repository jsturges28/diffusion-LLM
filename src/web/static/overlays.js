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

// Edit markers: the tint behind an entropy column and the dashed line
// over it, on the generator's profile and on the Analytics chart
// alike. Shared because the two surfaces draw the same mark and had
// drifted into keeping identical values under different names.
//
// A marker whose frame is known takes its hue from commitColor
// instead, so an edit reads against the same early-to-late scale as
// the tokens do. These stay as the fallback for an edit that cannot
// be placed in the run, which is the shape a pre-frame_index edit log
// has, and they are still the colour .token-remasked uses in
// style.css, so the fallback matches the tokens it annotates.
var OVERLAYS_EDIT_COLOR = "#ff9f1c";
// Drawn through globalAlpha rather than baked into the colour, since
// commitColor returns an opaque hsl() string and the alpha has to
// apply to either source. The line sits under full strength: it
// competes with the entropy bars now, and at full opacity it read as
// the loudest thing in a strip it is only annotating.
var OVERLAYS_EDIT_LINE_ALPHA = 0.62;
var OVERLAYS_EDIT_TINT_ALPHA = 0.15;

// One marker's colour: the run-relative hue of the frame the edit was
// made at, or the flat fallback when the frame is unknown. Kept here
// rather than in either page so a marker cannot mean one thing on the
// generator and another in Analytics.
function overlaysEditColor(frame, maxFrame) {
  if (typeof frame === "number" && frame >= 0) {
    return commitColor(frame, maxFrame);
  }
  return OVERLAYS_EDIT_COLOR;
}

// Whether a token's text renders with no horizontal extent, so the
// cross-highlight has to stand a marker where it sits instead of
// tinting a box that is zero pixels wide.
//
// Decided from the text rather than measured, because measuring means
// reading a layout box inside a hover handler. Line breaks and the
// empty string are the cases that occur: a space and a tab both have
// width, so they are deliberately not included.
var OVERLAYS_ZERO_WIDTH_TEXT = /^[\r\n]*$/;

function overlaysTokenIsZeroWidth(text) {
  if (typeof text !== "string") {
    return false;
  }
  return OVERLAYS_ZERO_WIDTH_TEXT.test(text);
}

// Divergence coloring: changed tokens glow magenta, unchanged tokens
// fade to a dim neutral so an intervention's footprint stands out.
function diffColor(changed) {
  if (changed) {
    return "hsl(320, 80%, 66%)";
  }
  return "hsl(0, 0%, 45%)";
}

// How many remasked positions a tooltip names before it starts
// counting instead. A chart tooltip is sized by its longest line, so
// an unbounded list is a box that grows with the edit: 44 positions
// ran it off the side of the chart. Five is enough to see where an
// edit landed, and the exact identity of the 44th is not something
// anyone reads off a hover.
var OVERLAYS_REMASK_LIST_MAX = 5;

// A frame's remask selection, as the lines a chart tooltip should
// draw. Two of them rather than one sentence, because the box is
// sized by its longest line and it is drawn onto the chart canvas,
// so it cannot escape a 380px column no matter where it is placed.
// Truncating to five alone still left 60 characters against a budget
// of about 57; splitting the count from the list drops the longest
// line to the mid thirties and gives the numbers room to grow.
function overlaysRemaskSummary(positions) {
  var count = positions.length;
  var plural = count !== 1 ? "s" : "";
  var shown = positions.slice(0, OVERLAYS_REMASK_LIST_MAX);
  var rest = count - shown.length;
  var body = shown.join(", ");
  if (rest > 0) {
    body += ", ... and " + rest + " others";
  }
  return [
    "User remasked " + count + " token" + plural + ":",
    "[" + body + "]",
  ];
}

// A still-unsettled position fades by how sure the model is of what
// it is holding there, so a canvas shows its own certainty forming
// rather than a flat field.
//
// The curve is concave on purpose, and it was chosen from the data
// rather than by eye. Measured across a 128-step LLaDA run, the
// median confidence of a masked position sits between 0.11 and 0.21
// for the whole run, so a linear map crowds nearly every position
// into the bottom of the range: the previous ramp, linear to a 0.4
// cap over a 0.35 floor, put a typical frame between 0.48 and 0.65,
// a spread too narrow to see on 14px text. Taking the square root
// spends the channel where the values actually are.
var MASK_OPACITY_FLOOR = 0.05;

function overlaysMaskOpacity(c) {
  // Absent is not zero, and the difference matters at this floor.
  // Zero means the model was asked and had no idea, which earns the
  // near-invisible end of the ramp. Absent means nothing was ever
  // measured here: LLaDA's opening frame, and runs saved before
  // their model measured every position. Grading those would draw a
  // confident claim about a number nobody has, on a whole canvas at
  // once, so they stay solid.
  //
  // DiffusionGemma used to belong on that list whenever its Entropy
  // Signal was off. It no longer has one, so for that model this is
  // now a statement about old runs rather than a live case.
  //
  // A position with no token at all is a third case and does not
  // reach here: the callers send it to the floor, because a hole is
  // structural padding that exists only so two stacked layers line
  // up, and drawing it at full strength would make the emptier layer
  // the loudest thing on the canvas.
  if (typeof c !== "number") {
    return 1;
  }
  var clamped = Math.max(0, Math.min(1, c));
  return MASK_OPACITY_FLOOR
    + (1 - MASK_OPACITY_FLOOR) * Math.sqrt(clamped);
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

// Faded twin, for positions a scrubbed frame has not reached yet.
// Alpha rather than a darker hue, so the ramp is still legible and
// the bar still reads as itself: the shape of the tail is what makes
// scrubbing back through a run worth doing.
//
// Its own function because withAlpha takes hex and this ramp is
// generated in HSL. Baked into the per-bar fill on the Analytics
// chart, since Chart.js has no per-bar opacity of its own.
var ENTROPY_DIM_ALPHA = 0.2;

function entropyDimColor(e) {
  var frac = overlaysEntropyFraction(e);
  var sat = Math.round(55 + 30 * frac);
  var light = Math.round(52 + 8 * frac);
  return "hsla(" + overlaysEntropyHue(e) + ", " + sat + "%, "
    + light + "%, " + ENTROPY_DIM_ALPHA + ")";
}

// The window forgetting is drawn over. Forgetting is the share of a
// state-space model's recurrent state that reading one token erased,
// and on the pinned Mamba-3 real text puts the middle ninety percent
// of its tokens between about 0.12 and 0.30 (two saved runs, one per
// device), with thin tails: digits below, line breaks and sentence
// openers above. A ramp from 0 to the busiest token seen left the
// middle half of a run within four points of lightness of itself,
// which read as one colour, so this one spans where tokens fall and
// the tails clamp to its ends.
//
// Fixed rather than fitted to each run, on purpose: stretching a run
// over its own range would paint a flat one, a model repeating a
// single token, in full contrast from noise, which is the reading the
// forgetting check exists to rule out.
var OVERLAYS_FORGETTING_FLOOR = 0.12;
var OVERLAYS_FORGETTING_CEILING = 0.30;

// Where a value sits in that window, in [0,1].
function overlaysForgettingFraction(f) {
  if (typeof f !== "number" || !isFinite(f)) {
    return 0;
  }
  var span = OVERLAYS_FORGETTING_CEILING - OVERLAYS_FORGETTING_FLOOR;
  var at = (f - OVERLAYS_FORGETTING_FLOOR) / span;
  return Math.max(0, Math.min(1, at));
}

// Dim slate-violet for a token that erased little, bright lilac for
// one that erased much. Lightness and saturation carry most of it and
// a small turn of hue, blue-violet to pink-violet, carries the rest:
// how dark a dim token may go is capped by legibility on the canvas,
// and lightness alone measured under half the heatmap's contrast on
// real runs. The hue stays inside the violet band, clear of the blue
// end of the entropy ramp and of the diff's magenta.
function forgettingColor(f) {
  var frac = overlaysForgettingFraction(f);
  var hue = Math.round(250 + 40 * frac);
  var sat = Math.round(20 + 80 * frac);
  var light = Math.round(46 + 40 * frac);
  return "hsl(" + hue + ", " + sat + "%, " + light + "%)";
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

// Distance from the token to the popover's near edge. Deliberately a
// hairline rather than 0: the trip from a token up into the popover
// crosses this gap, so every pixel of it is reach the pointer has to
// survive, but at 0 the border stops reading as a separate surface
// and subpixel rounding of the token's rect can drop the box shadow
// onto the glyph being read.
var OVERLAYS_POPOVER_GAP = 2;

// Place it vertically, preferring above the token. That preference
// began as a workaround, since the browser drew a native title
// tooltip below the cursor that nothing could move; the tooltip is
// gone now (see overlaysRenderTokenMetrics) but above still reads
// better, because it leaves the text you are pointing at uncovered.
//
// ``canvasTop`` is the viewport y of the token canvas's own top edge,
// and it is what the popover has to clear, not the viewport's. The
// canvas starts well down the page, so a viewport-only test let a
// token in the first line or two push the popover up out of the
// canvas and over the metrics strip above it. Overlapping the tokens
// is the whole point; overlapping their readout is not.
function overlaysPopoverTop(tokenRect, popoverBox, canvasTop) {
  var ceiling = typeof canvasTop === "number"
    ? Math.max(8, canvasTop)
    : 8;
  var above =
    tokenRect.top - popoverBox.height - OVERLAYS_POPOVER_GAP;
  if (above >= ceiling) {
    return above;
  }
  // Below is safe wherever this branch is reached: it is reached only
  // for a token near the top of the canvas, which is the case with
  // the most room underneath it.
  var below = Math.min(
    tokenRect.bottom + OVERLAYS_POPOVER_GAP,
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
  heading.appendChild(overlaysBuildAltPagers(page, onPage));
  return heading;
}

// The two arrows, the one toward the page on show disabled.
function overlaysBuildAltPagers(page, onPage) {
  var pager = document.createElement("span");
  pager.className = "alt-pager";
  for (var i = 0; i < OVERLAYS_ALT_PAGES.length; i++) {
    pager.appendChild(
      overlaysBuildAltPager(
        OVERLAYS_ALT_PAGES[i], page, onPage
      )
    );
  }
  return pager;
}

// The heading for a diffusion run's candidates: the position, and the
// step they were read at. "As of" when that is an earlier step than
// the frame on screen, because the capture thinned to a stride on a
// long run and skipped this one. On an edited run ``page`` names the
// run being read, and ``onPage``, given only when the other run has
// candidates there too, adds the arrows after the step; null leaves
// the plain title.
function overlaysBuildStepHeading(pos, step, onScreen, page, onPage) {
  var heading = document.createElement("div");
  heading.className = "alt-heading";
  var title = document.createElement("span");
  title.textContent = "Position " + (pos + 1) + ": "
    + (page ? overlaysAltPageLabel(page) : "candidates");
  heading.appendChild(title);
  var end = document.createElement("span");
  end.className = "alt-heading-end";
  var when = document.createElement("span");
  when.className = "alt-step";
  when.textContent = step === onScreen
    ? "Step " + step
    : "As of step " + step;
  end.appendChild(when);
  if (page && onPage) {
    end.appendChild(overlaysBuildAltPagers(page, onPage));
  }
  heading.appendChild(end);
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

// One candidate row: token text, proportional bar, probability. Both
// pages had their own copy of this, identical but for returning a row
// against a fragment, and both needed the same hover wiring added, so
// they share one now for the same reason the metrics strip is shared.
//
// ``onHover`` is handed a reading on enter and null on leave, and is
// what feeds the strip's right-hand readout. Optional, because the
// row is also drawn where nothing is listening. ``index`` is the
// row's place in the list, which is also its rank; see
// overlaysAltRank.
function overlaysBuildAltRow(alt, chosenId, onHover, index) {
  var row = document.createElement("div");
  row.className = "alt-row";
  if (alt.id === chosenId) {
    row.classList.add("alt-row-chosen");
  }
  // An explicit rank means this is the appended entry: the token the
  // position committed, from outside the captured set. Marked so it
  // can read as an answer rather than an offer, since substituting
  // the token already sitting there would re-run to the same place.
  if (typeof alt.rank === "number") {
    row.classList.add("alt-row-outside");
  }
  row.setAttribute("data-alt-id", String(alt.id));

  var text = document.createElement("span");
  text.className = "alt-text";
  text.textContent = overlaysAltDisplay(alt.t);
  row.appendChild(text);

  var clamped = Math.max(0, Math.min(1, alt.p));
  var bar = document.createElement("span");
  bar.className = "alt-bar";
  var fill = document.createElement("span");
  fill.className = "alt-bar-fill";
  fill.style.width = Math.round(clamped * 100) + "%";
  bar.appendChild(fill);
  row.appendChild(bar);

  var prob = document.createElement("span");
  prob.className = "alt-prob";
  prob.textContent = (clamped * 100).toFixed(1) + "%";
  row.appendChild(prob);

  if (onHover) {
    overlaysBindAltHover(
      row,
      {
        t: alt.t,
        p: alt.p,
        rank: overlaysAltRank(alt, index),
      },
      onHover
    );
  }
  return row;
}

// A candidate's rank, which for the captured set is simply where it
// sits in the list: the sampler takes them with torch.topk, so the
// order is descending by construction and the index is the rank.
//
// An explicit ``rank`` wins where one exists, which is the token the
// run actually chose when it fell outside the captured set. That one
// is appended after the others, so its index would claim it was the
// sixth likeliest when it may have been the forty-thousandth.
function overlaysAltRank(alt, index) {
  if (typeof alt.rank === "number" && alt.rank > 0) {
    return alt.rank;
  }
  if (typeof index !== "number" || index < 0) {
    return null;
  }
  return index + 1;
}

// mouseenter and mouseleave rather than mouseover: these do not
// bubble, so the row's own children cannot retrigger them and the
// readout holds steady as the pointer crosses the bar and the
// percentage inside one row.
function overlaysBindAltHover(row, reading, onHover) {
  row.addEventListener("mouseenter", function () {
    onHover(reading);
  });
  row.addEventListener("mouseleave", function () {
    onHover(null);
  });
}

// Which tokenizer cut these candidates, as a caption at the foot of
// the popover. Not chrome: every row above it is a piece of one
// specific vocabulary, and which vocabulary that is decides whether
// a word is one token or three. Returns null when unknown, so the
// popover simply has no footer rather than an empty one.
//
// The two pages source this differently on purpose. The generator
// asks the resident worker, since the candidates were just produced
// by it; Analytics asks the run, since its checkpoint may since have
// been swapped out. Bracket access because "class" is the payload's
// field name; see describe_tokenizer in worker_base.py.
function overlaysBuildAltTokenizer(tokenizer) {
  var tok = tokenizer || {};
  var name = tok["class"];
  if (!name) {
    return null;
  }
  var footer = document.createElement("div");
  footer.className = "alt-tokenizer";
  var text = String(name);
  if (tok.vocab_size) {
    text += " \u00B7 "
      + overlaysCompactCount(tok.vocab_size)
      + " vocab";
  }
  footer.textContent = text;
  // The class name is the half that can be long, so it is the half
  // that ellipsizes; the vocab stays whole.
  footer.title = text;
  return footer;
}

// Thousands as "128k". The footer shares a 190px popover with five
// candidate rows, so a grouped six-digit figure would either wrap it
// or push the box wider than the tokens it is annotating.
function overlaysCompactCount(count) {
  var n = Number(count);
  if (!isFinite(n) || n < 1000) {
    return String(count);
  }
  return Math.round(n / 1000) + "k";
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

// ---- Token metrics strip ----
//
// The always-present readout above each page's token canvas. It
// replaced a native ``title`` tooltip, which had three problems this
// fixes: the browser delays it by around half a second and will not
// let that be configured, it cannot be styled or positioned, and it
// is bound to one element, so hovering the entropy chart could never
// feed it. One strip serves both hover sources on both pages.
//
// The pages own every decision about *what* is under the pointer
// (which frame, which overlay, which stacked layer) and hand this a
// plain reading. Keeping the formatting here is what stops the two
// pages drifting into two dialects of the same readout.
//
// A reading is null when nothing is hovered, or:
//
//   { position, total, tokenText, masked, maskChar,
//     confidence, entropy, extra, runLabel }
//
// confidence and entropy are null when the run did not record them,
// which is different from zero and is rendered differently.
// maskChar is the caller's own glyph, because the generator swaps
// MASK_CHAR per model and the strip has to draw what the canvas does.

var OVERLAYS_METRIC_BLANK = "\u2013";

// Build the strip's children once. Each page calls this at boot; the
// markup carries only the empty container, so the structure is
// defined in exactly one place.
function overlaysBuildTokenMetrics(el) {
  if (!el) {
    return;
  }
  el.textContent = "";
  var nodes = {
    token: overlaysMetricToken(el),
    position: overlaysMetricField(el, "Position", false),
    confidence: overlaysMetricField(el, "Confidence", true),
    entropy: overlaysMetricField(el, "Entropy", true),
    // After the four fixed fields, so nothing to its left moves when
    // it appears and disappears under the pointer.
    candidate: overlaysMetricCandidate(el),
    extra: overlaysMetricTrailer(el, "token-metrics-extra"),
    run: overlaysMetricTrailer(el, "token-metrics-run"),
  };
  // Cached rather than re-queried per hover. Mouseover fires on every
  // token the pointer crosses, and this keeps that to attribute
  // writes on nodes we already hold.
  el.overlaysMetricNodes = nodes;
  overlaysRenderTokenMetrics(el, null);
}

function overlaysMetricToken(el) {
  var span = document.createElement("span");
  span.className = "token-metrics-token";
  el.appendChild(span);
  return span;
}

// A label, its value, and optionally a bar that reuses the overlay
// ramps, so the strip reads in the same colors as the canvas above
// it rather than inventing a third language for the same numbers.
function overlaysMetricField(el, label, withBar) {
  var field = document.createElement("span");
  field.className = "token-metrics-field";
  var name = document.createElement("span");
  name.className = "token-metrics-label";
  name.textContent = label;
  field.appendChild(name);
  var value = document.createElement("span");
  value.className = "token-metrics-value";
  field.appendChild(value);
  var fill = null;
  if (withBar) {
    var bar = document.createElement("span");
    bar.className = "token-metrics-bar";
    fill = document.createElement("span");
    fill.className = "token-metrics-fill";
    bar.appendChild(fill);
    field.appendChild(bar);
  }
  el.appendChild(field);
  return { value: value, fill: fill };
}

function overlaysMetricTrailer(el, className) {
  var span = document.createElement("span");
  span.className = className;
  el.appendChild(span);
  return span;
}

// The detail readout for a candidate under the pointer in the
// popover. It lives here rather than on the row because the row has
// no width for it: the popover is 320px at its widest, shared with
// five rows, and the strip has half its length standing empty.
//
// A green chip heads it, mirroring the grey chip that heads the left
// group, and the colors carry the distinction: grey is the token the
// run committed, green is one it merely weighed.
function overlaysMetricCandidate(el) {
  var group = document.createElement("span");
  group.className = "token-metrics-candidate";

  var chip = document.createElement("span");
  chip.className = "token-metrics-alt";
  group.appendChild(chip);

  var probability = document.createElement("span");
  probability.className = "token-metrics-value";
  group.appendChild(probability);

  var rank = document.createElement("span");
  rank.className = "token-metrics-rank";
  group.appendChild(rank);

  el.appendChild(group);
  return { group: group, chip: chip, value: probability, rank: rank };
}

// Render a reading, or the idle state when it is null. Every field is
// written on every call, so no stale value can survive a move onto a
// token that lacks it.
function overlaysRenderTokenMetrics(el, reading) {
  if (!el || !el.overlaysMetricNodes) {
    return;
  }
  var nodes = el.overlaysMetricNodes;
  var idle = !reading;
  el.classList.toggle("is-idle", idle);
  nodes.token.textContent = idle
    ? OVERLAYS_METRIC_BLANK
    : overlaysMetricTokenText(reading);
  nodes.position.value.textContent = idle
    ? OVERLAYS_METRIC_BLANK
    : (reading.position + 1) + " / " + reading.total;
  overlaysMetricNumber(
    nodes.confidence,
    idle ? null : reading.confidence,
    overlaysMetricConfidenceBar
  );
  overlaysMetricNumber(
    nodes.entropy,
    idle ? null : reading.entropy,
    overlaysMetricEntropyBar
  );
  nodes.extra.textContent = idle ? "" : (reading.extra || "");
  nodes.run.textContent = idle ? "" : (reading.runLabel || "");
  overlaysRenderMetricCandidate(
    nodes.candidate, idle ? null : reading.candidate
  );
}

// Hidden outright when nothing is hovered in the popover, unlike the
// left group, which stays visible while idle as a key to what the
// strip reports. This one has no such duty: it appears only while
// you are reading a specific candidate, so an idle placeholder for it
// would be a label for a question nobody asked.
function overlaysRenderMetricCandidate(nodes, candidate) {
  if (!nodes) {
    return;
  }
  if (!candidate) {
    nodes.group.hidden = true;
    return;
  }
  nodes.group.hidden = false;
  nodes.chip.textContent = overlaysAltDisplay(candidate.text);
  // Full precision, which is the point of putting it here: the row
  // in the popover rounds to a tenth of a percent, and a token the
  // model gave 1e-5 rounds away to nothing there.
  nodes.value.textContent = overlaysMetricProbability(
    candidate.probability
  );
  nodes.rank.textContent = overlaysMetricRank(candidate);
}

// Significant figures rather than fixed decimals, so a probability
// stays legible across the five orders of magnitude a typed token can
// span. toPrecision holds fixed notation down to about 1e-6 and
// switches to exponential below, which is where fixed stops being
// readable anyway.
function overlaysMetricProbability(probability) {
  // Explicitly typed, not coerced: a pending measurement arrives as
  // null, and Number(null) is 0, which would report a token the model
  // has not been asked about yet as one it gave no weight to.
  if (typeof probability !== "number" || !isFinite(probability)) {
    return OVERLAYS_METRIC_BLANK;
  }
  if (probability === 0) {
    return "0";
  }
  return probability.toPrecision(3);
}

// How many tokens the model preferred. The reading that survives when
// the probability has collapsed: "#41,203 of 128,256" says what a
// rounded zero cannot. Omitted for a captured candidate, whose rank
// is its position in the list you are already looking at.
// The denominator is optional. Runs saved before the model's output
// width was recorded have none, and a bare "#3" still says the useful
// thing; inventing a width from the tokenizer's vocab_size beside it
// would be wrong wherever the embedding is padded.
function overlaysMetricRank(candidate) {
  if (!candidate.rank) {
    return "";
  }
  var rank = "#" + Number(candidate.rank).toLocaleString();
  if (!candidate.vocabSize) {
    return rank;
  }
  return rank + " of "
    + Number(candidate.vocabSize).toLocaleString();
}

// A masked position has no text of its own to show, so it reports the
// glyph the canvas is drawing there rather than an empty slot.
function overlaysMetricTokenText(reading) {
  if (reading.masked) {
    return reading.maskChar || OVERLAYS_MASK_CHAR;
  }
  return overlaysAltDisplay(reading.tokenText);
}

// Absent reads as a dash, not as zero: a run saved without the signal
// is not a run that was certain, and the old tooltip conflated those.
function overlaysMetricNumber(field, value, bar) {
  var known = typeof value === "number" && isFinite(value);
  field.value.textContent = known
    ? value.toFixed(3)
    : OVERLAYS_METRIC_BLANK;
  if (!field.fill) {
    return;
  }
  var shape = known ? bar(value) : { width: 0, color: "" };
  field.fill.style.width = shape.width + "%";
  field.fill.style.background = shape.color;
}

function overlaysMetricConfidenceBar(value) {
  var clamped = Math.max(0, Math.min(1, value));
  return { width: clamped * 100, color: heatColor(clamped) };
}

function overlaysMetricEntropyBar(value) {
  return {
    width: overlaysEntropyFraction(value) * 100,
    color: entropyColor(value),
  };
}

// The strip's overlay line while Forgetting is on, or "" for a token
// that carries none. The line rather than a field of its own, because
// three of the four models never report the value, and a permanent
// field would read as a dash on every run of theirs.
function overlaysForgettingReading(tok) {
  if (!tok || typeof tok.f !== "number" || !isFinite(tok.f)) {
    return "";
  }
  return "Forgetting: " + tok.f.toFixed(3);
}

// ---- KGW watermark membership and detector readout ----
//
// These colors describe keyed set membership only. They never mean
// correct/incorrect, confidence, quality, or authorship. A token with
// ``we=false`` was excluded from detector evidence (the first output
// token and a user-forced What If token today), so it also gets the
// non-color outline/pattern class defined in style.css.
var OVERLAYS_WATERMARK_FAVORED = "#35d07f";
var OVERLAYS_WATERMARK_COMPLEMENT = "#ff6b6b";
var OVERLAYS_WATERMARK_EVIDENCE_MIN = 50;

function watermarkColor(tok) {
  if (!tok || typeof tok.g !== "boolean") {
    return null;
  }
  return tok.g
    ? OVERLAYS_WATERMARK_FAVORED
    : OVERLAYS_WATERMARK_COMPLEMENT;
}

function overlaysWatermarkTokenClass(tok) {
  if (!tok || typeof tok.g !== "boolean") {
    return "";
  }
  var classes = [
    tok.g
      ? "token-watermark-favored"
      : "token-watermark-complement",
  ];
  if (tok.we === false) {
    classes.push("token-watermark-excluded");
  }
  return classes.join(" ");
}

function overlaysWatermarkReading(tok) {
  if (!tok || typeof tok.g !== "boolean") {
    return "";
  }
  var set = tok.g ? "keyed favored set" : "keyed complement";
  if (tok.we === false) {
    return "Watermark: " + set + "; excluded from detector score";
  }
  if (tok.we === true) {
    return "Watermark: " + set + "; included in detector score";
  }
  return "Watermark: " + set + "; evidence flag unavailable";
}

function overlaysWatermarkDescription(tok) {
  var reading = overlaysWatermarkReading(tok);
  return reading ? reading.replace("Watermark: ", "") : "";
}

function overlaysTokensCarryWatermark(tokens) {
  if (!Array.isArray(tokens)) {
    return false;
  }
  for (var i = 0; i < tokens.length; i++) {
    if (tokens[i] && typeof tokens[i].g === "boolean") {
      return true;
    }
  }
  return false;
}

// Recompute one frame's score from durable token evidence. ``p0`` is
// the exact worker-attested null probability; membership alone cannot
// recover it for a padded output vocabulary, so an invalid or absent
// value returns null rather than inventing one.
function overlaysWatermarkStats(tokens, p0) {
  if (!overlaysTokensCarryWatermark(tokens)) {
    return null;
  }
  if (
    typeof p0 !== "number"
    || !isFinite(p0)
    || p0 <= 0
    || p0 >= 1
  ) {
    return null;
  }
  var green = 0;
  var scored = 0;
  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    if (
      !tok
      || typeof tok.g !== "boolean"
      || typeof tok.we !== "boolean"
    ) {
      return null;
    }
    if (tok.we !== true) {
      continue;
    }
    scored += 1;
    if (tok.g) {
      green += 1;
    }
  }
  var variance = scored * p0 * (1 - p0);
  var zScore = scored > 0
    ? (green - scored * p0) / Math.sqrt(variance)
    : 0;
  return {
    status: scored < OVERLAYS_WATERMARK_EVIDENCE_MIN
      ? "insufficient_evidence"
      : "scored",
    green_count: green,
    scored_count: scored,
    green_rate: scored > 0 ? green / scored : 0,
    z_score: zScore,
    p0: p0,
  };
}

function overlaysWatermarkDisplayStatus(stats, threshold) {
  if (
    !stats
    || stats.scored_count < OVERLAYS_WATERMARK_EVIDENCE_MIN
  ) {
    return "insufficient_evidence";
  }
  if (
    typeof threshold !== "number"
    || !isFinite(threshold)
    || threshold < 0
  ) {
    throw new RangeError(
      "watermark display threshold must be non-negative"
    );
  }
  return stats.z_score >= threshold
    ? "threshold_crossed"
    : "threshold_not_crossed";
}

function overlaysBuildWatermarkReadout(el) {
  if (!el) {
    return;
  }
  el.textContent = "";
  var nodes = {
    label: overlaysWatermarkReadoutNode(
      el, "watermark-readout-label", "Experimental KGW"
    ),
    counts: overlaysWatermarkReadoutNode(
      el, "watermark-readout-counts", ""
    ),
    rate: overlaysWatermarkReadoutNode(
      el, "watermark-readout-rate", ""
    ),
    score: overlaysWatermarkReadoutNode(
      el, "watermark-readout-score", ""
    ),
    nullRate: overlaysWatermarkReadoutNode(
      el, "watermark-readout-null", ""
    ),
    status: overlaysWatermarkReadoutNode(
      el, "watermark-readout-status", ""
    ),
  };
  el.overlaysWatermarkNodes = nodes;
  el.hidden = true;
}

function overlaysWatermarkReadoutNode(el, className, text) {
  var node = document.createElement("span");
  node.className = className;
  node.textContent = text;
  el.appendChild(node);
  return node;
}

// ``reading`` is {stats, threshold, recordConsistency?}. The threshold is
// explicitly called a display threshold because it changes no score,
// token, or statistical null and cannot identify who wrote text.
function overlaysRenderWatermarkReadout(el, reading) {
  if (!el || !el.overlaysWatermarkNodes) {
    return;
  }
  if (!reading || !reading.stats) {
    el.hidden = true;
    el.removeAttribute("title");
    return;
  }
  var stats = reading.stats;
  var threshold = reading.threshold;
  var status = overlaysWatermarkDisplayStatus(stats, threshold);
  var nodes = el.overlaysWatermarkNodes;
  nodes.counts.textContent =
    "green/scored "
    + stats.green_count + "/" + stats.scored_count;
  nodes.rate.textContent =
    "green rate " + (stats.green_rate * 100).toFixed(1) + "%";
  nodes.score.textContent = "z " + stats.z_score.toFixed(2);
  nodes.nullRate.textContent =
    "p0 " + overlaysWatermarkExactProbability(stats.p0);
  nodes.status.textContent = overlaysWatermarkStatusText(
    status, threshold, reading.recordConsistency
  );
  nodes.status.className =
    "watermark-readout-status watermark-status-" + status;
  if (reading.recordConsistency === "mismatch") {
    nodes.status.className += " watermark-record-mismatch";
  }
  el.title = (
    "Experimental keyed-set score. Green means the keyed favored"
    + " set and red its complement, never correctness or confidence."
    + " The configurable z threshold is display-only and is not an"
    + " AI/human or authorship verdict."
  );
  el.hidden = false;
}

function overlaysWatermarkExactProbability(value) {
  return String(value);
}

function overlaysWatermarkStatusText(
  status, threshold, recordConsistency
) {
  var text;
  if (status === "insufficient_evidence") {
    text = "insufficient evidence (<50 scored)";
  } else if (status === "threshold_crossed") {
    text = "threshold crossed @ display z " + threshold;
  } else {
    text = "threshold not crossed @ display z " + threshold;
  }
  if (recordConsistency === "consistent") {
    text += " \u00b7 record counts consistent";
  } else if (recordConsistency === "mismatch") {
    text += " \u00b7 record counts differ from attestation";
  } else if (recordConsistency === "unavailable") {
    text += " \u00b7 record consistency unavailable";
  }
  return text;
}

// Per-position commit step for a run: the step after which a position
// last changed to its final value. Derived purely from the frame
// token stream (the final frame is ground truth), so it is exact for
// LLaDA (resolved tokens are frozen) and a "settle" proxy for
// DiffusionGemma. Positions still unresolved at the last frame get
// -1 (left uncolored).
//
// Takes a reader and a count rather than an array, because the two
// pages no longer agree on what a run is stored as. A diffusion run
// really is a list of per-frame arrays; a run that only grows is one
// flat list whose frames are prefixes, and materialising those to
// walk them here would rebuild the exact N(N+1)/2 the storage change
// removed. Asking for frame f leaves that decision where it belongs.
//
// ``readFrame(f)`` returns that frame's token array, or null.
function overlaysComputeCommitSteps(readFrame, frameCount) {
  if (frameCount === 0) {
    return [];
  }
  var finalTokens = readFrame(frameCount - 1);
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
      var ft = readFrame(f);
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

// The commit steps of a run that only grows, without reading it.
//
// A position appears at its final value and nothing behind it moves,
// so every one of them settles at step 0. That is what the general
// walk above computes for such a run, checked against real saved
// runs rather than reasoned about; this returns it directly instead
// of assembling N prefixes to rediscover it.
function overlaysAppendCommitSteps(positions) {
  var steps = new Array(positions.length);
  for (var i = 0; i < positions.length; i++) {
    var token = positions[i];
    steps[i] = !token || token.m ? -1 : 0;
  }
  return steps;
}

// One array of per-frame token arrays, read the way the folder above
// wants. For the pages that still hold their run that way.
function overlaysFrameReader(frames) {
  return function (index) {
    var frame = frames[index];
    return frame === undefined ? null : frame;
  };
}

// ---- Entropy at a frame ----
//
// A DiffusionGemma canvas ends on a committed frame, and a commit
// carries no entropy of its own: the model accepted the canvas rather
// than drawing it from a distribution. So every view that reads
// entropy at a frame asks which frame's entropy describes it. A frame
// carrying any answers for itself. A commit borrows its canvas's last
// draft, which is what the model weighed when it committed, and the
// views say "as of step N", as the candidate popover does when it
// borrows an earlier frame.

// Whether any position of a frame carries entropy.
function overlaysFrameHasEntropy(tokens) {
  if (!tokens) {
    return false;
  }
  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    if (tok && typeof tok.e === "number" && isFinite(tok.e)) {
      return true;
    }
  }
  return false;
}

// The frame whose entropy describes frame `index`: `index` itself
// when it carries any, else the latest earlier frame on the same
// canvas that does, else -1. `frameAt(f)` reads a frame's tokens and
// `canvasOf(f)` names its canvas. On an append stream every earlier
// frame is a prefix of this one, so it holds nothing this frame does
// not, and the search is not made.
function overlaysEntropyFrame(frameAt, canvasOf, index, isAppend) {
  if (index < 0) {
    return -1;
  }
  if (overlaysFrameHasEntropy(frameAt(index))) {
    return index;
  }
  if (isAppend) {
    return -1;
  }
  var canvas = canvasOf(index);
  for (var f = index - 1; f >= 0; f--) {
    if (canvasOf(f) !== canvas) {
      return -1;
    }
    if (overlaysFrameHasEntropy(frameAt(f))) {
      return f;
    }
  }
  return -1;
}

// How a view says its entropy was read at an earlier draft.
function overlaysEntropyAsOf(step) {
  return "as of step " + step;
}

// The metrics strip's trailing line, with that note added when the
// entropy it shows was borrowed. Joined to whatever the active
// overlay already says there, so neither hides the other.
function overlaysEntropyNote(extra, asOfStep) {
  if (typeof asOfStep !== "number") {
    return extra;
  }
  var note = "entropy " + overlaysEntropyAsOf(asOfStep);
  return extra ? extra + "  \u2022  " + note : note;
}

// ---- Adaptive stopping ----
//
// DiffusionGemma ends a canvas once two things hold at once: it is
// steady, unchanged for `stability_threshold` steps, and confident,
// its mean entropy below `confidence_threshold` nats. Everything
// needed to watch that happen is already on the frames: every
// position of a draft carries its entropy, `e`, and whether it
// changed since the last draft, `m`. A committed canvas carries no
// entropy at all, which is how its frame is told apart.
//
// Read off the frames rather than reported by the worker, and checked
// before it was built: on five saved runs, nine canvases and 144
// drafts, "both hold" was true on exactly the frame each canvas
// stopped. What it can get wrong is bounded by what the page is
// sent, each position's entropy to four decimals from a bf16 copy of
// the logits the model judged, so a mean within about 0.0001 nats of
// the threshold could read either way. The verdict on a committed
// canvas does not depend on that; see overlaysStopReadingAt.
//
// A rule is { threshold, steadySteps, budget }: the two parameters
// and the canvas's step budget, `max_denoising_steps`.

var OVERLAYS_STOP_DRAFT = "draft";
var OVERLAYS_STOP_COMMIT = "commit";
var OVERLAYS_STOP_NONE = "none";

// The trace's vertical scale, in nats. A canvas opens near 4.5, so
// ten leaves headroom; the floor sits a decade under the threshold,
// so the dashed line is never on the frame's edge whatever the run
// set it to.
var OVERLAYS_STOP_TRACE_TOP_NATS = 10;
var OVERLAYS_STOP_TRACE_FLOOR_RATIO = 0.1;
// The resource meter's footprint, so the two read as one family.
var OVERLAYS_STOP_TRACE_WIDTH = 62;
var OVERLAYS_STOP_TRACE_HEIGHT = 13;
var OVERLAYS_STOP_TRACE_LINE = "#b8b8b8";
var OVERLAYS_STOP_TRACE_RULE = "rgba(0, 255, 65, 0.45)";
var OVERLAYS_STOP_TRACE_MET = "#00ff41";
var OVERLAYS_STOP_TRACE_DOT = "#e6e6e6";

// The rule a run stopped by, from its parameters by their registry
// names, or null when they do not make a rule. `fallback` supplies
// any the run did not record: the generator passes the model's
// defaults, Analytics passes nothing because the server already did.
function overlaysStopRuleFrom(params, fallback) {
  var pick = function (name) {
    if (params && params[name] !== undefined) {
      return Number(params[name]);
    }
    return fallback ? Number(fallback[name]) : NaN;
  };
  var rule = {
    threshold: pick("confidence_threshold"),
    steadySteps: pick("stability_threshold"),
    budget: pick("max_denoising_steps"),
  };
  if (!(isFinite(rule.threshold) && rule.threshold > 0)) {
    return null;
  }
  var steady = rule.steadySteps;
  if (!(Number.isInteger(steady) && steady >= 0)) {
    return null;
  }
  if (!(Number.isInteger(rule.budget) && rule.budget >= 1)) {
    return null;
  }
  return rule;
}

// One frame's two measurements: the mean of `e` when every position
// carries one, and how many positions changed. Entropy is null where
// no position carries it, which is a committed canvas or a run saved
// before entropy was recorded everywhere. A frame where only some
// positions carry it reads as nothing at all rather than as a mean
// over that subset, which was the bias the 2026-08-28 change removed.
function overlaysStopSummary(tokens) {
  if (!tokens || tokens.length === 0) {
    return null;
  }
  var sum = 0;
  var measured = 0;
  var changed = 0;
  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    if (tok && typeof tok.e === "number" && isFinite(tok.e)) {
      sum += tok.e;
      measured += 1;
    }
    if (tok && tok.m) {
      changed += 1;
    }
  }
  if (measured === tokens.length) {
    return { entropy: sum / measured, changed: changed };
  }
  if (measured === 0) {
    return { entropy: null, changed: changed };
  }
  return null;
}

// The stopping state of every frame of a run, in one pass.
//
// `source` reads the run the way the pages hold it:
//   { count, readFrame(f), canvasAt(f), segmentStarts }
// where segmentStarts lists the first frame of each resumed segment,
// which is each edit's frame_index: the page truncates at that frame
// and the resume's first frame takes its place.
//
// Each entry is { kind, entropy, changed, steady, step, canvas }.
// `steady` counts consecutive drafts with nothing changed, as
// transformers keeps it: it restarts with every canvas and every
// resume, since each begins a history that cannot yet be steady.
// `step` is a draft's place in its canvas, from 1, and a resume
// carries on counting the canvas it branched from; a commit holds
// the number of drafts before it.
function overlaysStopTrack(source) {
  var starts = {};
  var segments = source.segmentStarts || [];
  for (var s = 0; s < segments.length; s++) {
    starts[segments[s]] = true;
  }
  var state = { canvas: null, steady: 0, step: 0 };
  var track = [];
  for (var f = 0; f < source.count; f++) {
    track.push(overlaysStopStep(
      state,
      overlaysStopSummary(source.readFrame(f)),
      source.canvasAt(f),
      starts[f] === true
    ));
  }
  return track;
}

// One frame of the track, advancing `state` in place.
function overlaysStopStep(state, summary, canvas, segmentStart) {
  if (canvas !== state.canvas) {
    state.canvas = canvas;
    state.steady = 0;
    state.step = 0;
  }
  var entry = {
    kind: OVERLAYS_STOP_NONE,
    entropy: null,
    changed: summary ? summary.changed : 0,
    steady: 0,
    step: state.step,
    canvas: canvas,
  };
  if (summary === null) {
    return entry;
  }
  if (summary.entropy === null) {
    // Only a commit if drafts with entropy came before it in this
    // canvas; otherwise this is a run that never measured any.
    if (state.step > 0 && summary.changed === 0) {
      entry.kind = OVERLAYS_STOP_COMMIT;
    }
    return entry;
  }
  state.step += 1;
  // A canvas's first draft and a resume's first frame begin a fresh
  // history, which cannot be steady whatever `m` says about them.
  var fresh = segmentStart || state.step === 1;
  if (fresh || summary.changed > 0) {
    state.steady = 0;
  } else {
    state.steady += 1;
  }
  entry.kind = OVERLAYS_STOP_DRAFT;
  entry.entropy = summary.entropy;
  entry.steady = state.steady;
  entry.step = state.step;
  return entry;
}

// What the readout shows for frame `index`, or null for nothing.
//
// The verdict on a committed canvas compares its draft count with the
// budget rather than re-judging its last draft. A canvas ends early
// only by the rule, a stopped run commits nothing, and a resumed
// canvas keeps frame_index drafts and is given budget - frame_index
// more, so "fewer drafts than the budget" is exactly "stopped by the
// rule", with no rounding to get wrong.
function overlaysStopReadingAt(track, index, rule) {
  var entry = track[index];
  if (!rule || !entry || entry.kind === OVERLAYS_STOP_NONE) {
    return null;
  }
  var reading = {
    kind: entry.kind,
    canvas: entry.canvas,
    step: entry.step,
    trace: overlaysStopTrace(track, index),
    rule: rule,
  };
  if (entry.kind === OVERLAYS_STOP_COMMIT) {
    reading.stopped = entry.step < rule.budget;
    return reading;
  }
  reading.entropy = entry.entropy;
  reading.changed = entry.changed;
  reading.steady = entry.steady;
  reading.entropyMet = entry.entropy < rule.threshold;
  reading.steadyMet = entry.steady >= rule.steadySteps;
  return reading;
}

// The drafts of frame `index`'s canvas up to it, as
// { step, entropy }, which is everything the trace draws.
function overlaysStopTrace(track, index) {
  var canvas = track[index].canvas;
  var start = index;
  while (start > 0 && track[start - 1].canvas === canvas) {
    start -= 1;
  }
  var points = [];
  for (var f = start; f <= index; f++) {
    if (track[f].kind === OVERLAYS_STOP_DRAFT) {
      points.push({ step: track[f].step, entropy: track[f].entropy });
    }
  }
  return points;
}

// Two significant figures below one nat and two decimals above, so
// 4.51 and 0.0047 take the same room and 0.0047 still shows how close
// it is to 0.005.
function overlaysStopEntropyText(value) {
  if (value === 0) {
    return "0";
  }
  if (value >= 1) {
    return value.toFixed(2);
  }
  return value.toPrecision(2);
}

function overlaysStopPlural(count, noun) {
  return count === 1 ? noun : noun + "s";
}

// The readout's words, as clauses that each know whether they are
// met. A part is plain text or a value, which the strip's idiom
// draws brighter.
function overlaysStopClauses(reading) {
  if (reading.kind === OVERLAYS_STOP_COMMIT) {
    return [overlaysStopVerdictClause(reading)];
  }
  var clauses = [{
    met: reading.entropyMet,
    parts: [
      { text: "entropy " },
      { text: overlaysStopEntropyText(reading.entropy), value: true },
      { text: " of " + reading.rule.threshold },
    ],
  }];
  if (reading.rule.steadySteps > 0) {
    clauses.push({ met: false, parts: [{ text: ", " }] });
    clauses.push(overlaysStopSteadyClause(reading));
  }
  return clauses;
}

// The same words as plain text, for a surface that cannot colour
// them: the Stopping chart's tooltip says what the readout says.
function overlaysStopWords(reading) {
  var clauses = overlaysStopClauses(reading);
  var text = "";
  for (var c = 0; c < clauses.length; c++) {
    for (var p = 0; p < clauses[c].parts.length; p++) {
      text += clauses[c].parts[p].text;
    }
  }
  return text;
}

// "13 changing" while anything moves, then "steady 1 of 2" as the
// count builds, and "steady" once it is enough.
function overlaysStopSteadyClause(reading) {
  if (reading.changed > 0) {
    return {
      met: false,
      parts: [
        { text: String(reading.changed), value: true },
        { text: " changing" },
      ],
    };
  }
  if (reading.steadyMet) {
    return { met: true, parts: [{ text: "steady" }] };
  }
  return {
    met: false,
    parts: [
      { text: "steady " },
      { text: String(reading.steady), value: true },
      { text: " of " + reading.rule.steadySteps },
    ],
  };
}

function overlaysStopVerdictClause(reading) {
  var steps = reading.step;
  var noun = overlaysStopPlural(steps, " step");
  var parts = [
    { text: "Canvas " },
    { text: String(reading.canvas + 1), value: true },
  ];
  if (reading.stopped) {
    parts.push({ text: " stopped after " });
  } else {
    parts.push({ text: " used all " });
  }
  parts.push({ text: String(steps), value: true });
  parts.push({ text: noun });
  return { met: reading.stopped, parts: parts };
}

// The rule in a sentence, with the run's own numbers, for the
// readout's tooltip.
function overlaysStopRuleText(rule) {
  var steps = rule.steadySteps;
  var steady = steps > 0
    ? " and no position has changed for " + steps
      + overlaysStopPlural(steps, " step")
    : "";
  return "A canvas stops once its mean entropy is below "
    + rule.threshold + " nats" + steady
    + ", or once it has used all " + rule.budget
    + overlaysStopPlural(rule.budget, " step") + ".";
}

// Build the readout's children once; each page calls this at boot,
// as it does for the strip, so the structure lives in one place.
function overlaysBuildStopReadout(el) {
  if (!el) {
    return;
  }
  el.textContent = "";
  var label = document.createElement("span");
  label.className = "stop-readout-label";
  label.textContent = "Stop";
  var trace = document.createElement("canvas");
  trace.className = "stop-readout-trace";
  var text = document.createElement("span");
  text.className = "stop-readout-text";
  el.appendChild(label);
  el.appendChild(trace);
  el.appendChild(text);
  el.overlaysStopNodes = { trace: trace, text: text };
  el.hidden = true;
}

// Render a reading, or hide the readout when it is null. Hidden
// rather than blanked, because a model that does not stop adaptively
// has nothing to report and an empty frame would read as one waiting.
function overlaysRenderStopReadout(el, reading) {
  if (!el || !el.overlaysStopNodes) {
    return;
  }
  if (!reading) {
    el.hidden = true;
    el.removeAttribute("title");
    return;
  }
  el.hidden = false;
  el.setAttribute("title", overlaysStopRuleText(reading.rule));
  overlaysStopWriteClauses(
    el.overlaysStopNodes.text, overlaysStopClauses(reading)
  );
  overlaysDrawStopTrace(el.overlaysStopNodes.trace, reading);
}

function overlaysStopWriteClauses(target, clauses) {
  target.innerHTML = "";
  for (var c = 0; c < clauses.length; c++) {
    var clause = document.createElement("span");
    clause.className = clauses[c].met
      ? "stop-readout-clause is-met"
      : "stop-readout-clause";
    var parts = clauses[c].parts;
    for (var p = 0; p < parts.length; p++) {
      var part = document.createElement("span");
      if (parts[p].value) {
        part.className = "stop-readout-value";
      }
      part.textContent = parts[p].text;
      clause.appendChild(part);
    }
    target.appendChild(clause);
  }
}

// Where the trace's marks fall in a box of `width` by `height`: the
// threshold's height, and one point per draft. Log scale, because a
// canvas's entropy falls three or four orders of magnitude on its
// way to the threshold; across the step budget, so a canvas that
// runs to its limit reaches the right edge.
function overlaysStopTraceLayout(reading, width, height) {
  var top = Math.log(OVERLAYS_STOP_TRACE_TOP_NATS);
  var floor =
    reading.rule.threshold * OVERLAYS_STOP_TRACE_FLOOR_RATIO;
  var span = top - Math.log(floor);
  var yOf = function (entropy) {
    var clamped = Math.min(
      OVERLAYS_STOP_TRACE_TOP_NATS, Math.max(floor, entropy)
    );
    return ((top - Math.log(clamped)) / span) * height;
  };
  var budget = reading.rule.budget;
  var xOf = function (step) {
    if (budget <= 1) {
      return width / 2;
    }
    return (Math.min(step, budget) - 1) / (budget - 1) * width;
  };
  var points = [];
  for (var i = 0; i < reading.trace.length; i++) {
    var mark = reading.trace[i];
    points.push({ x: xOf(mark.step), y: yOf(mark.entropy) });
  }
  return { threshold: yOf(reading.rule.threshold), points: points };
}

function overlaysDrawStopTrace(canvas, reading) {
  var ratio = window.devicePixelRatio || 1;
  var width = canvas.clientWidth || OVERLAYS_STOP_TRACE_WIDTH;
  var height = canvas.clientHeight || OVERLAYS_STOP_TRACE_HEIGHT;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  var ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  var layout = overlaysStopTraceLayout(reading, width, height);
  ctx.setLineDash([2, 2]);
  ctx.strokeStyle = OVERLAYS_STOP_TRACE_RULE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, layout.threshold);
  ctx.lineTo(width, layout.threshold);
  ctx.stroke();
  ctx.setLineDash([]);
  overlaysDrawStopTraceLine(ctx, layout.points, reading);
}

function overlaysDrawStopTraceLine(ctx, points, reading) {
  if (points.length === 0) {
    return;
  }
  ctx.strokeStyle = OVERLAYS_STOP_TRACE_LINE;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (var i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
  var last = points[points.length - 1];
  var final = reading.trace[reading.trace.length - 1];
  ctx.fillStyle = final.entropy < reading.rule.threshold
    ? OVERLAYS_STOP_TRACE_MET
    : OVERLAYS_STOP_TRACE_DOT;
  ctx.beginPath();
  ctx.arc(last.x, last.y, 1.6, 0, Math.PI * 2);
  ctx.fill();
}

// Let the readout's words give way before the strip's own content is
// cut. The strip is nowrap with its overflow hidden, so without this
// whatever is widest simply falls off its end. Measured with the
// words shown and hidden only if the strip then comes up short, or
// its overlay note is cut; the label and trace stay, so the canvas's
// progress is still on screen. Each page calls it after drawing
// either the strip or the readout.
function overlaysFitStopReadout(strip, readout) {
  if (!strip || !readout) {
    return;
  }
  readout.removeAttribute("data-compact");
  if (readout.hidden) {
    return;
  }
  if (overlaysStripClipped(strip)) {
    readout.setAttribute("data-compact", "");
  }
}

function overlaysStripClipped(strip) {
  if (strip.scrollWidth > strip.clientWidth) {
    return true;
  }
  var nodes = strip.overlaysMetricNodes;
  var extra = nodes ? nodes.extra : null;
  return !!extra && extra.scrollWidth > extra.clientWidth;
}

// ---- Revisions ----
//
// A revision is a position settling on a different token from the
// last one it settled on, in the same canvas. Only DiffusionGemma
// does it: LLaDA never revisits a settled position, and a model that
// appends never revisits anything, so on those runs every count here
// comes out zero and nothing is offered.
//
// A first settle is a birth, which the worker reports and the birth
// glow marks, so it is never also a revision; nor is a return to the
// token a position already held. An edit resets the positions it
// remasked, so their next settle is a birth as well. That is what the
// worker reports on a resume, on both models, and a change the user
// asked for is not the model changing its mind.
//
// The fold is what one frame hands the next: the canvas it belongs
// to, and the id each position last settled on.
function overlaysRevisionFold() {
  return { canvas: null, settled: [] };
}

// One frame's revised positions, and the fold after it. ``remasked``
// lists the positions an edit sent back at this frame; they forget
// their token before the frame is read. A new canvas starts empty,
// because its positions are unrelated to the last canvas's.
function overlaysRevisionStep(fold, tokens, canvas, remasked) {
  var settled = fold.canvas === canvas ? fold.settled.slice() : [];
  for (var r = 0; r < remasked.length; r++) {
    settled[remasked[r]] = undefined;
  }
  var revised = [];
  for (var i = 0; i < tokens.length; i++) {
    var token = tokens[i];
    if (!token || token.m) {
      continue;
    }
    var last = settled[i];
    if (typeof last === "number" && last !== token.id) {
      revised.push(i);
    }
    settled[i] = token.id;
  }
  return {
    revised: revised,
    fold: { canvas: canvas, settled: settled },
  };
}

// Every frame's revised positions, one array per frame. Takes a
// reader and a count for the reason overlaysComputeCommitSteps does,
// and the run's edit log so a remasked position starts over at the
// frame its edit's branch begins.
function overlaysComputeRevisions(
  readFrame, frameCount, canvasOf, edits
) {
  if (typeof readFrame !== "function") {
    throw new Error("revisions: readFrame must be a function");
  }
  if (typeof canvasOf !== "function") {
    throw new Error("revisions: canvasOf must be a function");
  }
  var log = edits || [];
  var revisions = new Array(frameCount);
  var fold = overlaysRevisionFold();
  for (var f = 0; f < frameCount; f++) {
    var step = overlaysRevisionStep(
      fold,
      readFrame(f) || [],
      canvasOf(f),
      overlaysRemaskedAt(log, f)
    );
    revisions[f] = step.revised;
    fold = step.fold;
  }
  return revisions;
}

// The positions an edit log remasked at ``frame``: the frame its
// branch begins at, which replaced the frame the edit was made on.
function overlaysRemaskedAt(edits, frame) {
  var positions = [];
  for (var e = 0; e < edits.length; e++) {
    if (edits[e].frame_index === frame) {
      positions = positions.concat(edits[e].token_positions || []);
    }
  }
  return positions;
}

// Whether a run revised anything at all, which is what decides if
// the overlay is offered.
function overlaysHasRevisions(revisions) {
  for (var f = 0; f < revisions.length; f++) {
    if (revisions[f].length > 0) {
      return true;
    }
  }
  return false;
}

// How many times each position had been revised by ``frame``,
// counted from the first frame of that frame's canvas. Sparse: a
// position never revised has no entry.
function overlaysRevisionCounts(revisions, frame, canvasOf) {
  var counts = [];
  if (frame < 0 || frame >= revisions.length) {
    return counts;
  }
  var canvas = canvasOf(frame);
  for (var f = frame; f >= 0 && canvasOf(f) === canvas; f--) {
    var revised = revisions[f];
    for (var i = 0; i < revised.length; i++) {
      counts[revised[i]] = (counts[revised[i]] || 0) + 1;
    }
  }
  return counts;
}

// Cyan, deepening with how often a position changed its mind: pale
// for once, saturated for twice, deep for three times or more. Steps
// rather than a ramp, because the counts are small whole numbers (on
// saved DiffusionGemma runs nearly every revised position changed one
// to three times) and three swatches read at a glance where a
// gradient would need a scale. Its own hue: white is the birth glow,
// orange is an edit, and the heatmap and the mask are green.
var OVERLAYS_REVISION_COLORS = ["#7fe8ff", "#2fd4ff", "#00a8e0"];

function revisionColor(count) {
  if (typeof count !== "number" || count < 1) {
    return null;
  }
  var at = Math.min(count, OVERLAYS_REVISION_COLORS.length) - 1;
  return OVERLAYS_REVISION_COLORS[at];
}

// The metrics strip's line for a position under the Revisions
// overlay, blank where it has not changed.
function overlaysRevisionReading(count) {
  if (typeof count !== "number" || count < 1) {
    return "";
  }
  return "Revisions: " + count;
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
// Beyond the required colorFor, ``opts`` takes three optional
// callbacks, all defaulting to the plain Analytics behavior so that
// page passes none of them:
//
//   maskedFor(index, tok)           -> mask a resolved token
//   classFor(index, tok, masked)    -> extra classes
//   opacityFor(index, tok, masked)  -> inline opacity
//
// The generator needs all three. It draws the mask glyph over
// positions the user selected for remasking even though their tokens
// are resolved (hence maskedFor), and marks those and its clickable
// and substitutable positions with their own classes (classFor).
//
// Both pages pass opacityFor, and it is the one hook whose absence is
// usually a bug rather than a choice: a mask's fade is a property of
// the token's own confidence, which a saved run carries just as a
// live one does. The pages differ only in their exceptions, which is
// why it stays a callback: the generator holds a remask selection
// solid, and Analytics has no selection to hold.
//
// ``opts.revealMask`` is the user's setting, not a callback: with it
// on, an unsettled position draws the token it is holding instead of
// the glyph. Both pages pass it, and it defaults falsy so a caller
// that says nothing keeps drawing glyphs.
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
  // The position's own claim, kept apart from masked because the
  // hook below can mask a token that did settle. Only this one earns
  // the reveal: a hook-masked position is the app hiding a settled
  // token to show intent, and revealing it would undo the point.
  var unsettled = masked && !!tok;
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
  var text = mask;
  if (!masked) {
    text = tok.t;
  } else if (unsettled && opts.revealMask) {
    text = overlaysMaskCandidate(tok, mask);
  }
  if (span.textContent !== text) {
    span.textContent = text;
  }
  var description = opts.descriptionFor
    ? opts.descriptionFor(index, tok, masked)
    : "";
  if (description) {
    var accessibleText = String(text).trim();
    if (!accessibleText) {
      accessibleText = "Whitespace token";
    }
    span.setAttribute(
      "aria-label", accessibleText + ". " + description
    );
    span.setAttribute("title", description);
  } else {
    overlaysRemoveAttribute(span, "aria-label");
    overlaysRemoveAttribute(span, "title");
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
}

function overlaysRemoveAttribute(element, name) {
  if (typeof element.removeAttribute === "function") {
    element.removeAttribute(name);
    return;
  }
  if (element.getAttribute(name) !== null) {
    element.setAttribute(name, "");
  }
}

// What an unsettled position is currently holding, for the reveal.
// Falls back to the glyph on anything that would draw as nothing,
// because an empty span collapses and two stacked layers stop lining
// up. A saved run recorded before the samplers kept their guess has
// the glyph in tok.t already, so it falls through unchanged and the
// setting is simply inert there.
function overlaysMaskCandidate(tok, mask) {
  if (typeof tok.t !== "string" || tok.t === "") {
    return mask;
  }
  return tok.t;
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
// (originalOpacity / editedOpacity), a ``blend`` flag, the
// ``revealMask`` preference, and an ``opacityFor`` hook. The last two
// go to both layers together: they are two readings of the same
// canvas, and drawing one as faded words and the other as solid
// blocks would make the diff unreadable.
function overlaysBuildDiffLayers(
  origTokens, editedTokens, diff, opts, maskChar
) {
  var options = opts || {};
  var revealMask = !!options.revealMask;
  var opacityFor = options.opacityFor;
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
    revealMask: revealMask,
    opacityFor: opacityFor,
    colorFor: overlaysDiffColorFor(diff, true, blend),
  });

  var editLayer = overlaysBuildTokenLayer(editedTokens || [], {
    layerClass: "token-layer-edited",
    opacity: editedOpacity / 100,
    interactive: editedTakes,
    maskChar: maskChar,
    revealMask: revealMask,
    opacityFor: opacityFor,
    colorFor: overlaysDiffColorFor(diff, false, blend),
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

// Each flash as blurred copies of the text at 100% brightness, as the
// channels of an rgba(). A birth is two white copies, which merge
// with white text into a glow. Two cyan copies behind white text read
// as a tint rather than a glow, which is how the revision flash first
// looked on hardware, so a revision adds a bright pale core inside a
// wider halo, and its keyframes light the glyph itself (style.css).
// One brightness and one fade scale both.
var GLOW_BIRTH_RGB = "255, 255, 255";
var GLOW_REVISION_RGB = "0, 220, 255";
var GLOW_REVISION_CORE_RGB = "225, 252, 255";

var GLOW_BIRTH_LAYERS = [
  {
    blurPx: GLOW_INNER_BLUR_PX,
    alpha: GLOW_INNER_ALPHA,
    rgb: GLOW_BIRTH_RGB,
  },
  {
    blurPx: GLOW_OUTER_BLUR_PX,
    alpha: GLOW_OUTER_ALPHA,
    rgb: GLOW_BIRTH_RGB,
  },
];

var GLOW_REVISION_LAYERS = [
  { blurPx: 3, alpha: 0.95, rgb: GLOW_REVISION_CORE_RGB },
  { blurPx: 8, alpha: 0.9, rgb: GLOW_REVISION_RGB },
  { blurPx: 16, alpha: 0.55, rgb: GLOW_REVISION_RGB },
];

// The settings keys each model class reads, keyed on the family from
// ModelCapabilities. Family rather than generation shape: these are
// per-class visual preferences, so a state-space model wants its own
// pair even though it appends like an autoregressive one.
//
// Written out rather than derived from the class name so every key is
// greppable as a literal; a new class is one entry here plus an
// option in the Settings picker. The key strings themselves are
// persisted user settings, so they are named after the family and
// must not be renamed to follow a refactor.
var GLOW_KEYS = {
  diffusion: {
    brightness: "glowBrightnessDiffusion",
    fadeMs: "glowFadeMsDiffusion",
  },
  autoregressive: {
    brightness: "glowBrightnessAutoregressive",
    fadeMs: "glowFadeMsAutoregressive",
  },
  state_space: {
    brightness: "glowBrightnessStateSpace",
    fadeMs: "glowFadeMsStateSpace",
  },
};

var GLOW_CLASS_OPTIONS = [
  { value: "diffusion", label: "Diffusion" },
  { value: "autoregressive", label: "Autoregressive" },
  { value: "state_space", label: "State space" },
];

// What an unsettled diffusion position shows: the block glyph, the
// token the model is holding there, or its captured candidates,
// cycling. One choice rather than two toggles, because they are three
// readings of one position and never stack: cycling reserves room for
// its widest candidate, which would pad a canvas of guesses.
var UNSETTLED_SHOWS_OPTIONS = [
  { value: "glyph", label: "The mask glyph" },
  { value: "guess", label: "The model's guess" },
  { value: "candidates", label: "Its candidates, cycling" },
];

var SETTINGS_DEFAULTS = {
  highlightTokens: true,
  diffusionText: false,
  diffusionTextMode: "default",
  gpuTicker: true,
  tokenBirthGlow: true,
  revisionGlow: true,
  // The glyph by default. A canvas of blocks is what a diffusion run
  // looks like, and reading a page of plausible words that are not
  // the answer yet is a thing to opt into, not to be handed.
  unsettledShows: "glyph",
  glowBrightnessDiffusion: GLOW_BRIGHTNESS_DEFAULT,
  glowFadeMsDiffusion: GLOW_FADE_MS_DEFAULT,
  glowBrightnessAutoregressive: GLOW_BRIGHTNESS_DEFAULT,
  glowFadeMsAutoregressive: GLOW_FADE_MS_DEFAULT,
  glowBrightnessStateSpace: GLOW_BRIGHTNESS_DEFAULT,
  glowFadeMsStateSpace: GLOW_FADE_MS_DEFAULT,
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
    revisionGlow: SETTINGS_DEFAULTS.revisionGlow,
    unsettledShows: SETTINGS_DEFAULTS.unsettledShows,
    glowBrightnessDiffusion:
      SETTINGS_DEFAULTS.glowBrightnessDiffusion,
    glowFadeMsDiffusion: SETTINGS_DEFAULTS.glowFadeMsDiffusion,
    glowBrightnessAutoregressive:
      SETTINGS_DEFAULTS.glowBrightnessAutoregressive,
    glowFadeMsAutoregressive:
      SETTINGS_DEFAULTS.glowFadeMsAutoregressive,
    glowBrightnessStateSpace:
      SETTINGS_DEFAULTS.glowBrightnessStateSpace,
    glowFadeMsStateSpace: SETTINGS_DEFAULTS.glowFadeMsStateSpace,
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
      settings.revisionGlow = parsed.revisionGlow !== false;
      settings.unsettledShows = parseUnsettledShows(parsed);
      parseGlowInto(settings, parsed);
      settings.tpsMode =
        parsed.tpsMode === "last" ? "last" : "total";
    }
  } catch (_e) {
    // Corrupt storage: keep the defaults.
  }
  return settings;
}

// A stored choice counts when it is one of the three. A profile saved
// while this was a single toggle migrates: the reveal switched on was
// asking for the guess. Anything else keeps the glyph, unlike the
// highlight and the glow above, which default on when absent: this
// one changes what the canvas says rather than how it looks.
function parseUnsettledShows(parsed) {
  var stored = parsed.unsettledShows;
  for (var i = 0; i < UNSETTLED_SHOWS_OPTIONS.length; i++) {
    if (UNSETTLED_SHOWS_OPTIONS[i].value === stored) {
      return stored;
    }
  }
  return parsed.revealMaskCandidate ? "guess" : "glyph";
}

// Whether an unsettled position draws the token it is holding, which
// is the span builder's revealMask flag. True for cycling as well as
// for the guess, because a position shows its guess wherever there
// are no candidates to cycle through: while a run streams, on its
// opening frame, mid-edit, and with motion reduced.
function overlaysDrawsGuess(settings) {
  return settings.unsettledShows !== "glyph";
}

// Fold every class's glow pair out of stored state, clamped to
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

// A flash's shadow layers at full strength, plus the same layers at
// zero alpha for the animation to land on.
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
function overlaysGlowShadow(brightnessPercent, layers) {
  if (!Array.isArray(layers) || layers.length === 0) {
    throw new Error("glow: a shadow needs its layers");
  }
  var scale = clampGlowValue(
    brightnessPercent,
    GLOW_BRIGHTNESS_MIN,
    GLOW_BRIGHTNESS_MAX,
    GLOW_BRIGHTNESS_DEFAULT
  ) / 100;
  var peak = [];
  var off = [];
  for (var i = 0; i < layers.length; i++) {
    var layer = layers[i];
    var blur = (layer.blurPx * scale).toFixed(1);
    var alpha = Math.min(layer.alpha * scale, 1).toFixed(3);
    peak.push(glowShadowLayer(blur, alpha, layer.rgb));
    off.push(glowShadowLayer(blur, "0", layer.rgb));
  }
  return { peak: peak.join(", "), off: off.join(", ") };
}

function glowShadowLayer(blurPx, alpha, rgb) {
  return "0 0 " + blurPx + "px rgba(" + rgb + ", " + alpha + ")";
}

// Write the glow's custom properties onto `el`, which is where the
// keyframes read them from. Shared so the Settings page preview and
// the live canvas cannot drift apart. Both flashes take the class's
// one brightness and one fade.
function overlaysApplyGlowVars(el, brightnessPercent, fadeMs) {
  if (!el) {
    return;
  }
  var birth = overlaysGlowShadow(
    brightnessPercent, GLOW_BIRTH_LAYERS
  );
  var revision = overlaysGlowShadow(
    brightnessPercent, GLOW_REVISION_LAYERS
  );
  var duration = clampGlowValue(
    fadeMs,
    GLOW_FADE_MS_MIN,
    GLOW_FADE_MS_MAX,
    GLOW_FADE_MS_DEFAULT
  );
  el.style.setProperty("--token-birth-shadow", birth.peak);
  el.style.setProperty("--token-birth-shadow-off", birth.off);
  el.style.setProperty("--token-revision-shadow", revision.peak);
  el.style.setProperty(
    "--token-revision-shadow-off", revision.off
  );
  el.style.setProperty(
    "--token-birth-duration", duration + "ms"
  );
}

// The glow pair a model class reads, falling back to the diffusion
// pair for a family that has no entry yet.
function overlaysGlowFor(settings, family) {
  var keys = GLOW_KEYS[family] || GLOW_KEYS.diffusion;
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
    && a.revisionGlow === b.revisionGlow
    && a.unsettledShows === b.unsettledShows
    && a.glowBrightnessDiffusion === b.glowBrightnessDiffusion
    && a.glowFadeMsDiffusion === b.glowFadeMsDiffusion
    && a.glowBrightnessAutoregressive
      === b.glowBrightnessAutoregressive
    && a.glowFadeMsAutoregressive === b.glowFadeMsAutoregressive
    && a.glowBrightnessStateSpace === b.glowBrightnessStateSpace
    && a.glowFadeMsStateSpace === b.glowFadeMsStateSpace
    && a.tpsMode === b.tpsMode
  );
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
