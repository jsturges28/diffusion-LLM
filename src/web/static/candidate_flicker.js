// A diffusion position's candidates, cycling on the canvas.
//
// Loaded as a classic global script after overlays.js and
// run_candidates.js, and touching no DOM until a page starts it,
// which is what lets a test drive the schedule in a `vm`.
//
// Each unsettled position cycles through the candidates its frame
// captured, holding each for its probability's share of a second, and
// shows the mask glyph for the rest: whatever the model gave tokens
// outside the five. So the time-average of what a position shows is
// the distribution itself. A calm position barely moves and a
// contested one churns; confidence is the amount of motion, and no
// rule has to pick which positions deserve it.
//
// Only on a finished run, because the candidates arrive as it ends,
// and only at unsettled positions. Each one reserves the width of its
// longest scheduled text, so settled text keeps its layout and
// nothing moves while a frame plays. A literal stack of the five,
// overlaid at their shares, was mocked on a real frame first: it
// smeared every contested stretch and spread the canvas out.

"use strict";

// One cycle, stepped through in slots. A slot is the finest share a
// candidate can hold, 5% at these values, so one worth less than half
// a slot does not show at all.
var FLICKER_CYCLE_MS = 1000;
var FLICKER_SLOTS = 20;
var FLICKER_SLOT_MS = FLICKER_CYCLE_MS / FLICKER_SLOTS;
// A motion budget rather than a selection rule. It binds only early
// in a run, where most of a canvas is undetermined, and keeps the
// most contested positions moving.
var FLICKER_MAX_POSITIONS = 64;
// How far apart neighbouring positions start their cycles: the golden
// ratio's fraction, so no run of nearby positions changes together.
var FLICKER_PHASE_STEP = 0.6180339887;
// A control token as every model here spells one: a name in angle
// brackets, with or without pipes (<|endoftext|>, <eos>, <|turn>).
var FLICKER_CONTROL_TOKEN = /^<\|?[A-Za-z_]+\|?>$/;

if (FLICKER_CYCLE_MS % FLICKER_SLOTS !== 0) {
  throw new Error("flicker: the slots must divide the cycle evenly");
}

// What is cycling now: one {span, position, texts} per position, and
// the timer stepping them. One flicker per page at a time.
var flickerEntries = [];
var flickerTimer = null;

// Start cycling `layers`, each {spans, tokens, sets}: the layer's
// spans indexed by position, its tokens, and its captured frame's
// sets or null. Stops whatever cycled before. Under reduced motion
// nothing starts, so every position keeps the guess it was drawn
// with. Call it once the spans are on the page, because a tick that
// finds them detached stops.
function flickerStart(layers, mask) {
  flickerStop();
  if (prefersReducedMotion()) {
    return;
  }
  var entries = [];
  for (var i = 0; i < layers.length; i++) {
    entries = entries.concat(flickerBind(layers[i], mask));
  }
  if (entries.length === 0) {
    return;
  }
  flickerEntries = entries;
  flickerTick();
  flickerTimer = setInterval(flickerTick, FLICKER_SLOT_MS);
}

function flickerStop() {
  if (flickerTimer !== null) {
    clearInterval(flickerTimer);
    flickerTimer = null;
  }
  flickerEntries = [];
}

// Step every cycling position to its slot for now. Writes a span only
// when its text changes, since each write relayouts its line. Stops
// itself once its spans have left the page, which is how a render
// path that forgot to stop it still ends it.
function flickerTick() {
  if (flickerEntries.length === 0) {
    return;
  }
  if (flickerEntries[0].span.isConnected === false) {
    flickerStop();
    return;
  }
  if (typeof document !== "undefined" && document.hidden) {
    return;
  }
  var now = flickerNow();
  for (var i = 0; i < flickerEntries.length; i++) {
    var entry = flickerEntries[i];
    var text = entry.texts[flickerSlotAt(now, entry.position)];
    if (entry.span.textContent !== text) {
      entry.span.textContent = text;
    }
  }
}

// The clock the cycle reads, a function so a test can set it.
function flickerNow() {
  return Date.now();
}

// One layer as flickerStart takes it: its spans by position, its
// tokens, and its run's captured sets at the frame it shows, or null
// where the store has none there. Shared by both pages, which differ
// only in which store and frame each of their layers reads.
function flickerLayer(spans, tokens, store, frame, canvasOf) {
  var found = store && frame >= 0
    ? runCandidatesAt(store, frame, canvasOf)
    : null;
  return {
    spans: spans,
    tokens: tokens,
    sets: found === null ? null : found.sets,
  };
}

// Mark the spans a layer's plan cycles and reserve their width.
function flickerBind(layer, mask) {
  var plan = flickerPlan(layer.tokens || [], layer.sets || [], mask);
  var spans = layer.spans || [];
  var bound = [];
  for (var i = 0; i < plan.length; i++) {
    var span = spans[plan[i].position];
    if (!span) {
      continue;
    }
    span.setAttribute("data-cycling", "");
    span.style.width = plan[i].width + "ch";
    bound.push({
      span: span,
      position: plan[i].position,
      texts: plan[i].texts,
    });
  }
  return bound;
}

// The positions of one layer that cycle, as {position, texts, width,
// top}: unsettled ones with a captured set whose schedule shows more
// than one text. At most FLICKER_MAX_POSITIONS, keeping the most
// contested, the ones whose likeliest candidate holds the least.
function flickerPlan(tokens, sets, mask) {
  var entries = [];
  var count = Math.min(tokens.length, sets.length);
  for (var i = 0; i < count; i++) {
    var entry = flickerEntry(i, tokens[i], sets[i], mask);
    if (entry !== null) {
      entries.push(entry);
    }
  }
  entries.sort(function (a, b) {
    return a.top - b.top || a.position - b.position;
  });
  return entries.slice(0, FLICKER_MAX_POSITIONS);
}

function flickerEntry(position, token, set, mask) {
  if (!token || !token.m) {
    return null;
  }
  if (!set || !Array.isArray(set.c) || set.c.length === 0) {
    return null;
  }
  var texts = flickerSchedule(set, mask);
  if (!flickerVaries(texts)) {
    return null;
  }
  return {
    position: position,
    texts: texts,
    width: flickerWidth(texts),
    top: flickerProbability(set.c[0].p),
  };
}

// The texts a position shows over one cycle, a slot each: its
// candidates in rank order, each for its probability's share, then
// the glyph for the probability outside them. Slots go by largest
// remainder, so they always total FLICKER_SLOTS and a share is
// rounded rather than truncated away.
function flickerSchedule(set, mask) {
  var shares = [];
  var covered = 0;
  for (var i = 0; i < set.c.length; i++) {
    var share = flickerProbability(set.c[i].p);
    shares.push(share);
    covered += share;
  }
  shares.push(Math.max(0, 1 - covered));
  var slots = flickerApportion(shares, FLICKER_SLOTS);
  var texts = [];
  for (var j = 0; j < shares.length; j++) {
    var text = j < set.c.length
      ? flickerText(set.c[j].t, mask)
      : mask;
    for (var s = 0; s < slots[j]; s++) {
      texts.push(text);
    }
  }
  return texts;
}

// `count` slots divided among `shares` by largest remainder, ties to
// the earlier share, which is the likelier candidate. A share list
// worth nothing gives every slot to its last entry, the glyph.
function flickerApportion(shares, count) {
  var total = 0;
  for (var i = 0; i < shares.length; i++) {
    total += shares[i];
  }
  var slots = [];
  var remainders = [];
  var used = 0;
  for (var j = 0; j < shares.length; j++) {
    var quota = total > 0 ? (shares[j] / total) * count : 0;
    var whole = Math.floor(quota);
    slots.push(whole);
    remainders.push(quota - whole);
    used += whole;
  }
  if (total <= 0) {
    slots[slots.length - 1] = count;
    return slots;
  }
  for (var k = 0; k < count - used; k++) {
    var best = flickerLargest(remainders);
    slots[best] += 1;
    remainders[best] = -1;
  }
  return slots;
}

function flickerLargest(values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

// A candidate as the canvas draws a guess. The worker strips control
// tokens from what a position holds, and the canvas draws the empty
// result as the glyph, so they draw as the glyph here too rather than
// widening a cell for an end-of-text marker. A newline or tab draws
// as a visible stand-in, because breaking the line would move the
// canvas while it plays.
function flickerText(text, mask) {
  if (typeof text !== "string" || text === "") {
    return mask;
  }
  if (FLICKER_CONTROL_TOKEN.test(text)) {
    return mask;
  }
  return text.replace(/\n/g, "\u21B5").replace(/\t/g, "\u21E5");
}

function flickerProbability(value) {
  if (typeof value !== "number" || !isFinite(value) || value < 0) {
    return 0;
  }
  return Math.min(1, value);
}

function flickerVaries(texts) {
  for (var i = 1; i < texts.length; i++) {
    if (texts[i] !== texts[0]) {
      return true;
    }
  }
  return false;
}

// The characters a position needs: its longest scheduled text, which
// a monospace canvas measures in `ch`.
function flickerWidth(texts) {
  var widest = 0;
  for (var i = 0; i < texts.length; i++) {
    widest = Math.max(widest, texts[i].length);
  }
  return widest;
}

// Which slot a position shows at `nowMs`. Every position's cycle is
// offset by its own phase, so the canvas never changes all at once.
function flickerSlotAt(nowMs, position) {
  var phase = position * FLICKER_PHASE_STEP;
  var cycles = nowMs / FLICKER_CYCLE_MS + phase;
  var fraction = cycles - Math.floor(cycles);
  var slot = Math.floor(fraction * FLICKER_SLOTS);
  return Math.min(slot, FLICKER_SLOTS - 1);
}
