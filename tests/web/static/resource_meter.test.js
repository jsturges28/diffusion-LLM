// The footer meter shows what the worker volunteers, and no more.
//
// Strategy: load the generator into the DOM stub beside this file and
// feed it samples through the same handler the socket reaches. The
// stub's canvas context is a no-op, so the drawing is not
// the subject; what is asserted is the data behind it, which is where
// every decision actually lives: whether the row appears, what it is
// labelled, what the readout says, and how much history is kept.
//
// The bound matters more than it looks. This is the only thing on the
// page that grows purely with elapsed time, and the socket stays open
// for as long as the tab does, so an unbounded buffer is a leak that
// only shows up after an hour.
//
// Passing proves the row stays hidden until there is something to
// report, the label and readout follow the sample's kind rather than
// the page's guess, the history is bounded and keeps the newest, a
// change of kind does not splice two different measurements into one
// line, and a malformed or unknown sample is ignored, not drawn.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

const GIB = 1024 * 1024 * 1024;

function vramSample(fraction) {
  return {
    type: "resource_sample",
    kind: "vram",
    fraction: fraction,
    used_bytes: Math.round(fraction * 24 * GIB),
    total_bytes: 24 * GIB,
  };
}

function cpuSample(fraction) {
  return {
    type: "resource_sample",
    kind: "cpu",
    fraction: fraction,
    busy_cores: fraction * 32,
    total_cores: 32,
  };
}

// Driven through handleMessage rather than the handler directly, so
// the dispatch entry is part of what passes: a sample the switch does
// not route is indistinguishable from one the handler ignored.
function feed(context, sample) {
  context.handleMessage(sample);
}

function page() {
  const { context, registry } = loadPage({});
  const row = registry.get("status-resource");
  // The stub synthesises elements from ids and never reads the HTML,
  // so the `hidden` attribute the markup carries has to be stated
  // here. Stated rather than skipped, because "does not appear" is
  // half of what this file is about: the page ships it hidden and
  // only a usable sample may reveal it.
  row.hidden = true;
  return {
    context,
    row,
    label: registry.get("status-resource-label"),
    value: registry.get("status-resource-value"),
  };
}

// -- appearing --

test("the meter is hidden before any sample", () => {
  // A host with no card and no /proc sends nothing at all, so this is
  // the resting state on such a machine, not just a first frame.
  const { row } = page();

  assert.equal(row.hidden, true);
});

test("the first sample reveals it", () => {
  const { context, row } = page();

  feed(context, vramSample(0.5));

  assert.equal(row.hidden, false);
});

// -- what it says --

test("a card sample is labelled and read in gibibytes", () => {
  // Bytes rather than the bare percentage, because "71%" of a card
  // whose size you cannot remember says less than the pair does.
  const { context, label, value } = page();

  feed(context, vramSample(0.5));

  assert.equal(label.textContent, "VRAM");
  assert.equal(value.textContent, "12.0 GiB / 24.0 GiB");
});

test("a CPU sample is labelled and read against the cores", () => {
  // The machine's width is part of the reading: a quarter of 32 cores
  // and a quarter of 4 are very different facts.
  const { context, label, value } = page();

  feed(context, cpuSample(0.25));

  assert.equal(label.textContent, "CPU");
  assert.equal(value.textContent, "25% of 32 cores");
});

test("the readout follows the newest sample", () => {
  const { context, value } = page();

  feed(context, vramSample(0.25));
  feed(context, vramSample(0.75));

  assert.equal(value.textContent, "18.0 GiB / 24.0 GiB");
});

// -- history --

test("samples accumulate in arrival order", () => {
  const { context } = page();

  feed(context, vramSample(0.1));
  feed(context, vramSample(0.2));
  feed(context, vramSample(0.3));

  assert.deepEqual(
    Array.from(context.resourceHistory), [0.1, 0.2, 0.3]
  );
});

test("the history is bounded", () => {
  // The leak this exists to prevent. At the worker's cadence it
  // is reached in a minute and then never exceeded.
  const { context } = page();
  const bound = context.RESOURCE_HISTORY_MAX;

  for (let i = 0; i < bound + 50; i++) {
    feed(context, vramSample(0.5));
  }

  assert.equal(context.resourceHistory.length, bound);
});

test("the bound drops the oldest, not the newest", () => {
  // Which end is discarded is the whole point of a sparkline: keeping
  // the wrong end would freeze the line at startup.
  const { context } = page();
  const bound = context.RESOURCE_HISTORY_MAX;

  for (let i = 0; i < bound; i++) {
    feed(context, vramSample(0));
  }
  feed(context, vramSample(1));

  const history = context.resourceHistory;
  assert.equal(history.length, bound);
  assert.equal(history[history.length - 1], 1);
  assert.equal(history[0], 0);
});

test("a change of kind starts a new series", () => {
  // Activating a CPU model after a GPU one changes what the meter
  // measures. Joining the two would draw one line describing two
  // different quantities, and there is no conversion between them.
  const { context, label } = page();
  feed(context, vramSample(0.9));
  feed(context, vramSample(0.9));

  feed(context, cpuSample(0.1));

  assert.deepEqual(Array.from(context.resourceHistory), [0.1]);
  assert.equal(label.textContent, "CPU");
});

test("a fraction outside the axis is clamped", () => {
  // The worker clamps too, so this is the paired check on the far
  // of the wire: the line has a top and a bottom either way.
  const { context } = page();

  feed(context, vramSample(0));
  context.handleMessage({
    type: "resource_sample", kind: "vram", fraction: 1.4,
    used_bytes: 1, total_bytes: 1,
  });

  assert.equal(context.resourceHistory[1], 1);
});

// -- going away with the model --

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// A page together with the socket it opened, so these two exercise the
// wiring rather than the function: a clear that was never hooked to
// onclose would pass a direct call and still leave a stale line on
// screen after a switch.
//
// Two ticks, for two different reasons. The first drains connects still
// pending from the pages the tests above built, since those never await
// and ``FakeSocket.opened`` is static and shared; without it the index
// taken below can belong to somebody else's page, which is a genuinely
// confusing failure. The second waits for this page's own connect,
// which happens in a promise chain behind fetchModels.
async function pageWithSocket() {
  await tick();
  const mark = FakeSocket.opened.length;
  const built = page();
  await tick();
  const opened = FakeSocket.opened;
  assert.equal(
    opened.length, mark + 1, "expected exactly this page's socket"
  );
  return { ...built, socket: opened[mark] };
}

test("losing the connection puts the meter away", async () => {
  // A model switch looks like a dropped socket from here. Leaving the
  // row up would present the last reading as current while nothing is
  // sampling at all, which is the one thing a live meter must not do.
  const { context, row, socket } = await pageWithSocket();
  feed(context, vramSample(0.6));

  socket.close();

  assert.equal(row.hidden, true);
});

test("the history does not span the gap", async () => {
  // Two models on the same device would otherwise join into one line
  // running through a period nothing was measured in. The kind-change
  // reset cannot catch that case, because the kind does not change.
  const { context, socket } = await pageWithSocket();
  feed(context, vramSample(0.6));
  feed(context, vramSample(0.7));

  socket.close();
  feed(context, vramSample(0.2));

  assert.deepEqual(Array.from(context.resourceHistory), [0.2]);
});

// -- what it refuses --

test("an unknown kind is ignored", () => {
  // A future sample this build cannot label. Dropping it leaves a gap
  // in the line, which is honest; drawing it would put a number under
  // the wrong word.
  const { context, row } = page();

  context.handleMessage({
    type: "resource_sample", kind: "tokens_per_joule", fraction: 0.5,
  });

  assert.equal(row.hidden, true);
  assert.equal(context.resourceHistory.length, 0);
});

test("a sample with no fraction is ignored", () => {
  // The one field the line cannot do without.
  const { context, row } = page();

  context.handleMessage({ type: "resource_sample", kind: "vram" });

  assert.equal(row.hidden, true);
  assert.equal(context.resourceHistory.length, 0);
});

test("a missing byte count reads as unknown, not as zero", () => {
  // Defensive rather than expected. A worker that reported a fraction
  // without the absolutes would still get a line, and the readout
  // it cannot show the figures instead of inventing 0.0 GiB.
  const { context, value } = page();

  context.handleMessage({
    type: "resource_sample", kind: "vram", fraction: 0.5,
  });

  assert.match(value.textContent, /\?/);
});
