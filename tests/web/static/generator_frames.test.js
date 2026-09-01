// What the generator page does with a run's frames, driven for real.
//
// Strategy: load the whole generator page into the DOM stub beside
// this file, feed it the frames a worker would send, and read back
// what it retained, what it renders at each scrub position, and what
// it would post on save. Nothing is mocked between the frame arriving
// and the answer coming out.
//
// This is a **characterization** test. It was written against the
// full-snapshot protocol before any of it changed, so the numbers
// below are not a specification anyone reasoned out: they are what
// the page did on the day, recorded so that replacing per-token
// snapshots with an append stream can be shown to change nothing a
// user sees. A test written afterwards would only prove the new code
// agrees with itself.
//
// Passing proves the retained frames grow the way the page expects,
// that scrubbing to frame N yields exactly the tokens of frame N,
// that the pre-edit baseline freezes at first completion, and that
// the save payload carries one full record set per frame. Those are
// the four things the append work has to preserve.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

// One AR frame as `_build_frame` in ar_sampler.py emits it: a full
// snapshot of every position decoded so far.
function snapshotFrame(index, words, options) {
  const settings = options || {};
  const tokens = words.slice(0, index).map((word, position) => ({
    t: word,
    m: false,
    id: 1000 + position,
    c: +(0.5 + position / 100).toFixed(4),
    e: +(1.5 - position / 100).toFixed(4),
  }));
  const frame = {
    type: "frame",
    index: index,
    total_steps: settings.totalSteps || words.length,
    canvas_index: 0,
    mean_conf: tokens.length
      ? +(tokens.reduce((sum, t) => sum + t.c, 0)
        / tokens.length).toFixed(4)
      : 0,
    text: tokens.map((t) => t.t).join(""),
    tokens: tokens,
    revealed: index > 0 ? [index - 1] : [],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (settings.alts) {
    frame.alts = [{ id: 42, t: " other", p: 0.2 }];
  }
  return frame;
}

// The same run as `_build_append_frame` in ar_sampler.py emits it:
// the one position this frame added, and nothing the receiver
// already holds.
function appendFrame(index, words, options) {
  const settings = options || {};
  const position = index - 1;
  const confidences = words
    .slice(0, index)
    .map((_, at) => +(0.5 + at / 100).toFixed(4));
  const frame = {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: settings.totalSteps || words.length,
    canvas_index: 0,
    mean_conf: +(
      confidences.reduce((sum, c) => sum + c, 0) / index
    ).toFixed(4),
    token: {
      t: words[position],
      m: false,
      id: 1000 + position,
      c: confidences[position],
      e: +(1.5 - position / 100).toFixed(4),
    },
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (settings.alts) {
    frame.alts = [{ id: 42, t: " other", p: 0.2 }];
  }
  return frame;
}

function streamAppend(words, options) {
  const page = loadPage({});
  const settings = options || {};
  for (let index = 1; index <= words.length; index++) {
    page.context.handleFrame(
      appendFrame(index, words, {
        totalSteps: words.length,
        alts: !!settings.alts,
      })
    );
  }
  return page;
}

const WORDS = ["The", " cat", " sat", " on", " the", " mat"];

// Drive a whole run through the page and hand back the context.
function runThrough(words, options) {
  const page = loadPage({});
  const context = page.context;
  const settings = options || {};
  for (let index = 1; index <= words.length; index++) {
    context.handleFrame(
      snapshotFrame(index, words, {
        totalSteps: words.length,
        alts: !!settings.alts,
      })
    );
  }
  return page;
}

function frames(context) {
  return context.runFrames;
}

// -- what the page retains while a run streams --

test("a run retains one frame per token", () => {
  const { context } = runThrough(WORDS);

  assert.equal(frames(context).history.length, WORDS.length);
  assert.equal(frames(context).tokens.length, WORDS.length);
});

test("frame N holds N tokens, which is the quadratic", () => {
  // The shape of the defect, written down. Six frames hold 21 token
  // records between them, and at 2,048 tokens the same rule gives
  // 2,098,176. An append stream has to make this row read 1,2,3...
  // in what arrives while this assertion still passes for what the
  // page can hand out.
  const { context } = runThrough(WORDS);

  const counts = frames(context).tokens.map((t) => t.length);

  assert.deepEqual(
    Array.from(counts), [1, 2, 3, 4, 5, 6]
  );
});

test("each retained frame's text is the run so far", () => {
  const { context } = runThrough(WORDS);

  const texts = frames(context).history;

  assert.equal(texts[0], "The");
  assert.equal(texts[2], "The cat sat");
  assert.equal(texts[5], "The cat sat on the mat");
});

test("the per-frame scalars stay aligned with the frames", () => {
  const { context } = runThrough(WORDS);
  const held = frames(context);

  assert.equal(context.runFramesAligned(held), true);
  assert.equal(held.elapsed.length, WORDS.length);
  assert.equal(held.revealed.length, WORDS.length);
  assert.equal(held.canvasIndex.length, WORDS.length);
});

test("elapsed carries the worker's own reading", () => {
  const { context } = runThrough(WORDS);

  assert.deepEqual(
    Array.from(frames(context).elapsed),
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  );
});

test("candidates are kept by position, not by frame", () => {
  // Already linear, and it stays that way. Worth pinning because a
  // reader could reasonably assume alts ride the frames.
  const { context } = runThrough(WORDS, { alts: true });

  assert.equal(context.positionAlts.length, WORDS.length);
  assert.equal(context.positionAlts[3][0].t, " other");
});

// -- what scrubbing reads back --

test("scrubbing to frame N reads exactly N tokens", () => {
  // The property the append store must preserve. Every consumer of a
  // scrub position asks for a full token array at that frame, so
  // whatever backs it has to answer the same way.
  const { context } = runThrough(WORDS);

  for (let index = 0; index < WORDS.length; index++) {
    const at = frames(context).tokens[index];
    assert.equal(at.length, index + 1, `frame ${index}`);
    assert.equal(at[index].t, WORDS[index], `frame ${index} newest`);
  }
});

test("a scrubbed frame's tokens are the run's own prefix", () => {
  // Why an append stream is sound here: frame N is not a rewrite of
  // frame N-1, it is frame N-1 plus one. Nothing before the newest
  // position ever changes, which is exactly what diffusion does not
  // guarantee and why it keeps snapshots.
  const { context } = runThrough(WORDS);
  const held = frames(context).tokens;

  for (let index = 1; index < WORDS.length; index++) {
    const earlier = held[index - 1];
    const later = held[index].slice(0, index);
    assert.deepEqual(
      earlier.map((t) => t.t), later.map((t) => t.t),
      `frame ${index} changed a settled position`
    );
  }
});

test("navigating sets the scrub position the renderers read", () => {
  const { context } = runThrough(WORDS);
  context.activateScrubber();

  context.navigateToFrame(2);

  assert.equal(context.currentScrubFrame, 2);
});

// -- the pre-edit baseline --

test("the baseline freezes the run at first completion", () => {
  const { context } = runThrough(WORDS);

  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  assert.equal(context.originalRun.totalFrames, WORDS.length);
  assert.equal(context.originalRun.tokens.length, WORDS.length);
});

test("the baseline is a second copy of the same quadratic", () => {
  // Why an edited run pays twice: original_tokens.json is nearly as
  // large as tokens.json on disk.
  const { context } = runThrough(WORDS);
  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  const counts = context.originalRun.tokens.map((t) => t.length);

  assert.deepEqual(Array.from(counts), [1, 2, 3, 4, 5, 6]);
});

test("the baseline does not move when the live run is cut", () => {
  const { context } = runThrough(WORDS);
  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  context.runFramesTruncate(context.runFrames, 3);

  assert.equal(context.originalRun.totalFrames, WORDS.length);
  assert.equal(context.runFrames.tokens.length, 3);
});

// -- what a save would carry --

test("the save payload carries one record set per frame", () => {
  const { context } = runThrough(WORDS);

  const records = context.tokenRecordsFrom(context.runFrames.tokens);

  assert.equal(records.length, WORDS.length);
  assert.deepEqual(
    Array.from(records.map((r) => r.length)), [1, 2, 3, 4, 5, 6]
  );
});

test("a saved record keeps text, id, confidence and entropy", () => {
  const { context } = runThrough(WORDS);

  const records = context.tokenRecordsFrom(context.runFrames.tokens);
  const first = records[0][0];

  assert.equal(first.t, "The");
  assert.equal(first.id, 1000);
  assert.equal(first.m, false);
  assert.equal(typeof first.c, "number");
  assert.equal(typeof first.e, "number");
});

test("the payload's totals match the retained run", () => {
  // The invariant the server-side expansion has to reproduce: as
  // many text frames as token frames, each the right length.
  const { context } = runThrough(WORDS);

  const records = context.tokenRecordsFrom(context.runFrames.tokens);
  const texts = context.runFrames.history;

  assert.equal(records.length, texts.length);
  for (let index = 0; index < texts.length; index++) {
    assert.equal(
      records[index].map((r) => r.t).join(""), texts[index],
      `frame ${index} text disagrees with its tokens`
    );
  }
});

// -- truncation, which is how an edit rewinds --

test("cutting the run shortens every array together", () => {
  const { context } = runThrough(WORDS);

  context.runFramesTruncate(context.runFrames, 3);
  const held = frames(context);

  assert.equal(context.runFramesAligned(held), true);
  assert.equal(held.history.length, 3);
  assert.equal(held.tokens.length, 3);
});

test("a cut run continues from where it was cut", () => {
  // Substitution's shape: truncate to the edit point, then take new
  // frames from the worker. The append path has to behave the same,
  // since the seed frame becomes one appended token.
  const { context } = runThrough(WORDS);
  context.runFramesTruncate(context.runFrames, 3);

  context.handleFrame(snapshotFrame(4, [
    "The", " cat", " sat", " down",
  ]));

  const held = frames(context);
  assert.equal(held.tokens.length, 4);
  assert.equal(held.tokens[3][3].t, " down");
  assert.equal(held.history[3], "The cat sat down");
});

// -- the same run, delivered as an append stream --
//
// The point of the whole change: everything the block above pins for
// a snapshot run has to hold for a run assembled from one position
// per frame, or something a user can see has moved.

// Both runs are built inside their own vm context, so their arrays
// and objects carry that context's prototypes and a strict deepEqual
// between them fails on identity rather than on content. Bringing
// both into this realm is what makes the comparison about the data.
function hostCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

test("an append run reads back the same frames", () => {
  const snapshotRun = runThrough(WORDS).context;
  const appendRun = streamAppend(WORDS).context;

  for (let index = 0; index < WORDS.length; index++) {
    assert.deepEqual(
      hostCopy(
        appendRun.runFramesTokensAt(appendRun.runFrames, index)
      ),
      hostCopy(
        snapshotRun.runFramesTokensAt(snapshotRun.runFrames, index)
      ),
      `frame ${index} tokens`
    );
    assert.equal(
      appendRun.runFramesTextAt(appendRun.runFrames, index),
      snapshotRun.runFramesTextAt(snapshotRun.runFrames, index),
      `frame ${index} text`
    );
  }
});

test("an append run keeps one record per token, not per frame", () => {
  // The measurement, in the store rather than on the wire. Six
  // frames used to cost 21 token records between them; they now
  // cost six.
  const { context } = streamAppend(WORDS);

  assert.equal(context.runFrames.positions.length, WORDS.length);
  assert.equal(context.runFrames.tokens.length, 0);
  assert.equal(context.runFrames.history.length, 0);
});

test("an append run still counts its frames", () => {
  const { context } = streamAppend(WORDS);

  assert.equal(
    context.runFramesLength(context.runFrames), WORDS.length
  );
});

test("an append run's scalars stay per frame", () => {
  const { context } = streamAppend(WORDS);
  const held = context.runFrames;

  assert.equal(context.runFramesAligned(held), true);
  assert.deepEqual(
    Array.from(held.elapsed), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  );
});

test("an append run's candidates still land by position", () => {
  const { context } = streamAppend(WORDS, { alts: true });

  assert.equal(context.positionAlts.length, WORDS.length);
  assert.equal(context.positionAlts[3][0].t, " other");
});

test("an append run builds the same save records", () => {
  // What the server has to be handed, or expand to. If these two
  // disagree, a run saved after the change is not the run that was
  // saved before it.
  const snapshotRun = runThrough(WORDS).context;
  const appendRun = streamAppend(WORDS).context;

  const expected = snapshotRun.tokenRecordsFrom(
    snapshotRun.runFrames.tokens
  );
  const built = [];
  for (let index = 0; index < WORDS.length; index++) {
    built.push(
      appendRun.runFramesTokensAt(appendRun.runFrames, index)
    );
  }

  assert.deepEqual(
    hostCopy(appendRun.tokenRecordsFrom(built)), hostCopy(expected)
  );
});

test("an append run survives a session round trip whole", () => {
  // The failure this retires. A long run used to exceed the storage
  // quota, fall back to a light payload, and come back with no
  // hover, no candidates and no entropy. There is no light payload
  // for an append run because the full one is already linear.
  const { context } = streamAppend(WORDS);

  const written = context.runFramesToJson(context.runFrames);
  const read = context.runFramesFromJson(
    JSON.parse(JSON.stringify(written))
  );

  assert.equal(context.runFramesLackDetail(read), false);
  assert.equal(
    context.runFramesTextAt(read, 5), "The cat sat on the mat"
  );
});

test("a light field set still carries an append run whole", () => {
  // The quota fallback asks for three fields. An append run answers
  // with everything anyway, because dropping its positions would
  // drop the run.
  const { context } = streamAppend(WORDS);

  const light = context.runFramesToJson(
    context.runFrames, ["history", "elapsed", "revealed"]
  );

  assert.equal(light.framePositions.length, WORDS.length);
});

// -- substitution: cut the run, then splice --
//
// The plan named this test and it was not written, which is how What
// If shipped broken: the truncate before a substitution extended the
// two empty snapshot arrays with holes, so the seed frame arrived at
// a family whose five per-frame arrays disagreed with its positions.
// Every other append test appends to a run that was never cut.

test("an append run can be cut and continued", () => {
  const { context } = streamAppend(WORDS);

  context.truncateRunArraysAt(2);
  context.handleFrame(
    appendFrame(3, ["The", " cat", " ran"])
  );

  assert.equal(context.runFramesLength(context.runFrames), 3);
  assert.equal(
    context.runFramesTextAt(context.runFrames, 2), "The cat ran"
  );
});

test("cutting an append run leaves no phantom frames", () => {
  // The arrays a snapshot run uses stay empty rather than being
  // extended with holes, because for this shape they were never the
  // run and a length there is a lie about how many frames exist.
  const { context } = streamAppend(WORDS);

  context.truncateRunArraysAt(2);
  const held = context.runFrames;

  assert.equal(held.count, 2);
  assert.equal(held.positions.length, 2);
  assert.equal(held.history.length, 0);
  assert.equal(held.tokens.length, 0);
  assert.equal(context.runFramesAligned(held), true);
});

test("a cut append run keeps its scalars in step", () => {
  const { context } = streamAppend(WORDS);

  context.truncateRunArraysAt(2);
  const held = context.runFrames;

  assert.equal(held.elapsed.length, 2);
  assert.equal(held.revealed.length, 2);
  assert.equal(held.canvasIndex.length, 2);
  assert.equal(held.meanConf.length, 2);
});

test("a cut run's baseline still reads the original", () => {
  // What the Original/Edited comparison depends on: the branch
  // shortens the live run and the baseline must not move with it.
  const { context } = streamAppend(WORDS);
  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  context.truncateRunArraysAt(2);
  context.handleFrame(appendFrame(3, ["The", " cat", " ran"]));

  assert.equal(context.originalRun.totalFrames, WORDS.length);
  assert.equal(
    context.originalRunTextAt(context.originalRun, 5),
    "The cat sat on the mat"
  );
});

test("an edit session can be abandoned and the run restored", () => {
  // Retry and Exit both roll back to the pre-edit snapshot, which
  // now has to carry the positions as well as the scalars.
  const { context } = streamAppend(WORDS);
  const before = context.runFramesSnapshot(context.runFrames);

  context.truncateRunArraysAt(2);
  context.handleFrame(appendFrame(3, ["The", " cat", " ran"]));
  context.runFramesRestore(context.runFrames, before);

  assert.equal(
    context.runFramesLength(context.runFrames), WORDS.length
  );
  assert.equal(
    context.runFramesTextAt(context.runFrames, 5),
    "The cat sat on the mat"
  );
});

// -- a stream that lost a frame --

test("a gap in the stream stops the run", () => {
  // A snapshot stream repaired itself: a dropped frame cost one bad
  // render and the next frame restated everything. An append stream
  // cannot, so a gap has to be refused rather than absorbed.
  const { context } = loadPage({});
  context.handleFrame(appendFrame(1, WORDS));

  context.handleFrame(appendFrame(3, WORDS));

  assert.equal(context.runFramesLength(context.runFrames), 1);
});

test("a gap says so rather than failing silently", () => {
  const { context, registry } = loadPage({});
  context.handleFrame(appendFrame(1, WORDS));

  context.handleFrame(appendFrame(3, WORDS));

  const status = registry.get("status-message");
  assert.match(status.textContent, /out of order/);
});

test("a repeated frame is a gap too", () => {
  // The other direction, and the more likely one: a retry or a
  // duplicated delivery would otherwise insert a position twice and
  // shift the rest of the run.
  const { context } = loadPage({});
  context.handleFrame(appendFrame(1, WORDS));
  context.handleFrame(appendFrame(2, WORDS));

  context.handleFrame(appendFrame(2, WORDS));

  assert.equal(context.runFramesLength(context.runFrames), 2);
});
