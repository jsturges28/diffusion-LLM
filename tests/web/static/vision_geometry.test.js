// The tokeniser page's own arithmetic.
//
// Strategy: load `vision.js` into the shared stub with a narrowed
// script list, then drive the four pure functions it exposes. Passing
// proves a drawn grid lands where the geometry says, that a pointer
// maps to the token under it, and that the sentence under the cost
// table cannot contradict the table.
//
// The pipeline arithmetic is not retested here. Python owns that and
// holds it against the library in tests/inference. This file owns the
// part that exists only in the browser: fitting a picture into a
// canvas, and turning a pointer position into a token index.
//
// `visionPatchBlock` is the exception, duplicated on purpose: it also
// exists in Python as `patch_block`, because a hover cannot wait on a
// request. A test below pins the two to the same answers so the
// duplication stays honest.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

// The two shipping geometries, as the endpoint reports them.
const SMALL = {
  tile: 512, patch: 16, scale: 4, patch_side: 32,
  token_side: 8, tokens_per_tile: 64, patches_per_token: 16,
  unseen_edge: 0,
};
const LARGE = {
  tile: 384, patch: 14, scale: 3, patch_side: 27,
  token_side: 9, tokens_per_tile: 81, patches_per_token: 9,
  unseen_edge: 6,
};

function page() {
  const { context } = loadPage({
    scripts: ["overlays.js", "vision.js"],
  });
  return context;
}

// Objects returned from the page are built inside the `vm` realm, so
// their prototype is not this realm's `Object.prototype` and
// `deepEqual` from `assert/strict` rejects them on that alone. Fields
// are compared instead, which also states what is being asserted.
function assertBlock(block, expected, label) {
  assert.ok(block, `${label}: no block`);
  for (const key of ["row", "col", "height", "width"]) {
    assert.equal(block[key], expected[key], `${label}.${key}`);
  }
}

// -- fitting a picture into a canvas --

test("a square image fills a square box, less the padding", () => {
  const box = page().visionFitBox(100, 100, 220, 220, 10);

  assert.equal(box.width, 200);
  assert.equal(box.height, 200);
  assert.equal(box.x, 10);
  assert.equal(box.y, 10);
});

test("a wide image is centred vertically", () => {
  // The common case: a landscape photo in a squarer canvas leaves a
  // band above and below, and the grid has to sit in the picture
  // rather than in the canvas.
  const box = page().visionFitBox(200, 100, 220, 220, 10);

  assert.equal(box.width, 200);
  assert.equal(box.height, 100);
  assert.equal(box.x, 10);
  assert.equal(box.y, 60);
});

test("a tall image is centred horizontally", () => {
  const box = page().visionFitBox(100, 200, 220, 220, 10);

  assert.equal(box.height, 200);
  assert.equal(box.x, 60);
  assert.equal(box.y, 10);
});

test("the aspect ratio survives the fit", () => {
  // The fit must not distort, because the page's whole subject is
  // where distortion happens, and it is not here.
  const box = page().visionFitBox(1920, 1080, 520, 320, 10);

  const before = 1920 / 1080;
  assert.ok(Math.abs(box.width / box.height - before) < 1e-9);
});

test("the fit never leaves the box", () => {
  const context = page();

  for (const [width, height] of [[1, 1000], [1000, 1], [7, 13]]) {
    const box = context.visionFitBox(width, height, 200, 200, 10);
    assert.ok(box.x >= 10 - 1e-9, `${width}x${height} x`);
    assert.ok(box.y >= 10 - 1e-9, `${width}x${height} y`);
    assert.ok(box.x + box.width <= 190 + 1e-9);
    assert.ok(box.y + box.height <= 190 + 1e-9);
  }
});

test("padding larger than the box does not invert it", () => {
  // Negative space. A canvas smaller than its own padding would give
  // a negative usable size and a mirrored grid.
  const box = page().visionFitBox(100, 100, 10, 10, 20);

  assert.ok(box.width > 0);
  assert.ok(box.height > 0);
});

// -- a pointer to a token --

test("the top-left corner is token zero", () => {
  const context = page();
  const box = { x: 0, y: 0, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(1, 1, box, 9), 0);
});

test("the bottom-right corner is the last token", () => {
  const context = page();
  const box = { x: 0, y: 0, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(89, 89, box, 9), 80);
});

test("a token index is row-major", () => {
  // Second row, third column, on a 9-wide grid of 10px cells.
  const context = page();
  const box = { x: 0, y: 0, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(25, 15, box, 9), 9 + 2);
});

test("the box offset is honoured", () => {
  const context = page();
  const box = { x: 100, y: 50, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(101, 51, box, 9), 0);
  assert.equal(context.visionTokenAt(99, 51, box, 9), -1);
});

test("a point outside the picture is no token", () => {
  // Returned rather than clamped, because clamping would light a
  // block while the pointer sat in the padding.
  const context = page();
  const box = { x: 10, y: 10, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(5, 50, box, 9), -1);
  assert.equal(context.visionTokenAt(50, 5, box, 9), -1);
  assert.equal(context.visionTokenAt(105, 50, box, 9), -1);
  assert.equal(context.visionTokenAt(50, 105, box, 9), -1);
});

test("the far edge does not spill past the last token", () => {
  // The boundary that produces an index of `tokenSide * tokenSide`,
  // one past the end, if the comparison is wrong.
  const context = page();
  const box = { x: 0, y: 0, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(90, 90, box, 9), -1);
  assert.equal(context.visionTokenAt(89.999, 89.999, box, 9), 80);
});

test("a grid with no tokens reports none", () => {
  const context = page();
  const box = { x: 0, y: 0, width: 90, height: 90 };

  assert.equal(context.visionTokenAt(10, 10, box, 0), -1);
});

// -- the patch block a token covers --

test("token zero covers the first block", () => {
  const block = page().visionPatchBlock(0, 27, 3);

  assertBlock(block, { row: 0, col: 0, height: 3, width: 3 }, "0");
});

test("a block steps by the scale, not by one", () => {
  // The bug this guards: stepping by one would overlap every block
  // and claim nine tokens share the same patches.
  const context = page();

  assertBlock(
    context.visionPatchBlock(1, 27, 3),
    { row: 0, col: 3, height: 3, width: 3 }, "token 1"
  );
  assertBlock(
    context.visionPatchBlock(9, 27, 3),
    { row: 3, col: 0, height: 3, width: 3 }, "token 9"
  );
});

test("the last token covers the last block", () => {
  const block = page().visionPatchBlock(80, 27, 3);

  assertBlock(
    block, { row: 24, col: 24, height: 3, width: 3 }, "last"
  );
});

test("a token outside the grid has no block", () => {
  const context = page();

  assert.equal(context.visionPatchBlock(81, 27, 3), null);
  assert.equal(context.visionPatchBlock(-1, 27, 3), null);
  assert.equal(context.visionPatchBlock(64, 32, 4), null);
});

test("no two tokens claim the same patch", () => {
  // The property the highlight depends on. An overlap would light a
  // region belonging to a different token.
  const context = page();

  for (const encoder of [SMALL, LARGE]) {
    const seen = new Set();
    for (let token = 0; token < encoder.tokens_per_tile; token++) {
      const block = context.visionPatchBlock(
        token, encoder.patch_side, encoder.scale
      );
      for (let dr = 0; dr < block.height; dr++) {
        for (let dc = 0; dc < block.width; dc++) {
          const at = (block.row + dr) * encoder.patch_side
            + (block.col + dc);
          assert.ok(!seen.has(at), `patch ${at} claimed twice`);
          seen.add(at);
        }
      }
    }
    assert.equal(
      seen.size,
      encoder.tokens_per_tile * encoder.patches_per_token
    );
  }
});

test("a hover maps to the block the geometry names", () => {
  // The two functions joined, which is how the page uses them: the
  // pointer picks a token and the token picks a rectangle.
  const context = page();
  const box = { x: 0, y: 0, width: 270, height: 270 };

  const token = context.visionTokenAt(100, 40, box, LARGE.token_side);
  const block = context.visionPatchBlock(
    token, LARGE.patch_side, LARGE.scale
  );

  // A 30px cell, so x=100 is column 3 and y=40 is row 1.
  assert.equal(token, 1 * LARGE.token_side + 3);
  assertBlock(
    block, { row: 3, col: 9, height: 3, width: 3 }, "hover"
  );
});

// -- the sentence under the cost table --

function answer(tiles, perTile) {
  return {
    encoder: { tokens_per_tile: perTile },
    image: { tile_count: tiles, total_tokens: (tiles + 1) * perTile },
  };
}

test("the cost note names the ratio between encoders", () => {
  const context = page();
  context.visionEncoders = [
    { id: "a", display_name: "Cheap" },
    { id: "b", display_name: "Dear" },
  ];

  const rows = context.visionCostRows({
    a: answer(12, 64), b: answer(12, 81),
  });
  const note = context.visionCostNote(rows);

  assert.match(note, /Dear spends 1\.3 times what Cheap does/);
});

test("the note says so when the two agree", () => {
  const context = page();
  context.visionEncoders = [
    { id: "a", display_name: "One" },
    { id: "b", display_name: "Two" },
  ];

  const rows = context.visionCostRows({
    a: answer(4, 64), b: answer(4, 64),
  });

  assert.match(context.visionCostNote(rows), /same/);
});

test("one encoder alone makes no comparison", () => {
  // Negative space: a note claiming a ratio against nothing would be
  // the page inventing a comparison.
  const context = page();
  context.visionEncoders = [{ id: "a", display_name: "Only" }];

  const rows = context.visionCostRows({ a: answer(4, 64) });

  assert.equal(context.visionCostNote(rows), "");
});

test("an encoder with no answer is left out of the table", () => {
  // One encoder uncached and the other ready is a real state, and the
  // table must show the one it has rather than a blank row.
  const context = page();
  context.visionEncoders = [
    { id: "a", display_name: "Ready" },
    { id: "b", display_name: "Missing" },
  ];

  const rows = context.visionCostRows({ a: answer(4, 64) });

  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, "Ready");
});

test("the table keeps declaration order", () => {
  const context = page();
  context.visionEncoders = [
    { id: "a", display_name: "First" },
    { id: "b", display_name: "Second" },
  ];

  const rows = context.visionCostRows({
    b: answer(4, 81), a: answer(4, 64),
  });

  // `Array.from` rather than `rows.map`, which would hand back an
  // array from the page's realm and fail on its prototype alone.
  const names = Array.from(rows, (row) => row.name);
  assert.deepEqual(names, ["First", "Second"]);
});

test("a row reports the total the endpoint gave", () => {
  // Not recomputed here. The server owns the arithmetic, and a table
  // that recalculated could disagree with the diagram beside it.
  const context = page();
  context.visionEncoders = [{ id: "a", display_name: "One" }];

  const rows = context.visionCostRows({ a: answer(12, 81) });

  assert.equal(rows[0].total, 1053);
  assert.equal(rows[0].tiles, 12);
  assert.equal(rows[0].perTile, 81);
});
