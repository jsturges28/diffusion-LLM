// The image tokeniser view.
//
// Shows what a vision-language model does to a picture before any of
// it reaches the language model: rescale it to a whole number of
// tiles, cut it up, read each tile as a grid of patches, and fuse
// square blocks of patches into single tokens.
//
// The image never leaves the browser. Only its width and height go to
// the server, because the geometry depends on nothing else, and the
// drawing happens here over the file the reader chose. That is also
// why there is no upload, no temporary artifact and nothing to clean
// up afterwards.
//
// A classic script with no export tail, like every other page here,
// so the `vm` harness in tests/web/static can load it and call these
// functions directly. See AGENTS.md.

"use strict";

// Layout constants for the canvases. Named because three drawing
// functions share them and a stray literal in one would put its grid
// half a line off the others.
var VISION_PAD = 10;
var VISION_LINE_TILE = 1.5;
var VISION_LINE_PATCH = 0.4;
var VISION_LINE_FUSED = 1.1;

// Drawn thinner than a tile boundary and fainter than a fused block,
// because at 27 patches a side a solid grid is a grey rectangle.
var VISION_PATCH_ALPHA = 0.25;
var VISION_UNSEEN_ALPHA = 0.5;

// How many patches a side before the patch grid stops being drawn as
// lines. Past this it reads as fill rather than as a grid, and the
// fused blocks are the part worth seeing.
var VISION_PATCH_GRID_MAX = 64;

var visionEncoders = [];
var visionSelected = null;
var visionImage = null;
var visionGeometry = {};
var visionHoverToken = -1;

// ---- pure geometry helpers, which is what the tests drive ----

// The rectangle an image of `width` x `height` occupies inside a box,
// scaled to fit and centred. Returned rather than applied so the
// caller can place a grid in the same coordinates.
function visionFitBox(width, height, boxWidth, boxHeight, pad) {
  var usableWidth = Math.max(boxWidth - pad * 2, 1);
  var usableHeight = Math.max(boxHeight - pad * 2, 1);
  var scale = Math.min(usableWidth / width, usableHeight / height);
  var drawnWidth = width * scale;
  var drawnHeight = height * scale;
  return {
    x: pad + (usableWidth - drawnWidth) / 2,
    y: pad + (usableHeight - drawnHeight) / 2,
    width: drawnWidth,
    height: drawnHeight,
    scale: scale
  };
}

// Which token block a point falls in, or -1 when it is outside.
// Integer arithmetic on the token grid rather than a hit test per
// block, so this stays cheap on a pointer move.
function visionTokenAt(pointX, pointY, box, tokenSide) {
  if (tokenSide <= 0) {
    return -1;
  }
  var offsetX = pointX - box.x;
  var offsetY = pointY - box.y;
  if (offsetX < 0 || offsetY < 0) {
    return -1;
  }
  if (offsetX >= box.width || offsetY >= box.height) {
    return -1;
  }
  var col = Math.floor(offsetX / (box.width / tokenSide));
  var row = Math.floor(offsetY / (box.height / tokenSide));
  if (col >= tokenSide || row >= tokenSide) {
    return -1;
  }
  return row * tokenSide + col;
}

// The patch rectangle one token covers, mirroring `patch_block` in
// src/inference/vision_geometry.py. Duplicated rather than fetched
// because it is three lines and a hover cannot wait for a request;
// tests hold the two to the same answers.
function visionPatchBlock(tokenIndex, patchSide, scale) {
  var tokenSide = Math.floor(patchSide / scale);
  if (tokenSide <= 0 || tokenIndex < 0) {
    return null;
  }
  if (tokenIndex >= tokenSide * tokenSide) {
    return null;
  }
  return {
    row: Math.floor(tokenIndex / tokenSide) * scale,
    col: (tokenIndex % tokenSide) * scale,
    height: scale,
    width: scale
  };
}

// How the token cost compares across encoders, as the ordering the
// table draws. Cheapest first, so a reader sees the choice.
function visionCostRows(byEncoder) {
  var rows = [];
  for (var index = 0; index < visionEncoders.length; index++) {
    var encoder = visionEncoders[index];
    var answer = byEncoder[encoder.id];
    if (!answer) {
      continue;
    }
    rows.push({
      id: encoder.id,
      name: encoder.display_name,
      tiles: answer.image.tile_count,
      perTile: answer.encoder.tokens_per_tile,
      total: answer.image.total_tokens
    });
  }
  return rows;
}

// The sentence under the cost table, which is where the lesson lands.
// Built from the numbers rather than hard-coded, so it cannot claim
// something the table contradicts.
function visionCostNote(rows) {
  if (rows.length < 2) {
    return "";
  }
  var cheapest = rows[0];
  var dearest = rows[0];
  for (var index = 1; index < rows.length; index++) {
    if (rows[index].total < cheapest.total) {
      cheapest = rows[index];
    }
    if (rows[index].total > dearest.total) {
      dearest = rows[index];
    }
  }
  if (cheapest.total === dearest.total) {
    return "Both encoders spend the same on this shape.";
  }
  var ratio = (dearest.total / cheapest.total).toFixed(1);
  return (
    dearest.name + " spends " + ratio + " times what " +
    cheapest.name + " does on the same picture, because it fuses " +
    "fewer patches into each token."
  );
}

// ---- drawing ----

function visionClear(canvas) {
  var context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  return context;
}

function visionDrawImage(context, box) {
  if (visionImage) {
    context.drawImage(
      visionImage, box.x, box.y, box.width, box.height
    );
    return;
  }
  // No picture chosen: draw the frame so the geometry still reads.
  // Faint, because the grids are drawn in the same colour and a
  // stronger fill swallows the patch lines on top of it.
  context.save();
  context.globalAlpha = 0.07;
  context.fillStyle = "#00ff41";
  context.fillRect(box.x, box.y, box.width, box.height);
  context.restore();
}

// Step 1 and 2 together: the shape it arrives as, against the shape
// the encoder works in. Two outlines rather than one, because the
// change between them is the point.
function visionDrawResize(canvas, answer) {
  var context = visionClear(canvas);
  var image = answer.image;
  var box = visionFitBox(
    image.fitted_width, image.fitted_height,
    canvas.width, canvas.height, VISION_PAD
  );

  visionDrawImage(context, box);

  // Where the source shape sat inside the fitted one.
  var sourceScale = Math.min(
    image.fitted_width / image.source_width,
    image.fitted_height / image.source_height
  );
  var sourceWidth = image.source_width * sourceScale * box.scale;
  var sourceHeight = image.source_height * sourceScale * box.scale;
  context.save();
  context.strokeStyle = "#ff9f1c";
  context.lineWidth = VISION_LINE_TILE;
  context.setLineDash([5, 4]);
  context.strokeRect(
    box.x + (box.width - sourceWidth) / 2,
    box.y + (box.height - sourceHeight) / 2,
    sourceWidth, sourceHeight
  );
  context.restore();

  context.save();
  context.strokeStyle = "#00ff41";
  context.lineWidth = VISION_LINE_TILE;
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.restore();
}

// Step 3: the tile grid over the fitted image.
function visionDrawTiles(canvas, answer) {
  var context = visionClear(canvas);
  var image = answer.image;
  var box = visionFitBox(
    image.fitted_width, image.fitted_height,
    canvas.width, canvas.height, VISION_PAD
  );

  visionDrawImage(context, box);

  context.save();
  context.strokeStyle = "#00ff41";
  context.lineWidth = VISION_LINE_TILE;
  var stepX = box.width / Math.max(image.tile_cols, 1);
  var stepY = box.height / Math.max(image.tile_rows, 1);
  for (var col = 0; col <= image.tile_cols; col++) {
    context.beginPath();
    context.moveTo(box.x + col * stepX, box.y);
    context.lineTo(box.x + col * stepX, box.y + box.height);
    context.stroke();
  }
  for (var row = 0; row <= image.tile_rows; row++) {
    context.beginPath();
    context.moveTo(box.x, box.y + row * stepY);
    context.lineTo(box.x + box.width, box.y + row * stepY);
    context.stroke();
  }
  context.restore();
}

// Steps 4 and 5: one tile, its patch grid, and the blocks that fuse.
function visionDrawPatches(canvas, answer) {
  var context = visionClear(canvas);
  var encoder = answer.encoder;
  var box = visionFitBox(
    encoder.tile, encoder.tile,
    canvas.width, canvas.height, VISION_PAD
  );

  visionDrawImage(context, box);

  var perPixel = box.width / encoder.tile;

  // The patch grid, only while it still reads as a grid.
  if (encoder.patch_side <= VISION_PATCH_GRID_MAX) {
    context.save();
    context.globalAlpha = VISION_PATCH_ALPHA;
    context.strokeStyle = "#00ff41";
    context.lineWidth = VISION_LINE_PATCH;
    for (var index = 0; index <= encoder.patch_side; index++) {
      var at = index * encoder.patch * perPixel;
      context.beginPath();
      context.moveTo(box.x + at, box.y);
      context.lineTo(box.x + at, box.y + box.height);
      context.stroke();
      context.beginPath();
      context.moveTo(box.x, box.y + at);
      context.lineTo(box.x + box.width, box.y + at);
      context.stroke();
    }
    context.restore();
  }

  // The fused blocks, one per token.
  var blockPixels = encoder.patch * encoder.scale * perPixel;
  var fusedSpan = encoder.token_side * blockPixels;
  context.save();
  context.strokeStyle = "#00ff41";
  context.lineWidth = VISION_LINE_FUSED;
  for (var side = 0; side <= encoder.token_side; side++) {
    var offset = side * blockPixels;
    context.beginPath();
    context.moveTo(box.x + offset, box.y);
    context.lineTo(box.x + offset, box.y + fusedSpan);
    context.stroke();
    context.beginPath();
    context.moveTo(box.x, box.y + offset);
    context.lineTo(box.x + fusedSpan, box.y + offset);
    context.stroke();
  }
  context.restore();

  // The strip no patch covers, when there is one. Drawn because what
  // the model cannot see is as informative as what it can.
  if (encoder.unseen_edge > 0) {
    var seen = encoder.patch_side * encoder.patch * perPixel;
    var unseen = encoder.unseen_edge * perPixel;
    context.save();
    context.globalAlpha = VISION_UNSEEN_ALPHA;
    context.fillStyle = "#ff4444";
    context.fillRect(box.x + seen, box.y, unseen, box.height);
    context.fillRect(box.x, box.y + seen, box.width, unseen);
    context.restore();
  }

  // The hovered token's block, on top of everything.
  var block = visionPatchBlock(
    visionHoverToken, encoder.patch_side, encoder.scale
  );
  if (block) {
    context.save();
    context.fillStyle = "#ff9f1c";
    context.globalAlpha = 0.35;
    context.fillRect(
      box.x + block.col * encoder.patch * perPixel,
      box.y + block.row * encoder.patch * perPixel,
      block.width * encoder.patch * perPixel,
      block.height * encoder.patch * perPixel
    );
    context.restore();
  }
  return box;
}

// ---- wiring ----

function visionText(id, value) {
  var node = document.getElementById(id);
  if (node) {
    node.textContent = value;
  }
}

function visionDescribe(answer) {
  var image = answer.image;
  var encoder = answer.encoder;

  visionText("vision-resize-text",
    image.source_width + "x" + image.source_height + " becomes " +
    image.resized_width + "x" + image.resized_height +
    ", then rounds out to " + image.fitted_width + "x" +
    image.fitted_height + " so it divides into whole tiles." +
    (image.aspect_changed
      ? " That rounding changes the shape, so the model does not see"
        + " the framing you chose."
      : " The shape happens to survive this one."));

  visionText("vision-tiles-text",
    image.tile_rows + " by " + image.tile_cols + " tiles of " +
    encoder.tile + "x" + encoder.tile + ", plus one more tile" +
    " holding the whole picture shrunk down.");

  visionText("vision-patches-text",
    "One tile is " + encoder.patch_side + "x" + encoder.patch_side +
    " patches of " + encoder.patch + "px. Every " + encoder.scale +
    "x" + encoder.scale + " block of them, " +
    encoder.patches_per_token + " patches, fuses into a single" +
    " token, leaving " + encoder.tokens_per_tile + "." +
    (encoder.unseen_edge > 0
      ? " The red strip is the last " + encoder.unseen_edge +
        "px, which completes no patch and is never seen."
      : ""));

  visionText("vision-token-readout",
    visionHoverToken >= 0
      ? "Token " + visionHoverToken + " of " +
        encoder.tokens_per_tile + " covers " +
        encoder.patches_per_token + " patches."
      : "Hover the tile to see which patches make up a token.");
}

function visionRenderCost(byEncoder) {
  var rows = visionCostRows(byEncoder);
  var body = document.getElementById("vision-cost-rows");
  if (body) {
    body.textContent = "";
    for (var index = 0; index < rows.length; index++) {
      var row = rows[index];
      var tr = document.createElement("tr");
      var cells = [
        row.name, String(row.tiles) + " + 1",
        String(row.perTile), String(row.total)
      ];
      for (var cell = 0; cell < cells.length; cell++) {
        var td = document.createElement("td");
        td.textContent = cells[cell];
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
  }
  visionText("vision-cost-note", visionCostNote(rows));
}

function visionRender() {
  var answer = visionGeometry[visionSelected];
  if (!answer) {
    return;
  }
  visionDescribe(answer);
  visionDrawResize(
    document.getElementById("vision-canvas-resize"), answer
  );
  visionDrawTiles(
    document.getElementById("vision-canvas-tiles"), answer
  );
  visionDrawPatches(
    document.getElementById("vision-canvas-patches"), answer
  );
}

function visionDimensions() {
  if (visionImage) {
    return {
      width: visionImage.naturalWidth,
      height: visionImage.naturalHeight
    };
  }
  var width = document.getElementById("vision-width");
  var height = document.getElementById("vision-height");
  return {
    width: width ? Number(width.value) : 0,
    height: height ? Number(height.value) : 0
  };
}

function visionMeasure() {
  var size = visionDimensions();
  if (!(size.width > 0) || !(size.height > 0)) {
    return Promise.resolve();
  }

  var pending = visionEncoders.map(function (encoder) {
    var query = "?encoder=" + encodeURIComponent(encoder.id) +
      "&width=" + size.width + "&height=" + size.height;
    return fetch("/api/vision/geometry" + query)
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (body) {
            return { id: encoder.id, error: body.error || "failed" };
          });
        }
        return response.json().then(function (body) {
          return { id: encoder.id, answer: body };
        });
      })
      .catch(function (error) {
        return { id: encoder.id, error: String(error) };
      });
  });

  return Promise.all(pending).then(function (results) {
    var failures = [];
    visionGeometry = {};
    for (var index = 0; index < results.length; index++) {
      if (results[index].answer) {
        visionGeometry[results[index].id] = results[index].answer;
      } else {
        failures.push(results[index]);
      }
    }
    var unavailable = document.getElementById("vision-unavailable");
    if (unavailable) {
      unavailable.hidden = failures.length === 0;
      if (failures.length > 0) {
        visionText("vision-unavailable-text",
          failures.map(function (item) {
            return item.id + ": " + item.error;
          }).join("; "));
      }
    }
    if (!visionGeometry[visionSelected]) {
      var ids = Object.keys(visionGeometry);
      visionSelected = ids.length > 0 ? ids[0] : null;
    }
    visionRenderCost(visionGeometry);
    visionRender();
  });
}

function visionSelectEncoder(encoderId) {
  visionSelected = encoderId;
  visionHoverToken = -1;
  var tabs = document.querySelectorAll(".vision-tab");
  for (var index = 0; index < tabs.length; index++) {
    var tab = tabs[index];
    var active = tab.getAttribute("data-encoder") === encoderId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  visionRender();
}

function visionBuildTabs() {
  var host = document.getElementById("vision-encoder-tabs");
  if (!host) {
    return;
  }
  host.textContent = "";
  for (var index = 0; index < visionEncoders.length; index++) {
    var encoder = visionEncoders[index];
    var button = document.createElement("button");
    button.type = "button";
    button.className = "vision-tab";
    button.setAttribute("role", "tab");
    button.setAttribute("data-encoder", encoder.id);
    button.setAttribute(
      "aria-selected", index === 0 ? "true" : "false"
    );
    if (index === 0) {
      button.classList.add("is-active");
    }
    button.textContent = encoder.display_name;
    button.title = encoder.summary;
    (function (id) {
      button.addEventListener("click", function () {
        visionSelectEncoder(id);
      });
    })(encoder.id);
    host.appendChild(button);
  }
}

function visionWireHover() {
  var canvas = document.getElementById("vision-canvas-patches");
  if (!canvas) {
    return;
  }
  canvas.addEventListener("mousemove", function (event) {
    var answer = visionGeometry[visionSelected];
    if (!answer) {
      return;
    }
    var bounds = canvas.getBoundingClientRect
      ? canvas.getBoundingClientRect()
      : { left: 0, top: 0 };
    var box = visionFitBox(
      answer.encoder.tile, answer.encoder.tile,
      canvas.width, canvas.height, VISION_PAD
    );
    var found = visionTokenAt(
      event.clientX - bounds.left, event.clientY - bounds.top,
      box, answer.encoder.token_side
    );
    if (found !== visionHoverToken) {
      visionHoverToken = found;
      visionRender();
    }
  });
  canvas.addEventListener("mouseleave", function () {
    if (visionHoverToken !== -1) {
      visionHoverToken = -1;
      visionRender();
    }
  });
}

function visionWireFile() {
  var picker = document.getElementById("vision-file");
  if (!picker) {
    return;
  }
  picker.addEventListener("change", function () {
    var file = picker.files && picker.files[0];
    if (!file) {
      return;
    }
    var image = new Image();
    image.onload = function () {
      visionImage = image;
      var empty = document.getElementById("vision-empty");
      if (empty) {
        empty.hidden = true;
      }
      visionMeasure();
    };
    // Object URLs stay local; nothing is uploaded.
    image.src = URL.createObjectURL(file);
  });
}

function visionBoot() {
  var boot = window.__BOOT__;
  if (boot && boot.encoders) {
    visionEncoders = boot.encoders;
  }
  var ready = Promise.resolve();
  if (visionEncoders.length === 0) {
    // The harness and a directly-opened file have no boot state, so
    // every consumer here falls back to fetching, as the other pages
    // do for the same reason.
    ready = fetch("/api/vision/encoders")
      .then(function (response) { return response.json(); })
      .then(function (body) { visionEncoders = body.encoders || []; })
      .catch(function () { visionEncoders = []; });
  }
  return ready.then(function () {
    visionBuildTabs();
    if (visionEncoders.length > 0) {
      visionSelected = visionEncoders[0].id;
    }
    visionWireFile();
    visionWireHover();
    var apply = document.getElementById("vision-apply");
    if (apply) {
      apply.addEventListener("click", function () {
        visionImage = null;
        visionMeasure();
      });
    }
    return visionMeasure();
  });
}

visionBoot();
