// ASCII Scene Renderer — luminance-based ASCII art animation.
//
// Draws a procedural scene (title with diffusion reveal,
// palm trees, scrolling car, stars) onto a hidden canvas,
// samples pixel brightness, and renders the result as
// ASCII characters on a visible canvas.

"use strict";

(function () {
  var DENSITY_RAMP = " .:-=+*#%@";
  var CELL_W = 10;
  var CELL_H = 16;
  // Pre-rendered ASCII art title (patorjk.com "Slant").
  var TITLE_LINES = [
    "    ____   _  ____ ____              _                  __     __     __  ___",
    "   / __ \\ (_)/ __// __/__  __ _____ (_)____   ____     / /    / /    /  |/  /",
    "  / / / // // /_ / /_ / / / // ___// // __ \\ / __ \\   / /    / /    / /|_/ /",
    " / /_/ // // __// __// /_/ /(__  )/ // /_/ // / / /  / /___ / /___ / /  / /",
    "/_____//_//_/  /_/   \\__,_//____//_/ \\____//_/ /_/  /_____//_____//_/  /_/",
  ];
  var TITLE_WIDTH = 0;
  for (var tw = 0; tw < TITLE_LINES.length; tw++) {
    if (TITLE_LINES[tw].length > TITLE_WIDTH) {
      TITLE_WIDTH = TITLE_LINES[tw].length;
    }
  }

  // Count non-space characters for diffusion reveal.
  var TITLE_CHAR_COUNT = 0;
  for (var tc = 0; tc < TITLE_LINES.length; tc++) {
    for (var cc = 0; cc < TITLE_LINES[tc].length; cc++) {
      if (TITLE_LINES[tc][cc] !== " ") {
        TITLE_CHAR_COUNT++;
      }
    }
  }

  // Diffusion reveal timing (seconds).
  var REVEAL_DURATION = 3.0;
  var HOLD_DURATION = 4.0;
  var FADE_DURATION = 2.0;
  var CYCLE_DURATION =
    REVEAL_DURATION + HOLD_DURATION + FADE_DURATION;

  var NOISE_CHARS =
    "/\\|_()-=+*#%@!~:;.,<>[]{}^&$";

  var CAR_SPEED = 110; // px per second

  var hiddenCanvas = null;
  var hiddenCtx = null;
  var visibleCanvas = null;
  var visibleCtx = null;
  var rafId = null;
  var startTime = 0;

  // Pre-generated star positions (normalized 0-1).
  var stars = [];
  var STAR_COUNT = 80;
  for (var i = 0; i < STAR_COUNT; i++) {
    stars.push({
      x: Math.random(),
      y: Math.random() * 0.65,
      brightness: 0.3 + Math.random() * 0.7,
      twinkleRate: 1.5 + Math.random() * 3.0,
    });
  }

  // Palm tree positions (normalized x, 0-1).
  var PALM_POSITIONS = [0.12, 0.35, 0.62, 0.85];

  // Seagull flight parameters (normalized coords).
  var seagulls = [];
  var SEAGULL_COUNT = 5;
  for (var si = 0; si < SEAGULL_COUNT; si++) {
    seagulls.push({
      speed: 0.03 + Math.random() * 0.04,
      y: 0.38 + Math.random() * 0.15,
      phase: Math.random() * 6.28,
      bobRate: 2.0 + Math.random() * 2.0,
      bobAmp: 0.008 + Math.random() * 0.012,
      offset: Math.random(),
    });
  }

  // Sun/moon full cycle duration (seconds).
  var CELESTIAL_CYCLE = 40.0;

  // --------------------------------------------------
  // Scene drawing helpers
  // --------------------------------------------------

  function drawStars(ctx, width, height, time) {
    for (var i = 0; i < stars.length; i++) {
      var s = stars[i];
      var flicker =
        0.5 +
        0.5 * Math.sin(time * s.twinkleRate + i);
      var alpha = s.brightness * flicker;
      var radius = 1 + flicker;
      ctx.fillStyle =
        "rgba(255,255,255," + alpha.toFixed(2) + ")";
      ctx.beginPath();
      ctx.arc(
        s.x * width,
        s.y * height,
        radius,
        0,
        6.283
      );
      ctx.fill();
    }
  }

  function drawGround(ctx, width, height) {
    var groundY = height * 0.85;
    ctx.fillStyle = "#333";
    ctx.fillRect(0, groundY, width, height - groundY);
    ctx.fillStyle = "#666";
    ctx.fillRect(0, groundY, width, 4);
  }

  function drawSeagulls(ctx, width, height, time) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 6;
    ctx.lineCap = "round";

    for (var i = 0; i < seagulls.length; i++) {
      var g = seagulls[i];
      var normX =
        (g.offset + time * g.speed) % 1.3 - 0.15;
      var normY =
        g.y +
        Math.sin(time * g.bobRate + g.phase) *
          g.bobAmp;
      var sx = normX * width;
      var sy = normY * height;
      var wingSpan = width * 0.035;
      var wingDip =
        height * 0.018 *
        (1 + Math.sin(time * 5 + g.phase));

      // Two thick arcs forming a bird silhouette.
      ctx.beginPath();
      ctx.moveTo(sx - wingSpan, sy + wingDip);
      ctx.quadraticCurveTo(
        sx - wingSpan * 0.3, sy - wingDip * 0.6,
        sx, sy
      );
      ctx.quadraticCurveTo(
        sx + wingSpan * 0.3, sy - wingDip * 0.6,
        sx + wingSpan, sy + wingDip
      );
      ctx.stroke();
    }
  }

  function drawCelestialBody(
    ctx, width, height, time
  ) {
    // Half the cycle is sun, half is moon.
    var halfCycle = CELESTIAL_CYCLE / 2;
    var phase = time % CELESTIAL_CYCLE;
    var isSun = phase < halfCycle;
    var localT = isSun
      ? phase / halfCycle
      : (phase - halfCycle) / halfCycle;

    // Arc path: rises from bottom-right horizon,
    // peaks well above center, exits top-left.
    var horizonY = height * 0.85;
    var peakY = height * -0.05;
    var startX = width * 1.05;
    var endX = width * -0.05;
    var peakX = width * 0.45;

    // Quadratic interpolation along the arc.
    var t = localT;
    var oneMinusT = 1 - t;
    var bx =
      oneMinusT * oneMinusT * startX +
      2 * oneMinusT * t * peakX +
      t * t * endX;
    var by =
      oneMinusT * oneMinusT * horizonY +
      2 * oneMinusT * t * peakY +
      t * t * horizonY;

    // Fade in/out near the horizon edges.
    var alpha = 1.0;
    if (t < 0.1) {
      alpha = t / 0.1;
    } else if (t > 0.9) {
      alpha = (1.0 - t) / 0.1;
    }

    var radius = height * 0.09;

    if (isSun) {
      ctx.fillStyle =
        "rgba(255,255,200," +
        (alpha * 0.9).toFixed(2) + ")";
      ctx.beginPath();
      ctx.arc(bx, by, radius, 0, 6.283);
      ctx.fill();

      // Warm glow halo.
      ctx.fillStyle =
        "rgba(255,220,150," +
        (alpha * 0.15).toFixed(2) + ")";
      ctx.beginPath();
      ctx.arc(bx, by, radius * 1.8, 0, 6.283);
      ctx.fill();
    } else {
      // Moon: bright crescent with a glow halo.
      ctx.fillStyle =
        "rgba(200,210,255," +
        (alpha * 0.12).toFixed(2) + ")";
      ctx.beginPath();
      ctx.arc(bx, by, radius * 1.8, 0, 6.283);
      ctx.fill();

      ctx.fillStyle =
        "rgba(255,255,255," +
        (alpha * 0.95).toFixed(2) + ")";
      ctx.beginPath();
      ctx.arc(bx, by, radius, 0, 6.283);
      ctx.fill();

      // Subtract an offset circle to form crescent.
      ctx.fillStyle = "#000";
      ctx.beginPath();
      ctx.arc(
        bx + radius * 0.35,
        by - radius * 0.15,
        radius * 0.8,
        0,
        6.283
      );
      ctx.fill();
    }
  }

  function drawPalmTree(
    ctx, centerX, groundY, height, time, treeIdx
  ) {
    var trunkW = height * 0.03;
    var trunkH = height * 0.25;

    // Trunk — bold rectangle.
    ctx.fillStyle = "#aaa";
    ctx.fillRect(
      centerX - trunkW / 2,
      groundY - trunkH,
      trunkW,
      trunkH
    );

    // Gentle sway — each tree gets a unique phase
    // so they don't all move in lock-step.
    var sway =
      Math.sin(time * 1.2 + treeIdx * 2.1) * 0.12;

    // Fronds — thick arcs with tall canopy.
    var topY = groundY - trunkH;
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    var frondLength = height * 0.18;
    var angles = [
      -1.4, -0.9, -0.4, 0.4, 0.9, 1.4,
    ];
    for (var i = 0; i < angles.length; i++) {
      var angle = angles[i] + sway;
      ctx.beginPath();
      ctx.moveTo(centerX, topY);
      var spread = Math.abs(angles[i]);
      var cpX =
        centerX +
        Math.sin(angle) * frondLength * 0.55;
      var cpY =
        topY -
        frondLength * (0.55 - spread * 0.1);
      var endX =
        centerX + Math.sin(angle) * frondLength;
      var endY =
        topY +
        frondLength * (0.05 + spread * 0.15);
      ctx.quadraticCurveTo(cpX, cpY, endX, endY);
      ctx.stroke();
    }
    // Central tuft — sways with the fronds.
    ctx.beginPath();
    ctx.moveTo(centerX, topY);
    ctx.lineTo(
      centerX + Math.sin(sway) * frondLength * 0.15,
      topY - frondLength * 0.45
    );
    ctx.stroke();
  }

  function drawCar(ctx, x, groundY, scale) {
    var bodyW = 90 * scale;
    var bodyH = 26 * scale;
    var roofW = 50 * scale;
    var roofH = 20 * scale;
    var wheelR = 12 * scale;

    var bodyY = groundY - bodyH - wheelR * 0.6;

    ctx.fillStyle = "#ddd";
    ctx.fillRect(x, bodyY, bodyW, bodyH);

    // Roof (trapezoid).
    var roofX = x + (bodyW - roofW) * 0.45;
    var roofY = bodyY - roofH;
    ctx.beginPath();
    ctx.moveTo(roofX, bodyY);
    ctx.lineTo(roofX + 6 * scale, roofY);
    ctx.lineTo(roofX + roofW - 6 * scale, roofY);
    ctx.lineTo(roofX + roofW, bodyY);
    ctx.closePath();
    ctx.fill();

    // Wheels.
    ctx.fillStyle = "#fff";
    var wheelY = groundY - wheelR * 0.6;
    ctx.beginPath();
    ctx.arc(
      x + 18 * scale,
      wheelY,
      wheelR,
      0,
      6.283
    );
    ctx.fill();
    ctx.beginPath();
    ctx.arc(
      x + bodyW - 18 * scale,
      wheelY,
      wheelR,
      0,
      6.283
    );
    ctx.fill();
  }

  // Draws the ASCII art title directly on the visible
  // canvas so every character is pixel-perfect. Called
  // AFTER asciiConvert() to overlay on top of the
  // luminance-converted scene.
  function overlayAsciiTitle(
    vCtx, cols, rows, cellW, cellH, time
  ) {
    var cycleTime = time % CYCLE_DURATION;
    var revealFraction;
    if (cycleTime < REVEAL_DURATION) {
      revealFraction = cycleTime / REVEAL_DURATION;
    } else if (
      cycleTime < REVEAL_DURATION + HOLD_DURATION
    ) {
      revealFraction = 1.0;
    } else {
      var fadeTime =
        cycleTime - REVEAL_DURATION - HOLD_DURATION;
      revealFraction = 1.0 - fadeTime / FADE_DURATION;
    }
    if (revealFraction < 0) {
      revealFraction = 0;
    }

    var revealedCount = Math.floor(
      revealFraction * TITLE_CHAR_COUNT
    );

    // Center the title block horizontally and place
    // it near the top of the grid.
    var startCol = Math.floor(
      (cols - TITLE_WIDTH) / 2
    );
    var startRow = 1;
    if (startCol < 0) {
      startCol = 0;
    }

    var fontSize = Math.max(8, cellH - 2);
    vCtx.font = fontSize + "px monospace";
    vCtx.textBaseline = "top";

    var charsSeen = 0;
    for (var r = 0; r < TITLE_LINES.length; r++) {
      var line = TITLE_LINES[r];
      var gridRow = startRow + r;
      if (gridRow >= rows) {
        break;
      }

      for (var c = 0; c < line.length; c++) {
        var gridCol = startCol + c;
        if (gridCol >= cols) {
          break;
        }

        var ch = line[c];
        var px = gridCol * cellW;
        var py = gridRow * cellH;

        if (ch === " ") {
          continue;
        }

        // Clear the cell so the title overwrites
        // the scene beneath it.
        vCtx.fillStyle = "#000";
        vCtx.fillRect(px, py, cellW, cellH);

        charsSeen++;
        if (charsSeen <= revealedCount) {
          vCtx.fillStyle = "rgba(0,255,65,1.0)";
          vCtx.fillText(ch, px, py);
        } else {
          var noiseIdx =
            (charsSeen * 53 +
              Math.floor(time * 14)) %
            NOISE_CHARS.length;
          vCtx.fillStyle = "rgba(0,255,65,0.2)";
          vCtx.fillText(
            NOISE_CHARS[noiseIdx], px, py
          );
        }
      }
    }
  }

  // --------------------------------------------------
  // Full scene compositor
  // --------------------------------------------------

  function drawScene(ctx, width, height, time) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    drawCelestialBody(ctx, width, height, time);
    drawStars(ctx, width, height, time);

    var groundY = height * 0.85;
    drawGround(ctx, width, height);

    for (var i = 0; i < PALM_POSITIONS.length; i++) {
      drawPalmTree(
        ctx,
        PALM_POSITIONS[i] * width,
        groundY,
        height,
        time,
        i
      );
    }

    drawSeagulls(ctx, width, height, time);

    // Car scrolling right to left, wrapping.
    var carScale = height / 500;
    var carBodyW = 90 * carScale;
    var carX =
      width -
      ((time * CAR_SPEED) % (width + carBodyW * 2)) +
      carBodyW;
    drawCar(ctx, carX, groundY, carScale);
  }

  // --------------------------------------------------
  // Luminance-to-ASCII conversion
  // --------------------------------------------------

  function asciiConvert(
    hiddenC, visibleC, cols, rows, cellW, cellH
  ) {
    var hCtx = hiddenC.getContext("2d");
    var vCtx = visibleC.getContext("2d");
    var imgData = hCtx.getImageData(
      0, 0, hiddenC.width, hiddenC.height
    );
    var pixels = imgData.data;
    var srcW = hiddenC.width;

    vCtx.fillStyle = "#000";
    vCtx.fillRect(
      0, 0, visibleC.width, visibleC.height
    );

    var fontSize = Math.max(8, cellH - 2);
    vCtx.font = fontSize + "px monospace";
    vCtx.textBaseline = "top";

    var rampLen = DENSITY_RAMP.length;

    for (var row = 0; row < rows; row++) {
      for (var col = 0; col < cols; col++) {
        var sumBrightness = 0;
        var sampleCount = 0;

        // Sample every other pixel for speed.
        var pxStartX = Math.floor(col * cellW);
        var pxStartY = Math.floor(row * cellH);
        var pxEndX = Math.min(
          Math.floor((col + 1) * cellW), srcW
        );
        var pxEndY = Math.min(
          Math.floor((row + 1) * cellH),
          hiddenC.height
        );

        for (var py = pxStartY; py < pxEndY; py += 2) {
          for (
            var px = pxStartX; px < pxEndX; px += 2
          ) {
            var idx = (py * srcW + px) * 4;
            var r = pixels[idx];
            var g = pixels[idx + 1];
            var b = pixels[idx + 2];
            sumBrightness += (r + g + b) / 3;
            sampleCount++;
          }
        }

        if (sampleCount === 0) {
          continue;
        }

        var avgBrightness = sumBrightness / sampleCount;
        var charIdx = Math.floor(
          (avgBrightness / 255) * (rampLen - 1)
        );
        if (charIdx < 0) {
          charIdx = 0;
        }
        if (charIdx >= rampLen) {
          charIdx = rampLen - 1;
        }

        var ch = DENSITY_RAMP[charIdx];
        if (ch === " ") {
          continue;
        }

        var lum = (avgBrightness / 255).toFixed(2);
        vCtx.fillStyle =
          "rgba(0,255,65," + lum + ")";
        vCtx.fillText(
          ch, col * cellW, row * cellH
        );
      }
    }
  }

  // --------------------------------------------------
  // Public API
  // --------------------------------------------------

  function sizeCanvases(container) {
    var w = container.clientWidth;
    var h = container.clientHeight;
    if (visibleCanvas) {
      visibleCanvas.width = w;
      visibleCanvas.height = h;
    }
    if (hiddenCanvas) {
      hiddenCanvas.width = w;
      hiddenCanvas.height = h;
    }
  }

  function startAsciiScene(container) {
    if (rafId !== null) {
      return;
    }

    hiddenCanvas = document.createElement("canvas");
    hiddenCanvas.style.display = "none";
    container.appendChild(hiddenCanvas);

    visibleCanvas = document.createElement("canvas");
    visibleCanvas.id = "ascii-scene-canvas";
    container.appendChild(visibleCanvas);
    container.classList.add("scene-active");

    sizeCanvases(container);

    hiddenCtx = hiddenCanvas.getContext("2d");
    visibleCtx = visibleCanvas.getContext("2d");

    startTime = performance.now() / 1000;

    var resizeHandler = function () {
      sizeCanvases(container);
    };
    window.addEventListener("resize", resizeHandler);

    function frame() {
      if (visibleCanvas === null) {
        return;
      }

      var now = performance.now() / 1000;
      var elapsed = now - startTime;

      var w = visibleCanvas.width;
      var h = visibleCanvas.height;
      if (w === 0 || h === 0) {
        rafId = requestAnimationFrame(frame);
        return;
      }

      drawScene(hiddenCtx, w, h, elapsed);

      var cols = Math.floor(w / CELL_W);
      var rows = Math.floor(h / CELL_H);
      if (cols > 0 && rows > 0) {
        asciiConvert(
          hiddenCanvas,
          visibleCanvas,
          cols,
          rows,
          CELL_W,
          CELL_H
        );
        overlayAsciiTitle(
          visibleCanvas.getContext("2d"),
          cols,
          rows,
          CELL_W,
          CELL_H,
          elapsed
        );
      }

      rafId = requestAnimationFrame(frame);
    }

    rafId = requestAnimationFrame(frame);

    // Store the resize handler so we can remove it
    // on stop.
    visibleCanvas._resizeHandler = resizeHandler;
  }

  function stopAsciiScene() {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }

    if (visibleCanvas && visibleCanvas._resizeHandler) {
      window.removeEventListener(
        "resize", visibleCanvas._resizeHandler
      );
    }

    if (hiddenCanvas && hiddenCanvas.parentNode) {
      hiddenCanvas.parentNode.removeChild(hiddenCanvas);
    }
    if (visibleCanvas && visibleCanvas.parentNode) {
      visibleCanvas.parentNode.classList.remove(
        "scene-active"
      );
      visibleCanvas.parentNode.removeChild(
        visibleCanvas
      );
    }

    hiddenCanvas = null;
    hiddenCtx = null;
    visibleCanvas = null;
    visibleCtx = null;
  }

  window.startAsciiScene = startAsciiScene;
  window.stopAsciiScene = stopAsciiScene;
})();
