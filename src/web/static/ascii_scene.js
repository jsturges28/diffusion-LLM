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
  var TITLE_TEXT = "Diffusion LLM";

  // Diffusion reveal timing (seconds).
  var REVEAL_DURATION = 3.0;
  var HOLD_DURATION = 4.0;
  var FADE_DURATION = 2.0;
  var CYCLE_DURATION =
    REVEAL_DURATION + HOLD_DURATION + FADE_DURATION;

  var NOISE_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";

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

  function drawPalmTree(ctx, centerX, groundY, height) {
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
      var angle = angles[i];
      ctx.beginPath();
      ctx.moveTo(centerX, topY);
      var spread = Math.abs(angle);
      // Outer fronds droop more; inner fronds
      // reach higher.
      var cpX =
        centerX +
        Math.sin(angle) * frondLength * 0.55;
      var cpY =
        topY - frondLength * (0.55 - spread * 0.1);
      var endX =
        centerX + Math.sin(angle) * frondLength;
      var endY =
        topY + frondLength * (0.05 + spread * 0.15);
      ctx.quadraticCurveTo(cpX, cpY, endX, endY);
      ctx.stroke();
    }
    // Central tuft — a short upward spike.
    ctx.beginPath();
    ctx.moveTo(centerX, topY);
    ctx.lineTo(centerX, topY - frondLength * 0.45);
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

  function drawDiffusionTitle(
    ctx, width, height, time
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

    // Large, heavy font so each letter's strokes span
    // 2-3 ASCII columns after downsampling. Weight 900
    // plus a strokeText outline makes strokes ~20-25px
    // thick, which the 10px ASCII grid resolves cleanly.
    var fontSize = Math.max(
      48,
      Math.min(width * 0.12, 130)
    );
    ctx.font =
      "900 " + Math.round(fontSize) + "px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    var titleY = height * 0.22;
    var totalLen = TITLE_TEXT.length;
    var revealedCount = Math.floor(
      revealFraction * totalLen
    );

    var metrics = ctx.measureText(TITLE_TEXT);
    var totalWidth = metrics.width;
    var charWidth = totalWidth / totalLen;
    var startX =
      (width - totalWidth) / 2 + charWidth / 2;

    // Stroke width fattens each letter's edges so
    // they survive the luminance downsampling.
    var strokeW = Math.max(3, fontSize * 0.07);
    ctx.lineWidth = strokeW;

    for (var i = 0; i < totalLen; i++) {
      var cx = startX + i * charWidth;

      if (i < revealedCount) {
        ctx.strokeStyle = "#fff";
        ctx.strokeText(TITLE_TEXT[i], cx, titleY);
        ctx.fillStyle = "#fff";
        ctx.fillText(TITLE_TEXT[i], cx, titleY);
      } else {
        var noiseIdx =
          (i * 97 + Math.floor(time * 12)) %
          NOISE_CHARS.length;
        ctx.fillStyle = "rgba(255,255,255,0.2)";
        ctx.fillText(
          NOISE_CHARS[noiseIdx], cx, titleY
        );
      }
    }
  }

  // --------------------------------------------------
  // Full scene compositor
  // --------------------------------------------------

  function drawScene(ctx, width, height, time) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    drawStars(ctx, width, height, time);

    var groundY = height * 0.85;
    drawGround(ctx, width, height);

    for (var i = 0; i < PALM_POSITIONS.length; i++) {
      drawPalmTree(
        ctx,
        PALM_POSITIONS[i] * width,
        groundY,
        height
      );
    }

    // Car scrolling right to left, wrapping.
    var carScale = height / 500;
    var carBodyW = 90 * carScale;
    var carX =
      width -
      ((time * CAR_SPEED) % (width + carBodyW * 2)) +
      carBodyW;
    drawCar(ctx, carX, groundY, carScale);

    drawDiffusionTitle(ctx, width, height, time);
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
