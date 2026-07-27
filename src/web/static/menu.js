// Main Menu: model-selection landing page. Fetches the registry plus
// live GPU/VRAM info, renders one selectable row per model (greying
// out any that will not fit), and on selection activates the chosen
// worker before handing off to the generator page at /generate.
//
// Classic script (matches app.js / overlays.js): plain ES5, no module
// system, so it runs the moment the tag is parsed.

"use strict";

(function () {
  var GENERATE_URL = "/generate";
  var FLOATER_COUNT = 30;

  var video = document.getElementById("menu-video");
  var systemBox = document.getElementById("menu-system");
  var systemText = document.getElementById("menu-system-text");
  var modelList = document.getElementById("menu-model-list");
  var errorBox = document.getElementById("menu-error");

  console.assert(!!modelList, "menu: model list mount missing");
  console.assert(!!systemText, "menu: system text mount missing");

  // Guards a second activation while one is already in flight.
  var selecting = false;

  // ---- Background fallback (grid is CSS-only; add floaters too) ----

  function spawnFloaters() {
    var container = document.getElementById("bg-floaters");
    if (!container) {
      return;
    }
    var chars =
      "01\u2591\u2592\u2593\u2588\u2584\u2580"
      + "\u28FF\u2847\u283F\u28C0\u28E4\u28FF"
      + "\u03A3\u0394\u03A9\u03BB\u2202\u2207";
    for (var i = 0; i < FLOATER_COUNT; i++) {
      var el = document.createElement("span");
      el.className = "floater";
      el.textContent = chars[
        Math.floor(Math.random() * chars.length)
      ];
      el.style.left = Math.random() * 100 + "%";
      el.style.animationDuration =
        30 + Math.random() * 50 + "s";
      el.style.animationDelay =
        -(Math.random() * 60) + "s";
      el.style.fontSize = 10 + Math.random() * 8 + "px";
      container.appendChild(el);
    }
  }

  // ---- Title video ----

  function hideVideo() {
    document.body.classList.add("menu-video-off");
  }

  function setupVideo() {
    if (!video) {
      hideVideo();
      return;
    }
    video.addEventListener("error", hideVideo);
    // Some engines block autoplay even when muted; fall back to the
    // animated grid backdrop rather than leaving a black frame.
    var attempt = video.play();
    if (attempt && typeof attempt.catch === "function") {
      attempt.catch(hideVideo);
    }
  }

  // ---- System (GPU / VRAM) readout ----

  function formatFreeVram(gib) {
    if (typeof gib !== "number") {
      return "free VRAM unknown";
    }
    return gib.toFixed(1) + " GiB free";
  }

  function renderSystem(info) {
    systemBox.classList.remove("menu-system-checking");
    if (info.gpu_name) {
      systemText.textContent =
        info.gpu_name + "  \u00B7  "
        + formatFreeVram(info.free_vram_gib);
      return;
    }
    // No readable GPU name: explain why when we can (a driver/library
    // mismatch usually just needs a reboot).
    if (info.gpu_status === "mismatch") {
      systemText.textContent =
        "GPU present, driver/library mismatch"
        + " (a reboot may be needed)";
    } else {
      systemText.textContent =
        "GPU not detected  \u00B7  free VRAM unknown";
    }
  }

  function showSystemError(message) {
    systemBox.classList.remove("menu-system-checking");
    systemText.textContent = message;
  }

  // ---- Error line ----

  function showError(message) {
    if (!errorBox) {
      return;
    }
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    if (!errorBox) {
      return;
    }
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  // ---- Model rows ----

  function vramLabel(model) {
    var gib = model.min_vram_gib;
    if (typeof gib !== "number" || gib <= 0) {
      return "";
    }
    return "~" + Math.round(gib) + " GiB";
  }

  // Render the status indicator: Available + green check (fits),
  // Insufficient VRAM + red cross (does not fit), or Resident (active).
  function applyStatus(statusEl, model, fits) {
    statusEl.className = "menu-model-status";
    statusEl.textContent = "";
    if (model.status === "active") {
      statusEl.appendChild(
        document.createTextNode("Resident")
      );
      return;
    }
    var icon = document.createElement("span");
    icon.className = "menu-status-icon";
    icon.setAttribute("aria-hidden", "true");
    if (fits) {
      statusEl.appendChild(
        document.createTextNode("Available ")
      );
      icon.classList.add("menu-status-ok");
      icon.textContent = "\u2713";
    } else {
      statusEl.appendChild(
        document.createTextNode("Insufficient VRAM ")
      );
      icon.classList.add("menu-status-bad");
      icon.textContent = "\u2717";
    }
    statusEl.appendChild(icon);
  }

  function buildRow(model) {
    var fits = model.fits !== false;
    var li = document.createElement("li");
    li.className = "menu-model-row";
    li.setAttribute("role", "option");
    li.setAttribute("data-id", model.id);

    var name = document.createElement("span");
    name.className = "menu-model-name";
    name.textContent = model.display_name || model.id;
    var desc = document.createElement("span");
    desc.className = "menu-model-desc";
    desc.textContent = model.description || "";

    // VRAM + status stacked under the name/description (left-aligned),
    // so the status text changing length never reflows the row.
    var meta = document.createElement("div");
    meta.className = "menu-model-meta";
    var vram = document.createElement("span");
    vram.className = "menu-model-vram";
    vram.textContent = vramLabel(model);
    var status = document.createElement("span");
    status.className = "menu-model-status";
    applyStatus(status, model, fits);
    meta.appendChild(vram);
    meta.appendChild(status);

    // Meta (~X GiB + status) sits directly under the model name,
    // above the description.
    li.appendChild(name);
    li.appendChild(meta);
    li.appendChild(desc);

    if (fits) {
      wireRow(li, model);
    } else {
      li.classList.add("is-disabled");
      li.setAttribute("aria-disabled", "true");
      li.title =
        "Needs about " + Math.round(model.min_vram_gib)
        + " GiB of VRAM; not enough is free.";
    }
    return li;
  }

  function wireRow(li, model) {
    li.tabIndex = 0;
    li.addEventListener("click", function () {
      selectModel(model, li);
    });
    li.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectModel(model, li);
      }
    });
  }

  function renderModels(list) {
    modelList.innerHTML = "";
    for (var i = 0; i < list.length; i++) {
      modelList.appendChild(buildRow(list[i]));
    }
  }

  // ---- Loading diffusion cycle ("Loading...") ----

  // Compact, self-contained version of the generator's diffusion
  // reveal: loops noise -> "Loading..." at the status-bar pace while a
  // model activates, regardless of the text-effect setting (this is the
  // menu's own signal). Honors reduced motion.
  var DENOISE_GLYPHS = "\u2591\u2592\u2593";
  var LOADING_HOLD_MS = 700;
  var loadingCycleTimer = null;

  function prefersReducedMotion() {
    try {
      return window.matchMedia(
        "(prefers-reduced-motion: reduce)"
      ).matches;
    } catch (_e) {
      return false;
    }
  }

  function revealOnce(el, text, onDone) {
    if (el._denoiseTimer) {
      clearInterval(el._denoiseTimer);
      el._denoiseTimer = null;
    }
    var steps_total = 8;
    var step = 0;
    var render = function () {
      var revealed = Math.floor(
        (step / steps_total) * text.length
      );
      var out = "";
      for (var i = 0; i < text.length; i++) {
        if (i < revealed || text[i] === " ") {
          out += text[i];
        } else {
          out += DENOISE_GLYPHS[
            Math.floor(Math.random() * DENOISE_GLYPHS.length)
          ];
        }
      }
      el.textContent = out;
      step += 1;
      if (step > steps_total) {
        clearInterval(el._denoiseTimer);
        el._denoiseTimer = null;
        el.textContent = text;
        if (onDone) {
          onDone();
        }
      }
    };
    render();
    el._denoiseTimer = setInterval(render, 45);
  }

  function startLoadingCycle(el) {
    stopLoadingCycle(el);
    if (prefersReducedMotion()) {
      el.textContent = "Loading\u2026";
      return;
    }
    var runOnce = function () {
      revealOnce(el, "Loading\u2026", function () {
        loadingCycleTimer = setTimeout(
          runOnce, LOADING_HOLD_MS
        );
      });
    };
    runOnce();
  }

  function stopLoadingCycle(el) {
    if (loadingCycleTimer !== null) {
      clearTimeout(loadingCycleTimer);
      loadingCycleTimer = null;
    }
    if (el && el._denoiseTimer) {
      clearInterval(el._denoiseTimer);
      el._denoiseTimer = null;
    }
  }

  // ---- Selection -> activation -> generator ----

  function setRowLoading(li) {
    document.body.classList.add("menu-busy");
    li.classList.add("is-loading");
    var status = li.querySelector(".menu-model-status");
    if (status) {
      status.textContent = "";
      status.classList.add("is-loading-status");
      startLoadingCycle(status);
    }
  }

  function clearRowLoading(li, model) {
    document.body.classList.remove("menu-busy");
    li.classList.remove("is-loading");
    var status = li.querySelector(".menu-model-status");
    if (status) {
      stopLoadingCycle(status);
      status.classList.remove("is-loading-status");
      applyStatus(status, model, model.fits !== false);
    }
  }

  function selectModel(model, li) {
    if (selecting) {
      return;
    }
    selecting = true;
    clearError();
    setRowLoading(li);
    fetch(
      "/api/models/" + encodeURIComponent(model.id) + "/activate",
      { method: "POST" }
    )
      .then(function (response) {
        return response.json();
      })
      .then(function (result) {
        if (result && result.ok) {
          window.location.assign(GENERATE_URL);
          return;
        }
        throw new Error(
          (result && result.message) || "activation failed"
        );
      })
      .catch(function (err) {
        selecting = false;
        clearRowLoading(li, model);
        showError(
          "Could not load "
          + (model.display_name || model.id) + ": "
          + (err && err.message ? err.message : String(err))
        );
      });
  }

  // ---- Boot ----

  function loadModels() {
    fetch("/api/models")
      .then(function (response) {
        return response.json();
      })
      .then(function (info) {
        renderSystem(info);
        renderModels(info.models || []);
      })
      .catch(function (err) {
        showSystemError("Could not reach the server.");
        showError(
          err && err.message ? err.message : String(err)
        );
      });
  }

  spawnFloaters();
  setupVideo();
  loadModels();
})();
