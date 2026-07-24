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
    var gpu = info.gpu_name || "GPU not detected";
    systemText.textContent =
      gpu + "  \u00B7  " + formatFreeVram(info.free_vram_gib);
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

  function statusLabel(model, fits) {
    if (model.status === "active") {
      return "Resident";
    }
    if (!fits) {
      return "Insufficient VRAM";
    }
    return "Ready to load";
  }

  function buildRow(model) {
    var fits = model.fits !== false;
    var li = document.createElement("li");
    li.className = "menu-model-row";
    li.setAttribute("role", "option");
    li.setAttribute("data-id", model.id);

    var main = document.createElement("div");
    main.className = "menu-model-main";
    var name = document.createElement("span");
    name.className = "menu-model-name";
    name.textContent = model.display_name || model.id;
    var desc = document.createElement("span");
    desc.className = "menu-model-desc";
    desc.textContent = model.description || "";
    main.appendChild(name);
    main.appendChild(desc);

    var meta = document.createElement("div");
    meta.className = "menu-model-meta";
    var vram = document.createElement("span");
    vram.className = "menu-model-vram";
    vram.textContent = vramLabel(model);
    var status = document.createElement("span");
    status.className = "menu-model-status";
    status.textContent = statusLabel(model, fits);
    meta.appendChild(vram);
    meta.appendChild(status);

    li.appendChild(main);
    li.appendChild(meta);

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

  // ---- Selection -> activation -> generator ----

  function setRowLoading(li, model) {
    document.body.classList.add("menu-busy");
    li.classList.add("is-loading");
    var status = li.querySelector(".menu-model-status");
    if (status) {
      status.textContent =
        "Loading " + (model.display_name || model.id) + "\u2026";
    }
  }

  function clearRowLoading(li) {
    document.body.classList.remove("menu-busy");
    li.classList.remove("is-loading");
  }

  function selectModel(model, li) {
    if (selecting) {
      return;
    }
    selecting = true;
    clearError();
    setRowLoading(li, model);
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
        clearRowLoading(li);
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
