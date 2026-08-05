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
  // The supervisor samples the worker every 250ms, so reading at the
  // same rate is the fastest the bar can actually be told anything.
  // Anything slower stacks staleness on staleness, and a load that is
  // over in a couple of seconds gets only one or two readings before
  // the page moves on.
  var ACTIVATION_POLL_MS = 250;

  var video = document.getElementById("menu-video");
  var systemBox = document.getElementById("menu-system");
  var systemText = document.getElementById("menu-system-text");
  var modelList = document.getElementById("menu-model-list");
  var errorBox = document.getElementById("menu-error");
  var activationBox =
    document.getElementById("menu-activation");
  var activationProgress =
    document.getElementById("menu-activation-progress");
  var activationFill =
    document.getElementById("menu-activation-fill");
  var activationPct =
    document.getElementById("menu-activation-pct");
  var activationCancel =
    document.getElementById("menu-activation-cancel");
  var modelPager = document.getElementById("menu-pager");
  var modelPagePrev = document.getElementById("menu-page-prev");
  var modelPageNext = document.getElementById("menu-page-next");
  var modelPageCounter =
    document.getElementById("menu-page-counter");

  console.assert(!!modelList, "menu: model list mount missing");
  console.assert(!!systemText, "menu: system text mount missing");

  // Guards a second activation while one is already in flight.
  var selecting = false;
  // True while the select -> confirm prompt is showing (pre-activation).
  var confirming = false;
  // The in-flight selection ({ model, li }) and its status poll timer.
  var activeSelection = null;
  // Model pagination: show up to MODELS_PER_PAGE rows per page, with a
  // prev/counter/next pager in the panel's bottom-left corner.
  var MODELS_PER_PAGE = 3;
  var pagedModels = [];
  var pagedGpuPresent = false;
  var currentPage = 0;
  // Device the resident worker is running on, from /api/models. Only
  // an AR row can disagree with it, since every other row offers one
  // placement, so this is read solely to tell "the model you are
  // already running" apart from "that model, on the other device".
  var activeDevice = null;

  // Cross-page download navigation: the download is a global server task
  // (see download_toast.js), so the user can page/navigate away while it
  // runs. reattachSettled gates the toast until the initial re-attach has
  // paged to and bound the target row; lastDownloadStatus / prevDownload-
  // State track the toast module's poll for binding + the row fence.
  var reattachSettled = false;
  var lastDownloadStatus = null;
  var prevDownloadState = null;
  var pollTimer = null;
  // The row element whose weights are currently pre-downloading.
  var downloadRow = null;

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

  function formatFree(gib, unknownLabel) {
    if (typeof gib !== "number") {
      return unknownLabel;
    }
    return gib.toFixed(1) + " GiB free";
  }

  // One labeled readout line: an accent-green tag (GPU / CPU) styled
  // like the analytics chart headers, then the device detail.
  function makeSystemLine(tag, detail) {
    var line = document.createElement("span");
    line.className = "menu-system-line";
    var tagEl = document.createElement("span");
    tagEl.className = "menu-system-tag";
    tagEl.textContent = tag;
    var detailEl = document.createElement("span");
    detailEl.className = "menu-system-detail";
    detailEl.textContent = detail;
    line.appendChild(tagEl);
    line.appendChild(detailEl);
    return line;
  }

  function gpuDetail(info) {
    if (info.gpu_name) {
      return info.gpu_name + "  \u00B7  "
        + formatFree(info.free_vram_gib, "free VRAM unknown");
    }
    // No readable GPU name: explain why when we can (a driver/library
    // mismatch usually just needs a reboot).
    if (info.gpu_status === "mismatch") {
      return "driver/library mismatch (a reboot may be needed)";
    }
    return "not detected";
  }

  function renderSystem(info) {
    systemBox.classList.remove("menu-system-checking");
    systemText.textContent = "";
    systemText.appendChild(
      makeSystemLine("GPU:", gpuDetail(info))
    );
    var cpu = (info.cpu_name || "unknown") + "  \u00B7  "
      + formatFree(info.free_ram_gib, "RAM free unknown");
    systemText.appendChild(makeSystemLine("CPU:", cpu));
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

  // Autoregressive models run on CPU too, so they are never gated by
  // VRAM: a GPU-less or low-VRAM host just falls back to CPU.
  function isAutoregressive(model) {
    return !!(
      model.capabilities
      && model.capabilities.model_type === "autoregressive"
    );
  }

  // Model-family glyph pinned right of the name. The wrapper's title
  // gives a hover tooltip. (Both are first-pass shapes to iterate on
  // once rendered; the SVG path coordinates are cheap to nudge.)
  //
  // Autoregressive: an "@" that resolves into an "R". An inner "a"
  // counter sits under a head arch whose stroke loops over the top back
  // to the filled start node (the autoregressive feedback: build on
  // what was just emitted), dropping two matched legs (the "R" base).
  var _AR_ICON =
    '<svg viewBox="0 0 24 24" width="13" height="13" fill="none"'
    + ' stroke="currentColor" stroke-width="1.5" stroke-linecap="round"'
    + ' stroke-linejoin="round" aria-hidden="true">'
    + '<path d="M7 14 A5 5 0 0 0 17 14" />'
    + '<path d="M7 14 V19 M17 14 V19" />'
    + '<path d="M17 14 C 21 8 13 4 10 9" />'
    + '<path d="M10 10 H14 V13 H10 Z" />'
    + '<circle cx="10" cy="9" r="1.15" fill="currentColor"'
    + ' stroke="none" /></svg>';
  // Diffusion: a "D" and an "F" in superposition. The overlap (the D's
  // bowl plus the F's mid bar) reads as a backwards epsilon and is drawn
  // crisp at full opacity; the non-overlapping strokes (stems, the F top
  // bar, the D top/bottom) stay faint, so both letters still register.
  var _DIFFUSION_ICON =
    '<svg viewBox="0 0 24 24" width="13" height="13" fill="none"'
    + ' stroke="currentColor" stroke-linecap="round"'
    + ' stroke-linejoin="round" aria-hidden="true">'
    + '<path d="M7 5 v14 h4 a7 7 0 0 0 0 -14 z" opacity="0.3"'
    + ' stroke-width="1.3" />'
    + '<path d="M7 5 v14 M7 5 h8 M7 12 h6" opacity="0.3"'
    + ' stroke-width="1.3" />'
    + '<path d="M11 5 a7 7 0 0 1 0 14 M7 12 h6"'
    + ' stroke-width="1.9" /></svg>';

  function buildFamilyIcon(model) {
    var ar = isAutoregressive(model);
    var span = document.createElement("span");
    span.className = "model-family-icon";
    span.title = "Model Family: "
      + (ar ? "Autoregressive" : "Diffusion");
    span.innerHTML = ar ? _AR_ICON : _DIFFUSION_ICON;
    return span;
  }

  // Signed VRAM-headroom pill that extends left of the device tag:
  // green +X.X GiB when the model fits, red -X.X GiB when it is short.
  // Available = free + reclaimable (what you get after the current
  // model unloads); required = the model's min VRAM.
  function buildHeadroomOblong(model) {
    var headroom = model.vram_headroom_gib;
    if (typeof headroom !== "number") {
      return null;
    }
    var el = document.createElement("span");
    el.className = "device-headroom "
      + (headroom >= 0 ? "is-positive" : "is-negative");
    el.textContent = (headroom >= 0 ? "+" : "\u2212")
      + Math.abs(headroom).toFixed(1) + " GiB";
    var required = Math.round(model.min_vram_gib || 0);
    var available = (
      (model.min_vram_gib || 0) + headroom
    ).toFixed(1);
    el.title = (headroom >= 0
      ? "Fits. "
      : "Insufficient VRAM. ")
      + "Required: " + required + " GiB, Available: "
      + available + " GiB";
    return el;
  }

  // Status indicator: only "Resident" for the active model. Fit is
  // now shown by the headroom pill on the device tag.
  function applyStatus(statusEl, model) {
    statusEl.className = "menu-model-status";
    statusEl.textContent =
      model.status === "active" ? "Resident" : "";
  }

  // CPU/GPU segmented toggle for an autoregressive row. GPU is the
  // default when a GPU is present and the model fits; otherwise CPU
  // is forced and the GPU option is disabled with an explanatory
  // tooltip. Exposes getDevice() for the activation POST.
  function buildDeviceToggle(model, gpuPresent, fits) {
    var gpuOk = gpuPresent && fits;
    var wrap = document.createElement("div");
    wrap.className = "menu-model-device";
    var oblong = buildHeadroomOblong(model);
    if (oblong) {
      wrap.appendChild(oblong);
    }
    var current = gpuOk ? "cuda" : "cpu";
    var btns = {};

    function makeButton(value, label) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "menu-device-btn"
        + (value === current ? " is-active" : "");
      btn.textContent = label;
      if (value === "cuda" && !gpuOk) {
        btn.disabled = true;
        btn.classList.add("is-unavailable");
        btn.title = gpuPresent
          ? "Not enough free VRAM for GPU"
          : "No GPU detected";
      }
      btn.addEventListener("click", function (event) {
        // Do not let a toggle click activate the row.
        event.stopPropagation();
        if (btn.disabled) {
          return;
        }
        current = value;
        btns.cuda.classList.toggle(
          "is-active", current === "cuda"
        );
        btns.cpu.classList.toggle(
          "is-active", current === "cpu"
        );
      });
      return btn;
    }

    btns.cuda = makeButton("cuda", "GPU");
    btns.cpu = makeButton("cpu", "CPU");
    wrap.appendChild(btns.cuda);
    wrap.appendChild(btns.cpu);
    wrap.getDevice = function () {
      return current;
    };
    return wrap;
  }

  // Static, non-interactive device tag for GPU-only (diffusion) rows,
  // matching the AR toggle's active pill so all rows read consistently.
  function buildStaticDeviceTag(model, label) {
    var wrap = document.createElement("div");
    wrap.className = "menu-model-device menu-model-device-static";
    var oblong = buildHeadroomOblong(model);
    if (oblong) {
      wrap.appendChild(oblong);
    }
    var tag = document.createElement("span");
    tag.className = "menu-device-btn is-active";
    tag.textContent = label;
    wrap.appendChild(tag);
    return wrap;
  }

  function buildRow(model, gpuPresent) {
    var fits = model.fits !== false;
    var ar = isAutoregressive(model);
    var li = document.createElement("li");
    li.className = "menu-model-row";
    li.setAttribute("role", "option");
    li.setAttribute("data-id", model.id);

    var nameRow = document.createElement("div");
    nameRow.className = "menu-model-name-row";
    var name = document.createElement("span");
    name.className = "menu-model-name";
    name.textContent = model.display_name || model.id;
    nameRow.appendChild(name);
    nameRow.appendChild(buildFamilyIcon(model));
    var desc = document.createElement("span");
    desc.className = "menu-model-desc";
    desc.textContent = model.description || "";

    // Status ("Resident" or blank) under the name; fit/VRAM detail is
    // now carried by the device tag's headroom pill.
    var meta = document.createElement("div");
    meta.className = "menu-model-meta";
    var status = document.createElement("span");
    status.className = "menu-model-status";
    applyStatus(status, model);
    meta.appendChild(status);

    li.appendChild(nameRow);
    li.appendChild(meta);
    li.appendChild(desc);

    // A model whose weights are not cached yet gets a "Click to
    // Download" veneer; clicking it pre-fetches, then the row becomes
    // selectable. Tracked on the row so one click handler can dispatch.
    var needsDownload = !!(
      model.downloadable && !model.downloaded
    );
    li._needsDownload = needsDownload;

    // An AR row carries a CPU/GPU toggle and stays selectable even
    // when it will not fit on the GPU, since CPU is always a fallback.
    // Diffusion rows carry a static GPU-only tag.
    if (ar) {
      var toggle = buildDeviceToggle(model, gpuPresent, fits);
      li._getDevice = toggle.getDevice;
      li.appendChild(toggle);
    } else {
      li.appendChild(buildStaticDeviceTag(model, "GPU"));
    }

    if (needsDownload) {
      li.classList.add("needs-download");
      li.appendChild(buildDownloadVeneer());
      wireRow(li, model);
    } else if (ar || fits) {
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

  // Translucent overlay for uncached models. Three states: an idle
  // "Click to Download" label, a progress area (bar + percent), and a
  // message area (success/error + Ok) shown on finish.
  function buildDownloadVeneer() {
    var veneer = document.createElement("div");
    veneer.className = "menu-model-veneer";
    var label = document.createElement("span");
    label.className = "menu-model-veneer-label";
    label.textContent = "Click to Download";
    veneer.appendChild(label);
    var prog = document.createElement("div");
    prog.className = "menu-model-veneer-progress";
    prog.hidden = true;
    var pct = document.createElement("span");
    pct.className = "menu-model-veneer-pct";
    pct.textContent = "Downloading 0%";
    var bar = document.createElement("span");
    bar.className = "menu-model-veneer-bar";
    var fill = document.createElement("span");
    fill.className = "menu-model-veneer-fill";
    bar.appendChild(fill);
    prog.appendChild(pct);
    prog.appendChild(bar);
    veneer.appendChild(prog);
    var message = document.createElement("div");
    message.className = "menu-model-veneer-message";
    message.hidden = true;
    veneer.appendChild(message);
    return veneer;
  }

  // Fill the veneer's message area with a success/error line and an
  // Ok button; hides the label + progress. onOk fires on click.
  function showVeneerMessage(li, text, isError, onOk) {
    var veneer = li.querySelector(".menu-model-veneer");
    if (!veneer) {
      return;
    }
    var parts = veneerParts(li);
    if (parts.label) {
      parts.label.hidden = true;
    }
    if (parts.prog) {
      parts.prog.hidden = true;
    }
    var message = veneer.querySelector(".menu-model-veneer-message");
    if (!message) {
      return;
    }
    message.innerHTML = "";
    message.hidden = false;
    message.classList.toggle("is-error", !!isError);
    var msgText = document.createElement("span");
    msgText.className = "menu-model-veneer-message-text";
    msgText.textContent = text;
    var ok = document.createElement("button");
    ok.type = "button";
    ok.className = "menu-model-veneer-ok";
    ok.textContent = "Ok";
    ok.addEventListener("click", function (event) {
      event.stopPropagation();
      onOk();
    });
    message.appendChild(msgText);
    message.appendChild(ok);
  }

  function wireRow(li, model) {
    li.tabIndex = 0;
    var handler = function () {
      if (li._needsDownload) {
        beginDownload(model, li);
      } else {
        beginConfirm(model, li);
      }
    };
    li.addEventListener("click", handler);
    li.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        handler();
      }
    });
  }

  function renderModels(list, gpuPresent) {
    pagedModels = list || [];
    pagedGpuPresent = gpuPresent;
    currentPage = 0;
    renderCurrentPage();
  }

  function pageCount() {
    return Math.max(
      1, Math.ceil(pagedModels.length / MODELS_PER_PAGE)
    );
  }

  // Render just the current page of models and refresh the pager.
  function renderCurrentPage() {
    var pages = pageCount();
    currentPage = Math.min(Math.max(currentPage, 0), pages - 1);
    var start = currentPage * MODELS_PER_PAGE;
    var slice = pagedModels.slice(
      start, start + MODELS_PER_PAGE
    );
    modelList.innerHTML = "";
    for (var i = 0; i < slice.length; i++) {
      modelList.appendChild(
        buildRow(slice[i], pagedGpuPresent)
      );
    }
    if (modelPageCounter) {
      modelPageCounter.textContent =
        (currentPage + 1) + " / " + pages;
    }
    if (modelPagePrev) {
      modelPagePrev.disabled = currentPage <= 0;
    }
    if (modelPageNext) {
      modelPageNext.disabled = currentPage >= pages - 1;
    }
    updatePagerVisibility();
  }

  // The pager shows only when models exist and no confirm/selection is
  // in progress (paging mid-confirm would rebuild the contracted row).
  function updatePagerVisibility() {
    if (!modelPager) {
      return;
    }
    modelPager.hidden = !(
      pagedModels.length > 0 && !confirming && !selecting
    );
  }

  // Step one page within bounds and re-render.
  function goToPage(delta) {
    var next = currentPage + delta;
    if (next < 0 || next > pageCount() - 1) {
      return;
    }
    currentPage = next;
    renderCurrentPage();
    // Re-bind (or release) the download veneer for the new page and let
    // the toast reflect whether the download row is now visible.
    if (reattachSettled) {
      syncDownloadBinding();
      if (typeof downloadToastRefresh === "function") {
        downloadToastRefresh();
      }
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

  // Whether picking this row would land on the worker already
  // serving. The server treats that activation as a no-op, so the
  // only honest thing for the menu to do is navigate.
  //
  // Only an AR row carries a device toggle, so it is the only one
  // that can name a placement the resident worker is not using.
  // Everything else offers a single placement, which makes "this
  // model is active" the whole answer, and avoids having to mirror
  // the server's resolution of a device-less request.
  function isResidentSelection(model, li) {
    if (model.status !== "active") {
      return false;
    }
    if (li._getDevice) {
      return li._getDevice() === activeDevice;
    }
    return true;
  }

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
      applyStatus(status, model);
    }
  }

  function stopPolling() {
    if (pollTimer !== null) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  // Remove the select -> confirm animation state (restore the full
  // menu): un-contract the list and drop any confirm prompt.
  function resetMenuConfirm() {
    confirming = false;
    updatePagerVisibility();
    if (typeof downloadToastRefresh === "function") {
      downloadToastRefresh();
    }
    if (!modelList) {
      return;
    }
    modelList.classList.remove("is-confirming");
    var rows = modelList.querySelectorAll(
      ".menu-model-row.is-confirmed"
    );
    for (var i = 0; i < rows.length; i++) {
      rows[i].classList.remove("is-confirmed");
    }
    var prompts = modelList.querySelectorAll(
      ".menu-model-confirm"
    );
    for (var j = 0; j < prompts.length; j++) {
      prompts[j].parentNode.removeChild(prompts[j]);
    }
  }

  // Reset all in-flight selection UI (row highlight, activation panel,
  // and the confirm contraction).
  function finishSelecting() {
    selecting = false;
    stopPolling();
    if (activationBox) {
      activationBox.hidden = true;
    }
    if (activationProgress) {
      activationProgress.hidden = true;
    }
    if (activeSelection) {
      clearRowLoading(
        activeSelection.li, activeSelection.model
      );
      activeSelection = null;
    }
    resetMenuConfirm();
  }

  function selectionLabel() {
    var m = activeSelection ? activeSelection.model : null;
    return m ? m.display_name || m.id : "model";
  }

  // Selecting a model first contracts the menu to that row and shows a
  // confirm prompt (green check / red X). Check proceeds to load; X
  // reverses back to the full menu.
  function beginConfirm(model, li) {
    if (selecting || confirming) {
      return;
    }
    confirming = true;
    updatePagerVisibility();
    if (typeof downloadToastRefresh === "function") {
      downloadToastRefresh();
    }
    activeSelection = { model: model, li: li };
    clearError();
    modelList.classList.add("is-confirming");
    li.classList.add("is-confirmed");
    var device = li._getDevice ? li._getDevice() : "cuda";
    var resident = isResidentSelection(model, li);
    var box = document.createElement("div");
    box.className = "menu-model-confirm";
    // Clicks in the confirm box must not bubble to the row handler.
    box.addEventListener("click", function (event) {
      event.stopPropagation();
    });
    var msg = document.createElement("span");
    msg.className = "menu-model-confirm-msg";
    // Nothing is loaded when the worker is already serving this exact
    // model, so promising a load would be describing work that is not
    // going to happen.
    msg.textContent = resident
      ? "Go back to the Generation page?"
      : "Load " + (model.display_name || model.id)
        + " on " + (device === "cpu" ? "CPU" : "GPU") + "?";
    var actions = document.createElement("span");
    actions.className = "menu-model-confirm-actions";
    var yes = document.createElement("button");
    yes.type = "button";
    yes.className = "menu-confirm-yes";
    yes.title = "Confirm";
    yes.setAttribute(
      "aria-label",
      resident ? "Confirm and continue" : "Confirm and load"
    );
    yes.textContent = "\u2713";
    yes.addEventListener("click", function (event) {
      event.stopPropagation();
      confirmSelection();
    });
    var no = document.createElement("button");
    no.type = "button";
    no.className = "menu-confirm-no";
    no.title = "Cancel";
    no.setAttribute("aria-label", "Cancel");
    no.textContent = "\u2717";
    no.addEventListener("click", function (event) {
      event.stopPropagation();
      resetMenuConfirm();
      activeSelection = null;
    });
    actions.appendChild(yes);
    actions.appendChild(no);
    box.appendChild(msg);
    box.appendChild(actions);
    li.appendChild(box);
  }

  // Confirm accepted: drop the prompt (keep the contracted layout) and
  // start the real activation/loading.
  function confirmSelection() {
    if (!confirming || !activeSelection) {
      return;
    }
    confirming = false;
    var prompts = modelList.querySelectorAll(
      ".menu-model-confirm"
    );
    for (var i = 0; i < prompts.length; i++) {
      prompts[i].parentNode.removeChild(prompts[i]);
    }
    selectModel(activeSelection.model, activeSelection.li);
  }

  // ---- Download-only (pre-fetch weights, no VRAM) ----

  function veneerParts(li) {
    return {
      label: li.querySelector(".menu-model-veneer-label"),
      prog: li.querySelector(".menu-model-veneer-progress"),
      pct: li.querySelector(".menu-model-veneer-pct"),
      fill: li.querySelector(".menu-model-veneer-fill"),
    };
  }

  function beginDownload(model, li) {
    if (selecting || confirming || downloadRow) {
      return;
    }
    clearError();
    downloadRow = li;
    li.classList.add("is-downloading");
    document.body.classList.add("menu-busy");
    var parts = veneerParts(li);
    if (parts.label) {
      parts.label.hidden = true;
    }
    if (parts.prog) {
      parts.prog.hidden = false;
    }
    if (parts.fill) {
      parts.fill.style.width = "0%";
    }
    fetch(
      "/api/models/" + encodeURIComponent(model.id) + "/download",
      { method: "POST" }
    )
      .then(function (response) {
        return response.json();
      })
      .then(function (result) {
        if (result && result.ok) {
          pollDownload(model, li);
        } else {
          throw new Error(
            (result && result.message) || "download failed"
          );
        }
      })
      .catch(function (err) {
        downloadFailed(
          model, li,
          err && err.message ? err.message : String(err)
        );
      });
  }

  function pollDownload(model, li) {
    if (downloadRow !== li) {
      return;
    }
    fetch("/api/models/download-status")
      .then(function (response) {
        return response.json();
      })
      .then(function (status) {
        if (downloadRow !== li) {
          return;
        }
        if (status.state === "done") {
          finishDownload(model, li);
          return;
        }
        if (status.state === "error") {
          downloadFailed(
            model, li, status.message || "download failed"
          );
          return;
        }
        if (status.state === "idle") {
          resetDownload(li);
          return;
        }
        var pct = status.progress
          && typeof status.progress.fraction === "number"
          ? Math.round(status.progress.fraction * 100)
          : 0;
        var parts = veneerParts(li);
        if (parts.pct) {
          parts.pct.textContent = "Downloading " + pct + "%";
        }
        if (parts.fill) {
          parts.fill.style.width = pct + "%";
        }
        pollTimer = setTimeout(function () {
          pollDownload(model, li);
        }, 500);
      })
      .catch(function () {
        if (downloadRow !== li) {
          return;
        }
        pollTimer = setTimeout(function () {
          pollDownload(model, li);
        }, 800);
      });
  }

  // Restore the veneer to its idle "Click to Download" state.
  function resetDownload(li) {
    stopPolling();
    downloadRow = null;
    li.classList.remove("is-downloading");
    document.body.classList.remove("menu-busy");
    var parts = veneerParts(li);
    if (parts.prog) {
      parts.prog.hidden = true;
    }
    if (parts.label) {
      parts.label.hidden = false;
    }
    if (parts.fill) {
      parts.fill.style.width = "0%";
    }
    var veneer = li.querySelector(".menu-model-veneer");
    var message = veneer
      ? veneer.querySelector(".menu-model-veneer-message")
      : null;
    if (message) {
      message.hidden = true;
    }
  }

  // Download succeeded: keep the veneer, show a success message the
  // user acknowledges with Ok (frees the other rows meanwhile).
  function finishDownload(model, li) {
    stopPolling();
    downloadRow = null;
    li.classList.remove("is-downloading");
    document.body.classList.remove("menu-busy");
    showVeneerMessage(
      li,
      "Download successful for "
      + (model.display_name || model.id) + "!",
      false,
      function () {
        completeDownload(model, li);
      }
    );
  }

  // Ok on success: drop the veneer and denoise-reveal the (until now
  // hidden) model description; the row is then selectable.
  function completeDownload(model, li) {
    ackDownload();
    li.classList.remove("needs-download");
    li._needsDownload = false;
    model.downloaded = true;
    var veneer = li.querySelector(".menu-model-veneer");
    if (veneer) {
      veneer.parentNode.removeChild(veneer);
    }
    var desc = li.querySelector(".menu-model-desc");
    if (desc) {
      if (prefersReducedMotion()) {
        desc.textContent = model.description || "";
      } else {
        revealOnce(desc, model.description || "");
      }
    }
  }

  // Download failed: show the error on the veneer with Ok to retry.
  function downloadFailed(model, li, message) {
    stopPolling();
    downloadRow = null;
    li.classList.remove("is-downloading");
    document.body.classList.remove("menu-busy");
    showVeneerMessage(
      li,
      "Download attempt unsuccessful. Error: " + message,
      true,
      function () {
        ackDownload();
        resetDownload(li);
      }
    );
  }

  // ---- Cross-page download navigation ----

  function ackDownload() {
    if (typeof downloadToastAck === "function") {
      downloadToastAck();
    }
  }

  function modelIndexById(id) {
    for (var i = 0; i < pagedModels.length; i++) {
      if (pagedModels[i].id === id) {
        return i;
      }
    }
    return -1;
  }

  function modelById(id) {
    var idx = modelIndexById(id);
    return idx >= 0 ? pagedModels[idx] : null;
  }

  function rowById(id) {
    if (!modelList || !id) {
      return null;
    }
    var rows = modelList.querySelectorAll(".menu-model-row");
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].getAttribute("data-id") === id) {
        return rows[i];
      }
    }
    return null;
  }

  function ensureVeneer(li) {
    var veneer = li.querySelector(".menu-model-veneer");
    if (!veneer) {
      veneer = buildDownloadVeneer();
      li.appendChild(veneer);
    }
    return veneer;
  }

  // Bind the target row's veneer and resume progress polling (used when a
  // menu load or a page-back lands on the downloading model's row).
  function bindDownloadingRow(model, li) {
    ensureVeneer(li);
    stopPolling();
    downloadRow = li;
    li.classList.add("is-downloading");
    var parts = veneerParts(li);
    if (parts.label) {
      parts.label.hidden = true;
    }
    if (parts.prog) {
      parts.prog.hidden = false;
    }
    var frac = lastDownloadStatus && lastDownloadStatus.progress
      && typeof lastDownloadStatus.progress.fraction === "number"
      ? lastDownloadStatus.progress.fraction
      : 0;
    var pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
    if (parts.pct) {
      parts.pct.textContent = "Downloading " + pct + "%";
    }
    if (parts.fill) {
      parts.fill.style.width = pct + "%";
    }
    pollDownload(model, li);
  }

  // Show the terminal (done/error) veneer on the target row when the menu
  // was not bound as the download finished (i.e. the user was away). The
  // bound-row case is handled by the in-page poll's finish/fail handlers.
  function showDownloadResult(model, li, isError, message) {
    if (downloadRow === li) {
      return;
    }
    var veneer = ensureVeneer(li);
    var msg = veneer.querySelector(".menu-model-veneer-message");
    if (msg && !msg.hidden) {
      return;
    }
    li.classList.remove("is-downloading");
    if (isError) {
      showVeneerMessage(
        li,
        "Download attempt unsuccessful. Error: " + (message || ""),
        true,
        function () {
          ackDownload();
          resetDownload(li);
        }
      );
    } else {
      showVeneerMessage(
        li,
        "Download successful for "
        + (model.display_name || model.id) + "!",
        false,
        function () {
          completeDownload(model, li);
        }
      );
    }
  }

  // Bind/unbind the veneer for the current page against the latest known
  // download status. Does not touch the row fence (onDownloadStatus owns
  // menu-busy) or the current page (that is user/re-attach driven).
  function syncDownloadBinding() {
    var status = lastDownloadStatus;
    var state = status ? status.state : "idle";
    var target = status ? status.target : null;
    var active = state === "downloading"
      || state === "done" || state === "error";
    var row = (active && target) ? rowById(target) : null;
    if (!row) {
      if (downloadRow) {
        stopPolling();
        downloadRow = null;
      }
      return;
    }
    var model = modelById(target);
    if (!model) {
      return;
    }
    if (state === "downloading") {
      if (downloadRow !== row) {
        bindDownloadingRow(model, row);
      }
    } else if (state === "done") {
      showDownloadResult(model, row, false, null);
    } else if (state === "error") {
      showDownloadResult(model, row, true, status.message);
    }
  }

  // Toast subscriber: keeps the row fence and (once re-attached) the veneer
  // binding in sync with the global download as it progresses/finishes,
  // even when the menu is paged away from the row.
  function onDownloadStatus(status) {
    lastDownloadStatus = status;
    var state = status ? status.state : "idle";
    if (state === "downloading") {
      document.body.classList.add("menu-busy");
    } else if (
      prevDownloadState === "downloading"
      && !selecting && !confirming
    ) {
      // Download just ended: lift the fence, unless an activation holds it.
      document.body.classList.remove("menu-busy");
    }
    prevDownloadState = state;
    if (reattachSettled) {
      syncDownloadBinding();
    }
  }

  // The toast is suppressed on the menu while the inline veneer is visible:
  // the target row is bound on the current page and not confirm-collapsed.
  // Suppressed until re-attach settles so it does not flash on load.
  function menuDownloadInlineVisible() {
    if (!reattachSettled) {
      return true;
    }
    return !!downloadRow && !confirming;
  }

  // Toast click on the menu: exit any confirm, page to the download's row,
  // and bind it (no full reload).
  function pageToDownloadTarget() {
    var target = lastDownloadStatus ? lastDownloadStatus.target : null;
    if (!target) {
      return;
    }
    if (selecting || confirming) {
      finishSelecting();
    }
    var idx = modelIndexById(target);
    if (idx >= 0) {
      currentPage = Math.floor(idx / MODELS_PER_PAGE);
      renderCurrentPage();
    }
    syncDownloadBinding();
    if (typeof downloadToastRefresh === "function") {
      downloadToastRefresh();
    }
  }

  // On menu load: if a download is in flight or freshly finished, page to
  // its row so the veneer is visible, apply the fence, and bind it.
  function reattachDownload() {
    fetch("/api/models/download-status")
      .then(function (response) {
        return response.json();
      })
      .then(function (status) {
        lastDownloadStatus = status;
        prevDownloadState = status.state;
        var state = status.state;
        var active = state === "downloading"
          || state === "done" || state === "error";
        if (active && status.target) {
          var idx = modelIndexById(status.target);
          if (idx >= 0) {
            // Only re-render if the target is on a different page (avoid a
            // redundant innerHTML clear/rebuild that flickers on load).
            var targetPage = Math.floor(idx / MODELS_PER_PAGE);
            if (targetPage !== currentPage) {
              currentPage = targetPage;
              renderCurrentPage();
            }
          }
          if (state === "downloading") {
            document.body.classList.add("menu-busy");
          }
        }
        reattachSettled = true;
        syncDownloadBinding();
        if (typeof downloadToastRefresh === "function") {
          downloadToastRefresh();
        }
      })
      .catch(function () {
        reattachSettled = true;
        if (typeof downloadToastRefresh === "function") {
          downloadToastRefresh();
        }
      });
  }

  // Show the activation bar, for the download and for the read into
  // memory that follows it. The phrasing comes from the shared
  // reducer so this and the generator's overlay name the same moment
  // the same way.
  function updateActivationProgress(state, progress) {
    if (!activationProgress) {
      return;
    }
    var view = overlaysActivationProgress(state, progress);
    if (!view.determinate) {
      activationProgress.hidden = true;
      return;
    }
    activationProgress.hidden = false;
    if (activationFill) {
      activationFill.style.width = view.percent + "%";
    }
    if (activationPct) {
      activationPct.textContent =
        view.label + " " + view.percent + "%";
    }
  }

  // Fill the bar and let it be seen full before `done` navigates.
  //
  // Whether a bar was ever on screen is read off the element rather
  // than tracked beside it, so there is one source of truth. A
  // checkpoint whose size could not be worked out ran without a bar,
  // and must not have one appear for a fifth of a second at the end.
  function finishActivationProgress(done) {
    if (!activationProgress || activationProgress.hidden) {
      done();
      return;
    }
    updateActivationProgress("ready", null);
    setTimeout(done, OVERLAYS_LOAD_COMPLETE_HOLD_MS);
  }

  // Poll activation state until ready / error (or cancelled).
  function pollActivation() {
    if (!selecting) {
      return;
    }
    fetch("/api/models/activation")
      .then(function (response) {
        return response.json();
      })
      .then(function (status) {
        if (!selecting) {
          return;
        }
        if (status.state === "ready") {
          finishActivationProgress(function () {
            window.location.assign(GENERATE_URL);
          });
          return;
        }
        if (status.state === "error") {
          var label = selectionLabel();
          finishSelecting();
          showError(
            "Could not load " + label + ": "
            + (status.message || "load failed")
          );
          return;
        }
        updateActivationProgress(
          status.state, status.progress
        );
        pollTimer = setTimeout(
          pollActivation, ACTIVATION_POLL_MS
        );
      })
      .catch(function () {
        if (!selecting) {
          return;
        }
        pollTimer = setTimeout(pollActivation, 800);
      });
  }

  // Raise the "this is loading" UI: the row's cycling status and the
  // activation panel with its bar and cancel button.
  function beginLoadingUi(li) {
    setRowLoading(li);
    if (activationBox) {
      activationBox.hidden = false;
    }
  }

  function selectModel(model, li) {
    if (selecting) {
      return;
    }
    selecting = true;
    activeSelection = { model: model, li: li };
    var resident = isResidentSelection(model, li);
    if (!resident) {
      // Activating from here ends on the generator with a fresh
      // worker, so the previous model's run must not be waiting
      // there. The generator's own switch path does the same on its
      // way out. Re-selecting the resident model spawns nothing, so
      // its run is still the current one and has to survive.
      overlaysClearLastRun();
    }
    clearError();
    if (!resident) {
      beginLoadingUi(li);
    }
    // AR rows carry a device toggle; send the choice so the worker
    // loads on CPU or GPU. Other rows post no body (server default).
    var options = { method: "POST" };
    if (li._getDevice) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify({ device: li._getDevice() });
    }
    fetch(
      "/api/models/" + encodeURIComponent(model.id) + "/activate",
      options
    )
      .then(function (response) {
        return response.json();
      })
      .then(function (result) {
        if (!selecting) {
          return;
        }
        // Activation is non-blocking; the worker loads in the
        // background and we poll for progress until it is ready.
        if (result && result.ok) {
          // A no-op activation answers "ready" in the same breath,
          // which is the server confirming there was nothing to do.
          // Anything else means the worker had died since the menu
          // was drawn and this POST respawned it, so the load UI
          // still has to come up.
          if (resident && result.state === "ready") {
            window.location.assign(GENERATE_URL);
            return;
          }
          if (resident) {
            beginLoadingUi(li);
          }
          pollActivation();
          return;
        }
        var label = selectionLabel();
        finishSelecting();
        showError(
          "Could not load " + label + ": "
          + ((result && result.message) || "activation failed")
        );
      })
      .catch(function (err) {
        finishSelecting();
        showError(
          "Could not load "
          + (model.display_name || model.id) + ": "
          + (err && err.message ? err.message : String(err))
        );
      });
  }

  // Cancel an in-flight load: stop the worker (freeing VRAM) and
  // reset the UI so the user can pick again.
  function cancelSelection() {
    if (!selecting) {
      return;
    }
    stopPolling();
    fetch(
      "/api/models/activate/cancel", { method: "POST" }
    ).catch(function () {
      // Best-effort; the UI resets regardless.
    });
    finishSelecting();
    clearError();
  }

  // ---- "New runs" badge on the Analytics link ----

  // Mirror the generator's count on the menu: "N New" beside Analytics
  // when saved runs remain unopened (the shared set lives in
  // overlays.js and persists across app restarts).
  function refreshAnalyticsNewBadge() {
    var badge = document.getElementById("menu-analytics-new");
    if (!badge || typeof overlaysNewRunCount !== "function") {
      return;
    }
    var count = overlaysNewRunCount();
    if (count > 0) {
      badge.textContent = String(count);
      badge.hidden = false;
    } else {
      badge.textContent = "";
      badge.hidden = true;
    }
  }

  // ---- Boot ----

  function loadModels() {
    fetch("/api/models")
      .then(function (response) {
        return response.json();
      })
      .then(function (info) {
        activeDevice = info.active_device || null;
        renderSystem(info);
        renderModels(info.models || [], !!info.gpu_name);
        reattachDownload();
      })
      .catch(function (err) {
        showSystemError("Could not reach the server.");
        showError(
          err && err.message ? err.message : String(err)
        );
      });
  }

  if (activationCancel) {
    activationCancel.addEventListener("click", cancelSelection);
  }

  if (modelPagePrev) {
    modelPagePrev.addEventListener("click", function () {
      goToPage(-1);
    });
  }
  if (modelPageNext) {
    modelPageNext.addEventListener("click", function () {
      goToPage(1);
    });
  }

  // Wire the cross-page download toast (download_toast.js): the menu
  // suppresses it while the inline veneer shows, overrides its click to
  // page to the row, and reacts to the global download status.
  if (typeof downloadToastRegisterInlineCheck === "function") {
    downloadToastRegisterInlineCheck(menuDownloadInlineVisible);
    downloadToastRegisterNavigate(pageToDownloadTarget);
    downloadToastOnStatus(onDownloadStatus);
  }

  spawnFloaters();
  setupVideo();
  // Hydrate durable UI state from the server before reading the "new
  // run" cue, so the badge reflects saved runs across restarts. The
  // video and model list load immediately; only the badge waits.
  persistHydrate(refreshAnalyticsNewBadge);
  loadModels();
})();
