// Cross-page model-download toast.
//
// A model download is a global, server-side task (see server.py), so it
// keeps running across page navigations. This module runs on every page:
// it polls the download status and, whenever a download is active or has
// just finished but its inline progress veneer is NOT visible (the user
// navigated away, paged off the row, or the menu is mid-confirm), it shows
// a small upper-right toast. Clicking it returns to the download, where
// the menu re-attaches the veneer. The menu registers an inline-visibility
// predicate so the toast stays hidden while the real progress bar shows.
//
// Classic script (no modules); included after download_client.js on
// every page. The transport lives there since `TRUST-04`: this file
// used to run its own 1000ms loop from boot to unload while menu.js
// ran a second one at 500ms against the same URL, so sitting on a
// downloading row polled it twice on two clocks. Now both listen to
// one watcher and this file is only the toast.

"use strict";

(function () {
  var toastEl = null;
  var client = null;
  var lastStatus = null;
  // Page-provided hooks: the inline-veneer predicate (menu only), a click
  // override that navigates to the download without a reload, and a status
  // subscriber (the menu, to bind/unbind its veneer and the row fence).
  var inlineCheck = null;
  var navigateFn = null;
  var statusListener = null;

  // Drag-to-corner: the toast snaps to whichever screen quadrant it is
  // released in, and the choice persists across pages via the durable
  // UI-state layer. Default is the lower-left (clear of the header nav).
  var CORNER_KEY = "diffusion_download_toast_corner";
  var CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"];
  var DRAG_THRESHOLD_PX = 5;
  var EDGE_V = "20px";
  var EDGE_H = "24px";
  var pointerDown = false;
  var dragging = false;
  var justDragged = false;
  var startX = 0;
  var startY = 0;
  var grabX = 0;
  var grabY = 0;

  function inlineVisible() {
    if (typeof inlineCheck !== "function") {
      return false;
    }
    try {
      return !!inlineCheck();
    } catch (_e) {
      return false;
    }
  }

  function ensureToast() {
    if (toastEl || !document.body) {
      return toastEl;
    }
    var el = document.createElement("div");
    el.id = "download-toast";
    el.className = "download-toast";
    el.setAttribute("role", "status");
    el.addEventListener("click", onClick);
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    document.body.appendChild(el);
    toastEl = el;
    applyCorner(readCorner());
    return el;
  }

  function onClick() {
    // A drag ends with a click event; ignore it so a reposition never
    // navigates.
    if (justDragged) {
      return;
    }
    if (typeof navigateFn === "function") {
      navigateFn();
      return;
    }
    window.location.href = "/";
  }

  // ---- Drag to corner ----

  function readCorner() {
    try {
      var value = localStorage.getItem(CORNER_KEY);
      if (CORNERS.indexOf(value) !== -1) {
        return value;
      }
    } catch (_e) {
      // Storage unavailable: fall through to the default.
    }
    return "bottom-left";
  }

  function saveCorner(corner) {
    try {
      if (typeof persistSet === "function") {
        persistSet(CORNER_KEY, corner);
      } else {
        localStorage.setItem(CORNER_KEY, corner);
      }
    } catch (_e) {
      // Best-effort persistence.
    }
  }

  // Pin the toast to a corner via inline offsets (the two inactive edges
  // become "auto" so the anchor is unambiguous and resize-stable).
  function applyCorner(corner) {
    var el = ensureToast();
    if (!el) {
      return;
    }
    var top = corner.indexOf("top") === 0;
    var left = corner.indexOf("left") !== -1;
    el.style.top = top ? EDGE_V : "auto";
    el.style.bottom = top ? "auto" : EDGE_V;
    el.style.left = left ? EDGE_H : "auto";
    el.style.right = left ? "auto" : EDGE_H;
  }

  function onPointerDown(event) {
    if (event.button !== undefined && event.button !== 0) {
      return;
    }
    var el = toastEl;
    if (!el) {
      return;
    }
    pointerDown = true;
    dragging = false;
    startX = event.clientX;
    startY = event.clientY;
    var rect = el.getBoundingClientRect();
    grabX = event.clientX - rect.left;
    grabY = event.clientY - rect.top;
    try {
      el.setPointerCapture(event.pointerId);
    } catch (_e) {
      // Capture is best-effort.
    }
  }

  function onPointerMove(event) {
    if (!pointerDown || !toastEl) {
      return;
    }
    var movedX = Math.abs(event.clientX - startX);
    var movedY = Math.abs(event.clientY - startY);
    if (!dragging
      && (movedX > DRAG_THRESHOLD_PX || movedY > DRAG_THRESHOLD_PX)) {
      dragging = true;
      toastEl.classList.add("is-dragging");
    }
    if (dragging) {
      toastEl.style.top = (event.clientY - grabY) + "px";
      toastEl.style.left = (event.clientX - grabX) + "px";
      toastEl.style.bottom = "auto";
      toastEl.style.right = "auto";
      event.preventDefault();
    }
  }

  function onPointerUp(event) {
    if (!pointerDown) {
      return;
    }
    pointerDown = false;
    if (toastEl) {
      try {
        toastEl.releasePointerCapture(event.pointerId);
      } catch (_e) {
        // Ignore: capture may not be held.
      }
    }
    if (!dragging) {
      return;
    }
    dragging = false;
    toastEl.classList.remove("is-dragging");
    var rect = toastEl.getBoundingClientRect();
    var centerX = rect.left + rect.width / 2;
    var centerY = rect.top + rect.height / 2;
    var vert = centerY < window.innerHeight / 2 ? "top" : "bottom";
    var horiz = centerX < window.innerWidth / 2 ? "left" : "right";
    var corner = vert + "-" + horiz;
    applyCorner(corner);
    saveCorner(corner);
    // Suppress the click that fires on release after a drag.
    justDragged = true;
    setTimeout(function () {
      justDragged = false;
    }, 0);
  }

  function label(status) {
    var name = status.target_name || "model";
    if (status.state === "downloading") {
      var frac = status.progress
        && typeof status.progress.fraction === "number"
        ? status.progress.fraction
        : 0;
      var pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
      return "Downloading " + name + "\u2026 " + pct + "%";
    }
    if (status.state === "done") {
      return "\u2713 " + name + " downloaded \u00b7 open";
    }
    if (status.state === "error") {
      return "\u2715 " + name + " download failed \u00b7 open";
    }
    return "";
  }

  function render(status) {
    var el = ensureToast();
    if (!el) {
      return;
    }
    var active = !!status && (
      status.state === "downloading"
      || status.state === "done"
      || status.state === "error"
    );
    if (!active || inlineVisible()) {
      el.classList.remove("is-visible");
      return;
    }
    el.textContent = label(status);
    el.classList.toggle("is-error", status.state === "error");
    if (!el.classList.contains("is-visible")) {
      // Force a reflow so the first fade-in transition actually runs.
      void el.offsetWidth;
      el.classList.add("is-visible");
    }
  }

  // Re-evaluate the toast against the last known status immediately (the
  // menu calls this after a page or confirm change, without waiting for
  // the next poll).
  function refresh() {
    if (lastStatus) {
      render(lastStatus);
    }
  }

  // Every reading, from the one watcher this page runs.
  function onStatus(status) {
    lastStatus = status;
    if (typeof statusListener === "function") {
      try {
        statusListener(status);
      } catch (_e) {
        // A subscriber error must not stop the loop. The client
        // guards this too; kept because the listener is this
        // module's own contract with the menu, not the client's.
      }
    }
    render(status);
  }

  // Acknowledge a finished download (done/error -> idle) so the toast and
  // the menu re-attach stop firing. Best-effort; the poll reconciles.
  function ack() {
    return client.ack();
  }

  function init() {
    ensureToast();
    client = downloadClientCreate();
    client.subscribe(onStatus);
    // Observe rather than start: every page runs this, and none of
    // them began the fetch. Claiming an operation here would let a
    // page that is only watching cancel somebody else's download.
    client.observe();
  }

  // Public API (globals; classic scripts share one scope).
  window.downloadToastRegisterInlineCheck = function (fn) {
    inlineCheck = fn;
    refresh();
  };
  window.downloadToastRegisterNavigate = function (fn) {
    navigateFn = fn;
  };
  window.downloadToastOnStatus = function (fn) {
    statusListener = fn;
  };
  window.downloadToastRefresh = refresh;
  window.downloadToastAck = ack;
  // The page's one download transport, handed out so the menu can
  // start and cancel through the same watcher it is listening to
  // rather than opening a second one.
  window.downloadToastClient = function () {
    return client;
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
