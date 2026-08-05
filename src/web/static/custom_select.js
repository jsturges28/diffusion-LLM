// Shared themed in-app dropdown + text measurement.
//
// Native <select> option lists are drawn by the OS and ignore the
// app theme, so every dropdown in the app (model picker, param
// selects, analytics Group By, overlay picker) is built with
// createCustomSelect for a consistent, in-app look.
//
// Loaded before app.js and analytics.js on their respective pages.

"use strict";

// Widest string width in a set, using an element's computed font.
function measureTextWidth(texts, refEl) {
  var cs = window.getComputedStyle(refEl);
  var font =
    (cs.fontWeight || "400") + " "
    + (cs.fontSize || "12px") + " "
    + (cs.fontFamily || "monospace");
  var canvas =
    measureTextWidth._canvas
    || (measureTextWidth._canvas =
        document.createElement("canvas"));
  var ctx = canvas.getContext("2d");
  ctx.font = font;
  var max = 0;
  for (var i = 0; i < texts.length; i++) {
    max = Math.max(
      max, ctx.measureText(texts[i]).width
    );
  }
  return max;
}

// Size a custom select to its widest option label.
function sizeCustomSelect(widget) {
  var labels = widget._labels || [];
  if (!labels.length) {
    return;
  }
  var width = measureTextWidth(labels, widget);
  widget.style.minWidth = Math.ceil(width) + 48 + "px";
}

// Gap between a select and its option list, mirroring the 4px in the
// .custom-select-list rule.
var CUSTOM_SELECT_GAP_PX = 4;
// Ancestors walked looking for the box that would clip the list. A
// bound rather than a while-true: a detached or cyclic tree must not
// be able to hang a click.
var CUSTOM_SELECT_ANCESTOR_MAX = 32;

// Whether the option list should open upward instead of downward.
//
// Pure, and separated from the measuring, because this is the part
// with a decision in it. Opening up is only an improvement when the
// list genuinely does not fit below AND there is more room above:
// when neither side fits, flipping would trade one clipped list for a
// worse one.
function customSelectShouldDropUp(box) {
  var needed = box.listHeight + box.gap;
  var below = box.boundBottom - box.wrapBottom;
  var above = box.wrapTop - box.boundTop;
  if (below >= needed) {
    return false;
  }
  return above > below;
}

// Bottom/top of the nearest ancestor that would clip an overflowing
// list, falling back to the viewport when nothing clips.
//
// The drawer this most affects sits inside an output area that hides
// its overflow, so the list cannot simply spill out of it; the only
// way to stay visible near the bottom edge is to open the other way.
function customSelectClipBounds(el) {
  var node = el.parentElement;
  var steps = 0;
  while (node && steps < CUSTOM_SELECT_ANCESTOR_MAX) {
    if (node === document.body) {
      break;
    }
    var overflow = window.getComputedStyle(node).overflowY;
    if (overflow && overflow !== "visible") {
      var rect = node.getBoundingClientRect();
      return { top: rect.top, bottom: rect.bottom };
    }
    node = node.parentElement;
    steps += 1;
  }
  return { top: 0, bottom: window.innerHeight };
}

// "low_confidence" -> "Low confidence"; "random" -> "Random".
function prettifyOption(value) {
  var text = String(value).replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Themed in-app dropdown. Exposes `value` and `disabled` and
// fires a "change" event on selection.
//
// options: [{ value, label }]
function createCustomSelect(options, current) {
  var wrap = document.createElement("div");
  wrap.className = "custom-select";
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "listbox");
  var valueEl = document.createElement("span");
  valueEl.className = "custom-select-value";
  var caret = document.createElement("span");
  caret.className = "custom-select-caret";
  caret.setAttribute("aria-hidden", "true");
  caret.textContent = "\u25be";
  var list = document.createElement("ul");
  list.className = "custom-select-list";
  list.hidden = true;
  var value = current;

  function labelFor(target) {
    for (var i = 0; i < options.length; i++) {
      if (options[i].value === target) {
        return options[i].label;
      }
    }
    return String(target);
  }

  function renderList() {
    valueEl.textContent = labelFor(value);
    list.innerHTML = "";
    for (var i = 0; i < options.length; i++) {
      var option = options[i];
      var li = document.createElement("li");
      li.className =
        "custom-select-option"
        + (option.value === value ? " is-active" : "")
        + (option.disabled ? " is-disabled" : "");
      li.setAttribute("data-value", option.value);
      if (option.disabled) {
        li.setAttribute("aria-disabled", "true");
      }
      if (option.title) {
        li.title = option.title;
      }
      li.textContent = option.label;
      list.appendChild(li);
    }
  }

  function open() {
    if (wrap.classList.contains("disabled")) {
      return;
    }
    // Unhide before measuring: a hidden list has no height, so the
    // decision has to be made on the laid-out element. Reset the flip
    // first so the previous open's choice is not measured instead.
    wrap.classList.remove("drop-up");
    list.hidden = false;
    wrap.classList.add("open");
    var bounds = customSelectClipBounds(wrap);
    var rect = wrap.getBoundingClientRect();
    var flip = customSelectShouldDropUp({
      boundTop: bounds.top,
      boundBottom: bounds.bottom,
      wrapTop: rect.top,
      wrapBottom: rect.bottom,
      listHeight: list.offsetHeight,
      gap: CUSTOM_SELECT_GAP_PX,
    });
    wrap.classList.toggle("drop-up", flip);
  }

  function close() {
    list.hidden = true;
    wrap.classList.remove("open");
    wrap.classList.remove("drop-up");
  }

  wrap.appendChild(valueEl);
  wrap.appendChild(caret);
  wrap.appendChild(list);
  renderList();

  wrap.addEventListener("click", function (e) {
    var opt = e.target.closest(".custom-select-option");
    if (opt) {
      // Disabled options are inert: keep the list open, select nothing.
      if (opt.classList.contains("is-disabled")) {
        return;
      }
      value = opt.getAttribute("data-value");
      renderList();
      close();
      wrap.dispatchEvent(new Event("change"));
      return;
    }
    if (wrap.classList.contains("disabled")) {
      return;
    }
    if (list.hidden) {
      open();
    } else {
      close();
    }
  });
  wrap.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (list.hidden) {
        open();
      } else {
        close();
      }
    } else if (e.key === "Escape") {
      close();
    }
  });
  document.addEventListener("click", function (e) {
    if (!wrap.contains(e.target)) {
      close();
    }
  });

  Object.defineProperty(wrap, "value", {
    get: function () {
      return value;
    },
    set: function (v) {
      value = v;
      renderList();
    },
    configurable: true,
  });
  Object.defineProperty(wrap, "disabled", {
    get: function () {
      return wrap.classList.contains("disabled");
    },
    set: function (d) {
      wrap.classList.toggle("disabled", !!d);
      if (d) {
        close();
      }
    },
    configurable: true,
  });
  wrap._labels = options.map(function (o) {
    return o.label;
  });
  return wrap;
}
