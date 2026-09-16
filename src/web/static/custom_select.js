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

// ---- The travelling highlight ----
//
// One thing that moves, rather than a background appearing on one
// option as it disappears from another. Transitioning each option
// would cross-fade between two of them, which reads as two changes
// and is what made the highlight hard to follow.
//
// Drawn as a `::before` on the list and driven by custom properties,
// so it adds no children: both lists index their own children
// (`list.children[i]`, `modelRows()`), and a stray element in there
// would have to be filtered out at every one of those sites.
function selectCursorMove(list, option) {
  if (!list) {
    return;
  }
  if (!option) {
    list.classList.remove("has-cursor");
    return;
  }
  // offsetTop is measured from the list's border box; the cursor is
  // positioned against its padding box. Subtracting the border width
  // (clientTop) converts between the two, so the fill lands exactly
  // on the option rather than a border's width below it.
  var top = (option.offsetTop || 0) - (list.clientTop || 0);
  list.style.setProperty("--cursor-top", top + "px");
  list.style.setProperty(
    "--cursor-height", (option.offsetHeight || 0) + "px"
  );
  list.classList.add("has-cursor");
}

// ---- Outside-click ownership ----
//
// One listener for every select on the page, installed once here.
//
// Each widget used to install its own, closing over its own DOM, and
// nothing ever removed it. Analytics rebuilds its overlay picker for
// each run detail opened, so a long session accumulated one dead
// listener and one unreachable-but-retained tree per run, and every
// later click anywhere ran all of them. The generator worked around
// it by refusing to rebuild its picker unless the option set had
// changed, which is the kind of hidden rule a shared widget should
// not be asking its callers to remember.
//
// At most one select is open at a time, which was already true in
// practice (clicking a second one is an outside click for the first),
// so the open widget is held in one variable rather than found by
// querying the document. That keeps an outside click constant work
// instead of proportional to the page, and bounds what can be
// retained at one widget, cleared on the next open or close, against
// the unbounded growth it replaces.
var customSelectOpenWidget = null;

// Serial for the ids that tie a combobox to its listbox and its
// active option. Unique per document, which is what
// aria-activedescendant needs to resolve.
var customSelectSeq = 0;

// Close by manipulating the DOM rather than by calling a per-widget
// closure, so there is one closing path. A registry of closures would
// retain exactly what this exists to stop retaining.
function customSelectClose(wrap) {
  if (!wrap) {
    return;
  }
  var list = wrap.querySelector(".custom-select-list");
  if (list) {
    list.hidden = true;
  }
  wrap.classList.remove("open");
  wrap.classList.remove("drop-up");
  wrap.setAttribute("aria-expanded", "false");
  wrap.removeAttribute("aria-activedescendant");
  selectCursorMove(list, null);
  if (customSelectOpenWidget === wrap) {
    customSelectOpenWidget = null;
  }
}

function customSelectCloseOpen() {
  customSelectClose(customSelectOpenWidget);
}

document.addEventListener("click", function (e) {
  var open = customSelectOpenWidget;
  if (open && !open.contains(e.target)) {
    customSelectClose(open);
  }
});

// Themed in-app dropdown. Exposes `value` and `disabled` and
// fires a "change" event on selection.
//
// options: [{ value, label }]
function createCustomSelect(options, current) {
  var wrap = document.createElement("div");
  wrap.className = "custom-select";
  wrap.tabIndex = 0;
  // A combobox, not a listbox. The thing that expands is the
  // combobox; the listbox is the popup it controls. This said
  // `role="listbox"` on the collapsed control, which announced a
  // list of options where there was one value and no way to reach
  // them, so the semantics were not merely incomplete but wrong.
  wrap.setAttribute("role", "combobox");
  wrap.setAttribute("aria-expanded", "false");
  wrap.setAttribute("aria-haspopup", "listbox");
  customSelectSeq += 1;
  var listId = "custom-select-list-" + customSelectSeq;
  var optionIdPrefix = "custom-select-option-" + customSelectSeq + "-";
  wrap.setAttribute("aria-controls", listId);
  var valueEl = document.createElement("span");
  valueEl.className = "custom-select-value";
  var caret = document.createElement("span");
  caret.className = "custom-select-caret";
  caret.setAttribute("aria-hidden", "true");
  caret.textContent = "\u25be";
  var list = document.createElement("ul");
  list.className = "custom-select-list";
  list.id = listId;
  list.setAttribute("role", "listbox");
  list.hidden = true;
  var value = current;
  // Which option the keyboard is on, which is not the same as which
  // is selected: traversing moves this and leaves the selection alone
  // until Enter. -1 means nothing is being traversed.
  var activeIndex = -1;

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
        + (i === activeIndex ? " is-focused" : "")
        + (option.disabled ? " is-disabled" : "");
      li.id = optionIdPrefix + i;
      li.setAttribute("role", "option");
      li.setAttribute(
        "aria-selected", option.value === value ? "true" : "false"
      );
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
    syncActiveDescendant();
  }

  // Focus stays on the combobox and the active option is named by id,
  // rather than moving focus into the list. That keeps the existing
  // tabIndex and :focus styling working and means one Tab stop per
  // control, which is what a native select gives.
  function syncActiveDescendant() {
    if (activeIndex < 0 || list.hidden) {
      wrap.removeAttribute("aria-activedescendant");
      return;
    }
    wrap.setAttribute(
      "aria-activedescendant", optionIdPrefix + activeIndex
    );
  }

  function selectableIndexes() {
    var out = [];
    for (var i = 0; i < options.length; i++) {
      if (!options[i].disabled) {
        out.push(i);
      }
    }
    return out;
  }

  // Move to the next selectable option in `step` direction, or to the
  // first/last when nothing is active yet. Disabled options are
  // skipped rather than landed on and refused, matching the pointer,
  // which treats them as inert.
  function moveActive(step) {
    var usable = selectableIndexes();
    if (!usable.length) {
      return;
    }
    var at = usable.indexOf(activeIndex);
    if (at === -1) {
      // Start from the selection when there is one, so Down from a
      // chosen value goes to the next value rather than to the top.
      var fromValue = usable.indexOf(indexOfValue());
      at = fromValue === -1 ? (step > 0 ? -1 : 0) : fromValue;
    }
    var next = at + step;
    if (next < 0) {
      next = usable.length - 1;
    } else if (next >= usable.length) {
      next = 0;
    }
    setActive(usable[next]);
  }

  function setActive(index) {
    activeIndex = index;
    renderList();
    var el = list.children[index];
    selectCursorMove(list, el);
    if (el && el.scrollIntoView) {
      el.scrollIntoView({ block: "nearest" });
    }
  }

  function indexOfValue() {
    for (var i = 0; i < options.length; i++) {
      if (options[i].value === value) {
        return i;
      }
    }
    return -1;
  }

  function commitActive() {
    if (activeIndex < 0) {
      return false;
    }
    var option = options[activeIndex];
    if (!option || option.disabled) {
      return false;
    }
    value = option.value;
    renderList();
    close();
    wrap.dispatchEvent(new Event("change"));
    return true;
  }

  function open() {
    if (wrap.classList.contains("disabled")) {
      return;
    }
    // Whatever was open is not any more. Explicit rather than relying
    // on the outside click that opened this one, because a keyboard
    // open never produces one.
    if (customSelectOpenWidget && customSelectOpenWidget !== wrap) {
      customSelectClose(customSelectOpenWidget);
    }
    // Unhide before measuring: a hidden list has no height, so the
    // decision has to be made on the laid-out element. Reset the flip
    // first so the previous open's choice is not measured instead.
    wrap.classList.remove("drop-up");
    list.hidden = false;
    wrap.classList.add("open");
    wrap.setAttribute("aria-expanded", "true");
    customSelectOpenWidget = wrap;
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
    syncActiveDescendant();
  }

  function close() {
    // Traversal does not survive a close: reopening starts from the
    // selection again, which is what a native select does and what
    // stops a stale highlight sitting on an option nobody chose.
    activeIndex = -1;
    customSelectClose(wrap);
    renderList();
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
  // Every key a native select answers to except typeahead, which is
  // deliberately out for now. Before this the control opened and
  // closed from the keyboard and offered no way to choose anything,
  // so the whole parameter column, both overlay pickers and Settings
  // needed a pointer.
  wrap.addEventListener("keydown", function (e) {
    if (wrap.classList.contains("disabled")) {
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) {
        open();
      }
      moveActive(e.key === "ArrowDown" ? 1 : -1);
      return;
    }
    if (e.key === "Home" || e.key === "End") {
      if (list.hidden) {
        return;
      }
      e.preventDefault();
      var usable = selectableIndexes();
      if (usable.length) {
        setActive(
          e.key === "Home" ? usable[0] : usable[usable.length - 1]
        );
      }
      return;
    }
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (list.hidden) {
        open();
      } else if (!commitActive()) {
        // Open with nothing traversed: Enter closes rather than
        // picking whatever happens to be first.
        close();
      }
      return;
    }
    if (e.key === "Escape") {
      // Closes without selecting, so the value survives a browse.
      close();
      return;
    }
    if (e.key === "Tab") {
      // Leaving the control cannot leave its list hanging open over
      // whatever the focus moved to.
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
