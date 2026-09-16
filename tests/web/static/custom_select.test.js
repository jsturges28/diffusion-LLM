// The shared dropdown: who owns its listeners, and whether a
// keyboard can drive it.
//
// Strategy: load only `custom_select.js` into the DOM stub, build
// real widgets, and fire real events at them. Nothing is inspected as
// source text; every claim here is about what the widget does when
// something happens to it.
//
// Two defects, unrelated except that they live in the same file.
//
// The first is a leak. Every `createCustomSelect` installed its own
// document click listener, closing over its own DOM, and nothing ever
// removed it. Analytics rebuilds its overlay picker for each run
// detail opened, so a long session accumulated one dead listener and
// one retained tree per run, and every later click ran all of them.
// The generator carried a workaround (rebuild only when the option
// set changed) which is exactly the hidden rule a shared widget
// should not impose on callers.
//
// The second is that the control could be opened and closed from the
// keyboard and offered no way to choose anything, while announcing
// `role="listbox"` on the collapsed trigger, which is a list of
// options that could not be reached. Selection needed a pointer, so
// the parameter column, both overlay pickers and Settings did too.
//
// Passing proves the page holds one listener however many controls
// have existed, that an outside click costs the same at one control
// and at a thousand, and that a keyboard can open, traverse, select
// and escape while the control says what it is doing.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const OPTIONS = [
  { value: "none", label: "None" },
  { value: "conf", label: "Heatmap" },
  { value: "commit", label: "Commit Order" },
  { value: "diff", label: "Diff vs Original", disabled: true },
  { value: "entropy", label: "Entropy" },
];

function widgetPage() {
  return loadPage({ scripts: ["custom_select.js"] });
}

function build(page, options, current) {
  const wrap = page.context.createCustomSelect(
    options || OPTIONS, current === undefined ? "none" : current
  );
  page.document.body.appendChild(wrap);
  return wrap;
}

function key(wrap, name) {
  wrap.dispatch("keydown", {
    key: name,
    preventDefault() {},
  });
}

function optionEls(wrap) {
  return wrap.children.find((c) => c.tag === "ul").children;
}

function activeLabel(wrap) {
  const found = optionEls(wrap)
    .find((li) => li.classes.has("is-focused"));
  return found ? found.textContent : null;
}

function isOpen(wrap) {
  return wrap.classes.has("open");
}

// -- one listener, however many controls --

test("the page holds one document listener, not one per widget", () => {
  const page = widgetPage();
  const before = page.document.listenerCount("click");

  for (let i = 0; i < 1000; i += 1) {
    build(page);
  }

  assert.equal(
    page.document.listenerCount("click"), before,
    "each widget installed its own listener again"
  );
  assert.equal(before, 1, "the module should own exactly one");
});

test("an outside click costs the same at one widget and at a thousand", () => {
  // The finding's actual complaint was not the memory but the work:
  // every click anywhere ran every dead handler. Handlers run per
  // click is exactly the document listener count, so that number
  // holding flat across two orders of magnitude is the claim.
  const page = widgetPage();

  build(page);
  const withOne = page.document.listenerCount("click");

  for (let i = 0; i < 999; i += 1) {
    build(page);
  }
  const withAThousand = page.document.listenerCount("click");

  assert.equal(withOne, 1);
  assert.equal(withAThousand, withOne);
});

test("an outside click closes the open control", () => {
  const page = widgetPage();
  const wrap = build(page);
  wrap.dispatch("click", { target: wrap });
  assert.equal(isOpen(wrap), true);

  page.document.dispatch("click", { target: page.document.body });

  assert.equal(isOpen(wrap), false);
  assert.equal(wrap.getAttribute("aria-expanded"), "false");
});

test("a closed control is not held onto", () => {
  // The one thing this design can retain is the open widget, so it
  // has to let go on close. Without this, discarding a control while
  // its list happened to be open would keep that tree alive for the
  // rest of the session, which is a smaller version of the leak the
  // whole change is about.
  const page = widgetPage();
  const wrap = build(page);

  wrap.dispatch("click", { target: wrap });
  assert.equal(page.context.customSelectOpenWidget, wrap);

  page.document.dispatch("click", { target: page.document.body });

  assert.equal(page.context.customSelectOpenWidget, null);
});

test("escaping lets go too, not just clicking away", () => {
  const page = widgetPage();
  const wrap = build(page);
  key(wrap, "ArrowDown");

  key(wrap, "Escape");

  assert.equal(page.context.customSelectOpenWidget, null);
});

test("a click inside the control does not close it", () => {
  const page = widgetPage();
  const wrap = build(page);
  wrap.dispatch("click", { target: wrap });

  page.document.dispatch("click", { target: wrap });

  assert.equal(isOpen(wrap), true);
});

test("opening one control closes another", () => {
  // Previously each widget only closed on an outside click, so a
  // keyboard-opened second list left the first hanging open.
  const page = widgetPage();
  const first = build(page);
  const second = build(page);

  first.dispatch("click", { target: first });
  key(second, "ArrowDown");

  assert.equal(isOpen(first), false);
  assert.equal(isOpen(second), true);
});

// -- and a keyboard that can actually choose --

test("arrow keys open the list and move through it", () => {
  const page = widgetPage();
  const wrap = build(page);

  key(wrap, "ArrowDown");

  assert.equal(isOpen(wrap), true);
  assert.equal(activeLabel(wrap), "Heatmap");

  key(wrap, "ArrowDown");
  assert.equal(activeLabel(wrap), "Commit Order");

  key(wrap, "ArrowUp");
  assert.equal(activeLabel(wrap), "Heatmap");
});

test("traversal starts from the current selection", () => {
  // Down from a chosen value should go to the next one, not jump
  // back to the top of the list.
  const page = widgetPage();
  const wrap = build(page, OPTIONS, "commit");

  key(wrap, "ArrowDown");

  assert.equal(activeLabel(wrap), "Entropy");
});

test("disabled options are stepped over, not landed on", () => {
  // "Diff vs Original" sits between Commit Order and Entropy and is
  // disabled until a run has a branch to compare against.
  const page = widgetPage();
  const wrap = build(page, OPTIONS, "conf");

  key(wrap, "ArrowDown");
  assert.equal(activeLabel(wrap), "Commit Order");

  key(wrap, "ArrowDown");
  assert.equal(
    activeLabel(wrap), "Entropy",
    "traversal stopped on the disabled option"
  );
});

test("traversal wraps at both ends", () => {
  const page = widgetPage();
  const wrap = build(page);

  key(wrap, "ArrowUp");
  assert.equal(activeLabel(wrap), "Entropy");

  key(wrap, "ArrowDown");
  assert.equal(activeLabel(wrap), "None");
});

test("Home and End jump to the ends", () => {
  const page = widgetPage();
  const wrap = build(page);
  key(wrap, "ArrowDown");

  key(wrap, "End");
  assert.equal(activeLabel(wrap), "Entropy");

  key(wrap, "Home");
  assert.equal(activeLabel(wrap), "None");
});

test("Enter selects the active option and announces it", () => {
  const page = widgetPage();
  const wrap = build(page);
  let changes = 0;
  wrap.addEventListener("change", () => { changes += 1; });

  key(wrap, "ArrowDown");
  key(wrap, "Enter");

  assert.equal(wrap.value, "conf");
  assert.equal(changes, 1);
  assert.equal(isOpen(wrap), false);
});

test("Escape closes without changing the value", () => {
  // The point of a browse: looking at the options is not choosing
  // one, so backing out has to leave the value alone.
  const page = widgetPage();
  const wrap = build(page);
  let changes = 0;
  wrap.addEventListener("change", () => { changes += 1; });

  key(wrap, "ArrowDown");
  key(wrap, "ArrowDown");
  key(wrap, "Escape");

  assert.equal(wrap.value, "none");
  assert.equal(changes, 0);
  assert.equal(isOpen(wrap), false);
});

test("Tab closes the list rather than leaving it over the page", () => {
  const page = widgetPage();
  const wrap = build(page);
  key(wrap, "ArrowDown");

  key(wrap, "Tab");

  assert.equal(isOpen(wrap), false);
});

test("a disabled control ignores the keyboard", () => {
  const page = widgetPage();
  const wrap = build(page);
  wrap.disabled = true;

  key(wrap, "ArrowDown");

  assert.equal(isOpen(wrap), false);
});

test("reopening forgets where the last browse got to", () => {
  const page = widgetPage();
  const wrap = build(page);

  key(wrap, "ArrowDown");
  key(wrap, "ArrowDown");
  key(wrap, "Escape");
  key(wrap, "ArrowDown");

  assert.equal(
    activeLabel(wrap), "Heatmap",
    "a stale highlight survived the close"
  );
});

// -- the highlight that travels --

function listOf(wrap) {
  return wrap.children.find((c) => c.tag === "ul");
}

test("the cursor appears only once something is active", () => {
  const page = widgetPage();
  const wrap = build(page);
  const list = listOf(wrap);

  assert.equal(list.classes.has("has-cursor"), false);

  key(wrap, "ArrowDown");

  assert.equal(list.classes.has("has-cursor"), true);
});

test("it is told where to go, not just that it should", () => {
  // The offsets are what make it travel. In this stub they are all
  // zero, so what is checked is that they were set at all: JS that
  // never wrote them would leave a cursor pinned at the top while
  // the outline moved down the list.
  const page = widgetPage();
  const wrap = build(page);
  const list = listOf(wrap);

  key(wrap, "ArrowDown");

  assert.notEqual(list.style["--cursor-top"], undefined);
  assert.notEqual(list.style["--cursor-height"], undefined);
});

test("it goes away when the list closes", () => {
  // Otherwise it is still sitting on a row when the list reopens
  // somewhere else, and the first thing it does is slide.
  const page = widgetPage();
  const wrap = build(page);
  const list = listOf(wrap);
  key(wrap, "ArrowDown");

  key(wrap, "Escape");

  assert.equal(list.classes.has("has-cursor"), false);
});

// -- saying what it is, correctly --

test("the control is a combobox and the popup is the listbox", () => {
  // It used to claim `role="listbox"` on the collapsed trigger,
  // which announces a list of options where there is one value.
  const page = widgetPage();
  const wrap = build(page);

  assert.equal(wrap.getAttribute("role"), "combobox");
  assert.equal(wrap.getAttribute("aria-haspopup"), "listbox");

  const list = wrap.children.find((c) => c.tag === "ul");
  assert.equal(list.getAttribute("role"), "listbox");
  assert.equal(wrap.getAttribute("aria-controls"), list.id);
});

test("options carry their role and selected state", () => {
  const page = widgetPage();
  const wrap = build(page, OPTIONS, "conf");

  const items = optionEls(wrap);
  assert.equal(items[0].getAttribute("role"), "option");
  assert.equal(items[0].getAttribute("aria-selected"), "false");
  assert.equal(items[1].getAttribute("aria-selected"), "true");
  assert.equal(items[3].getAttribute("aria-disabled"), "true");
});

test("expanded state and the active option are announced", () => {
  const page = widgetPage();
  const wrap = build(page);

  assert.equal(wrap.getAttribute("aria-expanded"), "false");
  assert.equal(wrap.getAttribute("aria-activedescendant"), null);

  key(wrap, "ArrowDown");

  assert.equal(wrap.getAttribute("aria-expanded"), "true");
  const active = optionEls(wrap)
    .find((li) => li.classes.has("is-focused"));
  assert.equal(
    wrap.getAttribute("aria-activedescendant"), active.id
  );

  key(wrap, "Escape");
  assert.equal(wrap.getAttribute("aria-activedescendant"), null);
});

test("two controls on a page do not share option ids", () => {
  // aria-activedescendant resolves by id against the document, so a
  // duplicate points a screen reader at the wrong control.
  const page = widgetPage();
  const first = build(page);
  const second = build(page);

  const firstIds = optionEls(first).map((li) => li.id);
  const secondIds = optionEls(second).map((li) => li.id);

  for (const id of firstIds) {
    assert.equal(
      secondIds.includes(id), false, `duplicate option id ${id}`
    );
  }
  assert.notEqual(first.getAttribute("aria-controls"), null);
  assert.notEqual(
    first.getAttribute("aria-controls"),
    second.getAttribute("aria-controls")
  );
});
