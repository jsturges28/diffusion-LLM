// Help's side tabs show one panel at a time.
//
// Strategy: load the modal factory without app.js, build the same
// rail and scrolling body as the page, then drive selection through
// the buttons a reader uses. The controller discovers those nodes
// once and keeps them private.
//
// Passing proves exactly one panel is ever visible, the rail agrees
// with which one that is in both the class and the ARIA state, and
// switching returns the reader to the top of the new panel.
//
// The last of those is the one worth having. The panel scrolls, not
// the box, so without a reset a reader who was deep in Interventions
// opens Models already partway down it, which reads as missing copy
// rather than as a scroll position.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, makeElement } = require("./dom_stub.js");

const NAMES = ["start", "models", "running"];
const SCRIPTS = ["generator_modals.js"];

// A rail and its panels, wired into the page. Shaped like the real
// markup: both live inside .help-layout, and the panels inside the
// scrolling .help-body, because the controller resets that owned
// scroller whenever a tab is clicked.
function help(settings) {
  const config = settings || {};
  const page = loadPage({ scripts: SCRIPTS });
  const modal = page.document.getElementById("modal-help");

  const layout = makeElement(null);
  layout.className = "help-layout";
  modal.appendChild(layout);
  const body = makeElement(null);
  body.className = "modal-body help-body";
  layout.appendChild(body);

  const tabs = [];
  const panels = [];
  for (const name of NAMES) {
    const tab = makeElement(null);
    tab.className = "help-tab";
    tab.setAttribute("data-help-tab", name);
    layout.appendChild(tab);
    tabs.push(tab);

    const panel = makeElement(null);
    panel.className = "help-panel";
    panel.setAttribute("data-help-panel", name);
    body.appendChild(panel);
    panels.push(panel);
  }

  const controller = page.context.generatorModalsCreate({
    initialHelpTab: config.initialHelpTab || "start",
  });
  controller.wire();
  return { controller, tabs, panels, body };
}

function visible(panels) {
  return panels
    .filter((panel) => !panel.hidden)
    .map((panel) => panel.getAttribute("data-help-panel"));
}

function active(tabs) {
  return tabs
    .filter((tab) => tab.classList.contains("is-active"))
    .map((tab) => tab.getAttribute("data-help-tab"));
}

test("the initial tab and panel agree", () => {
  const { tabs, panels } = help();

  assert.deepEqual(visible(panels), ["start"]);
  assert.deepEqual(active(tabs), ["start"]);
  assert.equal(tabs[0].getAttribute("aria-selected"), "true");
});

test("selecting a tab shows only its panel", () => {
  const { tabs, panels } = help();

  tabs[1].click();

  assert.deepEqual(visible(panels), ["models"]);
});

test("selecting a tab marks only it active", () => {
  const { tabs } = help();

  tabs[2].click();

  assert.deepEqual(active(tabs), ["running"]);
});

test("the active tab is the one whose panel shows", () => {
  // The pair that matters: a rail highlighting one section while the
  // body shows another is worse than no highlight at all.
  const { tabs, panels } = help();

  tabs[1].click();

  assert.deepEqual(active(tabs), visible(panels));
});

test("aria-selected follows the class", () => {
  // Kept in step deliberately. Settings marks its active tab with a
  // class alone, which a screen reader cannot see.
  const { tabs } = help();

  tabs[2].click();

  const marked = tabs
    .filter((tab) => tab.getAttribute("aria-selected") === "true")
    .map((tab) => tab.getAttribute("data-help-tab"));
  assert.deepEqual(marked, ["running"]);
});

test("every other tab is explicitly not selected", () => {
  // The negative space. Leaving the previous tab's aria-selected on
  // announces two active tabs, so this checks the count and not just
  // that the new one is set.
  const { tabs } = help();

  tabs[1].click();

  const off = tabs.filter(
    (tab) => tab.getAttribute("aria-selected") === "false"
  );
  assert.equal(off.length, NAMES.length - 1);
});

test("clicking a tab switches the panel", () => {
  // Through the listener rather than the function, so the wiring is
  // part of what passes.
  const { tabs, panels } = help();

  tabs[1].click();

  assert.deepEqual(visible(panels), ["models"]);
});

test("clicking returns to the top of the new panel", () => {
  const { tabs, body } = help();
  body.scrollTop = 900;

  tabs[2].click();

  assert.equal(body.scrollTop, 0);
});

test("reselecting the open tab leaves it open", () => {
  // Idempotent, because a reader clicking the tab they are already on
  // should not see anything change.
  const { tabs, panels } = help();
  tabs[1].click();

  tabs[1].click();

  assert.deepEqual(visible(panels), ["models"]);
  assert.deepEqual(active(tabs), ["models"]);
});

test("an unknown name hides everything rather than guessing", () => {
  // Negative space. Nothing should call this with a name that has no
  // panel, so the honest outcome is a blank body, which is visibly
  // wrong. Falling back to the first panel would hide the typo.
  const { tabs, panels } = help({
    initialHelpTab: "nonexistent",
  });

  assert.deepEqual(visible(panels), []);
  assert.deepEqual(active(tabs), []);
});
