// Help's side tabs show one panel at a time.
//
// Strategy: the tab rail is selected by class, and the stub resolves
// elements by id, so `helpTabs` is empty when app.js loads. The rail
// is therefore built here and assigned over those two variables, then
// the page's own selectHelpTab and wireHelpTabs run against it. That
// keeps the function under test the shipped one, with only its input
// supplied by the test.
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

// A rail and its panels, wired into the page. Shaped like the real
// markup: both live inside .help-layout, and the panels inside the
// scrolling .help-body, because wireHelpTabs finds the scroller by
// walking up from the tab that was clicked.
function help() {
  const { context } = loadPage({});

  const layout = makeElement(null);
  layout.className = "help-layout";
  const body = makeElement(null);
  body.className = "modal-body help-body";
  body.parent = layout;
  layout.children.push(body);

  const tabs = [];
  const panels = [];
  for (const name of NAMES) {
    const tab = makeElement(null);
    tab.className = "help-tab";
    tab.setAttribute("data-help-tab", name);
    tab.parent = layout;
    layout.children.push(tab);
    tabs.push(tab);

    const panel = makeElement(null);
    panel.className = "help-panel";
    panel.setAttribute("data-help-panel", name);
    panel.parent = body;
    body.children.push(panel);
    panels.push(panel);
  }

  // The opening state the markup ships: first tab active, the rest of
  // the panels hidden.
  tabs[0].classList.add("is-active");
  tabs[0].setAttribute("aria-selected", "true");
  for (let i = 1; i < NAMES.length; i++) {
    tabs[i].setAttribute("aria-selected", "false");
    panels[i].hidden = true;
  }

  context.helpTabs = tabs;
  context.helpPanels = panels;
  context.wireHelpTabs();

  return { context, tabs, panels, body };
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

test("one panel is visible to begin with", () => {
  const { panels } = help();

  assert.deepEqual(visible(panels), ["start"]);
});

test("selecting a tab shows only its panel", () => {
  const { context, panels } = help();

  context.selectHelpTab("models");

  assert.deepEqual(visible(panels), ["models"]);
});

test("selecting a tab marks only it active", () => {
  const { context, tabs } = help();

  context.selectHelpTab("running");

  assert.deepEqual(active(tabs), ["running"]);
});

test("the active tab is the one whose panel shows", () => {
  // The pair that matters: a rail highlighting one section while the
  // body shows another is worse than no highlight at all.
  const { context, tabs, panels } = help();

  context.selectHelpTab("models");

  assert.deepEqual(active(tabs), visible(panels));
});

test("aria-selected follows the class", () => {
  // Kept in step deliberately. Settings marks its active tab with a
  // class alone, which a screen reader cannot see.
  const { context, tabs } = help();

  context.selectHelpTab("running");

  const marked = tabs
    .filter((tab) => tab.getAttribute("aria-selected") === "true")
    .map((tab) => tab.getAttribute("data-help-tab"));
  assert.deepEqual(marked, ["running"]);
});

test("every other tab is explicitly not selected", () => {
  // The negative space. Leaving the previous tab's aria-selected on
  // announces two active tabs, so this checks the count and not just
  // that the new one is set.
  const { context, tabs } = help();

  context.selectHelpTab("models");

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
  const { context, tabs, panels } = help();
  context.selectHelpTab("models");

  context.selectHelpTab("models");

  assert.deepEqual(visible(panels), ["models"]);
  assert.deepEqual(active(tabs), ["models"]);
});

test("an unknown name hides everything rather than guessing", () => {
  // Negative space. Nothing should call this with a name that has no
  // panel, so the honest outcome is a blank body, which is visibly
  // wrong. Falling back to the first panel would hide the typo.
  const { context, tabs, panels } = help();

  context.selectHelpTab("nonexistent");

  assert.deepEqual(visible(panels), []);
  assert.deepEqual(active(tabs), []);
});
