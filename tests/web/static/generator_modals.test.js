// The generator modal controller, driven without app.js.
//
// Strategy: build the About and Help dialog trees the factory reads,
// wire it twice, and exercise links, close buttons and backdrops.
// Import is present as a neighbouring dialog but is deliberately not
// handed to this controller.
//
// Passing proves the extraction owns real DOM and wiring state,
// preserves native modal opening and Escape ownership, and cannot
// accidentally take prompt-import confirmation from the composer.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, makeElement } = require("./dom_stub.js");

const SCRIPTS = ["generator_modals.js"];
const HELP_NAMES = ["start", "models", "running"];

function append(parent, child) {
  parent.appendChild(child);
  return child;
}

function addDialogBox(modal) {
  const box = makeElement(null);
  box.className = "modal-box";
  append(modal, box);
  const close = makeElement(null);
  close.className = "modal-close";
  append(box, close);
  return { box, close };
}

function loadModals(settings) {
  const config = settings || {};
  const page = loadPage({ scripts: SCRIPTS });
  const about = page.document.getElementById("modal-about");
  const help = page.document.getElementById("modal-help");
  const imports = page.document.getElementById("modal-import");
  const aboutBox = addDialogBox(about);
  const helpBox = addDialogBox(help);
  const importBox = addDialogBox(imports);
  const layout = makeElement(null);
  layout.className = "help-layout";
  append(helpBox.box, layout);
  const body = makeElement(null);
  body.className = "modal-body help-body";
  append(layout, body);
  const tabs = [];
  const panels = [];
  for (const name of HELP_NAMES) {
    const tab = makeElement(null);
    tab.className = "help-tab";
    tab.setAttribute("data-help-tab", name);
    append(layout, tab);
    tabs.push(tab);
    const panel = makeElement(null);
    panel.className = "help-panel";
    panel.setAttribute("data-help-panel", name);
    append(body, panel);
    panels.push(panel);
  }
  const controller = page.context.generatorModalsCreate({
    initialHelpTab: config.initialHelpTab || "start",
  });
  controller.wire();
  return {
    page,
    controller,
    about,
    help,
    imports,
    aboutBox,
    helpBox,
    importBox,
    tabs,
    panels,
    body,
  };
}

function openFrom(link) {
  let prevented = 0;
  link.dispatch("click", {
    preventDefault() {
      prevented += 1;
    },
  });
  return prevented;
}

test("the initial Help tab option is required", () => {
  const page = loadPage({ scripts: SCRIPTS });

  assert.throws(
    () => page.context.generatorModalsCreate(),
    /options object/
  );
  assert.throws(
    () => page.context.generatorModalsCreate({}),
    /initialHelpTab/
  );
  assert.throws(
    () => page.context.generatorModalsCreate({
      initialHelpTab: "",
    }),
    /non-empty/
  );
});

test("dialog DOM and wiring state stay in the closure", () => {
  const harness = loadModals({});

  for (const name of [
    "linkAbout",
    "linkHelp",
    "modalAbout",
    "modalHelp",
    "helpTabs",
    "helpPanels",
    "wired",
  ]) {
    assert.equal(harness.page.context[name], undefined, name);
  }
});

test("wiring twice does not duplicate listeners", () => {
  const harness = loadModals({});

  harness.controller.wire();

  const aboutLink =
    harness.page.registry.get("link-about");
  assert.equal(aboutLink.listeners.click.length, 1);
  assert.equal(harness.about.listeners.click.length, 1);
  assert.equal(harness.tabs[0].listeners.click.length, 1);
});

test("header links open native modal dialogs", () => {
  const harness = loadModals({});
  const aboutLink =
    harness.page.registry.get("link-about");

  const prevented = openFrom(aboutLink);

  assert.equal(prevented, 1);
  assert.equal(harness.about.open, true);
  assert.equal(harness.about.openedModally, true);
  assert.doesNotThrow(() => openFrom(aboutLink));
});

test(
  "close buttons and dialog backdrops close their own modal",
  () => {
    const harness = loadModals({});
    const aboutLink =
      harness.page.registry.get("link-about");
    const helpLink = harness.page.registry.get("link-help");
    openFrom(aboutLink);
    openFrom(helpLink);

    harness.aboutBox.close.click();

    assert.equal(harness.about.open, false);
    assert.equal(harness.help.open, true);
    harness.help.dispatch("click", { target: harness.help });
    assert.equal(harness.help.open, false);
  }
);

test("a click inside the dialog does not close it", () => {
  const harness = loadModals({});
  openFrom(harness.page.registry.get("link-about"));

  harness.about.dispatch(
    "click", { target: harness.aboutBox.box }
  );

  assert.equal(harness.about.open, true);
});

test("closeAll owns About and Help but not import", () => {
  const harness = loadModals({});
  openFrom(harness.page.registry.get("link-about"));
  openFrom(harness.page.registry.get("link-help"));
  harness.imports.showModal();

  harness.controller.closeAll();

  assert.equal(harness.about.open, false);
  assert.equal(harness.help.open, false);
  assert.equal(harness.imports.open, true);
  assert.equal(harness.importBox.close.listeners.click, undefined);
});

test("Escape remains native dialog behavior", () => {
  const harness = loadModals({});

  assert.equal(harness.page.document.listenerCount("keydown"), 0);
});
