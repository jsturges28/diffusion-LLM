// The generator's About and Help dialogs.
//
// Loaded as a classic script before app.js. The controller owns the
// two dialogs, their links, close and backdrop routes, and Help's tab
// state. Prompt-import confirmation remains with generator_composer;
// the loading curtain remains with generator_chrome.

"use strict";

function generatorModalsCreate(options) {
  if (!options) {
    throw new TypeError(
      "generatorModalsCreate needs an options object"
    );
  }
  var initialHelpTab = options.initialHelpTab;
  if (typeof initialHelpTab !== "string") {
    throw new TypeError(
      "generatorModalsCreate needs options.initialHelpTab"
    );
  }
  if (initialHelpTab === "") {
    throw new TypeError(
      "generatorModalsCreate needs a non-empty initialHelpTab"
    );
  }

  var linkAbout = requiredElement("link-about");
  var linkHelp = requiredElement("link-help");
  var modalAbout = requiredElement("modal-about");
  var modalHelp = requiredElement("modal-help");
  var helpTabs = modalHelp.querySelectorAll(".help-tab");
  var helpPanels = modalHelp.querySelectorAll(".help-panel");
  var helpBody = modalHelp.querySelector(".help-body");
  var modals = [modalAbout, modalHelp];
  var wired = false;

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error("Missing generator modal element #" + id);
    }
    return element;
  }

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    wireOpenLink(linkAbout, modalAbout);
    wireOpenLink(linkHelp, modalHelp);
    for (var index = 0; index < modals.length; index++) {
      wireModal(modals[index]);
    }
    for (var tabIndex = 0; tabIndex < helpTabs.length; tabIndex++) {
      wireHelpTab(helpTabs[tabIndex]);
    }
    selectHelpTab(initialHelpTab);
  }

  function wireOpenLink(link, modal) {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      openModal(modal);
    });
  }

  function wireModal(modal) {
    var closeButtons = modal.querySelectorAll(".modal-close");
    for (var index = 0; index < closeButtons.length; index++) {
      wireCloseButton(closeButtons[index], modal);
    }
    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal(modal);
      }
    });
  }

  function wireCloseButton(button, modal) {
    button.addEventListener("click", function () {
      closeModal(modal);
    });
  }

  function wireHelpTab(tab) {
    tab.addEventListener("click", function () {
      selectHelpTab(tab.getAttribute("data-help-tab"));
      if (helpBody) {
        helpBody.scrollTop = 0;
      }
    });
  }

  function selectHelpTab(name) {
    for (var index = 0; index < helpTabs.length; index++) {
      var active =
        helpTabs[index].getAttribute("data-help-tab") === name;
      helpTabs[index].classList.toggle("is-active", active);
      helpTabs[index].setAttribute(
        "aria-selected", active ? "true" : "false"
      );
    }
    for (var panelIndex = 0;
      panelIndex < helpPanels.length;
      panelIndex++
    ) {
      helpPanels[panelIndex].hidden =
        helpPanels[panelIndex].getAttribute(
          "data-help-panel"
        ) !== name;
    }
  }

  function openModal(modal) {
    // showModal traps focus, makes the page inert, and lets the
    // browser close only the topmost dialog when Escape is pressed.
    if (!modal.open) {
      modal.showModal();
    }
  }

  function closeModal(modal) {
    if (modal.open) {
      modal.close();
    }
  }

  function closeAll() {
    for (var index = 0; index < modals.length; index++) {
      closeModal(modals[index]);
    }
  }

  return {
    wire: wire,
    closeAll: closeAll,
  };
}
