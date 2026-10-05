// The generator's model selector and schema-driven parameter panel.
//
// Loaded as a classic script before app.js. Registry data, picker
// traversal, switch confirmation, parameter DOM references and form
// drafts stay private to the returned controllers. The reusable Run
// settings panel owns schema controls; this controller owns model
// selection and exposes narrow reads to the page.

"use strict";

function generatorModelPanelCreate(options) {
  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorModelPanelCreate needs options." + name
      );
    }
    return options[name];
  }

  var onSwitchRequested =
    requiredCallback("onSwitchRequested");
  var onValidationChanged =
    requiredCallback("onValidationChanged");
  var onParametersChanged =
    requiredCallback("onParametersChanged");
  var readReducedMotion =
    requiredCallback("readReducedMotion");
  var readGpuTicker = requiredCallback("readGpuTicker");

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error("Missing generator model panel element #" + id);
    }
    return element;
  }

  var modelSelect = requiredElement("model-select");
  var modelSelectValue = requiredElement("model-select-value");
  var modelSelectList = requiredElement("model-select-list");

  var PARAM_STATE_KEY = "diffusion_param_state";
  var MODEL_OPTION_ID_PREFIX = "model-select-option-";
  var MODEL_SELECT_TAB_INDEX = 0;
  var DEVICE_LABELS = { cuda: "GPU", cpu: "CPU" };
  var runSettingsPanel = runSettingsPanelCreate({
    idPrefix: "",
    root: requiredElement("run-settings"),
    widthTarget: document.documentElement,
    onValidationChanged: onValidationChanged,
    onParametersChanged: onParametersChanged,
    readPersistedState: readPersistedParamState,
    writePersistedState: writePersistedParamState,
  });

  var models = {};
  var modelList = [];
  var activeModelId = null;
  var activeModel = null;
  var activeDevice = null;
  var activeTokenizer = {};
  var activeContextLength = null;
  var gpuPresent = false;

  var modelSelectDisabled = false;
  var modelActiveRow = -1;
  var switchConfirmEl = null;
  var collapsedTickerTimer = null;
  var modelFontReadyPending = false;
  var wired = false;

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    runSettingsPanel.wire();
    wireModelPicker();
  }

  function wireModelPicker() {
    closeModelList();
    modelSelect.addEventListener("click", modelSelectClicked);
    modelSelectList.addEventListener(
      "click", modelSelectListClicked
    );
    modelSelect.addEventListener(
      "keydown", modelSelectKeydown
    );
    document.addEventListener("click", function (event) {
      if (!modelSelect.contains(event.target)) {
        closeModelList();
        closeSwitchConfirm();
      }
    });
  }

  function modelSelectClicked(event) {
    if (event.target.closest(".model-select-option")) {
      return;
    }
    if (modelSelectDisabled) {
      return;
    }
    toggleModelList();
  }

  function modelSelectListClicked(event) {
    var option = event.target.closest(".model-select-option");
    if (!option) {
      return;
    }
    var id = option.getAttribute("data-id");
    if (!id || id === activeModelId) {
      return;
    }
    requestSwitch(id, defaultDeviceFor(models[id]));
  }

  function modelSelectKeydown(event) {
    if (modelSelectDisabled) {
      return;
    }
    if (
      switchConfirmEl
      && switchConfirmEl.contains(event.target)
    ) {
      switchConfirmKeydown(event);
      return;
    }
    if (
      event.key === "ArrowDown"
      || event.key === "ArrowUp"
    ) {
      event.preventDefault();
      if (modelSelectList.hidden) {
        openModelList();
      }
      moveModelActive(event.key === "ArrowDown" ? 1 : -1);
      return;
    }
    if (
      event.key === "ArrowRight"
      || event.key === "ArrowLeft"
    ) {
      if (modelSelectList.hidden) {
        return;
      }
      event.preventDefault();
      moveModelDevice(event.key === "ArrowRight" ? 1 : -1);
      return;
    }
    modelSelectActionKeydown(event);
  }

  function switchConfirmKeydown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeSwitchConfirm();
    }
  }

  function modelSelectActionKeydown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (modelSelectList.hidden) {
        openModelList();
      } else if (!activateModelActive()) {
        closeModelList();
      }
      return;
    }
    if (event.key === "Escape") {
      closeModelList();
      return;
    }
    if (event.key === "Tab") {
      closeModelList();
    }
  }

  function configure(info) {
    if (!info || typeof info !== "object") {
      throw new TypeError(
        "generatorModelPanel.configure needs model information"
      );
    }
    modelList = modelClientList(info);
    models = {};
    indexModels(modelList);
    activeModelId =
      modelClientActiveId(info)
      || info.default
      || (modelList[0] && modelList[0].id)
      || null;
    activeModel =
      models[activeModelId] || modelList[0] || null;
    activeDevice = modelClientActiveDevice(info);
    activeTokenizer = info.active_tokenizer || {};
    activeContextLength = info.active_context_length;
    gpuPresent = modelClientGpuPresent(info);
    renderModelSelector(modelList, activeModelId);
    buildActiveParamPanel();
  }

  function indexModels(list) {
    for (var index = 0; index < list.length; index++) {
      models[list[index].id] = list[index];
    }
  }

  function buildActiveParamPanel() {
    var capabilities = activeModel
      ? activeModel.capabilities || {}
      : {};
    runSettingsPanel.configure({
      model: activeModel,
      models: modelList,
      modelId: activeModelId,
      modelDisplay: activeDisplayName(),
      device: activeDevice,
      inputMode: capabilities.input_mode || null,
    });
    remeasureModelSelectWhenFontReady();
  }

  function refresh(info) {
    if (!info || typeof info !== "object") {
      throw new TypeError(
        "generatorModelPanel.refresh needs model information"
      );
    }
    modelList = modelClientList(info);
    indexModels(modelList);
    renderModelSelector(modelList, activeModelId);
  }

  function activeModelRead() {
    return activeModel;
  }

  function activeModelIdRead() {
    return activeModelId;
  }

  function activeDeviceRead() {
    return activeDevice;
  }

  function activeTokenizerRead() {
    return activeTokenizer;
  }

  function activeContextRead() {
    return activeContextLength;
  }

  function capabilitiesRead() {
    if (!activeModel || !activeModel.capabilities) {
      return {};
    }
    return activeModel.capabilities;
  }

  function modelDisplayName(id) {
    var model = models[id];
    return model ? model.display_name : id;
  }

  function activeDisplayName() {
    return activeModel ? activeModel.display_name : "";
  }

  function supportedDevices(model) {
    var declared =
      model
      && model.capabilities
      && model.capabilities.supported_devices;
    return declared && declared.length ? declared : ["cuda"];
  }

  function deviceLabel(device) {
    return DEVICE_LABELS[device]
      || String(device).toUpperCase();
  }

  function defaultDeviceFor(model) {
    var devices = supportedDevices(model);
    if (gpuPresent && devices.indexOf("cuda") !== -1) {
      return "cuda";
    }
    if (!gpuPresent && devices.indexOf("cpu") !== -1) {
      return "cpu";
    }
    return devices[0];
  }

  function buildOptionInfo(model) {
    var required = Math.round(model.min_vram_gib || 0);
    var headroom = model.vram_headroom_gib;
    var pop = document.createElement("div");
    pop.className = "option-info";
    if (typeof headroom === "number") {
      return buildOptionHeadroom(pop, model);
    }
    if (required > 0) {
      pop.textContent = "Requires ~" + required + " GiB VRAM";
      return pop;
    }
    return null;
  }

  function buildOptionHeadroom(pop, model) {
    var required = Math.round(model.min_vram_gib || 0);
    var headroom = model.vram_headroom_gib;
    var minimumVram = model.min_vram_gib || 0;
    var available = (minimumVram + headroom).toFixed(1);
    var positive = headroom >= 0;
    var sign = (positive ? "+" : "\u2212")
      + Math.abs(headroom).toFixed(1);
    pop.appendChild(document.createTextNode(
      "Required " + required
      + " GiB \u00b7 Available " + available
      + " GiB \u00b7 "
    ));
    var head = document.createElement("span");
    head.className = "option-info-headroom "
      + (positive ? "is-positive" : "is-negative");
    head.textContent = sign;
    pop.appendChild(head);
    pop.classList.add(positive ? "is-positive" : "is-negative");
    return pop;
  }

  function buildDevicePill(label, active) {
    var pill = document.createElement("span");
    pill.className =
      "device-pill" + (active ? " is-active" : "");
    pill.textContent = label;
    return pill;
  }

  function stopCollapsedTicker() {
    if (collapsedTickerTimer !== null) {
      clearInterval(collapsedTickerTimer);
      collapsedTickerTimer = null;
    }
  }

  function startCollapsedTicker(settings) {
    stopCollapsedTicker();
    var label = settings.device === "cpu" ? "CPU" : "GPU";
    var textElement = document.createElement("span");
    textElement.className = "ticker-text";
    textElement.textContent = label;
    settings.pill.textContent = "";
    settings.pill.appendChild(textElement);
    var headroom = settings.model.vram_headroom_gib;
    var canTick = readGpuTicker()
      && !readReducedMotion()
      && settings.device !== "cpu"
      && typeof headroom === "number";
    if (!canTick) {
      return;
    }
    runCollapsedTicker({
      textElement: textElement,
      label: label,
      headroom: headroom,
    });
  }

  function runCollapsedTicker(settings) {
    var headLabel = (
      settings.headroom >= 0 ? "+" : "\u2212"
    ) + Math.abs(settings.headroom).toFixed(1);
    var showingDevice = true;
    collapsedTickerTimer = setInterval(function () {
      settings.textElement.classList.add("ticker-fade");
      setTimeout(function () {
        showingDevice = !showingDevice;
        settings.textElement.textContent =
          showingDevice ? settings.label : headLabel;
        settings.textElement.classList.toggle(
          "is-headroom", !showingDevice
        );
        settings.textElement.classList.remove("ticker-fade");
      }, 350);
    }, 2000);
  }

  function setModelSelectValue(id) {
    stopCollapsedTicker();
    var model = models[id];
    modelSelectValue.innerHTML = "";
    var nameElement = document.createElement("span");
    nameElement.className = "model-select-value-name";
    nameElement.textContent =
      model ? model.display_name : (id || "-");
    nameElement.title = model ? model.display_name : "";
    modelSelectValue.appendChild(nameElement);
    if (!model) {
      return;
    }
    var device = id === activeModelId && activeDevice
      ? activeDevice
      : defaultDeviceFor(model);
    var pill = buildDevicePill(
      device === "cpu" ? "CPU" : "GPU", false
    );
    pill.classList.add("device-pill-collapsed");
    modelSelectValue.appendChild(pill);
    startCollapsedTicker({
      pill: pill,
      model: model,
      device: device,
    });
  }

  function buildOptionDevice(model, activeId) {
    var wrap = document.createElement("span");
    wrap.className = "option-device";
    var supported = supportedDevices(model);
    if (supported.length < 2) {
      wrap.appendChild(
        buildDevicePill(deviceLabel(supported[0]), true)
      );
      return wrap;
    }
    var isActiveModel = model.id === activeId;
    var current = isActiveModel && activeDevice
      ? activeDevice
      : defaultDeviceFor(model);
    for (var index = 0; index < supported.length; index++) {
      buildOptionDeviceButton({
        wrap: wrap,
        model: model,
        device: supported[index],
        current: current,
        isActiveModel: isActiveModel,
      });
    }
    return wrap;
  }

  function buildOptionDeviceButton(settings) {
    var button = document.createElement("button");
    button.type = "button";
    button.tabIndex = -1;
    button.setAttribute("data-device", settings.device);
    var isCurrent = settings.isActiveModel
      && settings.device === activeDevice;
    button.className =
      "device-pill device-pill-btn"
      + (settings.device === settings.current
        ? " is-active" : "")
      + (isCurrent ? " is-current" : "");
    button.textContent = deviceLabel(settings.device);
    applyDeviceButtonAvailability(button, {
      device: settings.device,
      isCurrent: isCurrent,
    });
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      if (!button.disabled) {
        requestSwitch(settings.model.id, settings.device);
      }
    });
    settings.wrap.appendChild(button);
  }

  function applyDeviceButtonAvailability(button, settings) {
    if (settings.device === "cuda" && !gpuPresent) {
      button.disabled = true;
      button.title = "No GPU detected";
    }
    if (settings.isCurrent) {
      button.title = "Currently loaded";
    }
  }

  function setDisabled(disabled) {
    modelSelectDisabled = disabled;
    modelSelect.classList.toggle("disabled", disabled);
    modelSelect.setAttribute(
      "aria-disabled", disabled ? "true" : "false"
    );
    modelSelect.tabIndex = disabled
      ? -1
      : MODEL_SELECT_TAB_INDEX;
    runSettingsPanel.setDisabled(disabled);
    if (disabled) {
      closeModelList();
    }
  }

  function modelRows() {
    return Array.prototype.slice.call(modelSelectList.children);
  }

  function modelRowPills(row) {
    var found = row.querySelectorAll(".device-pill-btn");
    var usable = [];
    for (var index = 0; index < found.length; index++) {
      if (!found[index].disabled) {
        usable.push(found[index]);
      }
    }
    return usable;
  }

  function modelRowIndexOf(id) {
    var rows = modelRows();
    for (var index = 0; index < rows.length; index++) {
      if (rows[index].getAttribute("data-id") === id) {
        return index;
      }
    }
    return -1;
  }

  function renderModelActive() {
    var rows = modelRows();
    for (var index = 0; index < rows.length; index++) {
      var on = index === modelActiveRow;
      rows[index].classList.toggle("is-focused", on);
      var info = rows[index].querySelector(".option-info");
      if (info) {
        info.classList.toggle("is-visible", on);
      }
    }
    if (modelActiveRow < 0 || modelSelectList.hidden) {
      modelSelect.removeAttribute("aria-activedescendant");
      selectCursorMove(modelSelectList, null);
      return;
    }
    modelSelect.setAttribute(
      "aria-activedescendant", rows[modelActiveRow].id
    );
    selectCursorMove(modelSelectList, rows[modelActiveRow]);
  }

  function moveModelActive(step) {
    var rows = modelRows();
    if (rows.length === 0) {
      return;
    }
    var from = modelActiveRow;
    if (from < 0) {
      from = modelRowIndexOf(activeModelId);
    }
    modelActiveRow = nextModelRow({
      from: from,
      step: step,
      count: rows.length,
    });
    renderModelActive();
    if (rows[modelActiveRow].scrollIntoView) {
      rows[modelActiveRow].scrollIntoView({ block: "nearest" });
    }
  }

  function nextModelRow(settings) {
    if (settings.from < 0) {
      return settings.step > 0 ? 0 : settings.count - 1;
    }
    var next = settings.from + settings.step;
    if (next < 0) {
      return settings.count - 1;
    }
    if (next >= settings.count) {
      return 0;
    }
    return next;
  }

  function moveModelDevice(step) {
    var rows = modelRows();
    if (modelActiveRow < 0 || !rows[modelActiveRow]) {
      return;
    }
    var pills = modelRowPills(rows[modelActiveRow]);
    if (pills.length < 2) {
      return;
    }
    var at = activePillIndex(pills);
    var next = (at + step + pills.length) % pills.length;
    for (var index = 0; index < pills.length; index++) {
      pills[index].classList.toggle("is-active", index === next);
    }
  }

  function activePillIndex(pills) {
    for (var index = 0; index < pills.length; index++) {
      if (pills[index].classList.contains("is-active")) {
        return index;
      }
    }
    return 0;
  }

  function activateModelActive() {
    var rows = modelRows();
    if (modelActiveRow < 0 || !rows[modelActiveRow]) {
      return false;
    }
    var row = rows[modelActiveRow];
    var id = row.getAttribute("data-id");
    if (!id) {
      return false;
    }
    var device = activeDeviceForRow(row, id);
    if (id === activeModelId && device === activeDevice) {
      return false;
    }
    requestSwitch(id, device);
    return true;
  }

  function activeDeviceForRow(row, id) {
    var pills = modelRowPills(row);
    for (var index = 0; index < pills.length; index++) {
      if (pills[index].classList.contains("is-active")) {
        return pills[index].getAttribute("data-device");
      }
    }
    return defaultDeviceFor(models[id]);
  }

  function openModelList() {
    if (modelSelectDisabled) {
      return;
    }
    closeSwitchConfirm();
    modelSelectList.hidden = false;
    modelSelect.classList.add("open");
    modelSelect.setAttribute("aria-expanded", "true");
    renderModelActive();
  }

  function closeModelList() {
    modelSelectList.hidden = true;
    modelSelect.classList.remove("open");
    modelSelect.setAttribute("aria-expanded", "false");
    modelActiveRow = -1;
    renderModelActive();
  }

  function toggleModelList() {
    if (modelSelectList.hidden) {
      openModelList();
    } else {
      closeModelList();
    }
  }

  function renderModelSelector(list, activeId) {
    modelSelectList.innerHTML = "";
    for (var index = 0; index < list.length; index++) {
      modelSelectList.appendChild(
        buildModelOption({
          model: list[index],
          activeId: activeId,
          index: index,
        })
      );
    }
    setModelSelectValue(activeId);
    sizeModelSelect(list);
  }

  function buildModelOption(settings) {
    var model = settings.model;
    var option = document.createElement("li");
    option.className =
      "model-select-option"
      + (model.id === settings.activeId ? " is-active" : "");
    option.id = MODEL_OPTION_ID_PREFIX + settings.index;
    option.setAttribute("role", "option");
    option.setAttribute(
      "aria-selected",
      model.id === settings.activeId ? "true" : "false"
    );
    option.setAttribute("data-id", model.id);
    var nameElement = document.createElement("span");
    nameElement.className = "model-select-name";
    nameElement.textContent = model.display_name;
    nameElement.title = model.display_name;
    option.appendChild(nameElement);
    option.appendChild(
      buildOptionDevice(model, settings.activeId)
    );
    appendOptionInfo(option, model);
    return option;
  }

  function appendOptionInfo(option, model) {
    var info = buildOptionInfo(model);
    if (!info) {
      return;
    }
    option.appendChild(info);
    option.addEventListener("mouseenter", function () {
      info.classList.add("is-visible");
    });
    option.addEventListener("mouseleave", function () {
      info.classList.remove("is-visible");
    });
  }

  function closeSwitchConfirm() {
    if (switchConfirmEl && switchConfirmEl.parentNode) {
      switchConfirmEl.parentNode.removeChild(switchConfirmEl);
    }
    switchConfirmEl = null;
  }

  function requestSwitch(id, device) {
    closeModelList();
    var model = models[id];
    if (!model) {
      return;
    }
    if (id === activeModelId && device === activeDevice) {
      return;
    }
    openSwitchConfirm(id, device);
  }

  function openSwitchConfirm(id, device) {
    closeSwitchConfirm();
    var model = models[id];
    if (!model) {
      return;
    }
    var box = document.createElement("div");
    box.className = "switch-confirm";
    box.addEventListener("click", function (event) {
      event.stopPropagation();
    });
    box.appendChild(switchConfirmMessage(model, device));
    box.appendChild(switchConfirmActions(id, device));
    modelSelect.appendChild(box);
    switchConfirmEl = box;
  }

  function switchConfirmMessage(model, device) {
    var currentName = activeModel
      ? activeModel.display_name
      : "the current model";
    var message = document.createElement("span");
    message.className = "switch-confirm-msg";
    message.textContent =
      "Unload the current model " + currentName
      + " and load " + model.display_name + " on "
      + (device === "cpu" ? "CPU" : "GPU") + "?";
    return message;
  }

  function switchConfirmActions(id, device) {
    var actions = document.createElement("span");
    actions.className = "switch-confirm-actions";
    actions.appendChild(
      switchConfirmButton({
        className: "switch-confirm-yes",
        title: "Confirm switch",
        label: "Confirm switch",
        text: "\u2713",
        onClick: function () {
          closeSwitchConfirm();
          onSwitchRequested(id, device);
        },
      })
    );
    actions.appendChild(
      switchConfirmButton({
        className: "switch-confirm-no",
        title: "Cancel",
        label: "Cancel",
        text: "\u2717",
        onClick: closeSwitchConfirm,
      })
    );
    return actions;
  }

  function switchConfirmButton(settings) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = settings.className;
    button.title = settings.title;
    button.setAttribute("aria-label", settings.label);
    button.textContent = settings.text;
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      settings.onClick();
    });
    return button;
  }

  function sizeModelSelect(list) {
    if (list.length === 0) {
      return;
    }
    var names = [];
    for (var index = 0; index < list.length; index++) {
      names.push(list[index].display_name);
    }
    var width = measureTextWidth(names, modelSelectValue);
    modelSelect.style.minWidth =
      Math.ceil(width) + 48 + "px";
  }

  function remeasureModelSelectWhenFontReady() {
    var fonts = document.fonts;
    if (!fonts || !fonts.ready || !fonts.ready.then) {
      return;
    }
    if (modelFontReadyPending) {
      return;
    }
    modelFontReadyPending = true;
    function remeasure() {
      modelFontReadyPending = false;
      sizeModelSelect(modelList);
    }
    fonts.ready.then(remeasure).catch(remeasure);
  }

  function parameterValuesRead() {
    return runSettingsPanel.parameterValues();
  }

  function parameterDefaultsRead() {
    return runSettingsPanel.parameterDefaults();
  }

  function experimentalRead() {
    return runSettingsPanel.experimental();
  }

  function thinkingRead() {
    return runSettingsPanel.thinking();
  }

  function outputBudgetRead() {
    return runSettingsPanel.outputBudget();
  }

  function validationRead() {
    return runSettingsPanel.validation();
  }

  function conversationConfigurationRead() {
    return runSettingsPanel.snapshot();
  }

  function readParamStateAll() {
    var raw = null;
    try {
      raw = sessionStorage.getItem(PARAM_STATE_KEY);
    } catch (_error) {
      return {};
    }
    if (!raw) {
      return {};
    }
    try {
      var parsed = JSON.parse(raw);
      if (
        parsed
        && typeof parsed === "object"
        && !Array.isArray(parsed)
      ) {
        return parsed;
      }
    } catch (_error) {
      // Corrupt storage falls back to a fresh form-state record.
    }
    return {};
  }

  function readPersistedParamState(id) {
    if (id === null) {
      return null;
    }
    return readParamStateAll()[id] || null;
  }

  function writePersistedParamState(id, panelState) {
    var all = readParamStateAll();
    var state = all[id];
    if (
      !state
      || typeof state !== "object"
      || Array.isArray(state)
    ) {
      state = {};
    }
    state.experimental = panelState.experimental;
    state.params = Object.assign({}, panelState.params);
    all[id] = state;
    try {
      sessionStorage.setItem(
        PARAM_STATE_KEY, JSON.stringify(all)
      );
    } catch (_error) {
      // Quota or private mode: the form simply will not persist.
    }
  }

  function saveParamState() {
    runSettingsPanel.persist();
  }

  function refreshSelector() {
    setModelSelectValue(activeModelId);
  }

  return {
    wire: wire,
    configure: configure,
    refresh: refresh,
    setDisabled: setDisabled,
    refreshSelector: refreshSelector,
    saveDraft: saveParamState,
    activeModel: activeModelRead,
    activeModelId: activeModelIdRead,
    activeDevice: activeDeviceRead,
    activeTokenizer: activeTokenizerRead,
    activeContext: activeContextRead,
    capabilities: capabilitiesRead,
    modelDisplayName: modelDisplayName,
    activeDisplayName: activeDisplayName,
    parameterValues: parameterValuesRead,
    parameterDefaults: parameterDefaultsRead,
    experimental: experimentalRead,
    thinking: thinkingRead,
    outputBudget: outputBudgetRead,
    validation: validationRead,
    conversationConfiguration: conversationConfigurationRead,
  };
}
