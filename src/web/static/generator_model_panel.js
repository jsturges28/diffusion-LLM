// The generator's model selector and schema-driven parameter panel.
//
// Loaded as a classic script before app.js. Registry data, picker
// traversal, switch confirmation, parameter DOM references and form
// drafts stay private to the returned controller. The page owns the
// activation policy and generation semantics through callbacks and
// narrow reads.

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

  var validationHint = requiredElement("validation-hint");
  var runSettings = requiredElement("run-settings");
  var runSettingsSummary =
    requiredElement("run-settings-summary");
  var runSettingsSummaryChips =
    requiredElement("run-settings-summary-chips");
  var toggleExperimental =
    requiredElement("toggle-experimental");
  var btnParamDefaults = requiredElement("btn-param-defaults");
  var modelSelect = requiredElement("model-select");
  var modelSelectValue = requiredElement("model-select-value");
  var modelSelectList = requiredElement("model-select-list");
  var paramFields = requiredElement("param-fields");
  var modeExtra = requiredElement("mode-extra");

  var PARAM_STATE_KEY = "diffusion_param_state";
  var MODEL_OPTION_ID_PREFIX = "model-select-option-";
  var DEVICE_LABELS = { cuda: "GPU", cpu: "CPU" };
  var PARAM_GROUP_LABELS = {
    general: "General",
    output: "Output",
    sampling: "Sampling",
    features: "Features",
  };

  var models = {};
  var modelList = [];
  var activeModelId = null;
  var activeModel = null;
  var activeDevice = null;
  var activeTokenizer = {};
  var activeContextLength = null;
  var gpuPresent = false;

  var paramInputs = {};
  var paramTooltips = {};
  var paramGroupMounts = {};
  var modeGroupMounts = {};
  var paramsValid = true;
  var modelSelectDisabled = false;
  var modelActiveRow = -1;
  var switchConfirmEl = null;
  var collapsedTickerTimer = null;
  var wired = false;

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    toggleExperimental.addEventListener(
      "change", experimentalChanged
    );
    btnParamDefaults.addEventListener(
      "click", resetParamsToDefaults
    );
    runSettings.addEventListener(
      "toggle", syncRunSettingsExpanded
    );
    syncRunSettingsExpanded();
    wireModelPicker();
  }

  function syncRunSettingsExpanded() {
    runSettingsSummary.setAttribute(
      "aria-expanded", runSettings.open ? "true" : "false"
    );
  }

  function experimentalChanged() {
    applyLimits();
    paramFormChanged();
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
    if (!activeModel) {
      paramInputs = {};
      paramTooltips = {};
      paramGroupMounts = {};
      modeGroupMounts = {};
      paramFields.innerHTML = "";
      modeExtra.innerHTML = "";
      runSettingsSummaryChips.innerHTML = "";
      updateParamDefaultsButton();
      return;
    }
    buildParamPanel(activeModel);
    applyUniformParamWidth(modelList);
    remeasureWhenFontReady();
    restoreParamState();
    updateParamDefaultsButton();
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
    toggleExperimental.disabled = disabled;
    if (disabled) {
      closeModelList();
    }
    var names = Object.keys(paramInputs);
    for (var index = 0; index < names.length; index++) {
      paramInputs[names[index]].disabled = disabled;
    }
    updateParamDefaultsButton();
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

  function numericSpecs() {
    var result = [];
    if (!activeModel) {
      return result;
    }
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      if (
        specs[index].type === "int"
        || specs[index].type === "float"
      ) {
        result.push(specs[index]);
      }
    }
    return result;
  }

  function specOverride(spec) {
    if (
      spec.overrides
      && activeDevice
      && spec.overrides[activeDevice]
    ) {
      return spec.overrides[activeDevice];
    }
    return null;
  }

  function specBounds(spec, experimental) {
    var override = specOverride(spec);
    if (override) {
      var bounds = experimental
        ? override.experimental
        : override.recommended;
      if (bounds) {
        return bounds;
      }
    }
    return experimental ? spec.experimental : spec.recommended;
  }

  function specDefault(spec) {
    var override = specOverride(spec);
    if (
      override
      && override.default !== null
      && override.default !== undefined
    ) {
      return override.default;
    }
    return spec.default;
  }

  function activeLimits() {
    var result = {};
    if (!activeModel) {
      return result;
    }
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      var bounds = specBounds(
        specs[index], toggleExperimental.checked
      );
      if (bounds) {
        result[specs[index].name] = {
          min: bounds[0],
          max: bounds[1],
        };
      }
    }
    return result;
  }

  function buildInfoIcon(spec) {
    var info = document.createElement("span");
    info.className = "info-icon info-icon-sm";
    info.textContent = "?";
    info.setAttribute("aria-label", spec.label + " info");
    var tip = document.createElement("span");
    tip.className = "tooltip";
    info.appendChild(tip);
    info.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
    });
    paramTooltips[spec.name] = tip;
    return info;
  }

  function buildParamField(spec, input, mount) {
    var group = document.createElement("div");
    group.className = "param-group";
    var label = document.createElement("label");
    label.setAttribute("for", "param-" + spec.name);
    label.appendChild(document.createTextNode(spec.label));
    label.appendChild(buildInfoIcon(spec));
    group.appendChild(label);
    group.appendChild(input);
    mount.appendChild(group);
  }

  function buildModeToggle(spec, checkbox, mount) {
    var wrap = document.createElement("span");
    wrap.className = "mode-toggle";
    var toggle = document.createElement("label");
    toggle.className = "toggle-switch";
    var slider = document.createElement("span");
    slider.className = "toggle-slider";
    toggle.appendChild(checkbox);
    toggle.appendChild(slider);
    var name = document.createElement("span");
    name.className = "toggle-label";
    name.textContent = spec.label;
    wrap.appendChild(toggle);
    wrap.appendChild(name);
    wrap.appendChild(buildInfoIcon(spec));
    mount.appendChild(wrap);
  }

  function buildParamPanel(model) {
    paramInputs = {};
    paramTooltips = {};
    paramGroupMounts = {};
    modeGroupMounts = {};
    paramFields.innerHTML = "";
    modeExtra.innerHTML = "";
    var specs = model.param_specs;
    for (var index = 0; index < specs.length; index++) {
      appendParam(specs[index]);
    }
    applyLimits();
  }

  function appendParam(spec) {
    var input = buildParamInput(spec);
    paramInputs[spec.name] = input;
    if (spec.type === "bool") {
      buildModeToggle(spec, input, modeGroupMount(spec));
    } else {
      buildParamField(spec, input, paramGroupMount(spec));
    }
    var eventName = (
      spec.type === "int" || spec.type === "float"
    ) ? "input" : "change";
    input.addEventListener(eventName, function () {
      validateAllParams();
      paramFormChanged();
    });
  }

  function specGroup(spec) {
    var group = typeof spec.group === "string"
      ? spec.group
      : "general";
    return PARAM_GROUP_LABELS[group] ? group : "general";
  }

  function paramGroupMount(spec) {
    var group = specGroup(spec);
    if (!paramGroupMounts[group]) {
      paramGroupMounts[group] = buildControlGroup(
        paramFields, group, "run-settings-group"
      );
    }
    return paramGroupMounts[group];
  }

  function modeGroupMount(spec) {
    var group = specGroup(spec);
    if (!modeGroupMounts[group]) {
      modeGroupMounts[group] = buildControlGroup(
        modeExtra, group, "run-settings-mode-group"
      );
    }
    return modeGroupMounts[group];
  }

  function buildControlGroup(host, group, className) {
    var section = document.createElement("section");
    section.className = className;
    section.setAttribute("data-param-group", group);
    var heading = document.createElement("h3");
    heading.className = "run-settings-group-label";
    heading.textContent = PARAM_GROUP_LABELS[group];
    var controls = document.createElement("div");
    controls.className = "run-settings-group-controls";
    section.appendChild(heading);
    section.appendChild(controls);
    host.appendChild(section);
    return controls;
  }

  function buildParamInput(spec) {
    var input;
    if (spec.type === "select") {
      var options = (spec.options || []).map(function (value) {
        return { value: value, label: prettifyOption(value) };
      });
      input = createCustomSelect(options, spec.default);
    } else if (spec.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(specDefault(spec));
    } else {
      input = document.createElement("input");
      input.type = "number";
      if (spec.step !== null && spec.step !== undefined) {
        input.step = String(spec.step);
      }
      input.value = String(specDefault(spec));
    }
    input.id = "param-" + spec.name;
    input.disabled = modelSelectDisabled;
    return input;
  }

  function paramRangeText(spec, limits) {
    if (spec.type === "select") {
      return (spec.options || []).map(prettifyOption).join(" / ");
    }
    if (spec.type === "bool") {
      return "on / off";
    }
    var bounds = limits[spec.name];
    if (bounds) {
      return "(" + bounds.min + "\u2013" + bounds.max + ")";
    }
    return "";
  }

  function updateRangeLabels() {
    if (!activeModel) {
      return;
    }
    var limits = activeLimits();
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      updateRangeLabel(specs[index], limits);
    }
  }

  function updateRangeLabel(spec, limits) {
    var tip = paramTooltips[spec.name];
    if (!tip) {
      return;
    }
    tip.innerHTML = "";
    var rangeLine = document.createElement("div");
    var emphasis = document.createElement("em");
    emphasis.textContent = "Range:";
    rangeLine.appendChild(emphasis);
    rangeLine.appendChild(document.createTextNode(
      " " + paramRangeText(spec, limits)
    ));
    tip.appendChild(rangeLine);
    if (spec.help) {
      var description = document.createElement("div");
      description.className = "tooltip-desc";
      description.textContent = spec.help;
      tip.appendChild(description);
    }
  }

  function remeasureWhenFontReady() {
    var fonts = document.fonts;
    if (!fonts || !fonts.ready || !fonts.ready.then) {
      return;
    }
    function remeasure() {
      applyUniformParamWidth(modelList);
      sizeModelSelect(modelList);
    }
    fonts.ready.then(remeasure).catch(remeasure);
  }

  function applyUniformParamWidth(allModels) {
    var refLabel = paramFields.querySelector("label");
    if (!refLabel) {
      return;
    }
    var refControl =
      paramFields.querySelector("input, .custom-select")
      || refLabel;
    var maxWidth = widestParamWidth({
      models: allModels,
      label: refLabel,
      control: refControl,
    });
    document.documentElement.style.setProperty(
      "--param-width", Math.ceil(maxWidth) + "px"
    );
  }

  function widestParamWidth(settings) {
    var letterSpacing = 0.8;
    var maxWidth = 90;
    for (var modelIndex = 0;
      modelIndex < settings.models.length;
      modelIndex++
    ) {
      var specs =
        settings.models[modelIndex].param_specs || [];
      for (var specIndex = 0;
        specIndex < specs.length;
        specIndex++
      ) {
        maxWidth = Math.max(
          maxWidth,
          paramWidth({
            spec: specs[specIndex],
            label: settings.label,
            control: settings.control,
            letterSpacing: letterSpacing,
          })
        );
      }
    }
    return maxWidth;
  }

  function paramWidth(settings) {
    if (settings.spec.type === "bool") {
      return 0;
    }
    var upper = String(settings.spec.label).toUpperCase();
    var labelWidth =
      measureTextWidth([upper], settings.label)
      + settings.letterSpacing
        * Math.max(0, upper.length - 1)
      + 26;
    if (settings.spec.type !== "select") {
      return labelWidth;
    }
    var options =
      (settings.spec.options || []).map(prettifyOption);
    var optionWidth =
      measureTextWidth(options, settings.control) + 40;
    return Math.max(labelWidth, optionWidth);
  }

  function applyLimits() {
    var limits = activeLimits();
    var names = Object.keys(limits);
    for (var index = 0; index < names.length; index++) {
      applyInputLimit(
        paramInputs[names[index]], limits[names[index]]
      );
    }
    updateRangeLabels();
    validateAllParams();
  }

  function applyInputLimit(input, bounds) {
    if (!input || input.type !== "number") {
      return;
    }
    input.min = bounds.min;
    input.max = bounds.max;
    var value = parseFloat(input.value);
    if (isNaN(value)) {
      return;
    }
    if (value < bounds.min) {
      input.value = bounds.min;
    } else if (value > bounds.max) {
      input.value = bounds.max;
    }
  }

  function validateAllParams() {
    var limits = activeLimits();
    var errors = [];
    var specs = numericSpecs();
    for (var index = 0; index < specs.length; index++) {
      var input = paramInputs[specs[index].name];
      if (input) {
        input.classList.remove("input-warn");
      }
    }
    for (var check = 0; check < specs.length; check++) {
      validateNumericParam({
        spec: specs[check],
        bounds: limits[specs[check].name],
        errors: errors,
      });
    }
    validateDivisibility(errors);
    setValidation(errors);
    updateSummaryChips();
  }

  function validateNumericParam(settings) {
    var input = paramInputs[settings.spec.name];
    if (!input) {
      return;
    }
    var raw = input.value.trim();
    var value = parseFloat(raw);
    if (raw === "" || isNaN(value)) {
      input.classList.add("input-warn");
      settings.errors.push(
        settings.spec.label + " is empty or invalid."
      );
      return;
    }
    if (settings.bounds && value < settings.bounds.min) {
      input.classList.add("input-warn");
      settings.errors.push(
        value < 0
          ? settings.spec.label + " cannot be negative."
          : settings.spec.label + " must be at least "
            + settings.bounds.min + "."
      );
      return;
    }
    if (settings.bounds && value > settings.bounds.max) {
      input.classList.add("input-warn");
      settings.errors.push(
        settings.spec.label + " must be at most "
        + settings.bounds.max + "."
      );
    }
  }

  function setValidation(errors) {
    paramsValid = errors.length === 0;
    validationHint.hidden = paramsValid;
    validationHint.textContent = paramsValid ? "" : errors[0];
    if (!paramsValid && !runSettings.open) {
      revealFirstInvalidControl();
    }
    onValidationChanged(validationRead());
  }

  function revealFirstInvalidControl() {
    runSettings.open = true;
    syncRunSettingsExpanded();
    var first = paramFields.querySelector(".input-warn");
    if (!first) {
      first = modeExtra.querySelector(".input-warn");
    }
    if (first && typeof first.focus === "function") {
      first.focus();
    }
  }

  function updateSummaryChips() {
    runSettingsSummaryChips.innerHTML = "";
    if (!activeModel) {
      return;
    }
    var specs = activeModel.param_specs || [];
    for (var index = 0; index < specs.length; index++) {
      if (specs[index].prominence === "primary") {
        appendSummaryChip(specs[index]);
      }
    }
  }

  function appendSummaryChip(spec) {
    var input = paramInputs[spec.name];
    if (!input) {
      return;
    }
    var chip = document.createElement("span");
    chip.className = "run-settings-chip";
    chip.setAttribute("data-param-name", spec.name);
    if (input.classList.contains("input-warn")) {
      chip.classList.add("is-invalid");
    }
    var label = document.createElement("span");
    label.className = "run-settings-chip-label";
    label.textContent = spec.label;
    var value = document.createElement("span");
    value.className = "run-settings-chip-value";
    value.textContent = summaryParamValue(spec, input);
    chip.appendChild(label);
    chip.appendChild(value);
    runSettingsSummaryChips.appendChild(chip);
  }

  function summaryParamValue(spec, input) {
    if (spec.type === "bool") {
      return input.checked ? "On" : "Off";
    }
    if (spec.type === "select") {
      return prettifyOption(input.value);
    }
    return input.value;
  }

  function validateDivisibility(errors) {
    var genInput = paramInputs.gen_length;
    var blockInput = paramInputs.block_length;
    var stepsInput = paramInputs.steps;
    if (!genInput || !blockInput || !stepsInput) {
      return;
    }
    var genLength = parseInt(genInput.value, 10);
    var blockLength = parseInt(blockInput.value, 10);
    var steps = parseInt(stepsInput.value, 10);
    var genOkay =
      !genInput.classList.contains("input-warn");
    var blockOkay =
      !blockInput.classList.contains("input-warn");
    var stepsOkay =
      !stepsInput.classList.contains("input-warn");
    if (
      genOkay
      && blockOkay
      && blockLength > 0
      && genLength % blockLength !== 0
    ) {
      addBlockLengthError({
        errors: errors,
        genInput: genInput,
        blockInput: blockInput,
        genLength: genLength,
        blockLength: blockLength,
      });
      return;
    }
    addStepsDivisibilityError({
      errors: errors,
      input: stepsInput,
      genLength: genLength,
      blockLength: blockLength,
      valuesOkay: genOkay && blockOkay && stepsOkay,
    });
  }

  function addBlockLengthError(settings) {
    settings.genInput.classList.add("input-warn");
    settings.blockInput.classList.add("input-warn");
    settings.errors.push(
      "Gen Length (" + settings.genLength
      + ") must be divisible by Block Length ("
      + settings.blockLength + ")."
    );
  }

  function addStepsDivisibilityError(settings) {
    if (!settings.valuesOkay || settings.blockLength <= 0) {
      return;
    }
    if (settings.genLength % settings.blockLength !== 0) {
      return;
    }
    var numBlocks =
      settings.genLength / settings.blockLength;
    var steps = parseInt(settings.input.value, 10);
    if (numBlocks > 0 && steps % numBlocks !== 0) {
      settings.input.classList.add("input-warn");
      settings.errors.push(
        "Steps (" + steps
        + ") must be divisible by num_blocks ("
        + numBlocks + ")."
      );
    }
  }

  function parameterValuesRead() {
    var result = {};
    if (!activeModel) {
      return result;
    }
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      var spec = specs[index];
      var input = paramInputs[spec.name];
      if (input) {
        result[spec.name] = parsedParamValue(spec, input);
      }
    }
    return result;
  }

  function parsedParamValue(spec, input) {
    if (spec.type === "int") {
      return parseInt(input.value, 10);
    }
    if (spec.type === "float") {
      return parseFloat(input.value);
    }
    if (spec.type === "bool") {
      return input.checked;
    }
    return input.value;
  }

  function parameterDefaultsRead() {
    var defaults = {};
    if (!activeModel) {
      return defaults;
    }
    var specs = activeModel.param_specs || [];
    for (var index = 0; index < specs.length; index++) {
      defaults[specs[index].name] = specDefault(specs[index]);
    }
    return defaults;
  }

  function experimentalRead() {
    return toggleExperimental.checked;
  }

  function thinkingRead() {
    return Boolean(parameterValuesRead().thinking);
  }

  function outputBudgetRead() {
    var values = parameterValuesRead();
    var budget = values.gen_length;
    if (typeof budget !== "number" || !isFinite(budget)) {
      budget = values.max_new_tokens;
    }
    if (typeof budget !== "number" || !isFinite(budget)) {
      return 0;
    }
    return Math.max(0, Math.round(budget));
  }

  function validationRead() {
    return {
      valid: paramsValid,
      message: validationHint.textContent,
    };
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

  function currentParamRawValues() {
    var result = {};
    var names = Object.keys(paramInputs);
    for (var index = 0; index < names.length; index++) {
      var input = paramInputs[names[index]];
      result[names[index]] = input.type === "checkbox"
        ? input.checked
        : input.value;
    }
    return result;
  }

  function saveParamState() {
    if (activeModelId === null) {
      return;
    }
    var all = readParamStateAll();
    var state = all[activeModelId];
    if (
      !state
      || typeof state !== "object"
      || Array.isArray(state)
    ) {
      state = {};
    }
    state.experimental = toggleExperimental.checked;
    state.params = currentParamRawValues();
    all[activeModelId] = state;
    try {
      sessionStorage.setItem(
        PARAM_STATE_KEY, JSON.stringify(all)
      );
    } catch (_error) {
      // Quota or private mode: the form simply will not persist.
    }
  }

  function paramFormChanged() {
    saveParamState();
    updateParamDefaultsButton();
    onParametersChanged();
  }

  function restoreParamState() {
    if (activeModelId === null) {
      return;
    }
    var state = readParamStateAll()[activeModelId];
    if (!state) {
      return;
    }
    toggleExperimental.checked = !!state.experimental;
    applyParamRawValues(state.params);
    applyLimits();
  }

  function applyParamRawValues(values) {
    if (!values || !activeModel) {
      return;
    }
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      var stored = values[specs[index].name];
      if (stored !== undefined) {
        applyParamRawValue(specs[index], stored);
      }
    }
  }

  function applyParamRawValue(spec, stored) {
    var input = paramInputs[spec.name];
    if (!input) {
      return;
    }
    if (spec.type === "bool") {
      input.checked = !!stored;
    } else if (spec.type === "select") {
      if ((spec.options || []).indexOf(stored) >= 0) {
        input.value = stored;
      }
    } else {
      input.value = String(stored);
    }
  }

  function paramsAtDefaults() {
    if (toggleExperimental.checked) {
      return false;
    }
    if (!activeModel) {
      return true;
    }
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      if (!paramAtDefault(specs[index])) {
        return false;
      }
    }
    return true;
  }

  function paramAtDefault(spec) {
    var input = paramInputs[spec.name];
    if (!input) {
      return true;
    }
    if (spec.type === "bool") {
      return input.checked === Boolean(specDefault(spec));
    }
    if (spec.type === "select") {
      return input.value === spec.default;
    }
    return input.value === String(specDefault(spec));
  }

  function resetParamsToDefaults() {
    if (!activeModel || modelSelectDisabled) {
      return;
    }
    toggleExperimental.checked = false;
    var specs = activeModel.param_specs;
    for (var index = 0; index < specs.length; index++) {
      resetParamToDefault(specs[index]);
    }
    applyLimits();
    paramFormChanged();
  }

  function resetParamToDefault(spec) {
    var input = paramInputs[spec.name];
    if (!input) {
      return;
    }
    if (spec.type === "bool") {
      input.checked = Boolean(specDefault(spec));
    } else if (spec.type === "select") {
      input.value = spec.default;
    } else {
      input.value = String(specDefault(spec));
    }
  }

  function updateParamDefaultsButton() {
    btnParamDefaults.disabled =
      modelSelectDisabled || paramsAtDefaults();
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
  };
}
