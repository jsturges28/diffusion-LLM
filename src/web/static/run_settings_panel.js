// Mountable schema-driven Run settings disclosure.
//
// One instance owns one details element and all controls below it.
// Draft adopts the shipped subtree for first-paint compatibility;
// action surfaces can provide a mount and unique id prefix instead.

"use strict";

function runSettingsPanelCreate(options) {
  var settings = options || {};
  var core = runSettingsCoreCreate();
  var GROUP_LABELS = {
    general: "General",
    output: "Output",
    sampling: "Sampling",
    features: "Features",
    signals: "Signals",
  };

  function optionalCallback(name) {
    var callback = settings[name];
    if (callback === undefined || callback === null) {
      return function () {};
    }
    if (typeof callback !== "function") {
      throw new TypeError(name + " must be a function");
    }
    return callback;
  }

  function optionalPersistence(name) {
    var callback = settings[name];
    if (callback === undefined || callback === null) {
      return null;
    }
    if (typeof callback !== "function") {
      throw new TypeError(name + " must be a function");
    }
    return callback;
  }

  var onValidationChanged =
    optionalCallback("onValidationChanged");
  var onParametersChanged =
    optionalCallback("onParametersChanged");
  var readPersistedState =
    optionalPersistence("readPersistedState");
  var writePersistedState =
    optionalPersistence("writePersistedState");
  var idPrefix = settings.idPrefix || "";

  if (typeof idPrefix !== "string") {
    throw new TypeError("idPrefix must be a string");
  }
  if (settings.root && settings.mount) {
    throw new TypeError(
      "Run settings needs root or mount, not both"
    );
  }
  if (!settings.root && !settings.mount) {
    throw new TypeError(
      "Run settings needs an existing root or a mount"
    );
  }
  if (!settings.root && idPrefix === "") {
    throw new TypeError(
      "Mounted Run settings needs a unique idPrefix"
    );
  }

  function elementId(name) {
    return idPrefix + name;
  }

  function requiredElement(name) {
    var id = elementId(name);
    var element = document.getElementById(id);
    if (!element) {
      throw new Error("Missing Run settings element #" + id);
    }
    return element;
  }

  function adoptDisclosure(root) {
    return {
      root: root,
      summary: requiredElement("run-settings-summary"),
      chips: requiredElement(
        "run-settings-summary-chips"
      ),
      body: requiredElement("run-settings-body"),
      modeRow: requiredElement("mode-row"),
      experimental: requiredElement(
        "toggle-experimental"
      ),
      reset: requiredElement("btn-param-defaults"),
      modeExtra: requiredElement("mode-extra"),
      paramRow: requiredElement("param-row"),
      fields: requiredElement("param-fields"),
      validation: requiredElement("validation-hint"),
    };
  }

  function appendText(parent, text) {
    parent.appendChild(document.createTextNode(text));
  }

  function buildSummary() {
    var summary = document.createElement("summary");
    summary.id = elementId("run-settings-summary");
    summary.className = "run-settings-summary";
    summary.setAttribute(
      "aria-controls", elementId("run-settings-body")
    );
    summary.setAttribute("aria-expanded", "false");
    var title = document.createElement("span");
    title.className = "run-settings-title";
    title.textContent = settings.title || "Run settings";
    var chips = document.createElement("span");
    chips.id = elementId("run-settings-summary-chips");
    chips.className = "run-settings-summary-chips";
    var chevron = document.createElement("span");
    chevron.className = "run-settings-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "\u25be";
    summary.appendChild(title);
    summary.appendChild(chips);
    summary.appendChild(chevron);
    return { summary: summary, chips: chips };
  }

  function buildExperimentalInfo() {
    var info = document.createElement("span");
    info.className = "info-icon info-icon-sm";
    info.setAttribute(
      "aria-label", "Experimental mode info"
    );
    info.textContent = "?";
    var tooltip = document.createElement("span");
    tooltip.className = "tooltip";
    tooltip.textContent =
      "Reveals experimental controls and removes recommended "
      + "parameter bounds. Extreme values may produce unstable "
      + "results.";
    info.appendChild(tooltip);
    info.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
    });
    return info;
  }

  function buildResetButton() {
    var button = document.createElement("button");
    button.id = elementId("btn-param-defaults");
    button.className = "run-settings-reset";
    button.type = "button";
    button.disabled = true;
    button.title = "Reset hyperparameters to their defaults";
    button.setAttribute(
      "aria-label",
      "Reset hyperparameters to their defaults"
    );
    var icon = document.createElement("span");
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "\u21ba";
    var text = document.createElement("span");
    text.textContent = "Reset";
    button.appendChild(icon);
    button.appendChild(text);
    return button;
  }

  function buildModeRow() {
    var row = document.createElement("div");
    row.id = elementId("mode-row");
    row.className = "run-settings-mode-row";
    var toggleLabel = document.createElement("label");
    toggleLabel.className = "toggle-switch";
    var experimental = document.createElement("input");
    experimental.type = "checkbox";
    experimental.id = elementId("toggle-experimental");
    var slider = document.createElement("span");
    slider.className = "toggle-slider";
    toggleLabel.appendChild(experimental);
    toggleLabel.appendChild(slider);
    var name = document.createElement("label");
    name.className = "toggle-label";
    name.setAttribute("for", experimental.id);
    name.textContent = "Experimental";
    var reset = buildResetButton();
    row.appendChild(toggleLabel);
    row.appendChild(name);
    row.appendChild(buildExperimentalInfo());
    row.appendChild(reset);
    return {
      row: row,
      experimental: experimental,
      reset: reset,
    };
  }

  function buildBody() {
    var body = document.createElement("div");
    body.id = elementId("run-settings-body");
    body.className = "run-settings-body";
    var mode = buildModeRow();
    var modeExtra = document.createElement("div");
    modeExtra.id = elementId("mode-extra");
    modeExtra.className = "run-settings-mode-extra";
    var paramRow = document.createElement("div");
    paramRow.id = elementId("param-row");
    paramRow.className = "run-settings-param-row";
    var fields = document.createElement("div");
    fields.id = elementId("param-fields");
    fields.className =
      "param-fields run-settings-param-fields";
    paramRow.appendChild(fields);
    var validation = document.createElement("div");
    validation.id = elementId("validation-hint");
    validation.className = "run-settings-validation-hint";
    validation.setAttribute("role", "status");
    validation.setAttribute("aria-live", "polite");
    validation.hidden = true;
    body.appendChild(mode.row);
    body.appendChild(modeExtra);
    body.appendChild(paramRow);
    body.appendChild(validation);
    return {
      body: body,
      modeRow: mode.row,
      experimental: mode.experimental,
      reset: mode.reset,
      modeExtra: modeExtra,
      paramRow: paramRow,
      fields: fields,
      validation: validation,
    };
  }

  function buildDisclosure(mount) {
    var root = document.createElement("details");
    root.id = elementId("run-settings");
    var summary = buildSummary();
    var body = buildBody();
    root.appendChild(summary.summary);
    root.appendChild(body.body);
    mount.appendChild(root);
    return {
      root: root,
      summary: summary.summary,
      chips: summary.chips,
      body: body.body,
      modeRow: body.modeRow,
      experimental: body.experimental,
      reset: body.reset,
      modeExtra: body.modeExtra,
      paramRow: body.paramRow,
      fields: body.fields,
      validation: body.validation,
    };
  }

  var builtRoot = !settings.root;
  var refs = settings.root
    ? adoptDisclosure(settings.root)
    : buildDisclosure(settings.mount);
  var widthTarget =
    settings.widthTarget || refs.root;

  function decorateDisclosure() {
    refs.root.classList.add("run-settings");
    refs.summary.classList.add("run-settings-summary");
    refs.chips.classList.add(
      "run-settings-summary-chips"
    );
    refs.body.classList.add("run-settings-body");
    refs.modeRow.classList.add("run-settings-mode-row");
    refs.reset.classList.add("run-settings-reset");
    refs.modeExtra.classList.add(
      "run-settings-mode-extra"
    );
    refs.paramRow.classList.add(
      "run-settings-param-row"
    );
    refs.fields.classList.add(
      "run-settings-param-fields"
    );
    refs.validation.classList.add(
      "run-settings-validation-hint"
    );
    if (settings.compact === true) {
      refs.root.classList.add("run-settings-compact");
    }
  }

  decorateDisclosure();

  var model = null;
  var modelId = null;
  var modelDisplay = "";
  var device = null;
  var inputMode = null;
  var allModels = [];
  var inputs = {};
  var tooltips = {};
  var controls = {};
  var paramGroupMounts = {};
  var modeGroupMounts = {};
  var validationState = Object.freeze({
    valid: true,
    message: "",
    errors: Object.freeze([]),
    invalidNames: Object.freeze([]),
  });
  var inputListeners = [];
  var wired = false;
  var disabled = false;
  var destroyed = false;
  var fontReadyPending = false;

  function ensureAlive() {
    if (destroyed) {
      throw new Error("Run settings panel is destroyed");
    }
  }

  function listen(element, eventName, callback, target) {
    element.addEventListener(eventName, callback);
    target.push({
      element: element,
      eventName: eventName,
      callback: callback,
    });
  }

  var rootListeners = [];

  function wire() {
    ensureAlive();
    if (wired) {
      return;
    }
    wired = true;
    listen(
      refs.experimental,
      "change",
      experimentalChanged,
      rootListeners
    );
    listen(
      refs.reset,
      "click",
      resetToDefaults,
      rootListeners
    );
    listen(
      refs.root,
      "toggle",
      syncExpanded,
      rootListeners
    );
    syncExpanded();
  }

  function syncExpanded() {
    refs.summary.setAttribute(
      "aria-expanded", refs.root.open ? "true" : "false"
    );
  }

  function experimentalChanged() {
    applyLimits();
    formChanged();
  }

  function clearListeners(records) {
    for (var index = 0; index < records.length; index++) {
      var record = records[index];
      record.element.removeEventListener(
        record.eventName, record.callback
      );
    }
    records.length = 0;
  }

  function clearControls() {
    clearListeners(inputListeners);
    var names = Object.keys(inputs);
    for (var index = 0; index < names.length; index++) {
      inputs[names[index]].disabled = true;
    }
    inputs = {};
    tooltips = {};
    controls = {};
    paramGroupMounts = {};
    modeGroupMounts = {};
    refs.fields.innerHTML = "";
    refs.modeExtra.innerHTML = "";
    refs.chips.innerHTML = "";
  }

  function specGroup(spec) {
    var group = typeof spec.group === "string"
      ? spec.group
      : "general";
    return GROUP_LABELS[group] ? group : "general";
  }

  function buildControlGroup(host, group, className) {
    var section = document.createElement("section");
    section.className = className;
    section.setAttribute("data-param-group", group);
    var heading = document.createElement("h3");
    heading.className = "run-settings-group-label";
    heading.textContent = GROUP_LABELS[group];
    var mount = document.createElement("div");
    mount.className = "run-settings-group-controls";
    section.appendChild(heading);
    section.appendChild(mount);
    host.appendChild(section);
    mount._section = section;
    return mount;
  }

  function paramGroupMount(spec) {
    var group = specGroup(spec);
    if (!paramGroupMounts[group]) {
      paramGroupMounts[group] = buildControlGroup(
        refs.fields, group, "run-settings-group"
      );
    }
    return paramGroupMounts[group];
  }

  function modeGroupMount(spec) {
    var group = specGroup(spec);
    if (!modeGroupMounts[group]) {
      modeGroupMounts[group] = buildControlGroup(
        refs.modeExtra,
        group,
        "run-settings-mode-group"
      );
    }
    return modeGroupMounts[group];
  }

  function selectOptions(spec) {
    var result = [];
    var options = spec.options || [];
    for (var index = 0; index < options.length; index++) {
      result.push({
        value: options[index],
        label: core.optionLabel(options[index]),
      });
    }
    return result;
  }

  function buildInput(spec) {
    var input;
    var initial = core.defaultValue(spec, device);
    if (spec.type === "select") {
      input = createCustomSelect(
        selectOptions(spec), initial
      );
    } else if (spec.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(initial);
    } else {
      input = document.createElement("input");
      input.type = "number";
      if (spec.step !== null && spec.step !== undefined) {
        input.step = String(spec.step);
      }
      input.value = String(initial);
    }
    input.id = elementId("param-" + spec.name);
    input.setAttribute("aria-label", spec.label);
    input.disabled = disabled;
    return input;
  }

  function buildInfoIcon(spec) {
    var info = document.createElement("span");
    info.className = "info-icon info-icon-sm";
    info.textContent = "?";
    info.setAttribute("aria-label", spec.label + " info");
    var tooltip = document.createElement("span");
    tooltip.className = "tooltip";
    info.appendChild(tooltip);
    info.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
    });
    tooltips[spec.name] = tooltip;
    return info;
  }

  function buildParamField(spec, input) {
    var group = document.createElement("div");
    group.className = "param-group";
    var label = document.createElement("label");
    label.setAttribute("for", input.id);
    appendText(label, spec.label);
    label.appendChild(buildInfoIcon(spec));
    group.appendChild(label);
    group.appendChild(input);
    paramGroupMount(spec).appendChild(group);
    return group;
  }

  function buildModeToggle(spec, input) {
    var wrap = document.createElement("span");
    wrap.className = "mode-toggle";
    var toggle = document.createElement("label");
    toggle.className = "toggle-switch";
    var slider = document.createElement("span");
    slider.className = "toggle-slider";
    toggle.appendChild(input);
    toggle.appendChild(slider);
    var name = document.createElement("label");
    name.className = "toggle-label";
    name.setAttribute("for", input.id);
    name.textContent = spec.label;
    wrap.appendChild(toggle);
    wrap.appendChild(name);
    wrap.appendChild(buildInfoIcon(spec));
    modeGroupMount(spec).appendChild(wrap);
    return wrap;
  }

  function appendParam(spec) {
    var input = buildInput(spec);
    inputs[spec.name] = input;
    var control = spec.type === "bool"
      ? buildModeToggle(spec, input)
      : buildParamField(spec, input);
    control.setAttribute(
      "data-experimental-only",
      spec.experimental_only ? "true" : "false"
    );
    controls[spec.name] = control;
    var eventName = (
      spec.type === "int" || spec.type === "float"
    ) ? "input" : "change";
    listen(input, eventName, function () {
      validateAndRender();
      formChanged();
    }, inputListeners);
  }

  function specsRead() {
    return model && Array.isArray(model.param_specs)
      ? model.param_specs
      : [];
  }

  function buildControls() {
    clearControls();
    var specs = specsRead();
    for (var index = 0; index < specs.length; index++) {
      appendParam(specs[index]);
    }
    applyLimits();
  }

  function currentRawValues() {
    var result = {};
    var names = Object.keys(inputs);
    for (var index = 0; index < names.length; index++) {
      var input = inputs[names[index]];
      result[names[index]] = input.type === "checkbox"
        ? input.checked
        : input.value;
    }
    return result;
  }

  function applyControlVisibility() {
    var specs = specsRead();
    for (var index = 0; index < specs.length; index++) {
      var control = controls[specs[index].name];
      if (control) {
        control.hidden = !core.included(
          specs[index], refs.experimental.checked
        );
      }
    }
    updateGroupVisibility(paramGroupMounts);
    updateGroupVisibility(modeGroupMounts);
  }

  function updateGroupVisibility(mounts) {
    var names = Object.keys(mounts);
    for (var index = 0; index < names.length; index++) {
      var mount = mounts[names[index]];
      var visible = false;
      for (
        var child = 0;
        child < mount.children.length;
        child++
      ) {
        if (!mount.children[child].hidden) {
          visible = true;
          break;
        }
      }
      if (mount._section) {
        mount._section.hidden = !visible;
      }
    }
  }

  function applyInputLimit(spec) {
    var input = inputs[spec.name];
    if (!input || input.type !== "number") {
      return;
    }
    var range = core.bounds(
      spec, device, refs.experimental.checked
    );
    if (!range) {
      input.min = "";
      input.max = "";
      return;
    }
    input.min = range.min;
    input.max = range.max;
    input.value = core.clampRawValue(
      spec,
      input.value,
      device,
      refs.experimental.checked
    );
  }

  function applyLimits() {
    applyControlVisibility();
    var specs = specsRead();
    for (var index = 0; index < specs.length; index++) {
      applyInputLimit(specs[index]);
    }
    updateTooltips();
    validateAndRender();
  }

  function rangeText(spec) {
    if (spec.type === "select") {
      var labels = [];
      var options = spec.options || [];
      for (var index = 0; index < options.length; index++) {
        labels.push(core.optionLabel(options[index]));
      }
      return labels.join(" / ");
    }
    if (spec.type === "bool") {
      return "on / off";
    }
    var range = core.bounds(
      spec, device, refs.experimental.checked
    );
    return range
      ? "(" + range.min + "\u2013" + range.max + ")"
      : "";
  }

  function updateTooltip(spec) {
    var tooltip = tooltips[spec.name];
    if (!tooltip) {
      return;
    }
    tooltip.innerHTML = "";
    var rangeLine = document.createElement("div");
    var emphasis = document.createElement("em");
    emphasis.textContent = "Range:";
    rangeLine.appendChild(emphasis);
    appendText(rangeLine, " " + rangeText(spec));
    tooltip.appendChild(rangeLine);
    if (spec.help) {
      var description = document.createElement("div");
      description.className = "tooltip-desc";
      description.textContent = spec.help;
      tooltip.appendChild(description);
    }
  }

  function updateTooltips() {
    var specs = specsRead();
    for (var index = 0; index < specs.length; index++) {
      updateTooltip(specs[index]);
    }
  }

  function clearWarnings() {
    var names = Object.keys(inputs);
    for (var index = 0; index < names.length; index++) {
      inputs[names[index]].classList.remove("input-warn");
    }
  }

  function applyWarnings(invalidNames) {
    clearWarnings();
    for (
      var index = 0;
      index < invalidNames.length;
      index++
    ) {
      var input = inputs[invalidNames[index]];
      if (input) {
        input.classList.add("input-warn");
      }
    }
  }

  function revealFirstInvalid() {
    ensureAlive();
    if (validationState.valid) {
      return false;
    }
    refs.root.open = true;
    syncExpanded();
    var firstName = validationState.invalidNames[0];
    var first = inputs[firstName];
    if (first && typeof first.focus === "function") {
      first.focus();
    }
    return true;
  }

  function validateAndRender() {
    validationState = core.validate({
      specs: specsRead(),
      rawValues: currentRawValues(),
      device: device,
      experimental: refs.experimental.checked,
    });
    applyWarnings(validationState.invalidNames);
    refs.validation.hidden = validationState.valid;
    refs.validation.textContent = validationState.message;
    if (!validationState.valid && !refs.root.open) {
      revealFirstInvalid();
    }
    updateSummaryChips();
    onValidationChanged(validationRead());
  }

  function appendSummaryChip(spec, rawValues) {
    var chip = document.createElement("span");
    chip.className = "run-settings-chip";
    chip.setAttribute("data-param-name", spec.name);
    if (
      validationState.invalidNames.indexOf(spec.name) !== -1
    ) {
      chip.classList.add("is-invalid");
    }
    var label = document.createElement("span");
    label.className = "run-settings-chip-label";
    label.textContent = spec.label;
    var value = document.createElement("span");
    value.className = "run-settings-chip-value";
    value.textContent = core.summaryValue(
      spec, rawValues[spec.name]
    );
    chip.appendChild(label);
    chip.appendChild(value);
    refs.chips.appendChild(chip);
  }

  function updateSummaryChips() {
    refs.chips.innerHTML = "";
    var specs = specsRead();
    var rawValues = currentRawValues();
    for (var index = 0; index < specs.length; index++) {
      if (specs[index].prominence === "primary") {
        appendSummaryChip(specs[index], rawValues);
      }
    }
  }

  function persistenceState() {
    return Object.freeze({
      experimental: refs.experimental.checked,
      params: Object.freeze(currentRawValues()),
    });
  }

  function persist() {
    ensureAlive();
    if (!writePersistedState || modelId === null) {
      return;
    }
    writePersistedState(modelId, persistenceState());
  }

  function formChanged() {
    persist();
    updateResetButton();
    onParametersChanged();
  }

  function storedValues(state) {
    if (!state || typeof state !== "object") {
      return null;
    }
    if (
      state.parameters
      && typeof state.parameters === "object"
    ) {
      return state.parameters;
    }
    if (state.params && typeof state.params === "object") {
      return state.params;
    }
    return null;
  }

  function applyRawValue(spec, stored) {
    var input = inputs[spec.name];
    if (!input) {
      return;
    }
    if (spec.type === "bool") {
      input.checked = !!stored;
      return;
    }
    if (spec.type === "select") {
      if ((spec.options || []).indexOf(stored) !== -1) {
        input.value = stored;
      }
      return;
    }
    input.value = String(stored);
  }

  function applyState(state) {
    ensureAlive();
    if (!model || !state) {
      return;
    }
    if (state.modelId && state.modelId !== modelId) {
      throw new Error(
        "Run settings seed belongs to another model"
      );
    }
    refs.experimental.checked = state.experimental === true;
    var values = storedValues(state);
    var specs = specsRead();
    if (values) {
      for (var index = 0; index < specs.length; index++) {
        if (values[specs[index].name] !== undefined) {
          applyRawValue(
            specs[index], values[specs[index].name]
          );
        }
      }
    }
    applyLimits();
    updateResetButton();
  }

  function resetInput(spec) {
    var input = inputs[spec.name];
    if (!input) {
      return;
    }
    var value = core.defaultValue(spec, device);
    if (spec.type === "bool") {
      input.checked = Boolean(value);
    } else {
      input.value = String(value);
    }
  }

  function resetToDefaults() {
    ensureAlive();
    if (!model || disabled) {
      return;
    }
    refs.experimental.checked = false;
    var specs = specsRead();
    for (var index = 0; index < specs.length; index++) {
      resetInput(specs[index]);
    }
    applyLimits();
    formChanged();
  }

  function updateResetButton() {
    refs.reset.disabled = disabled || core.valuesAtDefaults({
      specs: specsRead(),
      rawValues: currentRawValues(),
      device: device,
      experimental: refs.experimental.checked,
    });
  }

  function configure(configuration) {
    ensureAlive();
    if (
      !configuration
      || typeof configuration !== "object"
    ) {
      throw new TypeError(
        "Run settings configure needs model information"
      );
    }
    model = configuration.model || null;
    modelId = configuration.modelId || null;
    modelDisplay = configuration.modelDisplay || "";
    device = configuration.device || null;
    inputMode = configuration.inputMode || null;
    allModels = Array.isArray(configuration.models)
      ? configuration.models
      : (model ? [model] : []);
    refs.experimental.checked = false;
    if (!model) {
      clearControls();
      validationState = core.validate({
        specs: [],
        rawValues: {},
        device: device,
        experimental: false,
      });
      refs.validation.hidden = true;
      refs.validation.textContent = "";
      onValidationChanged(validationRead());
      updateResetButton();
      return;
    }
    buildControls();
    applyUniformWidth();
    remeasureWhenFontReady();
    var seed = configuration.seed;
    if (seed === undefined && readPersistedState) {
      seed = readPersistedState(modelId);
    }
    if (seed) {
      applyState(seed);
    }
    updateResetButton();
  }

  function parameterValuesRead() {
    ensureAlive();
    return core.parameterValues(
      specsRead(),
      currentRawValues(),
      refs.experimental.checked
    );
  }

  function parameterDefaultsRead() {
    ensureAlive();
    return core.defaults(specsRead(), device);
  }

  function experimentalRead() {
    ensureAlive();
    return refs.experimental.checked;
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
    ensureAlive();
    return Object.freeze({
      valid: validationState.valid,
      message: validationState.message,
    });
  }

  function snapshotRead() {
    ensureAlive();
    return core.configurationSnapshot({
      modelId: modelId,
      modelDisplay: modelDisplay,
      inputMode: inputMode,
      specs: specsRead(),
      rawValues: currentRawValues(),
      device: device,
      experimental: refs.experimental.checked,
    });
  }

  function setDisabled(nextDisabled) {
    ensureAlive();
    disabled = nextDisabled === true;
    refs.experimental.disabled = disabled;
    var names = Object.keys(inputs);
    for (var index = 0; index < names.length; index++) {
      inputs[names[index]].disabled = disabled;
    }
    updateResetButton();
  }

  function widestParamWidth() {
    var label = refs.fields.querySelector("label");
    if (!label) {
      return 0;
    }
    var control =
      refs.fields.querySelector("input, .custom-select")
      || label;
    var maxWidth = 90;
    for (
      var modelIndex = 0;
      modelIndex < allModels.length;
      modelIndex++
    ) {
      var specs = allModels[modelIndex].param_specs || [];
      for (
        var specIndex = 0;
        specIndex < specs.length;
        specIndex++
      ) {
        maxWidth = Math.max(
          maxWidth,
          paramWidth(specs[specIndex], label, control)
        );
      }
    }
    return maxWidth;
  }

  function paramWidth(spec, label, control) {
    if (spec.type === "bool") {
      return 0;
    }
    var letterSpacing = 0.8;
    var upper = String(spec.label).toUpperCase();
    var labelWidth =
      measureTextWidth([upper], label)
      + letterSpacing * Math.max(0, upper.length - 1)
      + 26;
    if (spec.type !== "select") {
      return labelWidth;
    }
    var labels = [];
    var options = spec.options || [];
    for (var index = 0; index < options.length; index++) {
      labels.push(core.optionLabel(options[index]));
    }
    var optionWidth =
      measureTextWidth(labels, control) + 40;
    return Math.max(labelWidth, optionWidth);
  }

  function applyUniformWidth() {
    var width = widestParamWidth();
    if (width <= 0) {
      return;
    }
    widthTarget.style.setProperty(
      "--param-width", Math.ceil(width) + "px"
    );
  }

  function remeasureWhenFontReady() {
    var fonts = document.fonts;
    if (!fonts || !fonts.ready || !fonts.ready.then) {
      return;
    }
    if (fontReadyPending) {
      return;
    }
    fontReadyPending = true;
    function remeasure() {
      fontReadyPending = false;
      if (!destroyed) {
        applyUniformWidth();
      }
    }
    fonts.ready.then(remeasure).catch(remeasure);
  }

  function rootRead() {
    ensureAlive();
    return refs.root;
  }

  function destroy() {
    if (destroyed) {
      return;
    }
    clearListeners(inputListeners);
    clearListeners(rootListeners);
    clearControls();
    if (builtRoot) {
      refs.root.remove();
    }
    destroyed = true;
  }

  return Object.freeze({
    wire: wire,
    configure: configure,
    apply: applyState,
    setDisabled: setDisabled,
    reset: resetToDefaults,
    persist: persist,
    revealFirstInvalid: revealFirstInvalid,
    parameterValues: parameterValuesRead,
    parameterDefaults: parameterDefaultsRead,
    experimental: experimentalRead,
    thinking: thinkingRead,
    outputBudget: outputBudgetRead,
    validation: validationRead,
    snapshot: snapshotRead,
    root: rootRead,
    destroy: destroy,
  });
}
