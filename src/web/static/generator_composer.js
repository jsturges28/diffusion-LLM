// The generator's prompt composer: text entry, prompt history, file
// import, draft persistence and the context-window readout.
//
// Loaded as a classic script before app.js. It defines one global
// factory and keeps both its DOM references and mutable state inside
// the returned controller's closure. The page remains the composition
// root: model-panel facts and WebSocket delivery arrive as callbacks
// and method calls rather than being read from app.js globals.

"use strict";

function generatorComposerCreate(options) {
  if (!options || typeof options.onSubmit !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.onSubmit"
    );
  }
  if (typeof options.onDraftChanged !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.onDraftChanged"
    );
  }
  if (typeof options.reportStatus !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.reportStatus"
    );
  }
  if (typeof options.readThinking !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.readThinking"
    );
  }
  if (typeof options.readOutputBudget !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.readOutputBudget"
    );
  }
  if (typeof options.isCountReady !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.isCountReady"
    );
  }
  if (typeof options.sendCountPrompt !== "function") {
    throw new TypeError(
      "generatorComposerCreate needs options.sendCountPrompt"
    );
  }

  var onSubmit = options.onSubmit;
  var onDraftChanged = options.onDraftChanged;
  var reportStatus = options.reportStatus;
  var readThinking = options.readThinking;
  var readOutputBudget = options.readOutputBudget;
  var isCountReady = options.isCountReady;
  var sendCountPrompt = options.sendCountPrompt;

  var promptInput = requiredElement("prompt-input");
  var promptLabel = requiredElement("prompt-label");
  var promptModeInfo = requiredElement("prompt-mode-info");
  var promptModeTip = requiredElement("prompt-mode-tip");
  var promptContextRow = requiredElement("prompt-context");
  var promptContextCount =
    requiredElement("prompt-context-count");
  var promptContextNote =
    requiredElement("prompt-context-note");
  var promptHistoryGroup = requiredElement("prompt-history");
  var btnPromptImport = requiredElement("btn-prompt-import");
  var promptFileInput = requiredElement("prompt-file-input");
  var btnPromptHistory = requiredElement("btn-prompt-history");
  var promptHistoryNav =
    requiredElement("prompt-history-nav");
  var btnHistPrev = requiredElement("btn-hist-prev");
  var btnHistNext = requiredElement("btn-hist-next");
  var btnHistDelete = requiredElement("btn-hist-delete");
  var promptHistoryCounter =
    requiredElement("prompt-history-counter");
  var btnHistConfirm = requiredElement("btn-hist-confirm");
  var btnHistCancel = requiredElement("btn-hist-cancel");
  var modalImport = requiredElement("modal-import");
  var importFileLabel = requiredElement("import-file-label");
  var btnImportClose = requiredElement("btn-import-close");
  var btnImportConfirm = requiredElement("btn-import-confirm");
  var btnImportCancel = requiredElement("btn-import-cancel");

  var PROMPT_HISTORY_KEY = "diffusion_prompt_history";
  var PROMPT_HISTORY_MAX = 30;
  var PROMPT_HISTORY_DELETE_LABEL =
    "Delete this prompt from history";
  var PROMPT_HISTORY_DELETE_ARMED_LABEL =
    "Press again to delete this prompt";
  var PROMPT_DRAFT_STATE_KEY = "diffusion_param_state";
  // Refuse an accidentally selected video before reading it. The
  // character cap matches the worker's count cap, so an accepted
  // import is always counted in full.
  var PROMPT_IMPORT_BYTES_MAX = 1048576;
  var PROMPT_IMPORT_CHARS_MAX = 200000;
  // Counting is a readout, not a step blocking the user's gesture,
  // so it waits longer than typed-token preview requests.
  var PROMPT_COUNT_DEBOUNCE_MS = 350;

  var PROMPT_MODE_COPY = {
    chat: {
      label: "Prompt",
      placeholder: "Enter a prompt...",
      hint: "",
    },
    completion: {
      label: "Text to continue",
      placeholder:
        "Text for the model to continue, e.g. \"A REST API is\"",
      hint:
        "This is a base model: it writes on from where your text"
        + " stops rather than replying to it, so a question tends to"
        + " get more questions. To get an explanation, begin it"
        + " yourself, for example: \"A REST API is\"",
    },
  };

  var modelId = null;
  var contextLength = null;
  var promptHistory = [];
  var promptHistoryIndex = -1;
  var promptHistoryDraft = null;
  var promptHistoryActive = false;
  var promptHistoryDeleteArmed = false;
  var pendingImportFile = null;
  var promptCountRequest = 0;
  var promptCountTimer = null;
  var promptCountLatest = null;
  var promptCountThinkingSent = false;
  var wired = false;

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error("Missing generator composer element #" + id);
    }
    return element;
  }

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    wirePrompt();
    wireHistory();
    wireImport();
  }

  function wirePrompt() {
    promptInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        onSubmit();
      }
    });
    promptInput.addEventListener("input", function () {
      saveDraft();
      promptTextChanged();
      onDraftChanged();
    });
  }

  function wireHistory() {
    btnPromptHistory.addEventListener("click", function () {
      if (promptHistoryActive) {
        cancelPromptHistory();
      } else {
        enterPromptHistory();
      }
    });
    btnHistPrev.addEventListener("click", function () {
      cyclePromptHistory(1);
    });
    btnHistNext.addEventListener("click", function () {
      cyclePromptHistory(-1);
    });
    btnHistDelete.addEventListener(
      "click", pressPromptHistoryDelete
    );
    btnHistDelete.addEventListener("mouseleave", function () {
      setPromptHistoryDeleteArmed(false);
    });
    btnHistDelete.addEventListener("blur", function () {
      setPromptHistoryDeleteArmed(false);
    });
    btnHistConfirm.addEventListener(
      "click", confirmPromptHistory
    );
    btnHistCancel.addEventListener(
      "click", cancelPromptHistory
    );
  }

  function wireImport() {
    btnPromptImport.addEventListener("click", function () {
      promptFileInput.click();
    });
    promptFileInput.addEventListener("change", function () {
      var file = promptFileInput.files[0];
      // Clearing first lets choosing the same file fire again.
      promptFileInput.value = "";
      beginPromptImport(file);
    });
    promptInput.addEventListener("dragover", function (event) {
      if (!dragCarriesFile(event)) {
        return;
      }
      // Without this the browser navigates to the dropped file and
      // loses the page and any run on it.
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      promptInput.classList.add("is-drop-target");
    });
    promptInput.addEventListener("dragleave", function () {
      promptInput.classList.remove("is-drop-target");
    });
    promptInput.addEventListener("drop", function (event) {
      if (!dragCarriesFile(event)) {
        return;
      }
      event.preventDefault();
      promptInput.classList.remove("is-drop-target");
      beginPromptImport(event.dataTransfer.files[0]);
    });
    btnImportConfirm.addEventListener(
      "click", confirmPromptImport
    );
    btnImportCancel.addEventListener("click", closeImport);
    btnImportClose.addEventListener("click", closeImport);
    modalImport.addEventListener("close", function () {
      pendingImportFile = null;
    });
    modalImport.addEventListener("click", function (event) {
      if (event.target === modalImport) {
        closeImport();
      }
    });
  }

  function boot() {
    loadPromptHistory();
    updatePromptHistoryUI();
  }

  function configure(config) {
    if (!config || typeof config !== "object") {
      throw new TypeError(
        "generatorComposer.configure needs a configuration"
      );
    }
    modelId = typeof config.modelId === "string"
      ? config.modelId
      : null;
    contextLength = contextLengthFrom(config.contextLength);
    applyPromptMode(config.inputMode);
    restoreDraft();
    renderPromptContext();
  }

  function contextLengthFrom(value) {
    if (typeof value !== "number" || !isFinite(value)) {
      return null;
    }
    if (value <= 0) {
      return null;
    }
    return Math.round(value);
  }

  function applyPromptMode(mode) {
    var copy = PROMPT_MODE_COPY[mode] || PROMPT_MODE_COPY.chat;
    promptLabel.textContent = copy.label;
    promptInput.placeholder = copy.placeholder;
    promptModeTip.textContent = copy.hint;
    promptModeInfo.hidden = copy.hint === "";
  }

  function value() {
    return promptInput.value;
  }

  function trimmedValue() {
    return promptInput.value.trim();
  }

  function restore(text) {
    if (typeof text === "string" && text !== "") {
      promptInput.value = text;
    }
  }

  function clear() {
    exitPromptHistoryUI();
    promptInput.value = "";
    promptTextChanged();
    promptInput.focus();
  }

  function setDisabled(disabled) {
    promptInput.disabled = disabled;
    btnPromptImport.disabled = disabled;
  }

  // The form-state record is shared temporarily with app.js's
  // parameter persistence. This controller is the sole writer of its
  // `prompt` member; app.js preserves that member while writing the
  // schema-driven fields that move in a later slice.
  function readDraftStateAll() {
    var raw = null;
    try {
      raw = sessionStorage.getItem(PROMPT_DRAFT_STATE_KEY);
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

  function saveDraft() {
    if (modelId === null) {
      return;
    }
    var all = readDraftStateAll();
    var state = all[modelId];
    if (
      !state
      || typeof state !== "object"
      || Array.isArray(state)
    ) {
      state = {};
    }
    state.prompt = promptInput.value;
    all[modelId] = state;
    try {
      sessionStorage.setItem(
        PROMPT_DRAFT_STATE_KEY, JSON.stringify(all)
      );
    } catch (_error) {
      // Quota or private mode: the draft simply will not persist.
    }
  }

  function restoreDraft() {
    if (modelId === null) {
      return;
    }
    var state = readDraftStateAll()[modelId];
    if (
      state
      && typeof state.prompt === "string"
      && state.prompt !== ""
    ) {
      promptInput.value = state.prompt;
    }
  }

  function loadPromptHistory() {
    promptHistory = [];
    try {
      var raw = localStorage.getItem(PROMPT_HISTORY_KEY);
      if (!raw) {
        return;
      }
      var parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        promptHistory = parsed.filter(function (prompt) {
          return (
            typeof prompt === "string"
            && prompt.length > 0
          );
        });
      }
    } catch (_error) {
      promptHistory = [];
    }
  }

  function savePromptHistoryStore() {
    // Write through so desktop restarts keep the same history.
    persistSet(
      PROMPT_HISTORY_KEY, JSON.stringify(promptHistory)
    );
  }

  function prepareGeneration(text) {
    exitPromptHistoryUI();
    pushPromptHistory(text);
  }

  function pushPromptHistory(text) {
    var prompt = (text || "").trim();
    if (!prompt) {
      return;
    }
    promptHistory = promptHistory.filter(function (stored) {
      return stored !== prompt;
    });
    promptHistory.unshift(prompt);
    if (promptHistory.length > PROMPT_HISTORY_MAX) {
      promptHistory.length = PROMPT_HISTORY_MAX;
    }
    savePromptHistoryStore();
    updatePromptHistoryUI();
  }

  function updatePromptHistoryUI() {
    promptHistoryGroup.hidden = promptHistory.length === 0;
  }

  function setPromptHistoryCounter() {
    // Storage is newest-first, but the visible count is
    // chronological: newest is N / N and moving left toward older
    // prompts counts down.
    promptHistoryCounter.textContent =
      (promptHistory.length - promptHistoryIndex)
      + " / " + promptHistory.length;
  }

  function enterPromptHistory() {
    if (promptHistory.length === 0 || promptHistoryActive) {
      return;
    }
    promptHistoryActive = true;
    promptHistoryDraft = promptInput.value;
    promptHistoryIndex = 0;
    promptInput.value = promptHistory[0];
    promptInput.readOnly = true;
    promptTextChanged();
    btnPromptHistory.classList.add("is-active");
    promptHistoryNav.hidden = false;
    setPromptHistoryCounter();
  }

  function cyclePromptHistory(delta) {
    if (!promptHistoryActive || promptHistory.length === 0) {
      return;
    }
    setPromptHistoryDeleteArmed(false);
    var count = promptHistory.length;
    promptHistoryIndex =
      (((promptHistoryIndex + delta) % count) + count) % count;
    promptInput.value = promptHistory[promptHistoryIndex];
    setPromptHistoryCounter();
    promptTextChanged();
  }

  function exitPromptHistoryUI() {
    setPromptHistoryDeleteArmed(false);
    promptHistoryActive = false;
    promptHistoryDraft = null;
    promptHistoryIndex = -1;
    promptInput.readOnly = false;
    btnPromptHistory.classList.remove("is-active");
    promptHistoryNav.hidden = true;
  }

  function confirmPromptHistory() {
    exitPromptHistoryUI();
    promptInput.focus();
  }

  function cancelPromptHistory() {
    if (promptHistoryDraft !== null) {
      promptInput.value = promptHistoryDraft;
    }
    exitPromptHistoryUI();
    promptTextChanged();
  }

  function pressPromptHistoryDelete() {
    if (!promptHistoryActive) {
      return;
    }
    // Two presses protect a prompt that may exist nowhere else. Any
    // route away from this button disarms the first press.
    if (!promptHistoryDeleteArmed) {
      setPromptHistoryDeleteArmed(true);
      return;
    }
    setPromptHistoryDeleteArmed(false);
    deleteShownPromptHistory();
  }

  function setPromptHistoryDeleteArmed(armed) {
    promptHistoryDeleteArmed = armed;
    var label = armed
      ? PROMPT_HISTORY_DELETE_ARMED_LABEL
      : PROMPT_HISTORY_DELETE_LABEL;
    btnHistDelete.classList.toggle("is-armed", armed);
    btnHistDelete.title = label;
    btnHistDelete.setAttribute("aria-label", label);
  }

  function deleteShownPromptHistory() {
    var index = promptHistoryIndex;
    if (index < 0 || index >= promptHistory.length) {
      return;
    }
    promptHistory.splice(index, 1);
    savePromptHistoryStore();
    if (promptHistory.length === 0) {
      cancelPromptHistory();
      updatePromptHistoryUI();
      promptInput.focus();
      return;
    }
    promptHistoryIndex =
      Math.min(index, promptHistory.length - 1);
    promptInput.value = promptHistory[promptHistoryIndex];
    setPromptHistoryCounter();
    promptTextChanged();
  }

  function beginPromptImport(file) {
    if (!file) {
      return;
    }
    if (!isImportableTextFile(file)) {
      reportStatus(
        "Only .txt and .md files can be imported.", true
      );
      return;
    }
    if (file.size > PROMPT_IMPORT_BYTES_MAX) {
      reportStatus(
        "That file is too large to import ("
        + formatKilobytes(file.size)
        + "; the limit is "
        + formatKilobytes(PROMPT_IMPORT_BYTES_MAX)
        + ").",
        true
      );
      return;
    }
    if (promptInput.value.trim() === "") {
      readPromptFile(file);
      return;
    }
    pendingImportFile = file;
    importFileLabel.textContent = file.name;
    openImport();
  }

  function openImport() {
    if (!modalImport.open) {
      modalImport.showModal();
    }
  }

  function closeImport() {
    if (modalImport.open) {
      modalImport.close();
    }
  }

  function readPromptFile(file) {
    // This stays client-side. Sending a prompt to the server only to
    // receive the same bytes back would turn import into an upload.
    file
      .text()
      .then(function (text) {
        applyImportedPrompt(file, text);
      })
      .catch(function () {
        reportStatus(
          "Could not read " + file.name + ".", true
        );
      });
  }

  function applyImportedPrompt(file, text) {
    if (promptHistoryActive) {
      exitPromptHistoryUI();
    }
    var clipped = text.slice(0, PROMPT_IMPORT_CHARS_MAX);
    promptInput.value = clipped;
    promptTextChanged();
    saveDraft();
    onDraftChanged();
    var message = "Imported " + file.name;
    if (clipped.length < text.length) {
      message +=
        ", truncated to "
        + PROMPT_IMPORT_CHARS_MAX.toLocaleString()
        + " characters";
    }
    reportStatus(message + ".", false);
    promptInput.focus();
  }

  function confirmPromptImport() {
    var file = pendingImportFile;
    closeImport();
    if (file) {
      readPromptFile(file);
    }
  }

  function formatKilobytes(bytes) {
    return Math.round(bytes / 1024).toLocaleString() + " KB";
  }

  function isImportableTextFile(file) {
    var name = (file.name || "").toLowerCase();
    var extensions = [".txt", ".md", ".markdown"];
    for (var index = 0; index < extensions.length; index++) {
      if (name.endsWith(extensions[index])) {
        return true;
      }
    }
    return (file.type || "").indexOf("text/") === 0;
  }

  function dragCarriesFile(event) {
    var transfer = event.dataTransfer;
    if (!transfer) {
      return false;
    }
    var types = transfer.types || [];
    for (var index = 0; index < types.length; index++) {
      if (types[index] === "Files") {
        return true;
      }
    }
    return false;
  }

  function promptTextChanged() {
    promptCountLatest = null;
    renderPromptContext();
    if (promptCountTimer !== null) {
      clearTimeout(promptCountTimer);
    }
    promptCountTimer = setTimeout(
      requestPromptCount, PROMPT_COUNT_DEBOUNCE_MS
    );
  }

  function requestPromptCount() {
    promptCountTimer = null;
    if (!isCountReady()) {
      return;
    }
    var text = promptInput.value;
    var thinking = Boolean(readThinking());
    if (text === "") {
      promptCountLatest = {
        count: 0,
        truncated: false,
        thinking: thinking,
      };
      renderPromptContext();
      return;
    }
    var requestId = promptCountRequest + 1;
    sendCountPrompt({
      type: "count_prompt",
      text: text,
      thinking: thinking,
      request_id: requestId,
    });
    promptCountRequest = requestId;
    promptCountThinkingSent = thinking;
  }

  function handleCountResult(message) {
    // A late answer describes text that has already been replaced.
    if (!message || message.request_id !== promptCountRequest) {
      return;
    }
    promptCountLatest = {
      count: Number(message.count) || 0,
      truncated: Boolean(message.truncated),
      thinking: promptCountThinkingSent,
    };
    renderPromptContext();
  }

  function parametersChanged() {
    if (
      promptCountLatest !== null
      && promptCountLatest.thinking !== Boolean(readThinking())
    ) {
      promptTextChanged();
      return;
    }
    renderPromptContext();
  }

  function outputBudgetTokens() {
    var budget = readOutputBudget();
    if (typeof budget !== "number" || !isFinite(budget)) {
      return 0;
    }
    return Math.max(0, Math.round(budget));
  }

  function renderPromptContext() {
    if (promptCountLatest === null) {
      // Keep the row in the layout, empty. Removing it would move
      // everything below the prompt when the first answer arrives.
      promptContextRow.classList.add("is-empty");
      return;
    }
    var count = promptCountLatest.count;
    var text = count.toLocaleString();
    if (contextLength !== null) {
      text += " / " + contextLength.toLocaleString();
    }
    text += count === 1 ? " token" : " tokens";
    promptContextCount.textContent = text;
    promptContextRow.classList.remove("is-empty");
    applyPromptContextWarning(
      count, promptCountLatest.truncated
    );
  }

  function applyPromptContextWarning(count, truncated) {
    var note = "";
    var over = false;
    if (contextLength !== null) {
      var budget = outputBudgetTokens();
      // A truncated count hit a cap beyond every supported context,
      // so truncation itself proves that the prompt is over.
      if (truncated || count > contextLength) {
        note = "over the context window";
        over = true;
      } else if (count + budget > contextLength) {
        note =
          "prompt + "
          + budget.toLocaleString()
          + " output exceeds the window";
      }
    }
    promptContextNote.textContent = note;
    promptContextRow.classList.toggle(
      "is-warning", note !== ""
    );
    promptContextRow.classList.toggle("is-over", over);
  }

  return {
    wire: wire,
    boot: boot,
    configure: configure,
    value: value,
    trimmedValue: trimmedValue,
    restore: restore,
    clear: clear,
    setDisabled: setDisabled,
    saveDraft: saveDraft,
    prepareGeneration: prepareGeneration,
    textChanged: promptTextChanged,
    parametersChanged: parametersChanged,
    handleCountResult: handleCountResult,
    closeImport: closeImport,
  };
}
