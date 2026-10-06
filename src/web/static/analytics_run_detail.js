// Reusable request-fenced orchestration for Analytics run detail.

"use strict";

function analyticsRunDetailCreate(options) {
  var owner = {
    panel: analyticsRunDetailRequired(options, "panel"),
    openModal: analyticsRunDetailCallback(options, "openModal"),
    closeModal: analyticsRunDetailCallback(options, "closeModal"),
    onStart: analyticsRunDetailCallback(options, "onStart"),
    onInvalid: analyticsRunDetailCallback(options, "onInvalid"),
    fetchMeta: analyticsRunDetailCallback(options, "fetchMeta"),
    fetchMetrics: analyticsRunDetailCallback(
      options, "fetchMetrics"
    ),
    fetchFrames: analyticsRunDetailCallback(
      options, "fetchFrames"
    ),
    onMeta: analyticsRunDetailCallback(options, "onMeta"),
    onMetaFailure: analyticsRunDetailCallback(
      options, "onMetaFailure"
    ),
    onMetrics: analyticsRunDetailCallback(options, "onMetrics"),
    onMetricsFailure: analyticsRunDetailCallback(
      options, "onMetricsFailure"
    ),
    onFrames: analyticsRunDetailCallback(options, "onFrames"),
    onFramesFailure: analyticsRunDetailCallback(
      options, "onFramesFailure"
    ),
    onClose: analyticsRunDetailCallback(options, "onClose"),
    requests: detailRequestsCreate(),
    active: null,
    returnFocus: null,
    wired: false,
  };
  return Object.freeze({
    wire: function () {
      analyticsRunDetailWire(owner);
    },
    show: function (input) {
      analyticsRunDetailShow(owner, input);
    },
    close: function () {
      owner.closeModal(owner.panel);
    },
    cancel: function () {
      owner.requests.cancel();
      owner.active = null;
    },
    activeKey: function () {
      return owner.active === null ? null : owner.active.key;
    },
  });
}

function analyticsRunDetailWire(owner) {
  if (owner.wired) {
    return;
  }
  owner.wired = true;
  owner.panel.addEventListener("close", function () {
    var focus = owner.returnFocus;
    owner.requests.cancel();
    owner.active = null;
    owner.returnFocus = null;
    owner.onClose();
    if (focus && focus.isConnected !== false) {
      focus.focus();
    }
  });
}

function analyticsRunDetailShow(owner, input) {
  analyticsRunDetailInput(input);
  owner.active = input;
  owner.returnFocus = input.returnFocus || null;
  var token = owner.requests.begin(input.key);
  owner.openModal(owner.panel);
  owner.onStart(input);
  if (input.invalid === true) {
    owner.onInvalid(input);
    return;
  }
  analyticsRunDetailLoadMeta(owner, input, token);
  analyticsRunDetailLoadMetrics(owner, input, token);
  analyticsRunDetailLoadFrames(owner, input, token);
}

function analyticsRunDetailLoadMeta(owner, input, token) {
  Promise.resolve(
    owner.fetchMeta(input, token.signal)
  ).then(function (metadata) {
    if (owner.requests.accepts(token)) {
      owner.onMeta(metadata, input);
    }
  }).catch(function (error) {
    if (analyticsRunDetailIgnore(owner, token, error)) {
      return;
    }
    owner.onMetaFailure(error, input);
  });
}

function analyticsRunDetailLoadMetrics(owner, input, token) {
  Promise.resolve(
    owner.fetchMetrics(input, token.signal)
  ).then(function (metrics) {
    if (owner.requests.accepts(token)) {
      owner.onMetrics(metrics, input);
    }
  }).catch(function (error) {
    if (analyticsRunDetailIgnore(owner, token, error)) {
      return;
    }
    owner.onMetricsFailure(error, input);
  });
}

function analyticsRunDetailLoadFrames(owner, input, token) {
  Promise.resolve(
    owner.fetchFrames(input, token.signal)
  ).then(function (frames) {
    if (owner.requests.accepts(token)) {
      owner.onFrames(frames, input);
    }
  }).catch(function (error) {
    if (analyticsRunDetailIgnore(owner, token, error)) {
      return;
    }
    owner.onFramesFailure(error, input);
  });
}

function analyticsRunDetailIgnore(owner, token, error) {
  if (!owner.requests.accepts(token)) {
    return true;
  }
  return Boolean(
    error
    && (
      error.name === "AbortError"
      || detailRequestsIsAbort(error)
    )
  );
}

function analyticsRunDetailInput(input) {
  if (!input || typeof input !== "object") {
    throw new TypeError("run detail input must be an object");
  }
  if (typeof input.key !== "string" || input.key === "") {
    throw new TypeError("run detail input needs a key");
  }
  if (!input.summary || typeof input.summary !== "object") {
    throw new TypeError("run detail input needs a summary");
  }
}

function analyticsRunDetailRequired(options, name) {
  if (!options || !options[name]) {
    throw new TypeError(
      "analyticsRunDetailCreate needs options." + name
    );
  }
  return options[name];
}

function analyticsRunDetailCallback(options, name) {
  var callback = analyticsRunDetailRequired(options, name);
  if (typeof callback !== "function") {
    throw new TypeError(
      "analyticsRunDetailCreate needs callback options." + name
    );
  }
  return callback;
}
