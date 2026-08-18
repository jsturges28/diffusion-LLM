// One reading of what the server says about models.
//
// Loaded as a classic global script before the page scripts, like the
// other extracted modules, and like them it touches no DOM.
//
// The problem it solves is agreement, not transport. Four pages each
// fetched `/api/models` and then each decided for itself what the
// answer meant: whether a model counts as resident, what an absent
// device should default to, whether a missing `gpu_name` is false or
// undefined. Two of them, Analytics and Settings, implement the same
// rule for the same link, and the second says so in a comment rather
// than in code. A reading that drifts is worse than a fetch that is
// written twice, because the pages then disagree about the same
// server response while both look correct.
//
// There is deliberately no request epoch here, though `ORG-02` asks
// for one on the API clients. Every page calls this exactly once, at
// boot, so nothing can be in flight twice and a fence would be
// machinery with no caller. The pattern exists already in
// detail_requests.js for the page that genuinely needs it, and this
// is where to add it if a page ever reloads its model list.

"use strict";

var MODEL_CLIENT_ENDPOINT = "/api/models";

// `fetchImpl` is for tests; pages pass nothing and get the global.
function modelClientLoad(fetchImpl) {
  var get = fetchImpl || fetch;
  return get(MODEL_CLIENT_ENDPOINT).then(function (response) {
    return response.json();
  });
}

// The resident model's id, or null. This is the one every page asks
// about, and the answer gates the Generation link on two of them.
function modelClientActiveId(info) {
  if (!info || typeof info.active !== "string" || !info.active) {
    return null;
  }
  return info.active;
}

function modelClientHasActive(info) {
  return modelClientActiveId(info) !== null;
}

function modelClientList(info) {
  if (!info || !Array.isArray(info.models)) {
    return [];
  }
  return info.models;
}

// Where the resident model was placed, or null when nothing is
// resident. Not defaulted to "cuda": a wrong guess here labels a run
// with hardware it did not use.
function modelClientActiveDevice(info) {
  if (!info || typeof info.active_device !== "string") {
    return null;
  }
  return info.active_device;
}

// Whether this host has a GPU at all, which decides what a plain
// row-click targets and whether CPU-only rows are offered.
function modelClientGpuPresent(info) {
  return !!(info && info.gpu_name);
}

function modelClientGpuName(info) {
  return modelClientGpuPresent(info) ? info.gpu_name : null;
}

function modelClientFind(info, id) {
  var list = modelClientList(info);
  for (var i = 0; i < list.length; i++) {
    if (list[i].id === id) {
      return list[i];
    }
  }
  return null;
}

// The resident model itself, or null. Two lookups in one because
// every caller that wants it already had to ask twice.
function modelClientActiveModel(info) {
  var id = modelClientActiveId(info);
  if (id === null) {
    return null;
  }
  return modelClientFind(info, id);
}
