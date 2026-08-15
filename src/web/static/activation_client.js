// One way to start, watch and cancel a model activation. Loaded as a
// classic global script before app.js and menu.js (same pattern as
// overlays.js and detail_requests.js), so it must not depend on
// either page's state. It touches no DOM, which is what makes it
// testable away from a browser.
//
// The problem it solves: activation display was already shared, via
// overlaysActivationProgress, but the transport was not. The
// generator and the menu each owned a POST, a poll loop, a retry
// schedule and a terminal decision, in four readers of one endpoint
// with two copies of the interval between them. A recorded load-bar
// correction needed coordinated edits in both, and the pass that
// added the LIFE-06 changes did the same thing again: the discard
// went into both files and the failure surfacing into only one.
//
// The split is transport here, presentation and navigation in the
// page. This file decides what the server was asked and what it
// said; what a bar looks like and where "ready" navigates to are the
// parts that genuinely differ between the two entry points, and they
// stay where they are.

"use strict";

// Matched to the supervisor's own sampling of the worker, so the bar
// is told as soon as there is anything to tell it.
var ACTIVATION_CLIENT_POLL_MS = 250;

// A failing poll backs off rather than hammering. The load is still
// running; only our view of it is missing.
var ACTIVATION_CLIENT_RETRY_MS = 800;

var ACTIVATION_ENDPOINT = "/api/models/activation";
var ACTIVATION_CANCEL_ENDPOINT = "/api/models/activate/cancel";

// States the server can report. Two of them end a watch; the rest
// mean work is still underway and another poll is due.
var ACTIVATION_TERMINAL_STATES = ["ready", "error"];

function activationClientIsTerminal(state) {
  return ACTIVATION_TERMINAL_STATES.indexOf(state) !== -1;
}

// Create a client for one page. A factory rather than module globals
// so tests can run independent instances, and so the two pollers the
// generator runs (its boot watch and its switch) cannot tread on
// each other's timers.
//
// Every option has a working default; the injected ones exist so a
// test needs neither a network nor a real clock.
function activationClientCreate(options) {
  var settings = options || {};
  var fetchImpl = settings.fetchImpl
    || function (url, init) { return fetch(url, init); };
  var schedule = settings.schedule
    || function (fn, ms) { return setTimeout(fn, ms); };
  var unschedule = settings.unschedule
    || function (handle) { clearTimeout(handle); };
  var pollMs = typeof settings.pollMs === "number"
    ? settings.pollMs
    : ACTIVATION_CLIENT_POLL_MS;
  var retryMs = typeof settings.retryMs === "number"
    ? settings.retryMs
    : ACTIVATION_CLIENT_RETRY_MS;

  var onProgress = settings.onProgress || function () {};
  var onReady = settings.onReady || function () {};
  var onFailed = settings.onFailed || function () {};

  var timer = null;
  // The activation this client is responsible for, as the server
  // numbered it. Null while merely observing somebody else's.
  var operation = null;

  function stopTimer() {
    if (timer !== null) {
      unschedule(timer);
      timer = null;
    }
  }

  // Whether a status report describes the activation we started.
  // Observing (operation null) accepts whatever is running, because
  // that is the whole point of observing. A client that started one
  // accepts only its own: another window can supersede it between
  // two polls, and acting on the replacement's "ready" is how one
  // window ends up navigating for another's model.
  function ours(status) {
    if (operation === null) {
      return true;
    }
    if (!status || status.operation === undefined) {
      return true; // a server that does not number them yet
    }
    return status.operation === operation;
  }

  function read() {
    return fetchImpl(ACTIVATION_ENDPOINT).then(
      function (response) {
        return response.json();
      }
    );
  }

  // One poll, then either finish or book the next. Errors are not
  // terminal: the activation is server-side and outlives a dropped
  // request, so a failed read just leaves the last reading up and
  // tries again more slowly.
  function tick() {
    read().then(
      function (status) {
        if (timer === null) {
          return; // stopped while the request was in flight
        }
        if (!ours(status)) {
          // Somebody else's activation is what is running now. Ours
          // is gone, and there is nothing further to report on it.
          stopTimer();
          return;
        }
        onProgress(status.state, status.progress);
        if (status.state === "ready") {
          stopTimer();
          onReady();
          return;
        }
        if (status.state === "error") {
          stopTimer();
          onFailed(status.message || "load failed");
          return;
        }
        timer = schedule(tick, pollMs);
      },
      function () {
        if (timer === null) {
          return;
        }
        timer = schedule(tick, retryMs);
      }
    );
  }

  function watch() {
    stopTimer();
    timer = schedule(tick, 0);
  }

  // Ask for a model and follow it to a terminal state. Resolves once
  // the request is accepted, not once the model is ready; readiness
  // arrives through onReady, because activation is deliberately
  // non-blocking on the server.
  function start(modelId, startOptions) {
    var opts = startOptions || {};
    var init = { method: "POST" };
    if (opts.device) {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify({ device: opts.device });
    }
    var url =
      "/api/models/" + encodeURIComponent(modelId) + "/activate";
    return fetchImpl(url, init)
      .then(function (response) {
        return response.json();
      })
      .then(function (result) {
        if (!result || !result.ok) {
          var message =
            (result && result.message) || "activation failed";
          throw new Error(message);
        }
        operation =
          typeof result.operation === "number"
            ? result.operation
            : null;
        watch();
        return result;
      });
  }

  // Follow an activation this page did not start: the generator at
  // boot, arriving while a worker is already coming up. No operation
  // is claimed, so nothing here can cancel or navigate on its
  // behalf.
  function observe() {
    operation = null;
    watch();
  }

  // A single read, for a page that wants the current state without
  // committing to a loop. Used to surface a load that failed before
  // this page was open.
  function readOnce() {
    return read();
  }

  // Stop the activation we started. Sends the operation so the
  // server can refuse if it belongs to somebody else by now; the
  // caller sees that refusal rather than a button that did nothing.
  function cancel() {
    stopTimer();
    var init = { method: "POST" };
    if (operation !== null) {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify({ operation: operation });
    }
    var claimed = operation;
    operation = null;
    return fetchImpl(ACTIVATION_CANCEL_ENDPOINT, init).then(
      function (response) {
        return response.json().then(function (body) {
          if (response.ok && body && body.ok) {
            return body;
          }
          var message =
            (body && body.message) || "could not cancel";
          var error = new Error(message);
          error.operation = claimed;
          throw error;
        });
      }
    );
  }

  // Stop watching without cancelling: the page is leaving, or the
  // caller has taken over. The activation keeps running, which is
  // the point of the distinction from cancel().
  function stop() {
    stopTimer();
  }

  function currentOperation() {
    return operation;
  }

  return {
    start: start,
    observe: observe,
    readOnce: readOnce,
    cancel: cancel,
    stop: stop,
    operation: currentOperation,
  };
}
