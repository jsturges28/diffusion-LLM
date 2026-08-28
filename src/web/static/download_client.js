// One way to start, watch and cancel a weight download. Loaded as a
// classic global script before menu.js and download_toast.js (same
// pattern as activation_client.js, which this deliberately mirrors),
// so it must not depend on either consumer's state. It touches no
// DOM, which is what makes it testable away from a browser.
//
// The problem it solves is the one ORG-04 solved for activation, in
// the place ORG-02 said was next. Three readers of one endpoint:
// menu.js polled a bound row every 500ms, re-read the status once
// more at page load, and download_toast.js ran its own loop every
// 1000ms from boot to unload. Sitting on the downloading row ran two
// of them at once against the same URL, on different clocks, each
// with its own idea of what a terminal state meant.
//
// One watcher with observers instead. The split is transport here,
// presentation in the page: what a bar looks like, where a toast
// sits and which row is bound stay where they are, because those are
// the parts that genuinely differ.
//
// It also carries something activation had and downloads did not: an
// operation number. Two windows both see one download, so cancelling
// needs to name which, or a stale page ends a fetch it is only
// watching.

"use strict";

// Matched to the supervisor's own sampling of the cache directory,
// so the bar is told as soon as there is anything to tell it. The
// two loops this replaces disagreed about this number.
var DOWNLOAD_CLIENT_POLL_MS = 500;

// A failing poll backs off rather than hammering. The fetch is a
// separate process and outlives a dropped request; only our view of
// it is missing.
var DOWNLOAD_CLIENT_RETRY_MS = 1000;

var DOWNLOAD_STATUS_ENDPOINT = "/api/models/download-status";
var DOWNLOAD_CANCEL_ENDPOINT = "/api/models/download/cancel";
var DOWNLOAD_ACK_ENDPOINT = "/api/models/download/ack";

// States the server can report. Two of them end a fetch; "idle"
// means there is nothing to report, which is not the same as either.
var DOWNLOAD_TERMINAL_STATES = ["done", "error"];

function downloadClientIsTerminal(state) {
  return DOWNLOAD_TERMINAL_STATES.indexOf(state) !== -1;
}

// Create a client for one page. A factory rather than module globals
// so tests can run independent instances, and so a page is free to
// hold one watcher while several parts of it listen.
//
// Every option has a working default; the injected ones exist so a
// test needs neither a network nor a real clock.
function downloadClientCreate(options) {
  var settings = options || {};
  var fetchImpl = settings.fetchImpl
    || function (url, init) { return fetch(url, init); };
  var schedule = settings.schedule
    || function (fn, ms) { return setTimeout(fn, ms); };
  var unschedule = settings.unschedule
    || function (handle) { clearTimeout(handle); };
  var pollMs = typeof settings.pollMs === "number"
    ? settings.pollMs
    : DOWNLOAD_CLIENT_POLL_MS;
  var retryMs = typeof settings.retryMs === "number"
    ? settings.retryMs
    : DOWNLOAD_CLIENT_RETRY_MS;

  var timer = null;
  // The download this client started, as the server numbered it.
  // Null while merely watching somebody else's, which is what stops
  // a page that only observes from cancelling one.
  var operation = null;
  // Everyone who wants to hear. A list rather than one callback
  // because the menu row and the toast both want the same reading,
  // and giving each its own poll is what this file exists to undo.
  var listeners = [];

  function stopTimer() {
    if (timer !== null) {
      unschedule(timer);
      timer = null;
    }
  }

  function announce(status) {
    for (var i = 0; i < listeners.length; i++) {
      try {
        listeners[i](status);
      } catch (_e) {
        // One listener's failure must not stop the loop or rob the
        // others of the reading.
      }
    }
  }

  function read() {
    return fetchImpl(DOWNLOAD_STATUS_ENDPOINT).then(
      function (response) {
        return response.json();
      }
    );
  }

  // One poll, then book the next. Unlike an activation watch this
  // does not stop on a terminal state: a finished download stays
  // reportable until it is acknowledged, and the page that shows the
  // completion toast may not be the page that started the fetch.
  function tick() {
    read().then(
      function (status) {
        if (timer === null) {
          return; // stopped while the request was in flight
        }
        announce(status);
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

  // Ask for a model's weights and follow the fetch. Resolves once
  // the request is accepted, not once the download finishes, which
  // arrives through the listeners.
  function start(modelId) {
    var url =
      "/api/models/" + encodeURIComponent(modelId) + "/download";
    return fetchImpl(url, { method: "POST" })
      .then(function (response) {
        return response.json();
      })
      .then(function (result) {
        if (!result || !result.ok) {
          var message =
            (result && result.message) || "download failed";
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

  // Follow a download this page did not start: any page loading
  // while a fetch is already running. No operation is claimed, so
  // nothing here can cancel on its behalf.
  function observe() {
    operation = null;
    watch();
  }

  // Hear every reading. Returns a function that stops listening,
  // because a menu row is bound and unbound as the user pages
  // around while the toast listens for the whole visit.
  function subscribe(listener) {
    listeners.push(listener);
    return function () {
      var at = listeners.indexOf(listener);
      if (at !== -1) {
        listeners.splice(at, 1);
      }
    };
  }

  // Stop the fetch. Names the operation so a window watching
  // somebody else's download cannot end it; the server refuses a
  // number that is not the current one.
  //
  // The partial blobs stay on disk, which is the point: a re-click
  // resumes from where this stopped rather than starting over.
  function cancel() {
    var init = { method: "POST" };
    if (operation !== null) {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify({ operation: operation });
    }
    var claimed = operation;
    operation = null;
    return fetchImpl(DOWNLOAD_CANCEL_ENDPOINT, init).then(
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

  // Clear a finished fetch so its completion notice fires once.
  function ack() {
    return fetchImpl(DOWNLOAD_ACK_ENDPOINT, {
      method: "POST",
    }).catch(function () {
      // Best effort: the next poll re-reads the state anyway, and a
      // failed ack must not break the caller's teardown.
    });
  }

  // A single read, for a page that wants the current state without
  // committing to a loop.
  function readOnce() {
    return read();
  }

  // Stop watching without cancelling: the page is leaving. The
  // download keeps running, which is the point of the distinction
  // from cancel().
  function stop() {
    stopTimer();
  }

  return {
    start: start,
    observe: observe,
    subscribe: subscribe,
    readOnce: readOnce,
    cancel: cancel,
    ack: ack,
    stop: stop,
    operation: function () {
      return operation;
    },
  };
}
