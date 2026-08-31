// The collections API, one function per user gesture. Loaded as a
// classic global script before analytics.js (same pattern as
// overlays.js and detail_requests.js), so it must not depend on that
// page's state. It touches no DOM, which is what makes it testable
// away from a browser.
//
// Why the page does not just write the list: it used to. Every
// gesture serialized the whole array and PUT it, so two windows
// hydrated from the same value each filed a different run and the
// later write dropped the earlier one without either window
// noticing. That is `DATA-02`, and the fix is that a client no
// longer says what the list should become. It says what the user
// did, and the server applies it to whatever is stored at that
// moment, under the lock.
//
// So every function here sends one gesture and resolves to the whole
// list as it stands afterwards. The caller adopts that rather than
// merging into what it had, which is what makes a window that was
// behind level again the moment it acts.

"use strict";

var COLLECTIONS_BASE = "/api/collections";

// Create a client. A factory taking its fetch rather than reaching
// for the global, so tests drive it without a network and a second
// caller could point it elsewhere.
function collectionsClientCreate(options) {
  var settings = options || {};
  var fetchImpl = settings.fetchImpl || fetch;

  // One request, resolved to the list or rejected with a reason.
  //
  // The reason matters more here than the status. A refusal is
  // usually something the user can act on (the collection cap, a
  // name too long, a run that has since been deleted), so the
  // message is the server's rather than one invented from a status
  // the page would have to interpret.
  function send(path, init) {
    return fetchImpl(COLLECTIONS_BASE + path, init).then(
      function (response) {
        return response.json().then(function (body) {
          if (response.ok && body && body.success) {
            return body.collections || [];
          }
          throw collectionsClientError(body);
        });
      }
    );
  }

  function post(path, payload) {
    return send(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
  }

  function remove(path) {
    return send(path, { method: "DELETE" });
  }

  // The current list, reconciled server-side against runs on disk.
  // A window uses this to catch up with another window's filing
  // without a reload.
  function list() {
    return send("", { method: "GET" });
  }

  // Create, optionally filing one run at the same time. Naming a
  // collection from a run's own dialog is asking for that run to go
  // in it, and sent as two calls the second can fail and leave an
  // empty collection the user did not ask for.
  function create(name, runId) {
    var payload = { name: name };
    if (runId) {
      payload.run_id = runId;
    }
    return post("", payload);
  }

  // The same, for a selection. Separate from create rather than an
  // overloaded argument so a caller cannot pass an array where the
  // server reads a string.
  function createWithRuns(name, runIds) {
    return post("", { name: name, run_ids: runIds });
  }

  function rename(collectionId, name) {
    return post(
      "/" + encodeURIComponent(collectionId) + "/rename",
      { name: name }
    );
  }

  function destroy(collectionId) {
    return remove("/" + encodeURIComponent(collectionId));
  }

  function addRun(collectionId, runId) {
    return post(
      "/" + encodeURIComponent(collectionId) + "/runs",
      { run_id: runId }
    );
  }

  // File a selection. One request, so the server files all of them
  // or none: six sequential adds can stop at four and leave a
  // half-applied gesture nobody can see the shape of.
  function addRuns(collectionId, runIds) {
    return post(
      "/" + encodeURIComponent(collectionId) + "/runs",
      { run_ids: runIds }
    );
  }

  function removeRun(collectionId, runId) {
    return remove(
      "/" + encodeURIComponent(collectionId)
      + "/runs/" + encodeURIComponent(runId)
    );
  }

  // The star. One call rather than several because it is one
  // gesture with two meanings: file into Favorites, or clear from
  // every collection. Composed client-side it could half-apply.
  function toggleFavorite(runId) {
    return post("/favorite", { run_id: runId });
  }

  return {
    list: list,
    create: create,
    createWithRuns: createWithRuns,
    rename: rename,
    destroy: destroy,
    addRun: addRun,
    addRuns: addRuns,
    removeRun: removeRun,
    toggleFavorite: toggleFavorite,
  };
}

// A refusal, as an Error carrying what the server said. The message
// is shown to the user, so it is the server's wording; the reason is
// for a caller that wants to branch rather than just report.
function collectionsClientError(body) {
  var reason = body && body.reason ? body.reason : "unavailable";
  var message = body && body.message
    ? body.message
    : "The change could not be saved.";
  var error = new Error(message);
  error.reason = reason;
  return error;
}
