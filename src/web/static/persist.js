// Durable UI state: what a page keeps across navigations and app
// restarts, and the session key two pages have to agree on.
//
// Loaded as a classic global script before overlays.js and
// download_toast.js, both of which write through it. It reaches for
// storage, the server's /api/ui-state and the page's lifecycle
// events, and calls nothing in the visual code: the dependency runs
// from overlays.js and the pages to here, never back. That is what
// lets a page that only needs its state kept, the menu among them,
// load this without the visual code.

"use strict";

// ---- Durable UI state (server-backed localStorage mirror) ----
//
// The desktop app's window origin (scheme://host:port) can change
// between launches because the launcher's port varies, which partitions
// localStorage and made Settings, prompt history, the analytics "new
// run" cue, and the generate teaser reset across restarts. The server
// persists these keys in results/ui_state.json (see src/web/ui_state.py).
// We hydrate localStorage from it once on boot and write through on
// change, so the fast synchronous localStorage reads elsewhere keep
// working unchanged.

var PERSIST_KEYS = [
  "diffusion_settings",
  "diffusion_new_runs",
  "diffusion_prompt_history",
  "diffusion_generate_teased",
  "diffusion_download_toast_corner",
  "diffusion_active_conversation",
  // One per page: the two drawers sit in containers of different
  // heights, so a shared offset would land sensibly on at most one.
  "diffusion_overlay_drawer_top_generator",
  "diffusion_overlay_drawer_top_analytics",
  // diffusion_collections used to be here, and was the only value in
  // the list that was not a cache. It came off because a key/value
  // mirror is the wrong shape for it: two windows each replacing the
  // whole array meant the later write erased the earlier one's
  // filing. It is stored in the same file, but only the server may
  // write it, through the operations in collections_client.js.
];

// Debounce PUTs per key so rapid writes (e.g. successive settings
// toggles) coalesce into one network call.
var PERSIST_PUT_DEBOUNCE_MS = 250;
var persistPutTimers = {};

// Written straight out, no debounce. The active conversation is user
// intent rather than a cache: losing the last 250 ms of a window
// switch could reopen the wrong transcript after a close.
var PERSIST_IMMEDIATE_KEYS = [
  "diffusion_active_conversation",
];

// Values whose PUT is still waiting on a timer, so the flush below
// can send them when the page is going away.
var persistPending = {};

// Keepalive requests share a small per-origin budget (64 KiB is the
// usual figure) and are rejected wholesale above it. Two of these
// keys are allowed to reach 262,144 characters, so the flag is only
// set when the body comfortably fits; a larger body goes as an
// ordinary request and takes its chances.
var PERSIST_KEEPALIVE_MAX_CHARS = 50000;

// Write `value` to localStorage immediately (so the many synchronous
// reads see it at once) and write through to the server, debounced
// unless the key cannot afford to wait. Unknown keys stay local only.
function persistSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_e) {
    // Non-fatal: fall through to the server write regardless.
  }
  if (PERSIST_KEYS.indexOf(key) === -1) {
    return;
  }
  if (PERSIST_IMMEDIATE_KEYS.indexOf(key) !== -1) {
    persistPutKey(key, value, false);
    return;
  }
  if (persistPutTimers[key]) {
    clearTimeout(persistPutTimers[key]);
  }
  persistPending[key] = value;
  persistPutTimers[key] = setTimeout(function () {
    persistPutTimers[key] = null;
    delete persistPending[key];
    persistPutKey(key, value, false);
  }, PERSIST_PUT_DEBOUNCE_MS);
}

// Send every debounced write that has not fired yet. Called when the
// page is being hidden or torn down, which is the moment a pending
// timer would otherwise be discarded along with the document.
function persistFlushPending() {
  var keys = Object.keys(persistPending);
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    var value = persistPending[key];
    if (persistPutTimers[key]) {
      clearTimeout(persistPutTimers[key]);
      persistPutTimers[key] = null;
    }
    delete persistPending[key];
    persistPutKey(key, value, true);
  }
}

// Per-key callbacks for a write that did not reach disk. Only keys
// whose loss the user needs to know about register one; for a cache
// a silent retry next session is the right amount of noise.
var persistFailureHandlers = {};

function persistOnFailure(key, handler) {
  persistFailureHandlers[key] = handler;
}

function persistReportFailure(key) {
  var handler = persistFailureHandlers[key];
  if (typeof handler !== "function") {
    return;
  }
  try {
    handler(key);
  } catch (_e) {
    // A broken reporter must not break the next write.
  }
}

function persistPutKey(key, value, urgent) {
  var body = JSON.stringify({ value: value });
  var init = {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: body,
  };
  if (urgent && body.length <= PERSIST_KEEPALIVE_MAX_CHARS) {
    // Lets the request outlive the document that started it.
    init.keepalive = true;
  }
  try {
    fetch("/api/ui-state/" + encodeURIComponent(key), init)
      .then(function (response) {
        // A 4xx or 5xx resolves rather than rejecting, so the status
        // has to be read: the server answers a rejected write with
        // {"success": false} and this used to ignore it.
        if (!response.ok) {
          persistReportFailure(key);
        }
      })
      .catch(function () {
        persistReportFailure(key);
      });
  } catch (_e) {
    persistReportFailure(key);
  }
}

// Armed once per page, from the one call every page already makes.
// `visibilitychange` is the reliable half: `pagehide` does not fire
// in every teardown, and `beforeunload` is worse still. Both are
// registered because hiding a tab is not always followed by
// unloading it, and unloading is not always preceded by hiding.
var persistFlushArmed = false;

function persistArmFlush() {
  if (persistFlushArmed) {
    return;
  }
  persistFlushArmed = true;
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") {
      persistFlushPending();
    }
  });
  window.addEventListener("pagehide", persistFlushPending);
}

// Read the durable state the server inlined when it served the page,
// or null when there is none. Pages that carry it skip a round trip
// that everything after it was waiting on.
function persistInlinedState() {
  var boot = window.__BOOT__;
  var state = boot ? boot.ui_state : null;
  // The typeof is what matters: a string here would be treated as
  // hydrated state, so the fetch that would have got the real thing
  // never happens and every key silently keeps its stale local copy.
  if (!state || typeof state !== "object") {
    return null;
  }
  return state;
}

// Mirror server state into localStorage, then run `onReady`. Always
// calls `onReady` exactly once (even on failure) so a page never hangs
// on a persistence hiccup. Server values overwrite any stale local
// copy left by a previous window origin.
//
// Synchronous when the state was inlined, which matters more than it
// sounds: `onReady` is the page's boot, so a fetch here means the
// whole page waits, and on the generator it meant a second fetch
// waited behind this one before anything could be drawn correctly.
// The fetch stays for pages served without the state, which is the vm
// test harness and a file opened directly.
function persistHydrate(onReady) {
  persistArmFlush();
  var inlined = persistInlinedState();
  if (inlined !== null) {
    persistApplyHydrated(inlined);
    onReady();
    return;
  }
  var done = false;
  function finish() {
    if (done) {
      return;
    }
    done = true;
    onReady();
  }
  try {
    fetch("/api/ui-state")
      .then(function (response) {
        return response.json();
      })
      .then(function (state) {
        persistApplyHydrated(state);
        finish();
      })
      .catch(finish);
  } catch (_e) {
    finish();
  }
}

function persistApplyHydrated(state) {
  if (!state || typeof state !== "object") {
    return;
  }
  for (var i = 0; i < PERSIST_KEYS.length; i++) {
    var key = PERSIST_KEYS[i];
    if (typeof state[key] !== "string") {
      continue;
    }
    try {
      localStorage.setItem(key, state[key]);
    } catch (_e) {
      // Non-fatal: this key just will not hydrate this session.
    }
  }
}

// ---- "New runs" registry (shared across generator + analytics) ----
//
// Run IDs saved since the user last viewed them in Analytics.
// Persisted server-side (via persistSet) so the generator's
// Analytics-link count and the analytics table's per-row dots agree
// across page navigations and survive restarts; a run is cleared
// individually when its detail is opened or when the run is deleted.

var PERSIST_NEW_RUNS_KEY = "diffusion_new_runs";

function persistReadNewRuns() {
  try {
    var raw = localStorage.getItem(PERSIST_NEW_RUNS_KEY);
    if (!raw) {
      return [];
    }
    var parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_e) {
    return [];
  }
}

function persistWriteNewRuns(ids) {
  // Write-through to the server so the cue survives restarts and stays
  // consistent across the generator, menu, and analytics pages.
  persistSet(PERSIST_NEW_RUNS_KEY, JSON.stringify(ids));
}

// Returns true if the run was newly added (was not already tracked),
// so callers can flash the "+1" cue only for genuinely new runs.
function persistAddNewRun(runId) {
  if (!runId) {
    return false;
  }
  var ids = persistReadNewRuns();
  if (ids.indexOf(runId) === -1) {
    ids.push(runId);
    persistWriteNewRuns(ids);
    return true;
  }
  return false;
}

function persistClearNewRun(runId) {
  var ids = persistReadNewRuns();
  var idx = ids.indexOf(runId);
  if (idx !== -1) {
    ids.splice(idx, 1);
    persistWriteNewRuns(ids);
  }
}

function persistNewRunCount() {
  return persistReadNewRuns().length;
}

function persistIsNewRun(runId) {
  return persistReadNewRuns().indexOf(runId) !== -1;
}

// ---- Last-run snapshot (generator, session-scoped) ----
//
// The generator's completed-run snapshot, written and read by app.js
// so a trip to Analytics and back restores the output. Only the key
// and the clear live here, because two pages have to drop it:
// activating a model ends in a location.reload() on the generator and
// on the menu alike, and by the time the generator boots it can no
// longer tell that reload from a navigation. The page that *starts*
// the switch can, so each clears on its way out and neither needs to
// know the other's storage.

var PERSIST_LAST_RUN_KEY = "diffusion_last_run";
var PERSIST_ACTIVE_CONVERSATION_KEY =
  "diffusion_active_conversation";

function persistClearLastRun() {
  try {
    sessionStorage.removeItem(PERSIST_LAST_RUN_KEY);
  } catch (_e) {
    // Storage unavailable: there is nothing to clear.
  }
}
