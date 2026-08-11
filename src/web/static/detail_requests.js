// Request fencing for the Analytics detail panel. Loaded as a
// classic global script before analytics.js (same pattern as
// overlays.js and custom_select.js), so it must not depend on that
// page's state. It touches no DOM, which is what makes it testable
// away from a browser.
//
// The problem it solves: opening a run starts two independent
// fetches, and their callbacks used to paint whatever came back.
// Open run A, open run B before A answers, and B's title could sit
// above A's charts. Closing cancelled nothing either, so a
// dismissed panel could repopulate itself. A wrong number that
// looks right is the worst failure this page has, because nothing
// on screen says it is wrong.
//
// The fix is an epoch. Every open takes a fresh one, every response
// carries the token it was issued, and a response may paint only
// while its token is still current. The run id alone would not be
// enough: closing and reopening the same run has to discard the
// first attempt's answers too, or a slow response from the first
// open lands on top of the second.

"use strict";

// The epoch counter starts here and only ever increases, so a token
// minted before the first begin() can never match a live one.
var DETAIL_REQUESTS_NO_EPOCH = 0;

// Create a fence for one panel. A factory rather than a single
// module global so that tests can run independent instances, and so
// a second fenced surface can have its own counter instead of
// sharing this one.
function detailRequestsCreate() {
  var epoch = DETAIL_REQUESTS_NO_EPOCH;
  var openRunId = null;
  var controller = null;

  // Retire every outstanding token and stop whatever is in flight.
  // Bumping the epoch is what does the retiring; the abort is an
  // optimization on top, so a response that has already been
  // received but not yet delivered is still refused.
  function abandon() {
    if (controller !== null) {
      controller.abort();
      controller = null;
    }
    epoch += 1;
  }

  // Start an attempt and issue the token its fetches must carry.
  // The signal travels inside the token so a caller cannot pair one
  // attempt's epoch with another attempt's AbortController.
  function begin(runId) {
    abandon();
    openRunId = runId;
    controller = new AbortController();
    return {
      epoch: epoch,
      runId: runId,
      signal: controller.signal,
    };
  }

  // Abandon the current attempt without starting another: the panel
  // closed, a delete removed the run it was showing, or the compare
  // view took the screen. All three used to leave fetches running
  // into a panel that was no longer there.
  function cancel() {
    abandon();
    openRunId = null;
  }

  // Whether a response holding this token may still be painted.
  // Both halves are checked. The epoch catches a superseded or
  // cancelled attempt, and the run id catches a token that outlived
  // the run it described.
  function accepts(token) {
    if (!token) {
      return false;
    }
    if (token.epoch !== epoch) {
      return false;
    }
    return token.runId === openRunId;
  }

  return {
    begin: begin,
    cancel: cancel,
    accepts: accepts,
  };
}

// Whether a rejected fetch was our own cancellation rather than a
// real failure. Aborting rejects the promise, and rendering "failed
// to load" because the user closed the panel would be a lie. The
// name is checked rather than the class because DOMException is not
// constructible in every context this file is loaded into.
function detailRequestsIsAbort(error) {
  if (!error) {
    return false;
  }
  return error.name === "AbortError";
}
