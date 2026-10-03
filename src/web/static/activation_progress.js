// What the loading UI shows for one poll of an activation: a filled
// bar, a sweep, or nothing.
//
// Loaded as a classic global script before activation_client.js on
// the two pages that draw an activation, the generator's overlay and
// the menu's bar. Pure: it reads no page and no storage, so both
// pages draw the same moment the same way and it can be tested with
// nothing loaded beside it. activation_client.js asks the server what
// is happening; this decides what the answer looks like.

"use strict";

// ---- Activation progress (generator overlay + menu bar) ----
//
// How long a completed bar is left at 100% before the page moves on.
// Long enough to read as finishing rather than vanishing, and short
// enough to disappear into a load measured in seconds.
var ACTIVATION_PROGRESS_HOLD_MS = 180;
//
// One poll of /api/models/activation, reduced to what the loading UI
// should show. Pure and shared because two pages render the same
// moment: the generator's full-screen overlay during a switch, and
// the menu's inline bar during a first activation. They looked
// different only because each had written its own wording.
//
// `mode` is one of:
//   "fill"  a real measurement: draw the track filled to `percent`.
//   "sweep" a phase with no measurable target: draw the track with an
//           indeterminate sweep and no number.
//   "hidden" nothing to show.
//
// The sweep is why there are three modes rather than a determinate
// flag. An activation opens with several seconds of work that cannot
// be measured at all: a worker process spawning, importing torch and
// transformers in its own virtualenv, and only then reading the
// checkpoint headers that give the bar its target. Parking a real bar
// at 0% through that reads as hung, and showing nothing (which is
// what the menu used to do) reads as nothing happening. A sweep is
// honest about having no number while still saying work is underway.
//
// A "loading" state with no usable progress is the same situation:
// see load_progress.load_target_bytes, which returns a zero total
// rather than guess at an unfamiliar checkpoint layout.
function activationProgressView(state, progress) {
  // The worker reaches ready in the same breath as its last progress
  // sample, and the supervisor drops progress on that transition, so
  // the closing 100% never survives the trip to the browser. Naming
  // the completed state here is what lets the bar finish instead of
  // disappearing at whatever the last poll happened to catch.
  if (state === "ready") {
    return { mode: "fill", percent: 100, label: "Ready" };
  }
  // Named rather than folded into "Loading" because it is a different
  // wait with a different cause, and saying which one is running is
  // the difference between a slow start and an apparently hung one.
  if (state === "starting") {
    return {
      mode: "sweep", percent: 0, label: "Starting worker",
    };
  }
  var out = { mode: "sweep", percent: 0, label: "Loading" };
  if (state === "downloading") {
    out.label = "Downloading";
  } else if (state === "loading") {
    // The sampler names the counter it is reporting, so this tracks
    // the weights whether they route through RAM first or stream
    // straight to the GPU. Before the first sample there is no stage
    // to name yet, and the generic label stands in.
    if (progress) {
      out.label =
        progress.stage === "device"
          ? "Moving to GPU"
          : "Loading weights";
    }
  } else {
    // idle, error, or a state this build does not know about: no
    // activation is in flight to draw one way or the other.
    return { mode: "hidden", percent: 0, label: "Loading" };
  }
  if (!progress || typeof progress.fraction !== "number") {
    return out;
  }
  // A zero total is the "could not measure this checkpoint" signal.
  // Keep the label, keep sweeping.
  if (!(progress.total_bytes > 0)) {
    return out;
  }
  var fraction = progress.fraction;
  if (fraction < 0) {
    fraction = 0;
  } else if (fraction > 1) {
    fraction = 1;
  }
  out.mode = "fill";
  out.percent = Math.round(fraction * 100);
  return out;
}
