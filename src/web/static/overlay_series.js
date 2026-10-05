// A saved run's frames and its signal manifest, read the same way
// whichever shape the server sent them in.
//
// Loaded as a classic global script after overlays.js, whose
// commit, entropy and token-display helpers it reads, and before
// analytics.js, and like the other extracted modules it reaches for
// no page and no storage. Everything here answers a question about a
// series or a payload it is handed: how long a run is, what a frame
// holds, where it ends, when each position settled, what the stopping
// track reads, which frame a channel is read at, and whether entropy
// or forgetting can be drawn. The viewer state those questions are
// asked from, the open run, the scrubbed frame and the memoized
// revisions, stays in analytics.js, which passes in what is needed.
//
// Its tests read payloads the frames endpoint really produced, kept
// in tests/web/static/fixtures/, so a field the server stops sending
// fails there rather than passing against a hand-built stand-in.

"use strict";

// ---- A run's frames, whichever way the server sent them ----
//
// Two arrangements reach this page. A run whose positions change
// arrives as one array per frame, because there is no smaller
// truthful description of it. A run that only grows arrives as one
// flat list, because frame N is its first N+1 entries and sending
// them all again N times is what made a 2,048-token run a 123 MiB
// download.
//
// Everything below asks a series for a frame rather than indexing
// an array, so the two shapes are one thing to the rest of the page.

function overlaySeries(positions, frames) {
  return {
    positions: Array.isArray(positions) ? positions : null,
    frames: Array.isArray(frames) ? frames : null,
  };
}

function overlaySeriesLength(series) {
  if (!series) {
    return 0;
  }
  if (series.positions) {
    return series.positions.length;
  }
  return series.frames ? series.frames.length : 0;
}

function overlaySeriesPresent(series) {
  return overlaySeriesLength(series) > 0;
}

// The token array at ``index``, or null when there is no such frame.
// Null renders as a blank canvas, which is what an early all-masked
// frame with no records looks like.
function overlaySeriesAt(series, index) {
  if (!series || index < 0 || index >= overlaySeriesLength(series)) {
    return null;
  }
  if (series.positions) {
    return series.positions.slice(0, index + 1);
  }
  var frame = series.frames[index];
  return frame === undefined ? null : frame;
}

// Index of the last frame carrying records. The scrubber opens here,
// so the viewer shows the resolved output rather than a blank early
// canvas. Zero when none qualify.
function overlaySeriesFinalIndex(series) {
  if (!series) {
    return 0;
  }
  if (series.positions) {
    // Every position is a record, so the last frame is the last one.
    return series.positions.length > 0
      ? series.positions.length - 1
      : 0;
  }
  var frames = series.frames || [];
  for (var i = frames.length - 1; i >= 0; i--) {
    if (frames[i] && frames[i].length > 0) {
      return i;
    }
  }
  return 0;
}

function overlaySeriesFinal(series) {
  if (!series) {
    return null;
  }
  if (series.positions) {
    return series.positions.length > 0 ? series.positions : null;
  }
  var frames = series.frames || [];
  for (var i = frames.length - 1; i >= 0; i--) {
    if (frames[i] && frames[i].length > 0) {
      return frames[i];
    }
  }
  return null;
}

// A series built straight from a payload, for the handful of callers
// that are handed the response rather than the open run.
function overlaySeriesOf(data, baseline) {
  if (!data) {
    return overlaySeries(null, null);
  }
  if (baseline) {
    return overlaySeries(
      data.original_positions, data.original_frames
    );
  }
  return overlaySeries(data.positions, data.frames);
}

// Commit steps for a series, without assembling frames it does not
// have. A run that only grows settles every position the moment it
// appears, so the general walk over its prefixes would rebuild the
// whole quadratic to arrive at a column of zeros.
function overlaySeriesCommitSteps(series) {
  if (!series) {
    return [];
  }
  if (series.positions) {
    return overlaysAppendCommitSteps(series.positions);
  }
  return overlaysComputeCommitSteps(
    overlaysFrameReader(series.frames || []),
    overlaySeriesLength(series)
  );
}

// Every frame's revised positions for a series. A run that only grows
// never revisits a position, so it has none to find.
function overlaySeriesRevisions(series, canvasOf, edits) {
  if (!series || series.positions) {
    return [];
  }
  return overlaysComputeRevisions(
    overlaysFrameReader(series.frames || []),
    overlaySeriesLength(series),
    canvasOf,
    edits
  );
}

// A series as the stopping track reads it, from the payload it was
// read off. The branch carries the payload's canvas indices and
// starts a resumed segment at each edit's frame_index. The run it
// forked from is one canvas with no resumes, because DiffusionGemma
// resumes nothing longer.
function overlaySeriesStopSource(data, series, original) {
  var canvases = original ? null : data.canvas_index;
  var edits = original ? [] : data.remask_edits || [];
  return {
    count: overlaySeriesLength(series),
    readFrame: function (f) {
      return overlaySeriesAt(series, f);
    },
    canvasAt: function (f) {
      var canvas = canvases ? canvases[f] : 0;
      return typeof canvas === "number" ? canvas : 0;
    },
    segmentStarts: edits.map(function (edit) {
      return edit.frame_index;
    }),
  };
}

// ---- The signal manifest ----
//
// A run says what its signals measure and what they vary over, so a
// view does not have to guess from where a number is stored. Before
// this, everything read the final frame, which is right for an
// autoregressive position decided once and silently wrong for a
// diffusion position re-decided at every denoising step: the value
// shown was whatever the last step happened to hold.
//
// Absent on every run saved before the manifest existed, and those
// read exactly as they did before: one value per position.

// The shapes this page can actually draw. An axis pair outside this
// list is not a bug to hide, it is a channel a future model declared
// and this build has no view for, and saying so is the point.
var OVERLAY_SERIES_ENTROPY_SHAPES = ["position", "frame|position"];

// A channel's axes as one comparable string. Sorted deliberately not
// at all: "frame|position" is the declaration order, and treating
// ("frame","position") and ("position","frame") as different would
// invent a distinction nobody makes.
function overlaySeriesChannelShape(channel) {
  if (!channel || !channel.axes || !channel.axes.length) {
    return "";
  }
  return channel.axes.join("|");
}

// One declared channel by name, or null when the run declares none.
function overlaySeriesChannel(run, name) {
  var signals = (run && run.signals) || [];
  for (var i = 0; i < signals.length; i++) {
    if (signals[i] && signals[i].name === name) {
      return signals[i];
    }
  }
  return null;
}

// Which frame a channel should be read at. A per-position channel is
// the same in every frame, so the final one is as good as any and is
// what the charts already used. A channel that varies by frame has to
// follow the scrub, or the reader is shown one arbitrary step, so
// the caller passes the frame the scrub is on.
function overlaySeriesChannelFrame(channel, series, frameIndex) {
  var finalIndex = overlaySeriesFinalIndex(series);
  if (overlaySeriesChannelShape(channel) !== "frame|position") {
    return finalIndex;
  }
  if (frameIndex > finalIndex) {
    return finalIndex;
  }
  return frameIndex;
}

// Why the entropy view is or is not drawable: "ok", "absent" when the
// run captured none, or "unsupported" when it declared a shape no
// view here understands. Three answers rather than a boolean, because
// hiding the section for the third case is how a silently dropped
// channel would look.
function overlaySeriesEntropyAvailability(data) {
  var series = overlaySeriesOf(data, false);
  var channel = overlaySeriesChannel(data, "entropy");
  if (channel) {
    if (
      OVERLAY_SERIES_ENTROPY_SHAPES.indexOf(
        overlaySeriesChannelShape(channel)
      ) === -1
    ) {
      return "unsupported";
    }
    return overlaySeriesHasEntropy(series) ? "ok" : "absent";
  }
  // No manifest: a run from before this existed. Fall back to the
  // probe those runs were always read with.
  return overlaySeriesHasEntropy(series) ? "ok" : "absent";
}

// Whether the series carries entropy anywhere it can be read. Not
// only on its final frame: a DiffusionGemma run ends on a committed
// canvas, which carries none, while every draft before it does.
function overlaySeriesHasEntropy(series) {
  if (!series) {
    return false;
  }
  var last = overlaySeriesLength(series) - 1;
  return overlaySeriesEntropyFrame(
    series, last, overlaySeriesSingleCanvas
  ) >= 0;
}

// Whether the saved run carries per-token entropy.
function overlaySeriesCarriesEntropy(data) {
  return overlaySeriesHasEntropy(overlaySeriesOf(data, false));
}

// The frame whose entropy describes frame `index` of a series, or -1
// (see overlaysEntropyFrame), read through the series as the page
// holds it.
function overlaySeriesEntropyFrame(series, index, canvasOf) {
  return overlaysEntropyFrame(
    function (frame) {
      return overlaySeriesAt(series, frame);
    },
    canvasOf,
    index,
    !!series.positions
  );
}

// Per-position entropy for one frame series, read off its final
// frame: every position is sampled once in an autoregressive run, so
// its entropy never changes after the frame that introduced it.
// Mirrors the generator's entropyProfileValues. Runs over both the
// open run and its pre-edit snapshot, which is why it takes frames
// rather than the payload.
// The values at one frame. `at` is which frame to read, which the
// caller takes from the channel's declared axes: the final frame for
// a per-position channel, the scrubbed one for a trajectory.
function overlaySeriesEntropyValues(series, at) {
  var index = typeof at === "number"
    ? at
    : overlaySeriesFinalIndex(series);
  var final = overlaySeriesAt(series, index) || [];
  var values = [];
  var texts = [];
  for (var i = 0; i < final.length; i++) {
    var tok = final[i] || {};
    values.push(
      typeof tok.e === "number" ? +tok.e.toFixed(3) : null
    );
    texts.push(
      typeof tok.t === "string" ? overlaysAltDisplay(tok.t) : ""
    );
  }
  return { values: values, texts: texts };
}

// Whether any token in the series' final frame carries a number under
// `key`. The final frame is the series' ground truth.
function overlaySeriesHasTokenValue(series, key) {
  var final = overlaySeriesFinal(series);
  if (!final) {
    return false;
  }
  for (var i = 0; i < final.length; i++) {
    if (final[i] && typeof final[i][key] === "number") {
      return true;
    }
  }
  return false;
}

function overlaySeriesHasTokenBoolean(series, key) {
  var final = overlaySeriesFinal(series);
  if (!final) {
    return false;
  }
  for (var i = 0; i < final.length; i++) {
    if (final[i] && typeof final[i][key] === "boolean") {
      return true;
    }
  }
  return false;
}

// Whether the saved run carries per-token forgetting to colour by.
// The run's own manifest decides the shape when it has one, as it
// does for entropy: forgetting is one value per position, and a run
// declaring it any other way has a channel this page cannot draw. A
// run saved without a manifest falls back to the data.
function overlaySeriesCarriesForgetting(data) {
  var channel = overlaySeriesChannel(data, "forgetting");
  if (channel && overlaySeriesChannelShape(channel) !== "position") {
    return false;
  }
  return overlaySeriesHasTokenValue(
    overlaySeriesOf(data, false), "f"
  );
}

// Whether durable records can draw keyed membership. New runs also
// declare both channels as one value per position; a legacy run with
// boolean ``g`` records but no manifest remains readable.
function overlaySeriesCarriesWatermark(data) {
  var membership =
    overlaySeriesChannel(data, "watermark_membership");
  var evidence =
    overlaySeriesChannel(data, "watermark_evidence");
  if (
    membership
    && overlaySeriesChannelShape(membership) !== "position"
  ) {
    return false;
  }
  if (
    evidence
    && overlaySeriesChannelShape(evidence) !== "position"
  ) {
    return false;
  }
  return overlaySeriesHasTokenBoolean(
    overlaySeriesOf(data, false), "g"
  );
}

// One canvas for the whole run, which is how the adapter asks
// whether entropy exists anywhere a reader could find it.
function overlaySeriesSingleCanvas() {
  return 0;
}
