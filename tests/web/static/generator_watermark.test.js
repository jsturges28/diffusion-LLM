// The pasted-text detector dialog, driven without app.js.
//
// Strategy: use the shipped DOM ids and controller, record outbound
// requests, and deliver replies out of order. Passing proves the
// capability gate, raw text request, configurable display threshold,
// stale-reply fence, and request-scoped key error copy.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

function harness() {
  const page = loadPage({
    scripts: ["overlays.js", "generator_watermark.js"],
  });
  page.sent = [];
  page.parameters = {
    watermark_gamma: 0.2,
    watermark_z_threshold: 3.5,
  };
  page.controller = page.context.generatorWatermarkCreate({
    sendRequest(payload) {
      page.sent.push(JSON.parse(JSON.stringify(payload)));
      return true;
    },
    readParameters() {
      return page.parameters;
    },
  });
  page.controller.wire();
  return page;
}

function open(page) {
  page.controller.configure({ supports_watermark: true });
  page.registry.get("btn-watermark-detector").dispatch("click");
}

function detect(page, text) {
  page.registry.get("watermark-detector-text").value = text;
  page.registry.get("btn-watermark-detect").dispatch("click");
  return page.sent[page.sent.length - 1];
}

function result(requestId, overrides) {
  return Object.assign({
    type: "detect_watermark_result",
    request_id: requestId,
    key_id: "0123456789abcdef",
    model_id: "smollm3",
    gamma: 0.25,
    vocab_size: 128256,
    tokenizer_fingerprint: "ab".repeat(32),
    token_count: 80,
    green_count: 30,
    scored_count: 79,
    green_rate: 30 / 79,
    z_score: 3.25,
    p0: 0.25,
    status: "threshold_not_crossed",
    display_threshold: 3.5,
  }, overrides || {});
}

test("only watermark-capable models expose the dialog", () => {
  const page = harness();
  const button = page.registry.get("btn-watermark-detector");

  page.controller.configure({ supports_watermark: false });
  assert.equal(button.hidden, true);

  page.controller.configure({ supports_watermark: true });
  assert.equal(button.hidden, false);
});

test("detect sends raw text and experimental controls", () => {
  const page = harness();
  open(page);

  const request = detect(page, "  raw text\n");

  assert.equal(request.type, "detect_watermark");
  assert.equal(request.text, "  raw text\n");
  assert.equal(request.gamma, 0.2);
  assert.equal(request.z_threshold, 3.5);
  assert.equal(request.request_id, 1);
});

test("a stale reply cannot replace the latest result", () => {
  const page = harness();
  open(page);
  const first = detect(page, "first");
  page.controller.close();
  open(page);
  const second = detect(page, "second");
  const output = page.registry.get("watermark-detector-result");

  page.controller.handleResult(result(first.request_id, {
    key_id: "aaaaaaaaaaaaaaaa",
  }));
  assert.equal(output.textContent, "Tokenizing and scoring...");

  page.controller.handleResult(result(second.request_id));
  assert.match(output.textContent, /Key 0123456789abcdef/);
  assert.match(output.textContent, /model smollm3/);
  assert.match(output.textContent, /vocab 128256/);
  assert.match(output.textContent, /tokenizer abababababababab/);
  assert.match(output.textContent, /green\/scored 30\/79/);
  assert.match(output.textContent, /threshold not crossed/);
});

test("key mismatch errors stay in the detector dialog", () => {
  const page = harness();
  open(page);
  const request = detect(page, "text");

  const owned = page.controller.handleError({
    type: "error",
    request_type: "detect_watermark",
    request_id: request.request_id,
    code: "watermark_key_mismatch",
    message: "backend detail",
  });

  assert.equal(owned, true);
  assert.equal(
    page.registry.get("watermark-detector-result").textContent,
    "Key mismatch: backend detail"
  );
});

test("closing the dialog fences an in-flight reply", () => {
  const page = harness();
  open(page);
  const request = detect(page, "text");
  page.controller.close();
  const output = page.registry.get("watermark-detector-result");
  const before = output.textContent;

  page.controller.handleResult(result(request.request_id));

  assert.equal(output.textContent, before);
});

test("changing any detector input invalidates pending work", () => {
  const cases = [
    ["watermark-detector-text", "different text"],
    ["watermark-detector-gamma", "0.3"],
    ["watermark-detector-threshold", "5"],
    ["watermark-detector-key", "aaaaaaaaaaaaaaaa"],
  ];
  for (const [id, value] of cases) {
    const page = harness();
    open(page);
    const request = detect(page, "first text");
    const input = page.registry.get(id);
    input.value = value;
    input.dispatch("input");
    const output =
      page.registry.get("watermark-detector-result");

    assert.equal(output.textContent, "Inputs changed. Detect again.");
    assert.equal(
      output.classList.contains("is-stale"), true
    );
    page.controller.handleResult(result(request.request_id));
    assert.equal(output.textContent, "Inputs changed. Detect again.");
    assert.equal(
      page.registry.get("btn-watermark-detect").disabled,
      false
    );
  }
});
