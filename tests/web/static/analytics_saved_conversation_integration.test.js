// Full Analytics wiring for saved-conversation deep links.
//
// Strategy: boot the shipped page scripts at ?conversation=..., serve
// one snapshot and one pinned exchange, then activate View XAI.
// Passing proves the top-level mode, deep link, bounded turn request,
// and shared run-detail controller compose in the real script order.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  ANALYTICS_SCRIPTS,
  loadPage,
} = require("./dom_stub");

const SNAPSHOT_ID = "a".repeat(32);
const TURN_ID = "b".repeat(32);

function response(body, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  });
}

function fetchImpl(url) {
  const value = String(url);
  if (value === "/api/analytics/runs") {
    return response([]);
  }
  if (value === "/api/analytics/conversations") {
    return response([{
      snapshot_id: SNAPSHOT_ID,
      title: "Saved path",
      title_revision: 1,
      created_at: "2026-10-06T00:00:00Z",
      exchange_count: 1,
      xai_count: 1,
      text_only_count: 0,
      unavailable_count: 0,
    }]);
  }
  if (value.endsWith("/metadata")) {
    if (value.includes("/run/")) {
      return response({
        backend: "smollm3",
        model_type: "autoregressive",
        prompt: "Why?",
      });
    }
    return response({
      snapshot_id: SNAPSHOT_ID,
      title: "Saved path",
      title_revision: 1,
      exchange_count: 1,
      xai_count: 1,
      text_only_count: 0,
      unavailable_count: 0,
      source: { branch_id: "branch" },
    });
  }
  if (value.includes("/turns?")) {
    return response({
      snapshot_id: SNAPSHOT_ID,
      turns: [
        {
          index: 1,
          turn_id: "c".repeat(32),
          role: "user",
          text: "Why?",
          partial: false,
        },
        {
          index: 2,
          turn_id: TURN_ID,
          role: "assistant",
          text: "Because.",
          partial: false,
          model_id: "smollm3",
          xai: {
            status: "pinned",
            model_type: "autoregressive",
          },
        },
      ],
      next_before: null,
      has_more: false,
    });
  }
  if (value.endsWith("/metrics")) {
    return response({
      run_id: TURN_ID,
      model_type: "autoregressive",
      convergence: [],
      convergence_basis: "tokens",
      total_frames: 1,
      tokens_produced: [1],
    });
  }
  if (value.endsWith("/frames")) {
    return response({
      run_id: TURN_ID,
      positions: [{ t: "Because.", m: false, id: 1 }],
      original_positions: null,
      frames: null,
      original_frames: null,
      records_available: true,
      alternatives: null,
      alternatives_available: false,
      original_alternatives: null,
      remask_edits: [],
    });
  }
  if (value.includes("/api/ui-state")) {
    return response({ value: null });
  }
  if (value.includes("/api/collections")) {
    return response({ success: true, collections: [] });
  }
  if (value === "/api/analytics/system") {
    return response({});
  }
  return response({});
}

async function settle() {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("conversation deep link opens exchanges and shared XAI",
  async () => {
  const page = loadPage({
    scripts: ANALYTICS_SCRIPTS,
    fetchImpl,
    locationSearch: "?conversation=" + SNAPSHOT_ID,
  });
  await settle();

  assert.equal(
    page.registry.get("tab-conversations")
      .getAttribute("aria-selected"),
    "true"
  );
  assert.equal(
    page.registry.get("conversation-detail-modal").open,
    true
  );
  const exchanges = page.registry.get(
    "saved-conversation-exchanges"
  );
  const button = exchanges.querySelector(
    "[data-view-xai-turn]"
  );
  exchanges.dispatch("click", { target: button });
  await settle();

  assert.equal(page.registry.get("detail-modal").open, true);
  assert.match(
    page.registry.get("detail-title").textContent,
    /Saved response XAI/
  );
});
