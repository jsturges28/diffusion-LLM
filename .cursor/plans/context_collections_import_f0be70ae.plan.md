---
name: context collections import
overview: Add context-window metrics counted against the real templated prompt, a file-import control over the prompt box that honors them, and Analytics collections with tabs and star bookmarking. Three commits, sequenced so import can lean on the counting it needs.
todos:
  - id: context-worker
    content: Add describe_context_length to worker_base.py with the model_max_length sentinel guard, put it in the /health ready payload, cache it on ModelManager beside active_tokenizer, and expose it through _models_snapshot
    status: completed
  - id: context-count
    content: Extract build_llada_inputs to deduplicate streaming_sampler.py and llada_worker._store_state; add Backend.prompt_token_count with the templated default and the LLaDA override; add MSG_COUNT_PROMPT dispatched outside gen_lock with its own larger cap
    status: completed
  - id: context-ui
    content: Add the debounced prompt-count request in app.js with its own request-id counter and handler, the readout below the textarea, and the overflow warning keyed on gen_length or max_new_tokens per model
    status: completed
  - id: context-persist
    content: Emit prompt_len on the done frame from all three samplers, save a context block into metadata.json, and add the two Analytics detail rows that stay absent for older runs
    status: completed
  - id: import-prompt
    content: "Add #prompt-actions wrapping the import button and #prompt-history, the hidden file input with byte and character caps, drag-and-drop on the textarea, and a confirm modal before replacing a non-empty prompt"
    status: completed
  - id: collections-store
    content: Add the diffusion_collections key to ui_state.py with the set-membership shape, server-side pruning of deleted run ids on GET /api/ui-state, and the key in PERSIST_KEYS
    status: completed
  - id: collections-ui
    content: "Build the tab strip in #toolbar-left with rename and confirmed delete, the star plus caret in .col-actions with instant favorite and the chooser modal, and the persistently filled star"
    status: completed
  - id: collections-scope
    content: Scope renderTable filtering, onSelectAll and checkedRunIds to the active tab, and prune collection membership in applyDeletions
    status: completed
  - id: verify-docs
    content: Run pytest with the new tests, node --check, ReadLints, the column audit and ruff at 156; update README, ROADMAP, HANDOFF, About and Help; write the manual checklist continuing from item 113
    status: completed
isProject: false
---

# Context metrics, prompt import, and Analytics collections

Three independent arcs, three commits, in this order because the import control's "within the context" promise is only honest once counting exists.

## Commit 1: context window metrics

**The context length is read off the loaded object**, mirroring the reasoning already recorded for `describe_tokenizer`: registry data is static and served with no worker running, so a declared number is free to drift from the checkpoint. Add a sibling in [src/backends/worker_base.py](src/backends/worker_base.py) next to `describe_output_width` (line 276), reading `model.config.max_position_embeddings` with `tokenizer.model_max_length` as a fallback, and returning `None` rather than guessing when neither is sane. `model_max_length` is often a sentinel of `int(1e30)`, so it needs an upper bound check. It joins the `/health` ready payload (line 421) as `context`, is cached on `ModelManager` beside `active_tokenizer` ([server.py](src/web/server.py) line 622), and is exposed through `_models_snapshot` (line 889).

**The count is of the templated sequence, and must come from the code that builds the real inputs.** Counting raw prompt text would silently understate: the chat template adds role markers, and `enable_thinking` changes them. Add `Backend.prompt_token_count(prompt, *, thinking)` whose default mirrors `_build_inputs` in [ar_sampler.py](src/inference/ar_sampler.py) line 107 (SmolLM3 and DiffusionGemma inherit it unchanged), and override it in the LLaDA worker.

LLaDA needs a small refactor first: its template-and-encode block is duplicated between [streaming_sampler.py](src/inference/streaming_sampler.py) line 261 and `_store_state` in [llada_worker.py](src/backends/llada_worker.py) line 257. Extract one `build_llada_inputs(tokenizer, prompt)` and have all three callers use it, so the counted tokens are provably the generated tokens.

**Protocol.** New `MSG_COUNT_PROMPT` / `MSG_COUNT_PROMPT_RESULT` in [protocol.py](src/backends/protocol.py) line 100, dispatched **outside** `gen_lock` in `_ws` (line 514) exactly as `tokenize` is, since an encode is microseconds. This is a separate message rather than a flag on `tokenize`: that path caps at `TOKENIZE_TEXT_MAX_CHARS = 200` and returns one object per token, which for a 40 KB import would be tens of thousands of objects to answer with one integer. The new message returns a count and no pieces, with its own much larger cap.

**Client.** A debounced request on prompt input in [app.js](src/web/static/app.js), copying the `requestTypedPreview` shape (line 2934) but with its own request-id counter, since `handleTokenizeResult` (line 2955) is tightly bound to What If state. A readout below the textarea reads `1,240 / 65,536`, and enters a warning state when the prompt plus the output budget would exceed the window.

The budget key is model-dependent: `gen_length` for LLaDA, `max_new_tokens` for the other two, read through `getParamValues()`. DiffusionGemma chains 256-token canvases, so `max_new_tokens` legitimately exceeds one canvas; its guard compares against the context, not `canvas_length`.

**Persistence.** The samplers already compute `prompt_len` (`streaming_sampler.py` line 277, `dgemma_sampler.py` line 351, and `inputs["input_ids"].shape[-1]` for AR). Emit it on the `done` frame and save it, so a run records the authoritative count rather than the client's guess. Add a `context` block to `metadata.json` in `_save_run_blocking` (line 1359) and two rows to the Analytics run detail, absent for older runs the way the tokenizer rows already are.

## Commit 2: import a prompt from a file

`#prompt-history` is absolutely positioned at the top right of `#prompt-row` ([index.html](src/web/static/index.html) line 57) and is `hidden` whenever history is empty, so the import button cannot live inside it. Wrap both in a `#prompt-actions` flex container that is always present, with import to the left of history, styled off the existing `.prompt-hist-btn` rule ([style.css](src/web/static/style.css) line 375).

- A hidden `<input type="file" accept=".txt,.md,text/plain,text/markdown">` clicked by the button; the file is read with `file.text()`, entirely client-side.
- Drag-and-drop on the textarea, which is nearly free once the parsing exists. No file handling exists anywhere in the frontend today.
- A byte cap checked **before** reading, and a character cap on what is inserted.
- When the prompt box is non-empty, confirm before replacing. The generator page has no confirm modal, so add one following the Analytics `#modal-delete` pattern ([analytics.html](src/web/static/analytics.html) line 464) with `.modal-box-confirm` and `.modal-footer`.
- Markdown is inserted raw, since the model reads it fine and stripping it would misrepresent the file.

After an import the commit-1 readout immediately shows what fraction of the window the file consumes, which is the point of the ordering.

## Commit 3: Analytics collections

**Storage reuses [ui_state.py](src/web/ui_state.py)**, not a new file. A new `diffusion_collections` key in `UI_STATE_KEYS` (line 33) at the same 262,144 cap as `diffusion_new_runs`, holding a JSON string of `[{id, name, runs: [run_id]}]`. Membership is a **set**, so a run can sit in several collections without a later migration. Runs are keyed by `run_id`, which is the folder name (`metrics.py` line 278). Widen the module docstring: one key is now durable user intent rather than a cache.

Server-side pruning of ids for deleted runs on `GET /api/ui-state`, following `_reconcile_new_runs` ([server.py](src/web/server.py) line 1644). Add the key to `PERSIST_KEYS` in [overlays.js](src/web/static/overlays.js) line 927 so it hydrates and write-throughs for free.

**Tabs** go in `#toolbar-left` beside Group by ([analytics.html](src/web/static/analytics.html) line 47). "All" is a view rather than a collection; Favorites is created on first use. Long names truncate with an ellipsis. Rename and delete both ship, with a confirm on delete.

**The star** sits in `.col-actions` beside the existing trashcan, built in `renderTable()` ([analytics.js](src/web/static/analytics.js) line 868) and wired through the delegated `onRowClick` (line 4237). A plain click favorites immediately; a caret appearing on row hover opens the chooser modal. A collected run shows a **persistently filled** star, not one that appears on hover, otherwise the table still cannot be scanned for what was saved.

**Filtering** adds a step before `sortRuns` in `renderTable()`. Two things must follow the active tab or they will act on invisible rows: `onSelectAll` (line 4260) and `checkedRunIds` (line 631), which both iterate `allRuns` today. `applyDeletions` (line 4407) must also drop deleted ids from every collection.

Storage management is deliberately out of scope. Measured: 175 runs occupy 440 MB against 189 GB free, so the pressure this would relieve is roughly 75,000 runs away, and the table already has multi-select bulk delete.

## Verification

`pytest` with new tests for `describe_context_length` (including the `model_max_length` sentinel), `prompt_token_count` per backend, the count message's cap, and the `ui_state` collections key with its pruning. `node --check`, ReadLints, the 70-column audit, and ruff held at its 156 baseline. GPU and display work goes into a fresh `HANDOFF.md` checklist, continuing from item 113.