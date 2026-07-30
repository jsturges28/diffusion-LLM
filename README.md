# Discrete Diffusion LLM Visualizer

## Project Summary

This repository is a local **visual playground and analytics suite** for **discrete diffusion language models**: models that generate text not left-to-right (autoregressive decoding) but by **iteratively denoising a corrupted sequence** over many steps. A web UI (FastAPI + WebSocket) streams every intermediate frame to the browser so you can watch the sequence resolve, scrub back through the history, remask tokens and resume, color tokens by model confidence or the order in which they resolved, diff an edited run against the original, and compare runs in an analytics suite. The goal is an enjoyable, hands-on tool for building intuition about how diffusion LLMs behave, with a strong lean toward explainability (xAI).

The suite hosts two diffusion models plus a first **autoregressive baseline**, all running locally on a single 24 GB GPU (one resident at a time); the autoregressive model can also run on CPU, so a machine without a GPU can still try the suite:

- **[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct):** the first competitive large-scale discrete diffusion language model, pre-trained on 2.3T tokens and instruction-tuned to roughly LLaMA 3 8B quality ([paper](https://arxiv.org/abs/2502.09992)). Masked diffusion over a single canvas, run in bfloat16 (~17 GB VRAM). Supports interactive remasking and resume.
- **DiffusionGemma-26B-A4B:** Google's block-autoregressive text-diffusion model, a 26B-parameter Mixture-of-Experts (\~4B active) built on Gemma. Run here as a self-quantized 4-bit NF4 checkpoint (\~18 GB VRAM). Denoises 256-token canvases with adaptive stopping and an optional reasoning (thinking) channel. Single-canvas runs now also support interactive remasking and resume (via seed-canvas re-entry); multi-canvas resume is still on the roadmap.
- **[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) (autoregressive baseline):** a 3B decoder-only transformer that generates the ordinary way, left to right. Added as a contrast to the diffusion models: it streams token-by-token, one frame per new token, with per-token sampling confidence, reusing the same scrubber/save/analytics tooling as a left-to-right replay. It has an optional extended-reasoning (thinking) channel and runs in bfloat16 on GPU or CPU (chosen per activation). Diffusion-only affordances (Edit Frames, the Diff overlay, Commit Order, the convergence chart) are hidden for it, while timing, confidence, and the Heatmap remain.

Because these models depend on incompatible `transformers` versions and a single large model already saturates 24 GB, the app uses a **supervisor plus per-model worker** architecture: a lightweight server spawns exactly one model worker at a time, each in its own virtual environment, and proxies the browser WebSocket to it.


## How It Works

### Autoregressive vs diffusion

Autoregressive LLMs generate one token at a time, left to right:

$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{<t})$

Diffusion LLMs instead start from a corrupted sequence and refine the whole thing in parallel over *N* steps, using **bidirectional** attention (no causal mask), re-corrupting the least certain positions between steps until the sequence converges.

### LLaDA: masked discrete diffusion

- **Forward process (corruption):** independently replace each token with `[MASK]` with probability *t* in [0, 1]. At *t* = 0 the text is clean; at *t* = 1 everything is masked.
- **Reverse process (generation):** starting from a fully masked canvas, the Transformer predicts all masked positions at once, then **re-masks** the least confident predictions. Repeat for *N* steps until nothing is masked.

The training loss is cross-entropy on masked positions only, weighted by 1/*t*, which provides a variational upper bound on negative log-likelihood. This makes LLaDA a principled generative model, not a fill-in-the-blank system like BERT. In the UI, unresolved positions render as `░`; their opacity tracks the model's live predicted confidence for that position, so a mask starts solid and brightens toward a "shatter" as the model grows confident about what it will become, then resolves into the token.

### DiffusionGemma: block-autoregressive text diffusion

DiffusionGemma denoises a fixed **256-token canvas** on an encoder-decoder MoE backbone. Rather than a `[MASK]` placeholder, unresolved positions carry noisy tokens that the sampler renoises between steps under an entropy bound. Two properties make it distinct from LLaDA:

- **Adaptive stopping:** a canvas can finish in fewer than the configured maximum steps once its predictions stabilize, so simpler prompts run faster.
- **Block-autoregressive chaining:** for outputs longer than one canvas, it commits a canvas and then seeds the next, chaining multiple canvases. The status readout therefore reads `Step X, Canvas Y` rather than a fixed step total.

An optional **thinking** channel exposes a step-by-step reasoning pass, which the UI separates into a collapsible panel above the answer.

### Confidence and the heatmap

Every resolved token carries a **confidence** value in [0, 1], and every frame carries the mean confidence of its resolved tokens. The source differs per model, cheap by default:

- **LLaDA:** the softmax probability of the token at the moment it was unmasked (fixed thereafter, since resolved tokens are never revisited).
- **DiffusionGemma:** a lightweight **stability proxy** by default (how many consecutive steps a position has held the same prediction). Enabling the **Entropy signal** toggle switches to the true max-softmax probability from the model's logits (more faithful, but slower and heavier per step).

After a run, hovering any token shows its position and confidence for that frame, and the **Heatmap** overlay recolors resolved tokens by confidence. Per-frame mean confidence and canvas indices are also persisted for the analytics charts.

### Commit order and counterfactual diff

The frame history also drives two explainability overlays:

- **Commit order** colors each resolved token by *when* it settled into its final value, on a gradient from light green (early) to red-orange (late). This exposes the model's resolution trajectory across a run.
- **Diff vs Original** becomes available after you edit and resume a run. It compares the edited output against a snapshot of the original run, stacking the two with independent opacity sliders and an optional *difference blend* (matching tokens cancel to black, divergences glow), so you can see exactly how an intervention propagated.

Both overlays are derived from the recorded per-token frames. In the live generator they render across every frame; the underlying per-token data (display text, mask flag, vocab id, and confidence) is also persisted with each saved run, so both overlays are reviewable after the fact in the Analytics Suite (see below) rather than being lost on reload.

### Sampling parameters

**LLaDA**

| Parameter | Description |
|---|---|
| Steps | Number of denoising steps. More steps generally mean higher quality and slower generation. |
| Gen Length | Length of the masked canvas (output token count). Must be divisible by Block Length. |
| Block Length | Block size for semi-autoregressive sampling. When smaller than Gen Length, blocks resolve left-to-right with diffusion inside each block; set equal to Gen Length for pure diffusion. |
| Temperature | Gumbel noise temperature for categorical sampling. 0 is greedy (argmax). |
| CFG Scale | Classifier-free guidance strength. 0 disables it; higher values increase prompt adherence. |
| Seed | Random seed for reproducibility; -1 is nondeterministic. |
| Remasking | Strategy: `low_confidence` (default, re-mask least confident tokens) or `random`. |

**DiffusionGemma**

| Parameter | Description |
|---|---|
| Max Tokens | Output budget. Generation happens in 256-token canvases; larger budgets chain multiple canvases. |
| Denoising Steps | Upper bound on steps per canvas. Adaptive stopping may use fewer. |
| Temp Start / Temp End | Endpoints of a linear temperature schedule across the denoising steps (hotter early, cooler late). |
| Seed | Random seed for reproducibility; -1 is nondeterministic. |
| Thinking | Enables the step-by-step reasoning channel, shown in a separate panel. |
| Entropy signal | Computes true per-token confidence from logits for the heatmap (slower, off by default). |

All parameters are configurable in the web UI with recommended bounds enforced by default. An **Experimental** toggle lifts the bounds for exploratory use.


## Architecture

The single-model, single-process design has been replaced by a model-agnostic **supervisor plus workers** layout driven by a shared contract.

```
Browser (shared frontend)
  |  /ws + /api
  v
Supervisor  (.venv, no torch/transformers)
  - static assets + Analytics API + Save endpoint
  - Model Manager: spawns/stops one worker, VRAM-exclusive, pre-flight VRAM check
  - /ws bidirectional proxy to the active worker
  |
  |  spawn: <model venv> python -m src.backends.run_worker --model <id>
  v
Model Worker  (exactly one alive)
  - LLaDA worker          .venv          transformers 4.38.2
  - DiffusionGemma worker .venv-dgemma   transformers 5.13
  - SmolLM3 worker        .venv-ar       transformers >= 4.53
```

Pages and entry point: the app opens on a **Main Menu** at `/` (a looping title-screen video over a GPU/VRAM-aware model picker). Selecting a model activates its worker and enters the generator at `/generate`; the generator is gated behind model selection, so hitting it without an active model redirects back to the menu. The Analytics Suite at `/analytics.html` is model-agnostic and always available. The desktop app opens on the same menu.

Why process isolation: LLaDA loads through custom remote modeling that pins `transformers==4.38.2`, while DiffusionGemma requires `transformers` v5. They coexist only in separate virtual environments, and since a single model already saturates the 24 GB GPU, only one worker is ever alive. Switching models stops the current worker (freeing VRAM), runs a pre-flight VRAM check against the target model's requirement, then spawns the next worker and waits for it to report ready (or surfaces a clear error).

The contract lives in `src/backends/`:

- `protocol.py`: typed WebSocket and parameter schema. Per-token shape is `{t, m, id, c?}` where `m` marks an unresolved position and `c` is confidence; each frame also carries optional `canvas_index` and a `mean_conf`, with `total_steps` allowed to be null for adaptive runs.
- `registry.py`: data-only model registry (id, display name, venv Python, worker module, checkpoint, `min_vram_gib`, capabilities, and the parameter schema). Drives the frontend selector and the dynamic parameter panel.
- `worker_base.py`: shared worker FastAPI scaffolding (the `/ws` loop, cancel handling, elapsed timing, load-error reporting).
- `run_worker.py`: generic launcher that imports and serves the selected model's worker module.


## Project Structure

```
.
├── main.py                           # Supervisor entry point (uvicorn, browser UI)
├── desktop.py                        # Desktop launcher (pywebview native window)
├── requirements.txt                  # .venv: supervisor + LLaDA worker (transformers 4.38.2)
├── requirements-dgemma.txt           # .venv-dgemma: DiffusionGemma worker (transformers 5.13)
├── requirements-ar.txt               # .venv-ar: SmolLM3 autoregressive worker (transformers >= 4.53)
├── requirements-desktop.txt          # optional: pywebview[qt] desktop wrapper (installs into .venv)
├── README.md
├── LICENSE
├── src/
│   ├── backends/
│   │   ├── protocol.py               # Shared WS/param/model contract
│   │   ├── registry.py               # Model registry (models, params, capabilities, VRAM)
│   │   ├── worker_base.py            # Shared worker FastAPI scaffolding
│   │   ├── run_worker.py             # Generic per-model worker launcher
│   │   ├── llada_worker.py           # LLaDA backend
│   │   ├── dgemma_worker.py          # DiffusionGemma backend
│   │   └── smollm3_worker.py         # SmolLM3 autoregressive backend
│   ├── inference/
│   │   ├── llada_sampler.py          # Core LLaDA sampling loop + history recording
│   │   ├── streaming_sampler.py      # LLaDA live streaming + per-token confidence
│   │   ├── dgemma_sampler.py         # DiffusionGemma live streaming + confidence
│   │   ├── dgemma_nf4.py             # NF4 (4-bit) MoE-expert quantization
│   │   ├── ar_sampler.py             # Autoregressive token-by-token streaming + confidence
│   │   └── render_gif.py             # Render diffusion history frames to GIF
│   ├── analytics/
│   │   └── metrics.py                # Run parsing, convergence + canvas boundaries
│   └── web/
│       ├── server.py                 # Supervisor: model manager, /ws proxy, analytics, save, UI state
│       ├── ui_state.py               # Durable server-side UI state (Settings, cues) store
│       └── static/
│           ├── menu.html             # Main Menu (landing page, model selection)
│           ├── menu.js               # Menu: model/VRAM fetch + activate + navigate
│           ├── index.html            # Generator page (gated behind model selection)
│           ├── style.css             # Dark terminal aesthetic (shared)
│           ├── app.js                # WebSocket client + frame rendering + heatmap
│           ├── overlays.js           # Shared overlay math + layered diff builder
│           ├── analytics.html        # Analytics Suite page
│           ├── analytics.css         # Analytics-specific styles
│           ├── analytics.js          # Analytics charts + run browser
│           ├── custom_select.js      # Shared in-app dropdown widget
│           └── assets/               # Title-screen video (title-screen.webm/.mp4)
├── assets/
│   ├── icon.svg                      # App icon source (vector; app-menu launcher)
│   └── icon.png                      # Rasterized window icon (via scripts/render_icon.py)
├── scripts/
│   ├── quantize_diffusiongemma_nf4.py # Produce the NF4 checkpoint from the bf16 base
│   ├── install_desktop_entry.sh      # Generate a Linux .desktop launcher entry
│   ├── render_icon.py                # Render assets/icon.png from the icon geometry
│   ├── spike_diffusiongemma.py       # Standalone load + generate probe
│   └── ws_smoke_test.py              # End-to-end supervisor/worker smoke test
├── Results/                          # Saved runs from the web UI (Save button)
│   ├── ui_state.json                 # Durable UI state (Settings, "new run" cue, prompt history)
│   └── <timestamp>_<model>/
│       ├── metadata.json
│       ├── final.txt
│       ├── history.txt
│       ├── tokens.json               # Per-frame token records (durable overlays)
│       ├── original_tokens.json      # Pre-edit snapshot (edited runs only)
│       └── diffusion.gif
└── archive/                          # Old reference files and notes
```


## Setup

**Platform.** This app is currently built and tested only on **Ubuntu 24.04** (Linux). The desktop app, launcher script, and GPU/VRAM tooling assume a Linux environment, so other operating systems are unsupported for now; broader cross-OS support (Windows, macOS) is a future goal, not a current guarantee.

Requires Python 3.10+ and a CUDA GPU. The two models live in separate virtual environments because of their conflicting `transformers` versions.

**Supervisor and LLaDA (`.venv`, transformers 4.38.2):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LLaDA weights (~16 GB) download automatically from Hugging Face on first use. The supervisor itself runs in this environment and never imports torch or transformers.

**DiffusionGemma (`.venv-dgemma`, transformers 5.13), optional:**

```bash
python3 -m venv .venv-dgemma
.venv-dgemma/bin/pip install -r requirements-dgemma.txt
```

DiffusionGemma is gated on Hugging Face. Accept its license, download the bf16 base, then produce the local 4-bit checkpoint (only the MoE experts are quantized to NF4, which is what makes it fit in 24 GB):

```bash
.venv-dgemma/bin/python scripts/quantize_diffusiongemma_nf4.py
```

This writes the NF4 checkpoint to the path referenced by the registry (`~/models/diffusiongemma-26B-A4B-it-nf4`). If you only want LLaDA, you can skip this environment entirely; the model selector will still list DiffusionGemma but activation will fail gracefully with a clear message.

**SmolLM3 (`.venv-ar`, transformers >= 4.53), optional:**

```bash
python3 -m venv .venv-ar
.venv-ar/bin/pip install -r requirements-ar.txt
```

SmolLM3-3B weights (~6 GB) download automatically from Hugging Face on first activation. The pinned torch is the standard CUDA build, so this environment runs the model on the GPU when one is present and on CPU otherwise; the Main Menu row for SmolLM3 carries a CPU/GPU toggle. On a machine with no GPU, this is the model you can still run.

If you have no GPU and want to avoid downloading the large CUDA libraries, install the CPU-only torch wheel first, then the rest:

```bash
.venv-ar/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv-ar/bin/pip install -r requirements-ar.txt
```

**Desktop app (`.venv`, optional).** The same UI can run in a native window instead of a browser tab via [pywebview](https://pywebview.flowrl.com/). It is not part of the core install; add it into `.venv` with the optional requirements file, which pulls the Qt/QtWebEngine (Chromium) backend that renders most smoothly:

```bash
.venv/bin/pip install -r requirements-desktop.txt
```

`desktop.py` automatically prefers the Qt backend when present and gracefully falls back to GTK/WebKit otherwise. If you prefer the lighter GTK/WebKit backend, install its system libraries and binding instead (`sudo apt install libgirepository1.0-dev libcairo2-dev gir1.2-webkit2-4.1` then `.venv/bin/pip install "pywebview[gtk]"`). To add an app-menu launcher with an icon on Linux, run `scripts/install_desktop_entry.sh` (it generates a `.desktop` entry with the correct absolute paths for your checkout).

On an X11 session (rather than Wayland), the Qt backend needs the `xcb` platform plugin's runtime dependency, which Qt 6.5+ does not bundle: if the app aborts on launch with "xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin," install it with `sudo apt install libxcb-cursor0`. This is easy to hit after an NVIDIA driver update, which can flip the login session from Wayland to X11.


## Quickstart

```bash
python3 main.py            # or: python3 main.py --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). You land on the **Main Menu** (titled **LLM Visualizer**): a looping title screen over a model picker that shows the detected GPU (with free VRAM) and CPU (with free RAM). Each row carries a family glyph (diffusion vs autoregressive) beside its name and a device tag on the right: a static **GPU** for the diffusion models, and a **GPU / CPU** toggle for the autoregressive model. A small pill extending left of each tag shows the signed **VRAM headroom** (green `+X.X GiB` if it fits, red `-X.X GiB` if it is short); hovering it details required vs available VRAM. Available counts memory reclaimed from unloading the current model, so switching accounts for the resident one freeing up.

Selecting a model contracts the menu to that row and asks you to **confirm** (green check to load, red X to go back). **On the first activation its weights download from Hugging Face** (LLaDA ~16 GB, SmolLM3 ~6 GB, cached under `~/.cache/huggingface`); a model whose weights are not cached yet is gated behind a **Click to Download** veneer with a progress bar so you can pre-fetch it (its description stays hidden until the download finishes and is then revealed, and a success/error message with **Ok** confirms the result). A row reading **Resident** is the model currently loaded in memory. Type a prompt, adjust parameters, and click **Generate** to watch generation stream live. To switch models (or the autoregressive model's device) later, use the **Model** selector in the generator header, which confirms the switch the same way; its collapsed tag tickers between the device and VRAM headroom (a "Device tag ticker" Setting toggles this off), and hovering a listed model shows its full VRAM readout (only one model is resident at a time).

After a run completes, a **Save** button appears and a **frame scrubber** slides into view below the output area. While a save is in progress the status bar shows "Saving run…" and the scrubber is dimmed and frozen until it finishes.

**Prompt history.** A small history icon at the top-right of the prompt box (shown once you have run at least one prompt) recalls earlier prompts, persisted per-browser. Click it to browse: `‹` / `›` cycle through your prompts (with an `i / N` position counter), the green check keeps the shown prompt for editing, and the red cross restores what you had typed. Prompts are recorded automatically each time you Generate.

**New Run.** Once you have finalized a run by saving an edit (see Interactive remasking below), the **Generate** button becomes **New Run** in the same spot. Clicking it clears the canvas and prompt for a fresh start (the prompt box shows its "Enter a prompt" placeholder) and restores the **Generate** button.

**Desktop window.** To run the UI in a native window instead of a browser (after the optional pywebview setup in Setup):

```bash
.venv/bin/python desktop.py
```

This owns the server lifecycle: it starts the supervisor on a private localhost port, opens the window, and gracefully stops the active model worker (freeing its VRAM) when you close it. The browser path (`main.py`) still works and serves the same app.

**Freeing stuck GPU memory.** A worker is normally stopped when you switch models or close the app, and the supervisor sweeps stray workers on startup (plus a `PR_SET_PDEATHSIG` guard). If a hard crash ever leaves one behind holding VRAM, list GPU processes with `nvidia-smi` and clear them with `pkill -f "src.backends.run_worker"`, then relaunch.

#### Interactive remasking and resume

The scrubber steps through every intermediate frame. Navigate with the slider, the arrow buttons, or the keyboard (Left / Right arrows, Home / End). The remasking and resume tools work for LLaDA and for single-canvas DiffusionGemma runs; on multi-canvas DiffusionGemma runs the **Edit Frames** button is disabled (multi-canvas resume is on the roadmap).

**Guided multi-frame editing.** Click **Edit Frames** to chain edits across one or more frames:

1. **Select a frame:** the scrubber starts at frame 0 and only allows forward navigation. Navigate to the frame you want to edit and click **Select Frame**.
2. **Remask tokens:** click resolved tokens to remask them (they turn orange). Click again to deselect. When satisfied, click **Lock In**. For LLaDA the tokens are set back to `[MASK]`; for DiffusionGemma they are *renoised*, so committed neighbours may also shift on resume.
3. **Choose next action:**
   - **Edit Another Frame:** enters target selection mode. A faded preview of the original run is shown at each frame as a reference, noting that output will diverge from your edits. Navigate to the target frame and click **Run to Here**; the model resumes up to that frame and places you into edit mode on it.
   - **Resume to End:** resumes the model through all remaining steps to produce the final output.

A single edit followed by **Resume to End** is the simple case; you can also chain as many edits as you like. Each partial resume generates only the frames between your last edit and the next target, so earlier edits propagate forward. The scrubber enforces forward-only navigation; later edits cannot precede earlier ones. Clicking **Exit** discards the in-progress edits and restores the original run. All remask edits (frame indices and token positions) are recorded and saved with the run.

**Confirming an edit.** Clicking **Edit Frames** first auto-saves the current (original) run if you have not saved it yet, so the pre-edit run is always preserved as its own entry. After **Resume to End** completes, the editor stays open on the final frame and offers two choices in place of **Select Frame**: a green **Confirm** (checkmark), which saves the edited run, and a blue **Retry** (counter-clockwise arrow), which discards the edits and restarts editing from frame 0 (reusing the already-saved original, so it does not re-save). Once an edited run has been saved, **Edit Frames** is disabled for that run (with a "this run already has a saved edit" tooltip) until you **Generate** again, so a single run cannot accrue two conflicting saved edits.

#### Visual overlays and settings

A collapsible **Overlay** drawer in the top-right of the output area recolors the frame you are viewing. It defaults to **None** and offers:

- **Heatmap:** recolor resolved tokens by confidence (dim, desaturated tones for low, bright green for high).
- **Diff vs Original:** compare an edited run against the original. It is listed but disabled until you have edited and resumed a run. When active, a slim control row below the scrubber provides independent **Original** / **Edited** opacity sliders and a **Difference blend** toggle, alongside a `Diverged N/total` readout.

Persistent preferences live in **Settings** (in the header) and are saved per-browser; the modal stages changes behind **Save** / **Reset** with inline status feedback:

- **Show Commit Order:** tint resolved tokens by the step at which they settled, from light green (early) to red-orange (late), with a matching gradient legend in the status bar.
- **Highlight tokens:** add a light hover highlight so the full span of the hovered token is easy to see.
- **Render diffusion-style text:** dynamic status messages resolve from scrambled block-glyph noise, like a denoising pass, in the green palette (skipped automatically under `prefers-reduced-motion`). A **Mode** sub-setting picks **Default** (resolve once) or **Cycle** (keep re-diffusing while the status is active). The same effect drives small button interactions (Shuffle, the Generate/New Run idle cycle, and Lock In dissolving into mask glyphs).

The status bar reflects both (`Highlighted Tokens: On/Off`, `Show Commit Order: On/Off`). An explicit overlay selection (Heatmap or Diff) takes precedence; otherwise Commit Order, when enabled, is the ambient tint. Hovering any token still shows its position (`Token X/total` for LLaDA, `Token: X` for DiffusionGemma) and confidence for that frame (masked tokens report 0).

#### Analytics Suite

Click **Analytics** in the header (or navigate to `/analytics.html`) to open the Analytics Suite. It reads saved runs from `Results/` and provides interactive charts for comparing behavior across configurations and models.

- **Run browser:** group runs by date, model, prompt, or whether a run was edited. Columns are shared across models and ordered Date, Model, Prompt, Time, and a sortable **Edited** column (a checkmark, textured from the diffusion mask glyph, marks runs that carry a pre-edit snapshot for a Diff vs Original; blank otherwise). The leading Date column carries the pulsing green "new run" dot. Clicking a row opens a wide **detail modal** (fades in like About/Help; close with the X or by clicking outside) laid out with the token overlay canvas as the centerpiece on the left and the run's info plus the convergence, timing, and confidence charts stacked on the right.
- **Manage runs:** delete a saved run with the row's red trashcan action. Select rows with the checkboxes to enable **bulk delete** (a trashcan with the selected count appears in the actions header) and highlight the selected rows. Either path opens a confirmation modal ("Delete this run?" / "Delete N runs?") showing the folder path or count, and a toast confirms the deletion.
- **Convergence chart:** percentage of resolved tokens per frame. User remask edits are highlighted as blue segments with hover details.
- **Timing chart:** cumulative elapsed time per frame (accumulates across resumes). Remask transitions are highlighted in green; the detected GPU is shown in the header.
- **Confidence chart:** mean per-token confidence per frame, which climbs as a canvas converges. Shown for runs saved with confidence data.
- **Canvas boundaries:** for multi-canvas DiffusionGemma runs, dashed amber markers on the charts mark where one canvas commits and the next begins. Single-canvas runs show none.
- **Token overlay + per-frame scrubber:** a scrubbable view of the run's tokens inside the detail modal, with a corner **Overlay** drawer mirroring the generator's. A frame scrubber (prev / slider / next, `Frame i / N`) replays every saved frame through the active overlay, opening on the final frame. The drawer offers **None** and **Heatmap** for every run with token records (Heatmap recolors resolved tokens by their persisted confidence), plus **Commit Order** and **Diff vs Original** for diffusion runs. Commit Order tints each token by when it settled (early-to-late gradient legend); Diff vs Original (available only for edited runs with a saved snapshot) stacks the original and edited runs with independent **Original** / **Edited** opacity sliders and a **Difference blend** toggle, plus a `Diverged N/total` readout, matching the generator's layered diff (the original layer clamps to its final frame past its end). Autoregressive runs, which have no masked canvas, show only None + Heatmap. Hovering a token shows its position and persisted confidence. This makes the generator's explainability overlays durable and scrubbable post-hoc; runs saved before durable overlays (or without token data) show a short unavailable note.
- **Chart controls:** scroll-wheel zoom and +/-/Reset on every chart. Tooltips are kept fully inside the plot area (never spilling onto the axes) and each chart has a toggle to hide/show its tooltip box; when the box would cover a line, the covered segment and the hovered point glow through it.

#### Saving and reproducibility

Clicking **Save** writes a timestamped folder under `Results/` containing `metadata.json`, `final.txt`, `history.txt` (frame-by-frame snapshots), `tokens.json` (per-frame, per-token records: display text, mask flag, vocab id, and confidence), and `diffusion.gif`. Edited runs also write `original_tokens.json`, the pre-edit snapshot that powers the durable Diff vs Original overlay. The metadata captures the model, prompt, hyperparameters, any remask edits, per-frame timing, canvas indices, mean confidence, and reproducibility info: seed, GPU name, git commit, and the worker's torch/transformers versions.


## Implementation Status

- [x] Supervisor plus per-model worker architecture with process isolation (separate venvs)
- [x] Shared backend contract: protocol, model registry, worker scaffolding, generic launcher
- [x] Model selector with schema-driven dynamic parameter panel and per-model capabilities
- [x] LLaDA-8B-Instruct: masked diffusion, low-confidence remasking, CFG, semi-autoregressive blocks
- [x] DiffusionGemma-26B-A4B: self-quantized NF4 experts, 256-token canvases, adaptive stopping
- [x] DiffusionGemma thinking (reasoning) channel with split-panel view
- [x] Live diffusion visualization (FastAPI + WebSocket) with recommended bounds and Experimental mode
- [x] Real-time client-side validation (bounds, divisibility, negative values)
- [x] Interactive remasking and resume: frame scrubber, click-to-remask, resume from any frame (LLaDA and single-canvas DiffusionGemma via seed-canvas re-entry)
- [x] Guided multi-frame editing with faded original-run previews and partial resumes
- [x] Per-token confidence: softmax at reveal (LLaDA), stability proxy or true entropy (DiffusionGemma)
- [x] Grouped overlay picker (None / Heatmap / Diff vs Original), per-token hover tooltips, and token-hover highlight option
- [x] Commit-order (resolution-step) token coloring; counterfactual "Diff vs Original" overlay with opacity sliders and difference blend
- [x] Durable overlays: per-token records (text, mask, id, confidence) plus the pre-edit snapshot persisted per run, and a static commit-order / Diff-vs-Original viewer in the Analytics Suite
- [x] Analytics run detail as a wide fade-in modal with a corner overlay drawer, a sortable Edited column, and streamlined grouping
- [x] Analytics per-frame token scrubber and durable Heatmap: the detail modal's overlay replays every saved frame (None / Heatmap / Commit Order / Diff), Heatmap recoloring by persisted confidence, with Commit Order and Diff gated to diffusion runs (autoregressive runs get None + Heatmap)
- [x] Guided-edit confirm/retry review step and Edit-Frames lock after an edited run is saved
- [x] Persistent Settings (highlight tokens, commit order, diffusion text) with staged Save/Reset and status-bar readouts
- [x] Analytics Suite: model-aware run browser, convergence, timing, confidence, canvas-boundary markers
- [x] Analytics run deletion (confirmation modal + toast) and contained, toggleable chart tooltips with line burn-through
- [x] Reproducibility metadata (seed, GPU, git commit, library versions) and deterministic seeding
- [x] Graceful VRAM handling: pre-flight free-memory check and worker load-error reporting
- [x] Save runs (metadata, history, final text, GIF) with per-frame timing and confidence
- [x] Optional desktop app: pywebview native window that owns the server lifecycle (graceful shutdown frees VRAM) plus a Linux app-menu launcher
- [x] Prompt history (persisted per-browser) with a browse control, and a New Run flow that clears the canvas after a finalized run
- [x] Main Menu landing page: looping title-screen video (WebM/MP4) over a GPU/VRAM-aware model picker (Available / Insufficient VRAM) that greys out models that will not fit; generation gated behind model selection
- [x] Analytics layered "Diff vs Original" overlay (Original/Edited opacity sliders + difference blend) mirroring the generator
- [x] Opt-in "diffusion-style text" effect (scramble-to-resolve on status messages) with a Default/Cycle mode, honoring reduced motion, reused for Shuffle/Generate/Lock-In button micro-interactions
- [x] Confidence-driven mask rendering: masks use the accent color and their opacity tracks the model's live predicted confidence per position (LLaDA), rising to a "shatter" as a token nears its reveal
- [x] Randomize-remasks control (slider + N-of-M + Shuffle) in Edit Frames; Edit Frames opens on the first editable frame
- [x] Analytics "new run" cue: an unseen-run count badge on the Analytics link and Main Menu plus per-row green dots cleared when a run is opened; deleting a run decrements it, and the cue self-heals against runs that no longer exist
- [x] In-place edited-run save: an edited/bundled run updates its pre-edit folder so it is a single Analytics row rather than a duplicate
- [x] Robust GPU detection (resolves nvidia-smi across launch environments, with a driver/library-mismatch message)
- [x] Durable server-side UI state (`Results/ui_state.json` via `/api/ui-state`): Settings, the "new run" cue, prompt history, and the generate teaser survive restarts and are shared across the browser and desktop app, independent of the window origin
- [x] Analytics table rework: reordered columns (Date, Model, Prompt, Time, Edited), a diffusion-textured Edited checkmark, checkbox row highlighting, and multi-select bulk delete
- [x] Desktop launcher persistence: a stable window port (with ephemeral fallback) and a persistent web-storage profile
- [x] First autoregressive model (SmolLM3-3B) in a dedicated `.venv-ar`: token-by-token streaming with per-token confidence, per-activation CPU/GPU device selection (CPU-capable for GPU-less hosts), and a `model_type` gate that hides diffusion-only UI (Edit Frames, Diff, Commit Order, convergence) while keeping timing, confidence, and the Heatmap
- [x] Non-blocking activation with a menu progress bar and Cancel; a startup sweep + `PR_SET_PDEATHSIG` guard so a crashed supervisor cannot orphan a VRAM-holding worker
- [x] Signed VRAM-headroom pill on each device tag (green/red, accounting for the reclaimable resident model); menu GPU + CPU readout; model-family glyphs; select-to-confirm on the menu and dropdown
- [x] "Click to Download" veneer that pre-fetches an uncached model's weights (with a progress bar, no VRAM) before selection; Analytics **Processor** column and per-run timing header (GPU/CPU name)
- [x] Smooth download progress via a cache disk-size poller (repo total from Hub metadata, polling the `blobs/` directory including in-progress parts), with Xet disabled before the first Hub import so the classic downloader is used
- [x] Model dropdown polish: fixed-width device pill (the signed headroom is shrunk to fit), a collapsed-width option list with ellipsized names, and a hover VRAM side-popup whose trailing +/-X is tinted green (fits) or red (short)
- [x] Loaded model highlighted (and inert) in the dropdown with its loaded device locked while the other device stays switchable; the device-tag ticker is gated off on CPU (headroom is GPU-only)
- [x] Autoregressive step counter is 1-based, so a full N-token run reads "Step N/N" (matching the diffusion convention)


## Roadmap

Detailed, living notes for each item (technical hooks, files to touch, open questions) live in [ROADMAP.md](ROADMAP.md). Development conventions for agents and contributors live in [AGENTS.md](AGENTS.md), and the living per-session handoff (what shipped and where to pick up next) is [HANDOFF.md](HANDOFF.md).

**Phase 2 (shipped for single-canvas): DiffusionGemma interactive remask and resume.** Single-canvas runs can now be re-entered via `decoder_input_ids` as a seed canvas: remasked positions are renoised and denoising continues under a reduced step budget. The remaining work is multi-canvas resume, which must target the correct canvas while preserving already-committed prior canvases (encoder-decoder KV-cache and adaptive stopping make this the hard part).

**Phase 3: Multimodal image input.** Requires `AutoProcessor` plus torchvision and additional vision-tower VRAM, so it is deferred until the text foundations are solid.

**Experimental and xAI ideas (open for deliberation).** The suite is shaping up as an explainability playground. Two candidate directions shipped this cycle as overlays: commit-order (resolution-step) coloring and the counterfactual "Diff vs Original" comparison. Future sessions can explore further analysis features, for example top-k alternatives on hover, per-position uncertainty trajectories, cross-model comparisons on identical prompts, or an autoregressive baseline for contrast. These are intentionally open and to be scoped together.

### Possible extensions

- [ ] Shared tabbed **Settings page** (`/settings.html`) reached from a gear icon in the header and on the Main Menu
- [ ] Autoregressive analysis tools for SmolLM3: top-k alternatives on hover, "change the last token" resume, per-position entropy sparkline
- [ ] Side-by-side comparison with autoregressive generation
- [ ] Alignment experiments (RLHF / DPO) or fine-tuning on custom instruction data


## References

- **LLaDA paper:** Nie et al., "Large Language Diffusion Models," NeurIPS 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **LLaDA model:** [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on Hugging Face
- **DiffusionGemma model:** [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) on Hugging Face
