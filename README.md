# Discrete Diffusion LLM 

## Project Summary

This repository demonstrates **diffusion-style text generation**: instead of generating tokens left-to-right (autoregressive decoding), we generate by **iteratively denoising a fully masked sequence** using a masked diffusion model (MDM).

We use [LLaDA 8B Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) — the first competitive large-scale discrete diffusion language model — as our backbone. The model was pre-trained from scratch on 2.3T tokens and fine-tuned on 4.5M instruction pairs, achieving performance comparable to LLaMA 3 8B on standard benchmarks ([paper](https://arxiv.org/abs/2502.09992)).

The project has two interfaces:
- **CLI** — run a single prompt and save artifacts (final text, frame history, GIF animation).
- **Web UI** — launch a local server and watch the diffusion process unfold live in the browser, with streaming intermediate frames over WebSocket.


## How It Works

### Autoregressive vs Diffusion

Autoregressive LLMs generate one token at a time, left to right:

$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{<t})$

LLaDA instead uses a **masked diffusion** process:

- **Forward process (corruption):** independently replace each token with `[MASK]` with probability *t* ∈ [0, 1]. At *t* = 0 the text is clean; at *t* = 1 everything is masked.
- **Reverse process (generation):** starting from a fully masked sequence, a Transformer (with **bidirectional** attention — no causal mask) predicts all masked tokens simultaneously, then **re-masks** the least confident predictions. Repeat for *N* steps until the sequence is fully unmasked.

The training loss is cross-entropy on masked positions only, weighted by 1/*t*, which provides a variational upper bound on negative log-likelihood — making LLaDA a principled generative model, not just a fill-in-the-blank system like BERT.

### Sampling Parameters

| Parameter | Description |
|---|---|
| `--steps` | Number of denoising steps. More steps = higher quality, slower generation. |
| `--gen_length` | Length of the masked canvas (output token count). |
| `--block_length` | Block size for semi-autoregressive sampling. When < `gen_length`, blocks are generated left-to-right, with diffusion within each block. Set equal to `gen_length` for pure diffusion. |
| `--temperature` | Gumbel noise temperature for categorical sampling. 0 = greedy (argmax). |
| `--cfg_scale` | Classifier-free guidance strength. 0 = disabled. Higher values increase prompt adherence. |
| `--remasking` | Strategy: `low_confidence` (default, re-mask least confident tokens) or `random`. |


## Project Structure

```
.
├── main.py                           # CLI entry point (--sample, --serve)
├── README.md
├── requirements.txt
├── LICENSE
├── parameter_triples.txt             # Notes on valid parameter combinations
├── src/
│   ├── inference/
│   │   ├── llada_sampler.py          # Core LLaDA sampling loop + history recording
│   │   ├── streaming_sampler.py      # Async generator wrapper for live streaming
│   │   └── render_gif.py             # Render diffusion history frames to GIF
│   └── web/
│       ├── server.py                 # FastAPI + WebSocket server
│       └── static/
│           ├── index.html            # Single-page app
│           ├── style.css             # Dark terminal aesthetic
│           └── app.js                # WebSocket client + frame rendering
├── artifacts/                        # Auto-generated run outputs (--sample)
│   └── <timestamp>_llada/
│       ├── metadata.json             # Run config + final text
│       ├── final.txt                 # Decoded output
│       ├── history.txt               # All intermediate frames
│       └── diffusion.gif             # Animated diffusion visualization
├── Results/                          # Saved runs from the web UI (Save button)
│   └── <timestamp>_llada/
│       ├── metadata.json
│       ├── final.txt
│       ├── history.txt
│       └── diffusion.gif
└── archive/
    └── README_old.md                 # Previous project direction
```


## Setup

Requires Python 3.10+ and a CUDA GPU (LLaDA-8B in bfloat16 needs ~17 GB VRAM).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model weights (~16 GB) are downloaded automatically from Hugging Face on first run.


## Quickstart

### Web UI (recommended)

```bash
python3 main.py --serve
```

Open [http://localhost:8000](http://localhost:8000) in a browser. The model loads in the background (~30 seconds on first run) — a loading overlay shows progress. Once ready, type a prompt, adjust parameters, and click **Generate** to watch the diffusion process stream live.

The UI includes controls for all sampling parameters (steps, generation length, block length, temperature, CFG scale, remasking strategy) with recommended bounds enforced by default. An **Experimental** toggle lifts the bounds for exploratory use, with inline range hints and real-time validation that disables Generate when constraints are violated.

After a run completes, a **Save** button appears and a **frame scrubber** slides into view below the output area.

#### Interactive remasking

The scrubber lets you step through every intermediate frame of the diffusion process. Navigate with the slider, the arrow buttons, or the keyboard (Left / Right arrow keys, Home / End). At any frame you can click individual resolved tokens to **remask** them (they turn orange), then press **Resume** to re-run the remaining diffusion steps from that point with your edits applied. The model treats the entire generation region as a single block during resume, so remasked tokens in any position can be resolved.

You can resume multiple times — each resume extends the frame history and the scrubber updates accordingly. Remask edits (frame index and token positions) are recorded automatically and included in saved metadata.

#### Saving

Clicking **Save** writes the run results to a timestamped folder under `Results/` containing `metadata.json` (including any remask edits), `final.txt`, `history.txt` (frame-by-frame diffusion snapshots), and `diffusion.gif` (animated visualization).

Optional flags:

```bash
python3 main.py --serve --host 0.0.0.0 --port 8000
```

### CLI

```bash
python3 main.py --sample --prompt "Explain what a hash map is and give a Python example."
```

With custom parameters:

```bash
python3 main.py --sample \
  --prompt "Write a haiku about recursion." \
  --steps 128 \
  --gen_length 128 \
  --block_length 32 \
  --temperature 0.4 \
  --cfg_scale 0.0
```

Each run creates a timestamped directory under `artifacts/` containing the metadata, final text, frame history, and the diffusion GIF.


## Implementation Status

- [x] LLaDA-8B-Instruct model loading (bfloat16, `device_map="auto"`)
- [x] Iterative masked diffusion sampler with low-confidence remasking
- [x] Configurable steps, generation length, block length, temperature, CFG, remasking
- [x] Intermediate frame history recording
- [x] GIF rendering of the diffusion process
- [x] CLI with per-run artifact output (metadata, text, GIF)
- [x] Interactive web UI with live diffusion visualization (FastAPI + WebSocket)
- [x] Recommended parameter bounds with Experimental mode toggle
- [x] Real-time client-side validation (bounds, divisibility, negative values)
- [x] Save run results from the web UI (metadata, history, final text, GIF)
- [x] Interactive remasking: frame scrubber, click-to-remask tokens, resume diffusion from any frame
- [x] Remask edit metadata saved with run results

### Possible Extensions

- [ ] Alignment with reinforcement learning (RLHF / DPO)
- [ ] MDM fine-tuning on custom instruction data
- [ ] Side-by-side comparison with autoregressive generation


## References

- **LLaDA paper:** Nie et al., "Large Language Diffusion Models," NeurIPS 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **LLaDA model:** [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on Hugging Face
