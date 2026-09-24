# LLM Visualizer

A local visual playground and analytics suite for language models, with
its depth in **discrete diffusion**: models that generate text not
left-to-right but by **iteratively denoising a corrupted sequence** over
many steps. A FastAPI server streams every intermediate frame to the
browser, so you can watch a sequence resolve, scrub back through its
history, remask tokens and resume from there, colour tokens by the
model's confidence or by the order they settled in, and compare runs
afterwards. An autoregressive model runs alongside as a baseline.

It is built for building intuition, and it leans hard toward
explainability: most of what it draws is a signal the model produced,
not a decoration.

![A finished LLaDA run: the scrubber, the entropy-by-position strip, and
the VRAM meter in the status bar](assets/screenshot-generator.png)

Above, a finished run with the frame scrubber, the per-position entropy
strip, and the status bar's resource meter. Below, the same app running
the autoregressive model on CPU, where the meter reports cores instead
of VRAM.

![The same app mid-run on CPU, the meter reading 50% of 32
cores](assets/screenshot-cpu-run.png)

## The models

One is resident at a time; a single large model already saturates a
24 GB card. Each runs in its own virtual environment, because they need
mutually incompatible `transformers` versions.

| Model | Kind | Precision | VRAM | Notes |
|---|---|---|---|---|
| [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | Masked discrete diffusion, single canvas | bf16 | ~17 GB | The first competitive large-scale diffusion LLM ([paper](https://arxiv.org/abs/2502.09992)). Interactive remasking and resume. |
| DiffusionGemma-26B-A4B | Block-autoregressive diffusion, MoE (~4B active) | self-quantized 4-bit NF4 | ~18 GB | 256-token canvases, adaptive stopping, optional reasoning channel. Single-canvas runs support remask and resume. |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Autoregressive baseline | bf16 | ~6 GB | Runs on **GPU or CPU**, so a machine without a card can still use the suite. Per-token entropy, optional top-5 alternatives, What If? substitution. |

## How it works

An autoregressive model samples one token at a time, conditioned on
what came before. A diffusion LLM starts from a fully corrupted
sequence and refines all of it in parallel over *N* steps, using
bidirectional attention, re-masking the positions it is least sure
about between steps until the sequence converges.

That difference is what makes the app worth watching: an autoregressive
run has one frame per token and never revisits a position, while a
diffusion run re-decides every position at every step, so its frames
are a trajectory rather than a transcript.

[docs/GUIDE.md](docs/GUIDE.md) covers the mechanics per model, what each
overlay means, and every sampling parameter.

## Architecture

A model-agnostic **supervisor plus workers** layout, driven by a shared
contract in `src/backends/`.

```
Browser (shared frontend)
  |  /ws + /api
  v
Supervisor  (.venv, no torch/transformers)
  - static assets + Analytics API + Save endpoint
  - Model Manager: spawns/stops one worker, VRAM pre-flight,
    host-wide lease so two instances cannot both load
  - /ws bidirectional proxy to the active worker
  |
  |  spawn: <model venv> python -m src.backends.run_worker
  v
Model Worker  (exactly one alive)
  - LLaDA           .venv          transformers 4.38.2
  - DiffusionGemma  .venv-dgemma   transformers 5.13
  - SmolLM3         .venv-ar       transformers >= 4.53
```

The app opens on a **Main Menu** at `/`, a GPU-aware model picker.
Selecting a model activates its worker and enters the generator at
`/generate`, which is gated behind having a model. The Analytics Suite
at `/analytics.html` is model-agnostic and always available.

## Setup

**Platform.** Built and tested on **Ubuntu 24.04** only. The desktop
app, the launcher script and the GPU tooling assume Linux; other
operating systems are a future goal rather than a current guarantee.
Requires **Python 3.12** and, for the diffusion models, a CUDA GPU.

Each environment's direct dependencies are declared in one place,
`[tool.diffusion-llm]` in [pyproject.toml](pyproject.toml). The
`requirements*.txt` files are generated from those lists with every
transitive pin hashed, so install from them as usual and regenerate
only when a declared dependency changes:

```bash
.venv/bin/python scripts/lock_environments.py            # verify, offline
.venv/bin/python scripts/lock_environments.py --update   # resolve and write
```

**Supervisor and LLaDA** (`.venv`, required):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Weights (~16 GB) download on first use, at the commit the registry
pins, so the same app version always loads the same weights and remote
code. The supervisor runs here and never imports torch.

**SmolLM3** (`.venv-ar`, optional, the GPU-less path):

```bash
python3 -m venv .venv-ar
.venv-ar/bin/pip install -r requirements-ar.txt
```

**DiffusionGemma** (`.venv-dgemma`, optional) needs a license
acceptance on Hugging Face and a local 4-bit build step. See
[docs/GUIDE.md](docs/GUIDE.md) for the download, the quantize script,
and the artifact manifest it writes. Skipping it is fine: the selector
still lists the model and activation fails with a clear message.

**Desktop app** (optional) runs the same UI in a native window through
pywebview:

```bash
.venv/bin/pip install -r requirements-desktop.txt
```

## Quickstart

```bash
.venv/bin/python main.py        # then open http://localhost:8000
.venv/bin/python desktop.py     # or the native window
```

It binds to `127.0.0.1`. Serving to your network with `--host 0.0.0.0`
works and warns you, because there is no authentication: anyone who can
reach the port can load models, save runs and delete them.

Saved runs and durable UI state go to this repository's `results/`
whatever directory you start from. Point elsewhere with
`--results-dir`.

## What works

Diffusion generation for both diffusion models, streamed frame by
frame, with a scrubber over the full history. Autoregressive generation
alongside it, replayed through the same tooling. Interactive
**remasking and resume**: pick tokens at any frame, remask them, and
regenerate from there, keeping the pre-edit run for comparison.
**What If?** substitution for the autoregressive model. Four token
overlays: a confidence heatmap, commit order, entropy, and a diff
against the pre-edit run, with an Original/Edited crossfade between
them.

Per-token **confidence** and **entropy** on every model, declared by
the unit and the axes they vary over, so a reader knows whether a
signal belongs to a position or to a position at a frame. Optional
top-5 **alternatives** capture with a hover popover and true ranks.

An **Analytics Suite** with a run browser, collections and favourites,
a detail modal carrying the token canvas and four charts, run
comparison, and GIF export. Runs are published whole or not at all,
versioned, and carry the provenance the worker attested, including what
each run cost in VRAM.

A **desktop app**, durable server-side UI state, a host-wide lease so
two instances cannot both load a model, and a lint ratchet plus roughly
2,300 tests over the Python and browser code.

## Documentation

| File | What it is for |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | The manual: every feature, every parameter, and the mechanics behind them |
| [docs/HANDOFF.md](docs/HANDOFF.md) | A bounded cold start: what this is, how it fits together, where it stands |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Settled decisions, deliberate stopping points, and the backlog |
| [docs/TIGERSTYLE.md](docs/TIGERSTYLE.md) | The coding standard this repo is held to |
| [docs/MANUAL_VERIFICATION.md](docs/MANUAL_VERIFICATION.md) | Hardware scenarios worth re-running, since CI has no GPU |
| [AGENTS.md](AGENTS.md) | Working conventions for anyone, human or agent, picking this up |

The in-app **Help** modal covers the same ground as the guide, more
briefly, and is the right thing to reach for while using the app.

## Roadmap

Next up: multimodal image input, a live GPU and CPU meter that persists
its samples, top-k alternatives for the diffusion models, and
side-by-side comparison against autoregressive generation.
[docs/ROADMAP.md](docs/ROADMAP.md) carries the reasoning and the
backlog.

## References

- **LLaDA paper:** Nie et al., "Large Language Diffusion Models," NeurIPS 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **LLaDA model:** [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- **DiffusionGemma model:** [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- **SmolLM3 model:** [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
