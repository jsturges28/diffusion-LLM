# The Discrete Diffusion LLM

## Project Summary

This repository explores **diffusion-style text generation** for a seminar demo: instead of generating tokens left-to-right (autoregressive decoding), we generate by **iteratively denoising a masked sequence** (“masked diffusion / MDM”).

The main goal is to create a **visually compelling diffusion effect** (a step-by-step refinement animation) while remaining simple and robust enough to demo with **general instruction prompts**.

We pivot from an earlier embedding-noise prototype (DiT-style) to a simpler and increasingly common approach in text diffusion: **token masking + reconstruction**. This is the paradigm used by recent diffusion LMs such as [LLaDA](https://arxiv.org/abs/2502.09992), which models text via a forward masking process and a reverse process that predicts masked tokens. 

We also borrow practical ideas from [Open-dLLM’s](https://github.com/pengzhangzhi/Open-dLLM/tree/main) end-to-end diffusion LLM stack (notably iterative sampling with intermediate-state visualization).


## Table of Contents

- [Project Summary](#project-summary)
- [Diffusion vs Autoregressive Modeling](#diffusion-vs-autoregressive-modeling)
- [Approach](#approach)
  - [Backbone model](#backbone-model)
  - [Masked Diffusion Model (MDM)](#masked-diffusion-model-mdm)
  - [Sampling + Diffusion Effect Visualization](#sampling--diffusion-effect-visualization)
- [High-Level Pipeline](#high-level-pipeline)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Dataset Preparation (Optional)](#dataset-preparation-optional)
- [Current Implementation Status](#current-implementation-status)
- [TODO — Remaining Work](#todo--remaining-work)
- [Technical Notes](#technical-notes)


## Diffusion vs Autoregressive Modeling

Autoregressive language models factorize the joint distribution of tokens via the chain rule of probability:

$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{<t})$

That is, they model the probability of a full sequence as a product of next-token conditional probabilities.

Diffusion models instead define a forward **corruption** process, (here: masking tokens):

$q\left(x_t \mid x_0\right)$

and train a neural network to approximate the reverse process, called **denoising**:

$p_\theta\left(x_{t-1} \mid x_t\right)$

For text, this requires defining meaningful corruption processes over discrete token sequences.


## Approach

### Backbone model

We use a **small pretrained instruction-tuned model** without explicitly requiring us to train a Transformer from scratch.

Recommended default backbone:

- `Qwen/Qwen2.5-0.5B-Instruct` (0.49B params), which can be found [here](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

This model is small enough to run and fine-tune locally, while still supporting instruction-following behavior out of the box. 

> Note: Qwen2.5 models require a sufficiently recent `transformers` version; the model card explicitly warns older versions may error.

### Masked Diffusion Model (MDM)

Instead of Gaussian noise in embedding space, we use **token masking**:

1. Sample a mask ratio (often uniformly in \([0,1]\), as used in multiple masked diffusion recipes, see [Appendix A of LLaDA](https://arxiv.org/pdf/2502.09992)).
2. Replace a subset of tokens with a special `[MASK]` token.
3. Train the model to reconstruct masked tokens with **cross-entropy loss** computed only on masked positions.

This matches the conceptual framing in LLaDA (forward masking + reverse prediction) and is also the practical pattern shown in Open-dLLM’s sampling script (ensuring a mask token exists and iteratively generating).

### Sampling + Diffusion Effect Visualization

Sampling proceeds by iterative refinement:

- Start with a prompt + a masked “canvas” for the completion.
- Run `N` denoising steps.
- At each step, fill a subset of masked positions (e.g., confidence-based).
- Record intermediate sequences to produce an animation (“diffusion effect”).

Open-dLLM demonstrates this kind of iterative refinement in its demo assets and provides a sampling script built around diffusion generation and history output.


## High-Level Pipeline

**Sampling-first (demo path):**

Prompt → add `[MASK]` placeholders → iterative mask-and-fill → decoded text + history → GIF

**Training (optional, later):**

Instruction-style data (prompt/response) → mask response tokens (ratio schedule) → CE loss on masked positions → improved denoising


## Project Structure

(Keeping directory layout stable as the project pivots.)

```
.
├── main.py
├── README.md
├── data/
│   └── tinystories/ (optional)
├── src/
│   ├── config/
│   │   └── config.py
│   ├── preprocessing/
│   │   └── (legacy TinyStories sharding + token bin utilities; optional)
│   ├── models/
│   │   └── (MDM wrapper + sampler TODO)
│   ├── training/
│   │   └── (MDM fine-tune TODO)
│   ├── inference/
│   │   └── (sampling + GIF TODO)
│   └── utils/
```


## Setup

1. Create venv:
```bash
python3 -m venv .venv
```

2. Activate:
```
source .venv/bin/activate
```

3. Install:
```
pip install -r requirements.txt
```

## Quickstart

Goal: demonstrate diffusion-style generation on general instruction prompts using a pretrained backbone.

Planned command (sampling-first):
```
python3 main.py --sample --prompt "Explain what a hash map is and give a Python example."
```

Output:

- Final generated text
- Intermediate-step history
- Optional GIF (“diffusion effect”)

## Dataset Preparation (Optional)

TinyStories remains as a safe fallback demo dataset and a lightweight fine-tuning target.

Download and shard:
```
python3 scripts/download_dataset.py
```

If you also want to run preprocessing:
```
python3 main.py --prepare_data
```

(For the current masked-diffusion + pretrained-backbone direction, dataset prep is optional; sampling-first can run without it.)

## Current Implementation Status

Completed:

- [X] Project scaffolding
- [X] TinyStories download + sharding (optional path)
- [X] Preprocessing utilities (optional path)
- [X] README aligned to masked diffusion pivot

## TODO — Remaining Work

- [ ] Sampling-first (seminar-critical)
- [ ] Load pretrained instruction model (default: Qwen2.5-0.5B-Instruct)
- [ ] Ensure tokenizer has a [MASK] token (add if missing, resize embeddings)
- [ ] Implement iterative mask-and-fill sampler (steps, temperature, top-k)
- [ ] Record intermediate sequences (history) and render “diffusion effect” GIF

## Training (optional upgrade)

- [ ] Implement MDM fine-tune on prompt/response style data (mask only response tokens)
- [ ] Optional: fine-tune on TinyStories (fallback demo distribution)