# The Discrete Diffusion LLM


## Table of Contents

- [Project Summary](#project-summary)
- [Diffusion vs Autoregressive Modeling](#diffusion-vs-autoregressive-modeling)
- [Project Objectives](#project-objectives)
- [High-Level Pipeline](#high-level-pipeline)
- [Why GPT-2 BPE?](#why-gpt-2-bpe)
- [Dataset & Preprocessing](#dataset--preprocessing)
  - [Dataset](#dataset)
  - [Preprocessing](#preprocessing)
  - [Memory Mapping (numpy.memmap)](#memory-mapping-numpymemmap)
- [Model Architecture — DiT-Style Diffusion LM](#model-architecture--dit-style-diffusion-lm)
  - [Components](#components)
- [Diffusion Formulation](#diffusion-formulation)
	- [Forward Process](#forward-process)
	- [Training Objective (ε-prediction)](#training-objective-ε-prediction)
	- [Sampling (Reverse/Denoising Process)](#sampling-reversedenosing-process)
- [Bit-Level Diffusion (Planned)](#bit-level-diffusion-planned)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Current Implementation Status](#current-implementation-status)
- [TODO — Remaining Work](#todo--remaining-work)
- [Experimental Roadmap](#experimental-roadmap)
- [Design Philosophy](#design-philosophy)
- [Immediate Next Steps](#immediate-next-steps)
- [Technical Notes](#technical-notes)
  - [Time Embedding Formula](#time-embedding-formula)
- [Context Window](#context-window)
- [Long-Term Vision](#long-term-vision)


## Project Summary

This repository implements a diffusion-based language model that replaces traditional autoregressive next-token prediction with iterative denoising.

Modern large language models (LLMs) generate text sequentially, one token at a time, from left to right. Diffusion models, however, generate samples by starting from noise and progressively denoising toward structured data.

Applying diffusion to text is fundamentally more challenging than applying it to images because language is discrete. This project explores how to adapt diffusion methods to discrete token sequences using two approaches:

1. **Embedding-space diffusion (DiT-style)**
2. **Bit-level discrete diffusion (planned)**

The implementation draws conceptual inspiration from:

- [*Discrete Denoising Diffusion Probabilistic Models (D3PM)* (2021)](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)
- [*Diffusion-LM* (2022)]([https://arxiv.org/abs/2205.14217](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1be5bc25d50895ee656b8c2d9eb89d6a-Abstract-Conference.html))

The long-term goal is to build a clean, extensible research scaffold for experimenting with diffusion-based language modeling.



## Diffusion vs Autoregressive Modeling

Autoregressive language models factorize the joint distribution of tokens via the chain rule of probability:

$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{<t})$

That is, they model the probability of a full sequence as a product of next-token conditional probabilities.

Diffusion models instead define a forward corruption process:

$q\left(x_t \mid x_0\right)$

and train a neural network to approximate the reverse process:

$p_\theta\left(x_{t-1} \mid x_t\right)$

For text, this requires defining meaningful corruption processes over discrete token sequences.



## Project Objectives

1. Understand how discrete diffusion can be applied to text.
2. Build a minimal but scalable diffusion LLM pipeline.
3. Compare:
    - Embedding-space diffusion (DiT-style)
    - Bit-level discrete diffusion
4. Maintain explicit, readable, modular code.
5. Support scaling to large corpora via memory mapping.



## High-Level Pipeline

Raw Text (.txt)  
↓  
GPT-2 BPE Tokenization  
↓  
Flat Token Stream (.bin via memmap)  
↓  
Windowed Dataset (seq_len slicing)  
↓  
Diffusion Forward Process  
↓  
Transformer Backbone + Time Conditioning  
↓  
Denoising Objective (ε or x₀ prediction)



## Why GPT-2 BPE?

We use GPT-2 Byte-Pair Encoding (via `tiktoken`) because:
- Fixed vocabulary (~50,257 tokens)
- Deterministic integer token IDs
- No unknown tokens (byte-level encoding)
- Each token ID maps directly to an embedding row
- Standard in modern LLM practice

Conceptually:
- Token ID = index into embedding matrix
- Embedding matrix = learned semantic codebook
- Similar in structure to VQ codebooks, but derived from symbolic merges rather than learned quantization

This is ideal for:
- Continuous embedding diffusion
- Bit-level discrete diffusion



## Dataset & Preprocessing

### Dataset

Current dataset:

`roneneldan/TinyStories`

Downloaded and sharded into:

`data/tinystories/shard_XXXX/NNNNNN.txt`

Sharding avoids filesystem performance issues.

### Preprocessing

Implemented in:

`src/preprocessing/preprocess.py`

### Steps

1. Collect all `.txt` files (recursive).
2. Deterministically split into:
    - train
    - validation
    - test
3. Tokenize using GPT-2 BPE.
4. Insert BOS/EOS tokens per file.
5. Concatenate tokens.
6. Save:
    - `train.bin`
    - `val.bin`
    - `test.bin`

Each `.bin` file is:

- A flat contiguous array of token IDs.
- Stored as `uint16` or `uint32`.
- Designed for memory mapping.

### Memory Mapping (`numpy.memmap`)

We use `numpy.memmap` because:

- Token streams can be hundreds of millions of tokens.
- Full RAM loading is inefficient.
- `memmap` enables disk-backed arrays.
- Only required slices are loaded per batch.

This matches large-scale LLM training pipelines.



## Model Architecture — DiT-Style Diffusion LM

### Components

#### Token Embedding

`nn.Embedding(vocab_size, d_model)`

Maps discrete BPE token IDs to continuous embedding vectors in ℝᵈ:

$E \in \mathbb{R}^{|V| \times d}$

Each integer token ID indexes a row of this embedding matrix, functioning as a learned semantic codebook.

#### Positional Embedding

`nn.Embedding(context_window, d_model)`

Encodes token position within the fixed context window.  
Adds order information so the Transformer can distinguish between:

- “the cat sat”
- “sat the cat”

The embedding dimension matches `d_model` to allow direct addition to token embeddings.

#### Time Embedding

Encodes the diffusion timestep $t$ using sinusoidal frequency features followed by a small MLP projection.

- Frequencies span geometric scales.
- Timestep normalized to $[0,1]$.
- Output dimension matches `d_model`.

The time embedding conditions the Transformer on the current noise level.

#### Transformer Backbone

A stack of self-attention layers derived from `nn.TransformerEncoder`. These layers process the noisy embeddings conditioned on timestep information.

This backbone models interactions across the full context window.

#### Output Head

`Linear(d_model → d_model)`

Predicts either:
- $\epsilon$ (noise prediction (recommended))
- $x_0$ (clean embedding prediction)

## Diffusion Formulation

#### Forward Process

For timestep t:

$x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon$

where:
- $\epsilon \sim \mathcal{N}(0, I)$
- $\bar{\alpha}_t=\prod_{s=1}^t\left(1-\beta_s\right)$

### Training Objective (ε-prediction)

Minimize:

$\mathcal{L}=\mathbb{E}_{t, x_0, \epsilon}\left[\left\|\epsilon-\epsilon_\theta\left(x_t, t\right)\right\|^2\right]$

### Sampling (Reverse/Denoising Process)

Start from:

$x_T \sim \mathcal{N}(0, I)$

Iteratively compute:

$x_{t-1}=f_\theta\left(x_t, t\right)$

Final embeddings are projected back to token space via nearest-neighbor.



## Bit-Level Diffusion (Planned)

Alternative fully discrete approach:

1. Convert token IDs → fixed-width binary vectors.
2. Corrupt via bit flips.
3. Predict clean bits.
4. Decode bits → token IDs.

Loss: Binary Cross Entropy.

Advantages:
- Fully discrete diffusion.
- Closer alignment with categorical corruption.



## Project Structure

.  
├── main.py  
├── README.md  
├── data/  
│   └── tinystories/  
│       └── shard_XXXX/  
├── data-bin/  
│   ├── train.bin  
│   ├── val.bin  
│   └── test.bin  
└── src/  
    ├── config/  
    │   └── config.py  
    ├── preprocessing/  
    │   ├── tokenizer.py  
    │   └── preprocess.py  
    ├── models/  
    │   └── dit_lm.py  
    ├── training/  
    │   └── (TODO)  
    ├── inference/  
    │   └── (TODO)  
    └── utils/



## Setup

1. Create new virtual environment at project root:

```
python3 -m venv .venv
```

2. Activate environment:

```
source .venv/bin/activate
```

3. Install requirements:

```
pip install -r requirements.txt
```



## Dataset Preparation

Download and shard dataset:

```
python3 scripts/download_dataset.py
```

Run preprocessing:

```
python3 main.py --prepare_data
```

This generates:

```
data/splits/train.bin
data/splits/val.bin
data/splits/test.bin
```


## Current Implementation Status

### Completed

- [X] Project scaffolding
- [X] GPT-2 BPE tokenizer wrapper
- [X] Dataset download + sharding
- [X] File-level split
- [X] `.bin` generation
- [X] Memory mapping design
- [X] TimeEmbedding implementation
- [X] DiTLM architecture scaffold



## TODO — Remaining Work

### Dataset Layer

- [ ]  Implement `TokenBinDataset` using `memmap`
- [ ]  Create PyTorch DataLoader

### Training Loop

- [ ]  Implement beta schedule
- [ ]  Implement $\bar{\alpha}$ computation
- [ ]  Add optimizer (AdamW)
- [ ] Add gradient clipping
- [ ] Add validation loop
- [ ] Add checkpoint saving

### Sampling

- [ ] Implement reverse diffusion loop
- [ ] Add DDIM-style fast sampling
- [ ] Convert embeddings → token IDs
- [ ] Decode tokens → text

### Logging

- [ ] TensorBoard integration
- [ ] ClearML integration (future)

### Architecture Improvements

- [ ] GELU activation
- [ ] LayerNorm pre/post
- [ ] Rotary embeddings
- [ ] Classifier-free guidance

### Bit-Diffusion Implementation

- [ ] ID → bit conversion
- [ ] Bit-flip forward process
- [ ] BCE training loss
- [ ] Discrete sampling logic



## Experimental Roadmap

- Compare DiT vs Bit diffusion
- Analyze convergence rates
- Evaluate different T values
- Compare ε vs x₀ prediction
- Explore longer context windows
- Study sample diversity vs autoregressive baseline



## Design Philosophy

- Explicit code > abstraction
- Mathematical clarity
- Modular components
- Research extensibility
- Clean separation:
    - preprocessing
    - dataset
    - model
    - training
    - inference



## Immediate Next Steps

1. Implement TokenBinDataset
2. Implement training loop
3. Run 1 epoch sanity check
4. Confirm decreasing loss
5. Add sampling



## Technical Notes

### Time Embedding Formula

Frequencies:

$\omega_k=\exp \left(-\log (10000) \cdot \frac{k}{d / 2}\right)$

Embedding:

$\operatorname{emb}(t)=\left[\sin \left(t \omega_k\right), \cos \left(t \omega_k\right)\right]$

Then projected via MLP.



## Context Window

`seq_len` defines:
- Transformer maximum attention length
- Dataset slicing length
- Positional embedding size

Must be consistent across:
- DataLoader
- Model definition
- Training loop



## Long-Term Vision

This repository aims to become:
- A research sandbox for diffusion LLMs
- A comparative platform for discrete vs continuous diffusion
- A minimal, extensible LLM experimentation framework