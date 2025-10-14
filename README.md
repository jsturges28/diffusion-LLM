# The Discrete Diffusion LLM

In this tutorial, we'll explore how discrete diffusion models can be applied to text generation by building a character-level text diffusion model from scratch.

Most modern LLMs generate text sequentially, one token at a time, left to right, following autoregressive principles. On the other hand, diffusion models that are the main driver behind the recent successes of image and video generators take a very different approach, where they start by corrupting data with noise and then learn to denoise it.

Extending diffusion models to text, however, is not straightforward. Unlike images, which exist in a continuous space where adding and removing noise is easier, text is discrete, making the addition and removal of "noise" trickier. Since text is made of discrete symbols, "adding noise" here means flipping characters or tokens till it becomes gibberish. Teaching a model to undo this noise is far less straightforward.

To tackle this challenge, we'll begin with Andrej Karpathy's character-level baby GPT, a minimal yet mighty model for sequence modeling, and transform it into a character-level discrete diffusion model. Our implementation will closely follow the ideas presented in the papers [*Discrete Denoising Diffusion Probabilistic Models (D3PM)*](https://arxiv.org/abs/2107.03006) and [*Diffusion-LM*](https://arxiv.org/abs/2205.14217). 

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

We're going to utilize the huggingface [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset for our experiment. A preparation script is provided to make this easy; after installing your requirements, simply run:

```
python3 scripts/download_dataset.py
```