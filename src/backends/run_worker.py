"""Launch a single model worker in its own venv/process.

The supervisor spawns this module with the target venv's Python:

    <venv>/bin/python -m src.backends.run_worker \
        --model <id> --host 127.0.0.1 --port <port> \
        [--device cuda|cpu]

Only the selected model's worker module is imported, so a worker
venv never pulls in another model's dependencies.
"""

from __future__ import annotations

import argparse
import importlib
import os

import uvicorn

from src.backends.registry import REGISTRY
from src.backends.worker_base import create_worker_app

# Disable the Xet download client before the worker module import in
# main() (which transitively imports transformers -> huggingface_hub).
# None of the imports above load huggingface_hub, so setting the flag
# now, at module load, still precedes it. The flag is cached in hf
# constants at import time; the classic downloader routes weight fetches
# through our tqdm hook so the download bar fills smoothly, whereas Xet
# bypasses it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a diffusion model worker."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--host", type=str, default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Model placement; the supervisor picks this per"
        " activation (CPU for GPU-less hosts).",
    )
    args = parser.parse_args()

    if args.model not in REGISTRY:
        raise SystemExit(f"unknown model: {args.model}")

    info = REGISTRY[args.model]
    module = importlib.import_module(info.worker_module)
    backend = module.build_backend()
    app = create_worker_app(backend, device=args.device)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
