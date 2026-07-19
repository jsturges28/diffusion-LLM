"""Launch a single model worker in its own venv/process.

The supervisor spawns this module with the target venv's Python:

    <venv>/bin/python -m src.backends.run_worker \
        --model <id> --host 127.0.0.1 --port <port>

Only the selected model's worker module is imported, so a worker
venv never pulls in another model's dependencies.
"""

from __future__ import annotations

import argparse
import importlib

import uvicorn

from src.backends.registry import REGISTRY
from src.backends.worker_base import create_worker_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a diffusion model worker."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--host", type=str, default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    if args.model not in REGISTRY:
        raise SystemExit(f"unknown model: {args.model}")

    info = REGISTRY[args.model]
    module = importlib.import_module(info.worker_module)
    backend = module.build_backend()
    app = create_worker_app(backend)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
