"""Minimal WebSocket smoke test for the supervisor + worker.

Connects to the supervisor /ws, waits for the model to be ready,
runs one short generation, and reports frame/timing stats. Used
to regression-test the process-isolation refactor end to end.

    .venv/bin/python scripts/ws_smoke_test.py \
        --url ws://127.0.0.1:8000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request

import websockets

DEFAULT_LLADA_PARAMS = {
    "steps": 16,
    "gen_length": 32,
    "block_length": 32,
    "temperature": 0.0,
    "cfg_scale": 0.0,
    "remasking": "low_confidence",
    "seed": 0,
}


async def run(
    url: str,
    ready_timeout: float,
    params: dict,
    prompt: str,
) -> int:
    async with websockets.connect(url, max_size=None) as ws:
        # Wait for model_status: ready. The worker cold start
        # (import torch + load model) can take tens of seconds
        # before the first message arrives.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + ready_timeout
        ready = False
        while loop.time() < deadline:
            remaining = max(1.0, deadline - loop.time())
            raw = await asyncio.wait_for(
                ws.recv(), timeout=remaining
            )
            msg = json.loads(raw)
            if msg.get("type") == "model_status":
                print("model_status:", msg.get("status"))
                if msg.get("status") == "ready":
                    ready = True
                    break
        if not ready:
            print("FAIL: model never became ready")
            return 1

        payload = {
            "type": "generate",
            "prompt": prompt,
            "experimental": False,
        }
        payload.update(params)
        await ws.send(json.dumps(payload))

        frames = 0
        conf_frames = 0
        c_tokens = 0
        max_canvas = 0
        sample = None
        final_text = None
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "frame":
                frames += 1
                if isinstance(
                    msg.get("mean_conf"), (int, float)
                ):
                    conf_frames += 1
                canvas = msg.get("canvas_index")
                if isinstance(canvas, int) and canvas > max_canvas:
                    max_canvas = canvas
                for tok in msg.get("tokens") or []:
                    if isinstance(tok.get("c"), (int, float)):
                        c_tokens += 1
                        if sample is None:
                            sample = {
                                "t": tok.get("t"),
                                "c": tok.get("c"),
                                "mean_conf": msg.get("mean_conf"),
                                "canvas_index": canvas,
                            }
            elif mtype == "done":
                final_text = msg.get("final_text", "")
                thinking = msg.get("thinking")
                if thinking:
                    print(f"thinking: {thinking[:160]!r}")
                break
            elif mtype == "error":
                print("ERROR:", msg.get("message"))
                return 1

        print(f"frames received: {frames}")
        print(f"frames with mean_conf: {conf_frames}")
        print(f"resolved tokens carrying c: {c_tokens}")
        print(f"max canvas_index: {max_canvas}")
        print(f"sample resolved token: {sample}")
        print(f"final_text: {final_text!r}")
        if frames < 2 or final_text is None:
            print("FAIL: too few frames or no final text")
            return 1
        if conf_frames != frames:
            print("FAIL: not every frame carried mean_conf")
            return 1
        if c_tokens == 0:
            print("FAIL: no resolved token carried confidence")
            return 1
        print("SMOKE TEST OK")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8000/ws"
    )
    parser.add_argument(
        "--http-base", default="http://127.0.0.1:8000"
    )
    parser.add_argument("--activate", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument(
        "--prompt",
        default="In one short sentence, what is diffusion?",
    )
    parser.add_argument(
        "--ready-timeout", type=float, default=240.0
    )
    args = parser.parse_args()

    if args.activate:
        req = urllib.request.Request(
            f"{args.http_base}/api/models/"
            f"{args.activate}/activate",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            print("activate:", json.load(resp))

    params = (
        json.loads(args.params)
        if args.params
        else DEFAULT_LLADA_PARAMS
    )
    code = asyncio.run(
        run(args.url, args.ready_timeout, params, args.prompt)
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
