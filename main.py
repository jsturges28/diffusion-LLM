import argparse
import json
import torch

from datetime import datetime
from pathlib import Path
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM

from src.inference.llada_sampler import llada_generate_with_history
from src.inference.render_gif import history_to_gif

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Main controller")

    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--backend", type=str, default="llada", choices=["llada", "qwen_mdm"])

    ap.add_argument("--prompt", type=str, default="Explain what a hash map is and give a Python example.")
    ap.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")

    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--history_stride", type=int, default=1)

    ap.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))
    return ap.parse_args()


def sanitize_frame(s: str) -> str:
    s = s.replace("<|mdm_mask|>", "░")  # or "█" or "·"
    s = s.replace("<|eot_id|>", "")
    s = s.replace("<|endoftext|>", "")
    return s


def make_run_dir(base: Path, backend: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / f"{ts}_{backend}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    args = parse_args()
    #if args.prepare_data:
        #run(preprocess_config=PreprocessingConfig(), token_config=TokenizerConfig())

    if args.sample:
        # If using device_map="auto", don't call model.to(device)
        # If not using device_map, call model.to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.backend == "llada":

            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if tok.padding_side != "left":
                tok.padding_side = "left"

            model = AutoModel.from_pretrained(
                args.model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
                device_map="auto" if device.type == "cuda" else None,
            ).eval()

            final_text, history = llada_generate_with_history(
                model,
                tok,
                args.prompt,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking="low_confidence",
                history_stride=args.history_stride,
            )

        print("\n--- Prompt ---")
        print(args.prompt)
        print("\n--- Diffusion Output ---")
        print(sanitize_frame(final_text))

        run_dir = make_run_dir(args.artifacts_dir, args.backend)

        # ---- Save metadata ----
        metadata = {
            "backend": args.backend,
            "model": args.model,
            "prompt": args.prompt,
            "final_text": sanitize_frame(final_text),
            "params": {
                "steps": args.steps,
                "gen_length": args.gen_length,
                "block_length": args.block_length,
                "temperature": args.temperature,
                "cfg_scale": args.cfg_scale,
                "history_stride": args.history_stride,
                "remasking": "low_confidence",
            },
        }

        meta_path = run_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        # ---- Save final text ----
        final_path = run_dir / "final.txt"
        final_path.write_text(sanitize_frame(final_text), encoding="utf-8")

        # ---- Save history ----
        hist_path = run_dir / "history.txt"
        with hist_path.open("w", encoding="utf-8") as f:
            for i, frame in enumerate(history):
                frame = sanitize_frame(frame)
                f.write(f"\n===== FRAME {i} =====\n")
                f.write(frame)
                f.write("\n")

        # ---- Save GIF ----
        gif_path = run_dir / "diffusion.gif"
        gif_history = [sanitize_frame(frame) for frame in history]
        history_to_gif(gif_history, gif_path, header_text=args.prompt)

        print(f"\nSaved run artifacts to: {run_dir}")
        print(f"- {meta_path.name}, {final_path.name}, {hist_path.name}, {gif_path.name}")

        return
    
    print("No action specified. Use --sample. Example: python main.py --sample --gif")


if __name__ == '__main__':
    main()