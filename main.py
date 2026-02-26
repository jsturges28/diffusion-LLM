import argparse
from pathlib import Path
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM

from src.inference.llada_sampler import llada_generate_with_history
from src.inference.mdm_sampler import MDMSamplerConfig, mdm_generate_with_history
from src.inference.render_gif import history_to_gif
#from src.preprocessing.preprocess import run
#from src.config.config import PreprocessingConfig, TokenizerConfig

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

    ap.add_argument("--save_history", action="store_true")
    ap.add_argument("--history_path", type=Path, default=Path("artifacts/history.txt"))
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif_path", type=Path, default=Path("artifacts/diffusion.gif"))
    return ap.parse_args()


def sanitize_frame(s: str) -> str:
    s = s.replace("<|mdm_mask|>", "░")  # or "█" or "·"
    s = s.replace("<|eot_id|>", "")
    s = s.replace("<|endoftext|>", "")
    return s


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

        else:
            # Keep existing Qwen MDM demo as an alternate backend
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
                device_map="auto" if device.type == "cuda" else None,
            ).eval()
            #model.to(device)

            cfg = MDMSamplerConfig(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                steps=args.steps,
                max_new_tokens=args.gen_length,
            )
            final_text, history = mdm_generate_with_history(model, tok, args.prompt, cfg)

        print("\n--- Prompt ---")
        print(args.prompt)
        print("\n--- Diffusion Output ---")
        print(sanitize_frame(final_text))

        if args.save_history:
            # Save history frames as plain text for debugging / seminar notes
            out_txt = args.history_path
            out_txt.parent.mkdir(parents=True, exist_ok=True)

            with out_txt.open("w", encoding="utf-8") as f:
                for i, frame in enumerate(history):
                    frame = sanitize_frame(frame)
                    f.write(f"\n===== FRAME {i} =====\n")
                    f.write(frame)
                    f.write("\n")

            print(f"Saved history text: {out_txt}")

        if args.gif:
            history = [sanitize_frame(frame) for frame in history]
            history_to_gif(history, args.gif_path)
            print(f"\nSaved GIF: {args.gif_path}")

        return
    
    print("No action specified. Use --sample. Example: python main.py --sample --gif")


if __name__ == '__main__':
    main()