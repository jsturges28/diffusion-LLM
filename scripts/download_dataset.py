import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config.config import PrepareDatasetConfig


def save_dataset_to_txt(dataset, field: str, shard_size: int, save_dir: Path):
    """
    Write dataset samples to sharded subdirectories to avoid filesystem/UI slowdowns.
    Each shard contains at most `shard_size` files.

    Example layout:
      data/tinystories/
        shard_0000/000000.txt
        shard_0000/000001.txt
        ...
        shard_0001/001000.txt
        ...
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = list(save_dir.glob("shard_*"))
    if shard_dirs:
        print(f"Shards already exist in {save_dir}, skipping download.")
        return

    if hasattr(dataset, "__len__"):
        total = len(dataset)
    else:
        total = None

    print(f"Downloading dataset {dataset}, splitting into shard size {shard_size}...")

    for i, sample in enumerate(tqdm(dataset, total=total, unit="sample", desc="Downloading samples")):
        shard_id = i // shard_size
        shard_dir = save_dir / f"shard_{shard_id:04d}"
        shard_dir.mkdir(exist_ok=True)

        filename = shard_dir / f"{i:06d}.txt"
        text = sample[field]

        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
    
    print(f"Done, all shards written successfully.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=PrepareDatasetConfig.huggingface_dataset_path)
    parser.add_argument("--field", default=PrepareDatasetConfig.field)
    parser.add_argument("--shard_size", type=int, default=PrepareDatasetConfig.shard_size)
    parser.add_argument("--out", default=PrepareDatasetConfig.save_path)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset, split="train")
    
    save_dataset_to_txt(dataset=dataset, 
                        field=args.field, 
                        shard_size=args.shard_size, 
                        save_dir=args.out)
    

if __name__ == '__main__':
    main()