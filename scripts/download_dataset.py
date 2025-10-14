import sys
from pathlib import Path
from datasets import load_dataset

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config.config import PrepareDatasetConfig


def save_dataset_to_txt(dataset, field: str, save_dir: Path):
    """
    Save each sample’s text to a `.txt` file in out_dir.
    Each file name: e.g. “000000.txt”, “000001.txt”, ...
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(dataset):
        text = sample[field]
        filename = save_dir / f"{i:06d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)


def main():
    dataset = load_dataset(PrepareDatasetConfig.huggingface_dataset_path, 
                           split="train")
    
    save_dataset_to_txt(dataset=dataset, 
                        field=PrepareDatasetConfig.field, 
                        save_dir=PrepareDatasetConfig.save_path)
    

if __name__ == '__main__':
    main()