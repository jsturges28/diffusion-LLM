from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepareDatasetConfig:
    huggingface_dataset_path: str = "roneneldan/TinyStories"
    field: str = "text"
    shard_size: int = 1000
    save_path: Path = Path("data/tinystories")


@dataclass
class PreprocessingConfig:
    data_root: Path
    save_dir: Path
    train_ratio: float = 0.9
    tokenizer_name: str = "tiktoken-gpt2"
    bos_id: int = 50256
    eos_id: int = 50256
    force_uint16: bool = True
    shuffle_files: bool = True
    seed: int = 42


