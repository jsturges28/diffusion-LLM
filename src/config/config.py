from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepareDatasetConfig:
    huggingface_dataset_path: str = "roneneldan/TinyStories"
    field: str = "text"
    shard_size: int = 1000
    save_path: Path = Path("data/tinystories")


@dataclass
class TokenizerConfig:
    data_root: Path = PrepareDatasetConfig.save_path
    tokenizer_name: str = "tiktoken-gpt2"
    bos_id: int = 50256 # special '' token for GPT-2 BPE
    eos_id: int = 50256


@dataclass
class PreprocessingConfig:
    data_root = TokenizerConfig.data_root
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    save_path_splits: Path = Path("data/splits")
    save_train: str = "train.bin"
    save_val: str = "val.bin"
    save_test: str = "test.bin"
    #save_path_metadata: Path = Path("outputs/")
    MAX_VOCAB_SIZE = 65535 # largest value that fits in uint16
    shuffle: bool = True
    seed: int = 42
    force_uint16: bool = True
