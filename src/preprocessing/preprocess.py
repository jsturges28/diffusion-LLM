import random
import numpy as np
#import json
from pathlib import Path
from typing import Tuple, List

from src.config.config import PreprocessingConfig, TokenizerConfig
from src.preprocessing.tokenizer import BPETokenizer, get_text_files, tokenize_files, build_token_stream


def _choose_dtype(vocab_size: int, force_uint16: bool) -> np.dtype:
    if force_uint16 and vocab_size <= PreprocessingConfig.MAX_VOCAB_SIZE:
        return np.dtype(np.uint16)
    
    else:
        return np.dtype(np.uint32)
    

def _split_dataset(paths: List[Path], 
                   train_ratio: float, 
                   val_ratio: float,
                   shuffle: bool, 
                   seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    
    paths = paths[:] # copy
    if shuffle:
        rand = random.Random(seed)
        rand.shuffle(paths)

    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    #n_test = n_total - n_train - n_val

    train_dataset = paths[:n_train]
    val_dataset = paths[n_train:n_train + n_val]
    test_dataset = paths[n_train + n_val:]

    return train_dataset, val_dataset, test_dataset


def run(preprocess_config: PreprocessingConfig, token_config: TokenizerConfig) -> None:
    print(f"Starting preprocessing from data root: {token_config.data_root}")

    files = get_text_files(token_config.data_root)

    tokenizer = BPETokenizer(name=token_config.tokenizer_name,
                             bos_id=token_config.bos_id,
                             eos_id=token_config.eos_id)
    
    train_data, val_data, test_data = _split_dataset(paths=files,
                                                     train_ratio=preprocess_config.train_ratio,
                                                     val_ratio=preprocess_config.val_ratio,
                                                     shuffle=preprocess_config.shuffle,
                                                     seed=preprocess_config.seed)
    

    train_encodings = tokenize_files(train_data, tokenizer)
    val_encodings = tokenize_files(val_data, tokenizer)
    test_encodings = tokenize_files(test_data, tokenizer)

    dtype = _choose_dtype(tokenizer.vocab_size, force_uint16=preprocess_config.force_uint16)

    train_save_path = preprocess_config.save_path_splits / preprocess_config.save_train
    val_save_path = preprocess_config.save_path_splits / preprocess_config.save_val
    test_save_path = preprocess_config.save_path_splits / preprocess_config.save_test

    preprocess_config.save_path_splits.mkdir(parents=True, exist_ok=True)
    train_save_path.parent.mkdir(parents=True, exist_ok=True)
    val_save_path.parent.mkdir(parents=True, exist_ok=True)
    test_save_path.parent.mkdir(parents=True, exist_ok=True)

    train_save_path.write_bytes(train_encodings.astype(dtype).tobytes())
    val_save_path.write_bytes(val_encodings.astype(dtype).tobytes())
    test_save_path.write_bytes(test_encodings.astype(dtype).tobytes())

    # metadata = {
    #     "tokenizer": tok.name,
    #     "vocab_size": tok.vocab_size,
    #     "bos_id": tok.bos_id,
    #     "eos_id": tok.eos_id,
    #     "pad_id": None,
    #     "mask_id": None,
    #     "dtype": str(dtype),
    #     "split": {"train": preprocess_config.train_ratio, "val": preprocess_config.val_ratio, "test": 1.0 - preprocess_config.train_ratio - preprocess_config.val_ratio},
    #     "note": "File-level split; BPE ids for diffusion LMs.",
    # }
    # with open(preprocess_config.save_path_metadata, "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, indent=2)

    print(f"Saved train/val/test .bin files to {str(preprocess_config.save_path_splits)}")


if __name__ == '__main__':
    run(preprocess_config=PreprocessingConfig(), token_config=TokenizerConfig())