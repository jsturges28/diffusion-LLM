import tiktoken
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable

from src.config.config import TokenizerConfig


@dataclass
class BPETokenizer:
    '''
    Byte-Pair Encoding to tokenize the vocabulary
    '''
    name: str
    bos_id: int
    eos_id: int

    def __post_init__(self) -> None:
        if self.name != "tiktoken-gpt2":
            raise ValueError("Make sure 'tiktoken-gpt2' is implemented for preprocessing.")
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab
    
    def encode_with_specials(self, text: str) -> List[int]:
        ids = self._enc.encode(text)
        full_text = [self.bos_id] + ids + [self.eos_id]
        return full_text
    

def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()
    

def get_text_files(root: Path) -> List[Path]:
    """
    Glob .txt files under `root`. If `root` is a file, yield it directly.
    Supports sharded layouts like: data/tinystories/shard_0000/000000.txt, ...
    """
    if root.is_file():
        return [root]
    
    files = list(root.rglob("*.txt"))
    if not files:
        raise RuntimeError(f"No .txt files under {root}.")
    
    files.sort()

    return files


def tokenize_files(files: List[Path], tokenizer: BPETokenizer) -> np.ndarray:
    """
    Produce a single flat list of token IDs by concatenating
    BOS + encoded(doc) + EOS for each input file.
    """
    encodings: List[int] = []
    for file in tqdm(files, desc="Tokenizing files", unit="file"):
        text = read_text(file)
        if text:
            encoding = tokenizer.encode_with_specials(text)
            encodings.extend(encoding)

    return np.asarray(encodings, dtype=np.int64)
    

def build_token_stream(config: TokenizerConfig, tokenizer: BPETokenizer) -> np.ndarray:
    """
    Convenience to tokenize all files under the data root
    """
    files = get_text_files(config.data_root)
    if not files:
        raise RuntimeError(f"No .txt files under {config.data_root}")
    
    return tokenize_files(files, tokenizer)