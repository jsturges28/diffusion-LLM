import random
import glob
import tiktoken
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable

from src.config.config import PreprocessingConfig


@dataclass
class BPETokenizer:
    '''
    We use Byte-Pair Encoding to preprocess the vocabulary
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
    

def get_text_files(root: Path, shuffle: bool,seed: int) -> Iterable[Path]:
    