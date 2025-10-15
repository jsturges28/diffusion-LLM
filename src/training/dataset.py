from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryTokenDataset(Dataset):
    def __init__(self, bin_path: Path, seq_length: int):
        self.seq_length = seq_length