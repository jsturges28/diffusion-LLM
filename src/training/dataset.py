from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BinaryTokenDataset(Dataset):
    def __init__(self, bin_path: Path, context_window: int):
        self.context_window = context_window

        # numpy memory-map (memmap) allows us to read from disk directly, useful for large datasets
        self.arr = np.memmap(bin_path, dtype=np.uint16, mode="r")

        # take contiguous windows of length context_window
        self.n_tokens = self.arr.shape[0]
        self.n_items = self.n_tokens // self.context_window

    def __len__(self) -> int:
        return self.n_items
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.context_window
        end = start + self.context_window

        x = np.asarray(self.arr[start:end], dtype=np.int64) # embeddings as int64 for embedding lookup

        return torch.from_numpy(x)
    

def get_dataloader(bin_path: Path, context_window: int, batch_size: int, shffule: bool = True) -> DataLoader:
    dataset = BinaryTokenDataset(bin_path=bin_path, 
                                 context_window=context_window)

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shffule, 
                            drop_last=True)
    
    return dataloader