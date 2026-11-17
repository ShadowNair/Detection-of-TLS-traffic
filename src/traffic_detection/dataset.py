from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapPacketDataset(Dataset):
    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r") if y_path is not None else None
        self.indices = indices

    def __len__(self) -> int:
        return len(self.x) if self.indices is None else len(self.indices)

    def __getitem__(self, item: int):
        row = item if self.indices is None else int(self.indices[item])
        x = torch.from_numpy(np.asarray(self.x[row], dtype=np.int16).copy())
        if self.y is None:
            return x
        return x, int(self.y[row])

