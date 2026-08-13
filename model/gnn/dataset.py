"""
dataset.py

PyTorch Dataset for agonism prediction.
Loads pre-computed graphs from ml/data/graphs/*.pt (fast).
"""

import csv
import torch
from torch.utils.data import Dataset
from pathlib import Path


HOME           = Path("/home/jiasin/AMY123R_agonist_Design/model")
DEFAULT_GRAPH_DIR = HOME / "data/graphs_8A"


class AgonismDataset(Dataset):
    def __init__(
        self,
        meta_csv:  Path = HOME / "data/meta.csv",
        indices:   list[int] | None = None,
        rows:      list[dict] | None = None,
        graph_dir: Path | None = None,
    ):
        self.graph_dir = Path(graph_dir) if graph_dir else DEFAULT_GRAPH_DIR
        if rows is not None:
            all_rows = rows
        else:
            all_rows = list(csv.DictReader(open(meta_csv)))
            if indices is not None:
                all_rows = [all_rows[i] for i in indices]
        # only keep rows that have a cached graph; track original positions
        self.rows = []
        self.valid_positions = []   # positions within all_rows that have graphs
        for pos, r in enumerate(all_rows):
            if (self.graph_dir / f"{r['name']}.pt").exists():
                self.rows.append(r)
                self.valid_positions.append(pos)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return torch.load(self.graph_dir / f"{r['name']}.pt", weights_only=False)


def collate_skip_none(batch):
    """DataLoader collate that drops None items (missing files)."""
    from torch_geometric.data import Batch
    batch = [b for b in batch if b is not None]
    return Batch.from_data_list(batch) if batch else None
