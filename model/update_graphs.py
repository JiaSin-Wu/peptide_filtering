"""
update_graphs.py

Re-build precomputed PyG graphs for specific row_ids.
Updates both data/graphs_8A/ (GNN) and data/graphs_8A_egnn/ (EGNN).

Usage:
  python3 update_graphs.py --row_ids 512 531 ...
"""

import argparse
import csv
import torch
from pathlib import Path

HOME      = Path("/home/jiasin")
OUTPUT_DIR = HOME / "outputs"
DATA_DIR   = HOME / "AMY123R_agonist_Design/model/data"

GNN_DIR  = DATA_DIR / "graphs_8A"
EGNN_DIR = DATA_DIR / "graphs_8A_egnn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row_ids", nargs="+", type=int, required=True)
    parser.add_argument("--cutoff", type=float, default=8.0)
    args = parser.parse_args()

    target_ids = set(args.row_ids)
    meta = list(csv.DictReader(open(DATA_DIR / "meta.csv")))
    targets = [r for r in meta if int(r["row_id"]) in target_ids]

    print(f"Updating {len(targets)} graphs...")

    from gnn.graph import build_graph as build_gnn_graph
    from egnn.graph import build_graph as build_egnn_graph

    ok = 0
    for r in targets:
        name  = r["name"]
        label = r["label"]
        folder = OUTPUT_DIR / name
        print(f"  {name}")

        y     = torch.tensor([int(label == "agonist")], dtype=torch.float)
        group = name.split("_")[0]

        # GNN graph
        g = build_gnn_graph(folder, cutoff=args.cutoff)
        if g is not None:
            g.y = y
            g.group = group
            torch.save(g, GNN_DIR / f"{name}.pt")
        else:
            print(f"    [WARN] GNN graph failed")

        # EGNN graph
        eg = build_egnn_graph(folder, cutoff=args.cutoff)
        if eg is not None:
            eg.y = y
            eg.group = group
            torch.save(eg, EGNN_DIR / f"{name}.pt")
        else:
            print(f"    [WARN] EGNN graph failed")

        ok += 1

    print(f"Done. Updated {ok}/{len(targets)} graph pairs.")


if __name__ == "__main__":
    main()
