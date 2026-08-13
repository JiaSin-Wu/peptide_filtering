"""
precompute_graphs.py

Pre-build all PyG graphs from AF3 outputs and cache to ml/data/graphs/*.pt.
Only needs to run once; training then loads from disk (no HDD re-reads).

Usage:
  python3 precompute_graphs.py
  python3 precompute_graphs.py --cutoff 8.0 --workers 4
"""

import argparse
import csv
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from gnn.graph import build_graph

HOME       = Path("/home/jiasin")
OUTPUT_DIR = HOME / "outputs"


def _build_one(args):
    name, label, cutoff, graph_dir, overwrite = args
    out_path = Path(graph_dir) / f"{name}.pt"
    if out_path.exists() and not overwrite:
        return "skip"
    graph = build_graph(OUTPUT_DIR / name, cutoff=cutoff)
    if graph is None:
        return "fail"
    graph.y     = torch.tensor([int(label == "agonist")], dtype=torch.float)
    graph.group = name.split("_")[0]
    torch.save(graph, out_path)
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff",    type=float, default=8.0)
    parser.add_argument("--graph_dir", type=str,   default=None,
                        help="Output directory (default: ml/data/graphs_<cutoff>A)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers",   type=int,   default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir) if args.graph_dir else \
                HOME / f"AMY123R_agonist_Design/model/data/graphs_{args.cutoff:.0f}A"
    graph_dir.mkdir(parents=True, exist_ok=True)

    meta = list(csv.DictReader(open(HOME / "AMY123R_agonist_Design/model/data/meta.csv")))
    print(f"Total samples: {len(meta)}  workers={args.workers}", flush=True)

    tasks = [(r["name"], r["label"], args.cutoff, str(graph_dir), args.overwrite)
             for r in meta]

    ok = skip = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(_build_one, t): t[0] for t in tasks}
        for i, fut in enumerate(tqdm(as_completed(futs), total=len(tasks), desc="Building")):
            result = fut.result()
            if result == "ok":    ok   += 1
            elif result == "skip": skip += 1
            else:
                fail += 1
                print(f"  [FAIL] {futs[fut]}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"\nDone. built={ok}  skipped={skip}  failed={fail}")
    print(f"Output: {graph_dir}")


if __name__ == "__main__":
    main()
