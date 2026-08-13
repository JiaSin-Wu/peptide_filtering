"""
precompute_egnn_graphs.py

Build EGNN graphs (with Cα pos) from AF3 outputs.
Output: data/graphs_8A_egnn/*.pt

Usage:
  python3 precompute_egnn_graphs.py
  python3 precompute_egnn_graphs.py --workers 8
"""

import argparse, csv, torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from egnn.graph import build_graph

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
    parser.add_argument("--graph_dir", type=str,   default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers",   type=int,   default=4)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir) if args.graph_dir else \
                HOME / f"AMY123R_agonist_Design/model/data/graphs_{args.cutoff:.0f}A_egnn"
    graph_dir.mkdir(parents=True, exist_ok=True)

    meta  = list(csv.DictReader(open(HOME / "AMY123R_agonist_Design/model/data/meta.csv")))
    tasks = [(r["name"], r["label"], args.cutoff, str(graph_dir), args.overwrite)
             for r in meta]
    print(f"Total: {len(tasks)}  workers={args.workers}  output={graph_dir}", flush=True)

    ok = skip = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(_build_one, t): t[0] for t in tasks}
        for i, fut in enumerate(tqdm(as_completed(futs), total=len(tasks))):
            res = fut.result()
            if res == "ok":    ok   += 1
            elif res == "skip": skip += 1
            else:
                fail += 1
                print(f"  [FAIL] {futs[fut]}", flush=True)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)} ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"\nDone. built={ok}  skipped={skip}  failed={fail}")


if __name__ == "__main__":
    main()
