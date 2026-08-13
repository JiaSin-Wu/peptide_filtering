"""
run_docking.py — AF3 docking for 38 final GA sequences × 7 receptors.

Same logic as GA worker.py:
  - MSA on full 37aa sequence (PEPTIDE_PREFIX + variable region)
  - C-terminal NH2 → X in MSA query line
  - MSA(N+1) runs on CPU while AF3(N) runs on GPU (pipelined)

Outputs saved to: structures/<seq_id>/<receptor>/

Usage:
    conda run -n af3_ml python3 docking/run_docking.py
    conda run -n af3_ml python3 docking/run_docking.py --dry-run
    conda run -n af3_ml python3 docking/run_docking.py --skip-done
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
GA_DIR   = ROOT / "GA"
FILTER   = ROOT / "filter"
DOCK_OUT = ROOT / "structures"
AF3_OUT  = Path.home() / "outputs"

if "MMSEQS_BIN" not in os.environ:
    os.environ["MMSEQS_BIN"] = "/home/jiasin/miniconda3/envs/mmseqs2/bin/mmseqs"

sys.path.insert(0, str(GA_DIR))
sys.path.insert(0, str(GA_DIR / "ga"))
from worker import _msa_one
from oracle import (
    build_complex_json,
    run_inference_batch,
    PEPTIDE_PREFIX,
    RECEPTORS,
    OUT_DIR,
)

DAEMON_CONTAINER = os.environ.get("AF3_DAEMON_CONTAINER", "af3_daemon")


def already_done(seq_id: str, receptors: list[str]) -> bool:
    return all((DOCK_OUT / seq_id / r).exists() for r in receptors)


def compute_msa(seq_id: str, seq_24: str) -> str:
    """MSA on full 37aa, C-terminal NH2 → X. Same as worker._msa_one."""
    full_37 = PEPTIDE_PREFIX + seq_24
    af3_seq = full_37[:-1] + "X"
    t0 = time.time()
    msa = _msa_one(full_37, af3_seq)
    print(f"  [{seq_id}] MSA: {msa.count(chr(62))} seqs in {time.time()-t0:.0f}s",
          flush=True)
    return msa


def run_af3(seq_id: str, seq_24: str, msa: str, receptors: list[str], seed: int | None = None) -> list[str]:
    """Build complex JSONs and submit to AF3 daemon."""
    complex_jsons = []
    job_names     = []
    for receptor in receptors:
        cj = build_complex_json(seq_id, seq_24, msa, receptor)
        if seed is not None:
            cj["modelSeeds"] = [seed]
        complex_jsons.append(cj)
        job_names.append(cj["name"])

    print(f"  [{seq_id}] AF3: {len(complex_jsons)} jobs...", flush=True)
    t0 = time.time()
    run_inference_batch(complex_jsons)
    print(f"  [{seq_id}] AF3 done in {(time.time()-t0)/60:.1f} min", flush=True)
    return job_names


def save_outputs(seq_id: str, job_names: list[str]):
    """Copy AF3 outputs from ~/outputs/ to docking/outputs/<seq_id>/."""
    dest_base = DOCK_OUT / seq_id
    dest_base.mkdir(parents=True, exist_ok=True)
    for job_name in job_names:
        src = OUT_DIR / job_name
        if not src.exists():
            print(f"  [WARN] missing: {job_name}", flush=True)
            continue
        receptor = job_name.rsplit("_", 1)[-1]
        dest = dest_base / receptor
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(src), str(dest))
        print(f"  saved: {dest.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-done",  action="store_true")
    parser.add_argument("--seq",        help="Comma-separated seq IDs, e.g. GA_085,GA_102")
    parser.add_argument("--receptor",   help="Comma-separated receptors, e.g. AMY1R or AMY1R,AMY2R")
    parser.add_argument("--seed",       type=int, default=None, help="Override modelSeeds, e.g. 42")
    parser.add_argument("--all-seqs",   action="store_true", help="Use sequences_ga.csv (all 154) instead of ga_final_ranked.csv (38)")
    args = parser.parse_args()

    receptors = [r.strip() for r in args.receptor.split(",")] if args.receptor else RECEPTORS

    src_csv = FILTER / ("sequences_ga.csv" if args.all_seqs else "ga_final_ranked.csv")
    sequences = list(csv.DictReader(open(src_csv)))
    if args.seq:
        seq_filter = {s.strip() for s in args.seq.split(",")}
        sequences = [r for r in sequences if r["id"] in seq_filter]

    print(f"Sequences: {len(sequences)}  Receptors: {receptors}", flush=True)
    print(f"Total AF3 jobs: {len(sequences) * len(receptors)}\n", flush=True)

    if args.dry_run:
        for row in sequences:
            status = "done" if already_done(row["id"], receptors) else "todo"
            print(f"  {row['id']}  {row['seq_24aa']}  [{status}]")
        return

    DOCK_OUT.mkdir(parents=True, exist_ok=True)

    todo = [r for r in sequences
            if not (args.skip_done and already_done(r["id"], receptors))]
    print(f"To process: {len(todo)} sequences", flush=True)
    if not todo:
        print("All done.")
        return

    # ── Pipelined: MSA(N+1) on CPU while AF3(N) on GPU ───────────────────────
    with ThreadPoolExecutor(max_workers=1) as pool:
        msa_future = pool.submit(compute_msa, todo[0]["id"], todo[0]["seq_24aa"])

        for i, row in enumerate(todo):
            seq_id = row["id"]
            seq_24 = row["seq_24aa"]
            print(f"\n[{i+1:2d}/{len(todo)}] {seq_id}  {seq_24}", flush=True)

            msa = msa_future.result()

            if i + 1 < len(todo):
                nxt = todo[i + 1]
                msa_future = pool.submit(compute_msa, nxt["id"], nxt["seq_24aa"])

            job_names = run_af3(seq_id, seq_24, msa, receptors, seed=args.seed)

            save_outputs(seq_id, job_names)

    print(f"\n=== Done. Structures in: {DOCK_OUT} ===", flush=True)


if __name__ == "__main__":
    main()
