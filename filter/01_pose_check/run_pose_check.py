"""
run_pose_check.py — AF3 docking pose quality check.

Two geometric checks per (seq_id, receptor) pair:
  1. N-term in TMD  : min Cα dist, peptide res 1-13  → chain A TMD
  2. C-term to DEEP : min Cα dist, peptide res 26-37 → chain A ECD deep groove (res 80-131 CTR / 80-129 CLR)

ECD boundaries (GPCRdb + UniProt signal peptide):
  CTR  (calcr_human, chain A len=474) : signal peptide 1-24, ECD 25-131, TM1 132
  CLR  (calrl_human, chain A len=461) : signal peptide 1-22, ECD 23-129, TM1 130

Pass criteria (both must pass):
  d_nterm_tmd  <= 8 Å
  d_cterm_deep <= 8 Å

Output: pose_check/outputs/results.csv

Usage:
    conda run -n af3_ml python3 filter/pose_check/run_pose_check.py
    conda run -n af3_ml python3 filter/pose_check/run_pose_check.py --seq GA_074
    conda run -n af3_ml python3 filter/pose_check/run_pose_check.py --seq-file alive.txt --extra-id amylin
    conda run -n af3_ml python3 filter/pose_check/run_pose_check.py --receptor AMY1R,AMY2R
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import filter_lib as fl

ROOT     = Path(__file__).resolve().parents[2]
DOCK_OUT = ROOT / "structures"
OUT_CSV  = HERE / "outputs" / "results.csv"

RECEPTORS = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]

# ECD boundaries (GPCRdb, signal peptide excluded)
# Chain A len 474 → CTR (calcr_human), len 461 → CLR (calrl_human).
# 450/438 are the same two constructs with the N-terminal signal peptide
# trimmed before folding (AlphaFold Server submission, run_001's GA_155-158
# batch) — confirmed empirically the 450/438-residue chain A sequence is an
# exact substring of the 474/461 reference at offset 24/23 respectively, so
# every boundary below is just (reference boundary − that offset).
ECD_END    = {474: 131, 461: 129, 450: 131 - 24, 438: 129 - 23}  # last ECD residue (GPCRdb)
TM1_START  = {474: 132, 461: 130, 450: 132 - 24, 438: 130 - 23}  # first TMD residue
DEEP_START = {474: 80,  461: 80,  450: 80 - 24,  438: 80 - 23}   # deep groove start; end = ECD_END
TMD_END    = {474: 401, 461: 401, 450: 401 - 24, 438: 401 - 23}  # generous upper bound past all TM helices

PASS_Å = 8.0

FIELDS = ["id", "receptor",
          "d_nterm_tmd", "d_cterm_deep",
          "nterm_pass", "deep_pass", "pose_pass"]


def parse_ca(cif_path: Path) -> dict:
    atoms = {}
    for line in cif_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        p = line.split()
        if len(p) < 13 or p[3] != "CA":
            continue
        try:
            atoms[(p[6], int(p[8]))] = np.array([float(p[10]), float(p[11]), float(p[12])])
        except ValueError:
            pass
    return atoms


def min_dist(coords_a: list, coords_b: list) -> float:
    if not coords_a or not coords_b:
        return 999.0
    A = np.stack(coords_a)
    B = np.stack(coords_b)
    return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)).min())


def check_pose(seq_id: str, receptor: str) -> dict | None:
    cif = DOCK_OUT / seq_id / receptor / f"{seq_id}_{receptor}_model.cif"
    if not cif.exists():
        return None

    atoms = parse_ca(cif)
    # Peptide chain id depends on how the complex was built: AlphaFold-Server
    # submissions (older GA_155-158 batch) list the peptide as the 2nd entity
    # -> chain "B". Our own pipeline's build_complex_json always assigns the
    # peptide chain id "L" (ligand) regardless of receptor, appended after
    # receptor/RAMP/Gα -- "B" there is RAMP (or absent for RAMP-less
    # receptors like CTR), not the peptide. Prefer "L", fall back to "B".
    pep_chain = "L" if any(k[0] == "L" for k in atoms) else "B"
    if not any(k[0] == pep_chain for k in atoms):
        print(f"  [WARN] no peptide chain (L or B) in {cif.name}", flush=True)
        return None

    chain_a_len = sum(1 for (c, _) in atoms if c == "A")
    if chain_a_len not in ECD_END:
        print(f"  [WARN] unexpected chain A length {chain_a_len} in {seq_id}/{receptor}", flush=True)
        return None

    ecd_end     = ECD_END[chain_a_len]
    tm1_start   = TM1_START[chain_a_len]
    deep_start  = DEEP_START[chain_a_len]
    tmd_end     = TMD_END[chain_a_len]

    pep_nterm = [atoms[(pep_chain, r)] for r in range(1,  14) if (pep_chain, r) in atoms]
    pep_cterm = [atoms[(pep_chain, r)] for r in range(26, 38) if (pep_chain, r) in atoms]

    rec_deep = [atoms[("A", r)] for r in range(deep_start, ecd_end + 1) if ("A", r) in atoms]
    rec_tmd  = [atoms[("A", r)] for r in range(tm1_start,  tmd_end)    if ("A", r) in atoms]

    d_nterm = min_dist(pep_nterm, rec_tmd)
    d_deep  = min_dist(pep_cterm, rec_deep)

    nterm_pass = d_nterm <= PASS_Å
    deep_pass  = d_deep  <= PASS_Å
    pose_pass  = nterm_pass and deep_pass

    return {
        "id":           seq_id,
        "receptor":     receptor,
        "d_nterm_tmd":  f"{d_nterm:.2f}",
        "d_cterm_deep": f"{d_deep:.2f}",
        "nterm_pass":   int(nterm_pass),
        "deep_pass":    int(deep_pass),
        "pose_pass":    int(pose_pass),
    }


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",       help="Comma-separated seq IDs")
    parser.add_argument("--seq-file",  help="Path to a file with one seq ID per line")
    parser.add_argument("--extra-id",  help="Comma-separated extra IDs to always include (e.g. amylin)")
    parser.add_argument("--receptor",  help="Comma-separated receptors")
    args = parser.parse_args()

    all_seq_ids = sorted(d.name for d in DOCK_OUT.iterdir() if d.is_dir())
    seq_ids   = fl.resolve_ids(all_seq_ids, args.seq, args.seq_file, args.extra_id)
    receptors = RECEPTORS

    if args.receptor:
        receptors = [r.strip() for r in args.receptor.split(",")]

    print(f"Sequences: {len(seq_ids)}  Receptors: {receptors}\n", flush=True)

    rows = []
    for seq_id in seq_ids:
        for rec in receptors:
            row = check_pose(seq_id, rec)
            if row is None:
                continue
            rows.append(row)
            flag = "" if row["pose_pass"] else "  <-- FAIL"
            print(
                f"  {seq_id:12s} {rec:6s}  "
                f"N→TMD={row['d_nterm_tmd']:>6s}Å  "
                f"C→DEEP={row['d_cterm_deep']:>6s}Å"
                f"{flag}",
                flush=True,
            )

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(1 for r in rows if r["pose_pass"])
    print(f"\nTotal: {len(rows)}  PASS: {n_pass}  FAIL: {len(rows) - n_pass}")
    print(f"Results: {OUT_CSV}")


if __name__ == "__main__":
    main()
