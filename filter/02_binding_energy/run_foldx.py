"""
run_foldx.py — FoldX AnalyseComplex on AF3 docking structures.

Pipeline per structure:
  1. Convert AF3 CIF → PDB (biopython)
  2. FoldX RepairPDB  — optimize sidechains / fill missing atoms
  3. FoldX AnalyseComplex — pairwise interaction energies (A/B/C chains)
  4. Parse Interaction_*.fxout → results.csv

Chains:
  A = receptor (CTR or CLR)
  B = peptide
  C = RAMP
  D = Gα (excluded from AnalyseComplex)

Output: filter/02_binding_energy/foldx_results.csv
  Fields: id, receptor, dG_AB, dG_AC, dG_BC
    dG_AB = peptide ↔ receptor
    dG_AC = receptor ↔ RAMP
    dG_BC = peptide ↔ RAMP

Usage:
    conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py
    conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py --seq GA_051,GA_087
    conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py --skip-done
    conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py --workers 8
"""

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, Select

ROOT      = Path(__file__).resolve().parents[2]
DOCK_OUT  = ROOT / "structures"
FOLDX_DIR = Path(__file__).resolve().parent
FOLDX_BIN = FOLDX_DIR / "foldx"
ROTABASE  = FOLDX_DIR / "rotabase.txt"
OUT_CSV   = FOLDX_DIR / "foldx_results.csv"

RECEPTORS = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]

FIELDS = ["id", "receptor", "dG_AB", "dG_AC", "dG_BC"]


class ChainABCSelect(Select):
    """Keep only chains A, B, C (exclude Gα = chain D)."""
    def accept_chain(self, chain):
        return chain.id in ("A", "B", "C")


def cif_to_pdb(cif_path: Path, pdb_path: Path):
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("s", str(cif_path))
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(pdb_path), ChainABCSelect())


def run_foldx_cmd(cmd: list[str], workdir: Path) -> bool:
    r = subprocess.run(
        [str(FOLDX_BIN)] + cmd + [f"--rotabase={ROTABASE}"],
        cwd=str(workdir),
        capture_output=True, text=True,
    )
    if r.returncode != 0 or "run OK" not in r.stdout:
        return False, r.stdout[-500:]
    return True, ""


def parse_interaction(fxout_path: Path) -> dict | None:
    result = {}
    try:
        for line in fxout_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 6 or not parts[0].endswith(".pdb"):
                continue
            g1, g2 = parts[1].strip(), parts[2].strip()
            dg = float(parts[5])
            key = f"dG_{''.join(sorted([g1, g2]))}"
            result[key] = f"{dg:.4f}"
    except Exception:
        return None
    return result if result else None


def process_one(args: tuple) -> dict | None:
    seq_id, receptor = args
    cif = DOCK_OUT / seq_id / receptor / f"{seq_id}_{receptor}_model.cif"
    if not cif.exists():
        return None

    pdb_name = f"{seq_id}_{receptor}"

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        shutil.copy(ROTABASE, workdir / "rotabase.txt")
        pdb_path = workdir / f"{pdb_name}.pdb"

        # 1. CIF → PDB
        try:
            cif_to_pdb(cif, pdb_path)
        except Exception as e:
            print(f"  [CIF→PDB ERROR] {seq_id}/{receptor}: {e}", flush=True)
            return None

        # 2. RepairPDB
        ok, err = run_foldx_cmd(
            ["--command=RepairPDB", f"--pdb={pdb_name}.pdb",
             f"--output-dir={workdir}"], workdir)
        if not ok:
            print(f"  [RepairPDB ERROR] {seq_id}/{receptor}: {err}", flush=True)
            return None

        repaired_pdb = workdir / f"{pdb_name}_Repair.pdb"
        if not repaired_pdb.exists():
            print(f"  [WARN] RepairPDB output missing: {seq_id}/{receptor}", flush=True)
            return None

        # 3. AnalyseComplex
        ok, err = run_foldx_cmd(
            ["--command=AnalyseComplex",
             f"--pdb={pdb_name}_Repair.pdb",
             "--analyseComplexChains=A,B,C",
             f"--output-dir={workdir}"], workdir)
        if not ok:
            print(f"  [AnalyseComplex ERROR] {seq_id}/{receptor}: {err}", flush=True)
            return None

        # 4. Parse
        fxout = workdir / f"Interaction_{pdb_name}_Repair_AC.fxout"
        if not fxout.exists():
            print(f"  [WARN] fxout missing: {seq_id}/{receptor}", flush=True)
            return None

        energies = parse_interaction(fxout)
        if energies is None:
            print(f"  [PARSE ERROR] {seq_id}/{receptor}", flush=True)
            return None

    row = {"id": seq_id, "receptor": receptor,
           "dG_AB": energies.get("dG_AB", "NA"),
           "dG_AC": energies.get("dG_AC", "NA"),
           "dG_BC": energies.get("dG_BC", "NA")}
    print(f"  {seq_id:12s} {receptor:6s}  "
          f"dG_AB={row['dG_AB']:>8s}  dG_AC={row['dG_AC']:>8s}  dG_BC={row['dG_BC']:>8s}",
          flush=True)
    return row


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {(r["id"], r["receptor"]) for r in csv.DictReader(f)}


def save_row(row: dict, path: Path, lock: mp.Lock):
    with lock:
        exists = path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",       help="Comma-separated seq IDs")
    parser.add_argument("--receptor",  help="Comma-separated receptors")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--workers",   type=int, default=4)
    args = parser.parse_args()

    seq_ids   = sorted(d.name for d in DOCK_OUT.iterdir() if d.is_dir())
    receptors = RECEPTORS

    if args.seq:
        seq_filter = {s.strip() for s in args.seq.split(",")}
        seq_ids = [s for s in seq_ids if s in seq_filter]
    if args.receptor:
        receptors = [r.strip() for r in args.receptor.split(",")]

    done = load_done(OUT_CSV) if args.skip_done else set()
    todo = [(s, r) for s in seq_ids for r in receptors if (s, r) not in done]

    print(f"Sequences: {len(seq_ids)}  Receptors: {receptors}")
    print(f"Total: {len(todo)}  Workers: {args.workers}  (skip-done: {len(done)})")
    est_h = len(todo) * 143 / args.workers / 3600
    print(f"Estimated time: ~{est_h:.1f} h\n", flush=True)

    lock = mp.Manager().Lock()

    with mp.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(process_one, todo), 1):
            if row:
                save_row(row, OUT_CSV, lock)
            if i % 50 == 0:
                print(f"  --- checkpoint: {i}/{len(todo)} done ---", flush=True)

    total = sum(1 for _ in open(OUT_CSV)) - 1 if OUT_CSV.exists() else 0
    print(f"\n=== Done. {total} rows written to {OUT_CSV} ===")


if __name__ == "__main__":
    main()
