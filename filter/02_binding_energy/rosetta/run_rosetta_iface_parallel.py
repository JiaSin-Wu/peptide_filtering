"""
run_rosetta_iface_parallel.py — parallel variant of run_rosetta_iface.py,
using the same FastRelax + NH2-exclusion + no-chain-relabeling fixes.

Usage:
    conda run -n pyrosetta python3 run_rosetta_iface_parallel.py --seq GA_159,GA_160 --workers 10
    conda run -n pyrosetta python3 run_rosetta_iface_parallel.py --seq-file alive.txt --extra-id amylin

One of --seq / --seq-file is required -- unlike 01/03/05 there's no cheap
"everything" default here (each job is ~1-2 min x however many receptors).
"""
import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, Select

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import filter_lib as fl

ROOT     = Path(__file__).resolve().parents[3]
DOCK_OUT = ROOT / "structures"
OUT_CSV  = HERE / "rosetta_results.csv"
RECEPTORS = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]

_scorefxn = None


class KeepChainsSelect(Select):
    def __init__(self, keep: set[str]):
        self.keep = keep

    def accept_chain(self, chain):
        return chain.id in self.keep

    def accept_residue(self, residue):
        return residue.get_resname() != "NH2"


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> str:
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("s", str(cif_path))
    chain_ids = {c.id for c in struct[0]}
    pep_chain_id = "L" if "L" in chain_ids else "B"

    io = PDBIO()
    io.set_structure(struct)
    io.save(str(pdb_path), KeepChainsSelect({"A", pep_chain_id}))
    return pep_chain_id


def relax_pose(pose, scorefxn):
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.constraint_generator import (
        CoordinateConstraintGenerator, AddConstraints,
    )
    from pyrosetta.rosetta.core.scoring import ScoreType

    coord_gen = CoordinateConstraintGenerator()
    add_csts = AddConstraints()
    add_csts.add_generator(coord_gen)
    add_csts.apply(pose)

    relax_scorefxn = scorefxn.clone()
    relax_scorefxn.set_weight(ScoreType.coordinate_constraint, 1.0)

    fr = FastRelax(relax_scorefxn, 2)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


def _worker_init():
    global _scorefxn
    print(f"[worker {mp.current_process().pid}] initializing...", flush=True)
    import pyrosetta
    from pyrosetta import get_fa_scorefxn
    pyrosetta.init("-mute all -ignore_unrecognized_res -ex1 -ex2")
    _scorefxn = get_fa_scorefxn()
    print(f"[worker {mp.current_process().pid}] ready", flush=True)


def _process_one(args) -> dict | None:
    seq_id, receptor = args
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    cif = DOCK_OUT / seq_id / receptor / f"{seq_id}_{receptor}_model.cif"
    if not cif.exists():
        print(f"  [SKIP] no cif: {seq_id}/{receptor}", flush=True)
        return None
    pdb_path = HERE / f"{seq_id}_{receptor}_AB_{mp.current_process().pid}.pdb"
    try:
        pep_chain_id = cif_to_pdb(cif, pdb_path)
        pose = pose_from_pdb(str(pdb_path))
        relax_pose(pose, _scorefxn)
        mover = InterfaceAnalyzerMover(f"A_{pep_chain_id}")
        mover.apply(pose)
        dG = mover.get_interface_dG()
        dSASA = mover.get_interface_delta_sasa()
        print(f"  {seq_id:10s} {receptor:6s}  dG_AB(REU)={dG:8.2f}  dSASA={dSASA:7.1f}", flush=True)
        return {"id": seq_id, "receptor": receptor, "dG_AB_REU": dG, "dSASA": dSASA}
    except Exception as e:
        print(f"  [ERROR] {seq_id}/{receptor}: {e}", flush=True)
        return None
    finally:
        pdb_path.unlink(missing_ok=True)


def main():
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="Comma-separated seq IDs")
    ap.add_argument("--seq-file", help="Path to a file with one seq ID per line")
    ap.add_argument("--extra-id", help="Comma-separated extra IDs to always include (e.g. amylin)")
    ap.add_argument("--receptor", default=",".join(RECEPTORS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.seq and not args.seq_file:
        ap.error("one of --seq or --seq-file is required")

    seq_ids = fl.resolve_ids(None, args.seq, args.seq_file, args.extra_id)
    receptors = [r.strip() for r in args.receptor.split(",")]
    todo = [(s, r) for s in seq_ids for r in receptors]

    print(f"Total: {len(todo)}  Workers: {args.workers}", flush=True)

    write_header = not OUT_CSV.exists()
    lock = mp.Lock()

    def save(row):
        nonlocal write_header
        with lock:
            with open(OUT_CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "receptor", "dG_AB_REU", "dSASA"])
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(row)

    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_process_one, todo):
            if result is not None:
                save(result)

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
