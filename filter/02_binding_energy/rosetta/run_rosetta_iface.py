"""
run_rosetta_iface.py — SUPERSEDED by run_rosetta_iface_parallel.py (which
also runs fine sequentially with --workers 1, and is what run_pipeline.py
calls). Kept for reference/comparison only, not part of the funnel.

Original docstring follows:

run_rosetta_iface.py — cross-check FoldX dG_AB with Rosetta's InterfaceAnalyzerMover
on the same AF3 structures.

AF3 models are not relaxed for Rosetta's scorefunction (side-chain rotamers/
minor clashes score as huge positive energies if scored raw) -- side-chain
repack + a short constrained minimization is run first, same standard
practice as scoring any non-Rosetta-generated model.

Sequential, not parallel: multiprocessing.Pool (both "fork" and "spawn"
start methods) hung indefinitely at pyrosetta.init()/pool creation in this
environment -- not worth debugging further for a 49-job batch at
~1-2 min/job.

Usage:
    conda run -n pyrosetta python3 run_rosetta_iface.py --seq GA_159,GA_160 --receptor CTR,AM1R
"""
import argparse
import csv
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, Select

ROOT     = Path(__file__).resolve().parents[3]
DOCK_OUT = ROOT / "structures"
HERE     = Path(__file__).resolve().parent
OUT_CSV  = HERE / "rosetta_results.csv"
RECEPTORS = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]


class KeepChainsSelect(Select):
    """
    Keep only the given chains, and drop any residue literally named "NH2"
    (AF3's C-terminal amidation cap, represented as its own pseudo-residue
    with just 1-2 atoms rather than a patch on the real last residue).
    Confirmed by isolated step-by-step testing: every step up through
    relax_pose succeeds fine with it present, but
    InterfaceAnalyzerMover.apply() segfaults every time -- Rosetta's
    fa_standard residue-type set has no NH2-as-its-own-residue definition,
    unlike C-terminal amidation which Rosetta normally represents as a
    variant/patch on the preceding residue, not a separate one.

    Do NOT relabel the peptide chain id to "B" -- tried that, and for any
    receptor with RAMP present, RAMP already occupies "B", so the peptide's
    forced rename collides with it (confirmed: BiopythonWarning "id B is
    already used for a sibling", followed by another segfault, presumably
    on the resulting malformed structure). Keep the peptide's real chain id
    ("L" for our own pipeline, "B" for AlphaFold-Server-built structures)
    and build the InterfaceAnalyzerMover string from that instead.
    """
    def __init__(self, keep: set[str]):
        self.keep = keep

    def accept_chain(self, chain):
        return chain.id in self.keep

    def accept_residue(self, residue):
        return residue.get_resname() != "NH2"


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> str:
    """
    Keep receptor(A) + peptide only (see KeepChainsSelect for the NH2 fix).
    Peptide chain id depends on how the complex was built: AlphaFold-Server
    submissions list it 2nd -> "B" already; our own pipeline's
    build_complex_json always assigns it "L" (ligand). Returns whichever it
    is so the caller can build the matching InterfaceAnalyzerMover string.
    """
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("s", str(cif_path))
    chain_ids = {c.id for c in struct[0]}
    pep_chain_id = "L" if "L" in chain_ids else "B"

    io = PDBIO()
    io.set_structure(struct)
    io.save(str(pdb_path), KeepChainsSelect({"A", pep_chain_id}))
    return pep_chain_id


def relax_pose(pose, scorefxn):
    """
    Full FastRelax (backbone + side-chain, several pack/minimize cycles),
    coordinate-constrained back to the AF3-predicted structure so it fixes
    local clashes/rotamer mismatch without drifting into some unrelated
    Rosetta-favored conformation -- we want the energy of the pose AF3
    actually predicted, not of whatever FastRelax would idealize it into
    unconstrained.
    """
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

    fr = FastRelax(relax_scorefxn, 2)  # 2 repeat cycles, not the default 5 -- keeps a 49-job batch tractable
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


def process_one(seq_id, receptor, scorefxn):
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    cif = DOCK_OUT / seq_id / receptor / f"{seq_id}_{receptor}_model.cif"
    if not cif.exists():
        print(f"  [SKIP] no cif: {seq_id}/{receptor}", flush=True)
        return None
    pdb_path = HERE / f"{seq_id}_{receptor}_AB.pdb"
    try:
        pep_chain_id = cif_to_pdb(cif, pdb_path)
        pose = pose_from_pdb(str(pdb_path))
        relax_pose(pose, scorefxn)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--receptor", default=",".join(RECEPTORS))
    args = ap.parse_args()

    seq_ids = [s.strip() for s in args.seq.split(",")]
    receptors = [r.strip() for r in args.receptor.split(",")]
    todo = [(s, r) for s in seq_ids for r in receptors]

    print(f"Total: {len(todo)} (sequential)", flush=True)

    import pyrosetta
    from pyrosetta import get_fa_scorefxn
    pyrosetta.init("-mute all -ignore_unrecognized_res -ex1 -ex2")
    scorefxn = get_fa_scorefxn()

    rows = []
    for seq_id, receptor in todo:
        result = process_one(seq_id, receptor, scorefxn)
        if result is not None:
            rows.append(result)
            write_header = not OUT_CSV.exists()
            with open(OUT_CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "receptor", "dG_AB_REU", "dSASA"])
                if write_header:
                    w.writeheader()
                w.writerow(result)

    print(f"\n=== Done. {len(rows)} rows written to {OUT_CSV} ===")


if __name__ == "__main__":
    main()
