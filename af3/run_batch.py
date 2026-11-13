"""
run_batch.py — build AF3 inputs, run AlphaFold3 once over all of them, and copy the
resulting cif folders to structures/<seq_id>/<receptor>/ (what pose check reads).

No daemon: a single `docker run ... run_alphafold.py --input_dir` loads the model once
and folds every JSON in af3/input/. The JSONs carry their own MSA and
templates, so the AF3 data pipeline is off for folding. The peptide MSA is computed
beforehand by a separate, CPU-only data-pipeline run (build_inputs.compute_msas).

Default scope: 6 candidates x CTR.

Usage:
    python3 af3/run_batch.py --msa-only         # search peptide MSAs -> data/af3_ref/ligand/
    python3 af3/run_batch.py --dry-run --no-msa # only write JSONs to af3/input/
    python3 af3/run_batch.py                    # peptide MSA via jackhmmer, then fold
    python3 af3/run_batch.py --seq GA_005 --skip-done
Settings (image, weights, databases, GPU memory fraction) live in af3/config.py and can
be overridden via environment variables or a .env file; see .env.example.

Nothing is ever written outside this repo.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_inputs as bi   # noqa: E402

ROOT    = bi.ROOT
INF_DIR = ROOT / "af3" / "input"      # complex JSONs (also what --dry-run writes)
OUT_DIR = ROOT / "af3" / "output"     # raw AlphaFold 3 output, one folder per job
DEST    = ROOT / "structures"

DEFAULT_IDS       = ["GA_005", "GA_006", "GA_025", "GA_078", "GA_079", "GA_090"]
DEFAULT_RECEPTORS = ["CTR"]

IMAGE        = bi.IMAGE
MODEL_DIR    = bi.config.AF3_MODEL_DIR
MEM_FRACTION = bi.config.AF3_MEM_FRACTION


def run_alphafold() -> None:
    """One docker run over every JSON in INF_DIR (model is loaded once)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{INF_DIR}:/af3/inputs:ro",
        "-v", f"{OUT_DIR}:/af3/outputs",
        "-v", f"{MODEL_DIR}:/af3/models:ro",
        "-e", "PYTHONPATH=/app/alphafold/src",
        "-e", "HOME=/tmp",   # container runs as the host user; /root is not accessible
        "-e", f"XLA_CLIENT_MEM_FRACTION={MEM_FRACTION}",
        IMAGE,
        "python3", "/app/alphafold/run_alphafold.py",
        "--input_dir=/af3/inputs",
        "--output_dir=/af3/outputs",
        "--model_dir=/af3/models",
        "--run_data_pipeline=false",     # MSA/templates are already in the JSON
        "--num_diffusion_samples=1",     # AF3 default is 5
        "--num_recycles=10",
        "--max_template_date=2021-09-30",
        "--force_output_dir",
    ]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def save_outputs(job_names: list[str]) -> None:
    for job in job_names:
        src = OUT_DIR / job
        seq_id, receptor = job.rsplit("_", 1)
        if not (src / f"{job}_model.cif").exists():
            print(f"  [WARN] no model.cif for {job}", flush=True)
            continue
        dest = DEST / seq_id / receptor
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"  saved {dest.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="comma-separated ids (default: the 6 candidates)")
    ap.add_argument("--receptor", help="comma-separated receptors (default: CTR)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-msa", action="store_true",
                    help="peptide MSA = query only (skips the jackhmmer search)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write JSONs to af3/input/ and stop")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--msa-only", action="store_true",
                    help="only search/save peptide MSAs to data/af3_ref/ligand/, then stop")
    args = ap.parse_args()

    ids = [s.strip() for s in args.seq.split(",")] if args.seq else DEFAULT_IDS
    receptors = ([r.strip() for r in args.receptor.split(",")]
                 if args.receptor else DEFAULT_RECEPTORS)
    peptides = bi.load_peptides(ids)
    print(f"{len(ids)} peptides x {receptors} = {len(ids) * len(receptors)} jobs")

    if args.msa_only:
        bi.compute_msas(peptides, no_msa=args.no_msa)
        return

    jobs = [(i, r) for i in ids for r in receptors
            if not (args.skip_done and (DEST / i / r / f"{i}_{r}_model.cif").exists())]
    if not jobs:
        print("nothing to do")
        return

    # Peptide MSA: AF3 data pipeline (jackhmmer), once per peptide, cached in data/af3_ref/ligand/
    need = sorted({pid for pid, _ in jobs})
    msas = bi.compute_msas({p: peptides[p] for p in need}, no_msa=args.no_msa)

    # af3/input/ must hold exactly this run's JSONs: AF3 folds everything in it
    if INF_DIR.exists():
        shutil.rmtree(INF_DIR)
    names = []
    for pid, rec in jobs:
        msa = msas[pid]
        d = bi.build_complex_json(pid, peptides[pid], msa, rec, args.seed)
        bi.write_json(d, INF_DIR)
        names.append(d["name"])
        print(f"  built {d['name']}  msa_seqs={msa.count('>')}", flush=True)

    if args.dry_run:
        print(f"dry run: JSONs in {INF_DIR.relative_to(ROOT)}")
        return
    run_alphafold()
    save_outputs(names)


if __name__ == "__main__":
    main()
