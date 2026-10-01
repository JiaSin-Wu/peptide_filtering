"""
run_pipeline.py

Orchestrator for the 3_Filter pipeline: runs stages 01-05 as a real
sequential funnel -- only survivors of stage N are handed to stage N+1 --
in whatever order --order specifies. Order is just a comma-separated
string; change it and rerun, no code changes needed.

Default order: 03,04,05,01,02 -- run the three sequence-only checks
(aggregation/allergenicity/toxicity: cheap, no docked structures needed)
first and eliminate weak candidates, then only spend Rosetta time
(1-2 min/job x however many receptors) on the two structural checks
(pose_check, binding_energy) for whoever's left.

Gate-split correctness: see filter_lib.py's module docstring. Because
boolean AND is order-independent, any --order permutation must converge
on the same final surviving set as run_final_filter.py's one-shot AND --
verified with --replay-only (see filter.md's Verification section).

04 (AllerCatPro2) has no API -- it's a manual web tool. When this stage
hits candidates not yet covered by an existing
outputs/AllerCatPro2_prediction_*.csv, it writes a new subset FASTA batch,
prints upload instructions, and exits (EXIT_PAUSED). Re-run the same
command with --resume after uploading/downloading to continue --
already-completed stages are skipped via their checkpoint files.

Usage:
    python3 run_pipeline.py
    python3 run_pipeline.py --order 01,02,03,04,05
    python3 run_pipeline.py --resume
    python3 run_pipeline.py --replay-only --order 04,03,01,05,02
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import filter_lib as fl

HERE       = Path(__file__).resolve().parent
STATE_ROOT = HERE / "pipeline_state"

EXIT_PAUSED = 75

STAGE_ORDER_DEFAULT = ["03", "04", "05", "01"]
# 02 (Rosetta binding energy) is left out of the default order 2026-08-26 --
# it's the slowest stage (1-2 min/job) and needs `structures/` + a working
# pyrosetta env, neither reliably available; run it explicitly with
# --order 03,04,05,01,02 (or any other order including "02") when you have
# both. NOTE: `final_pass` in filter_lib.build_report_rows() still requires
# offt_pass (02's gate) regardless of --order -- leaving 02 out here only
# skips *orchestrating* that stage's computation, it doesn't relax the
# final_pass formula. If 02 hasn't been run for a candidate, it's
# conservatively offt_pass=0 in the final report.
#
# 2026-08-28: 02 is being deferred for the current batch. While it is,
# read the pass set off the report's `final_pass_no_energy` column
# (amy_pose_pass AND safety_pass) rather than `final_pass` -- see
# filter_lib.py's module docstring.

# 04 has no conda env -- it's a manual web-upload step, not a local script.
STAGE_CONDA_ENV = {"01": "af3_ml", "02": "pyrosetta", "03": None, "05": "raghava_tools"}

STAGE_SCRIPT = {
    "01": HERE / "01_pose_check" / "run_pose_check.py",
    "02": HERE / "02_binding_energy" / "rosetta" / "run_rosetta_iface_parallel.py",
    "03": HERE / "03_aggregation" / "run_tango.py",
    "05": HERE / "05_toxicity" / "run_toxinpred3.py",
}

STAGE_NAME = {
    "01": "01_pose_check",
    "02": "02_binding_energy",
    "03": "03_aggregation",
    "04": "04_allergenicity",
    "05": "05_toxicity",
}


def all_candidate_ids() -> list[str]:
    return [r["id"] for r in csv.DictReader(open(fl.SEQ_CSV))]


def build_cmd(stage: str, seqfile: Path, workers: int) -> list[str]:
    script = STAGE_SCRIPT[stage]
    tail = ["--seq-file", str(seqfile)]
    if stage in ("01", "02"):
        tail += ["--extra-id", fl.AMYLIN_ID]
    if stage == "02":
        tail += ["--workers", str(workers)]
    env = STAGE_CONDA_ENV[stage]
    prefix = ["conda", "run", "-n", env] if env else []
    return prefix + ["python3", str(script)] + tail


def append_audit(audit_csv: Path, stage: str, eliminated: list[str]) -> None:
    write_header = not audit_csv.exists()
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["id", "eliminated_at_stage"])
        for sid in eliminated:
            w.writerow([sid, STAGE_NAME[stage]])


def write_fasta_batches(ids: list[str], seq_lookup: dict, outdir: Path, prefix: str,
                          batch_size: int = 50) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        path = outdir / f"{prefix}_part{i // batch_size + 1}.fa"
        with open(path, "w") as f:
            for sid in batch:
                f.write(f">{sid}\n{seq_lookup[sid]}\n")
        paths.append(path)
    return paths


def run_auto_stage(stage: str, alive: list[str], run_dir: Path, workers: int) -> list[str]:
    seqfile = run_dir / f"input_{stage}.txt"
    ids_for_run = sorted(set(alive) | ({fl.AMYLIN_ID} if stage in ("01", "02") else set()))
    fl.write_ids(seqfile, ids_for_run)

    cmd = build_cmd(stage, seqfile, workers)
    print(f"\n[{stage}] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(HERE))

    gate_result = fl.GATES[stage](alive)
    survivors, eliminated = fl.apply_gate(alive, gate_result)
    fl.write_ids(run_dir / f"alive_after_{stage}.txt", survivors)
    append_audit(run_dir / "audit.csv", stage, eliminated)
    print(f"[{stage}] {len(survivors)}/{len(alive)} survived", flush=True)
    return survivors


def run_manual_stage_04(alive: list[str], run_dir: Path) -> list[str]:
    covered = set(fl.load_allercatpro().keys())
    missing = [i for i in alive if i not in covered]
    if missing:
        seq_lookup = {r["id"]: r["full_37aa"] for r in csv.DictReader(open(fl.SEQ_CSV))}
        batches = write_fasta_batches(
            missing, seq_lookup,
            HERE / "04_allergenicity" / "inputs",
            prefix=f"ga_sequences_pending_{run_dir.name}",
        )
        print(f"\n[04] {len(missing)} candidate(s) have no AllerCatPro2 result yet.")
        print(f"[04] Upload the batch(es) below to https://allercatpro.bioinf.nl/ and")
        print(f"     save the downloaded CSV(s) into 04_allergenicity/outputs/, then")
        print(f"     re-run this exact command with --resume to continue:")
        for p in batches:
            print(f"       {p}")
        sys.exit(EXIT_PAUSED)

    gate_result = fl.GATES["04"](alive)
    survivors, eliminated = fl.apply_gate(alive, gate_result)
    fl.write_ids(run_dir / "alive_after_04.txt", survivors)
    append_audit(run_dir / "audit.csv", "04", eliminated)
    print(f"[04] {len(survivors)}/{len(alive)} survived", flush=True)
    return survivors


def replay_stage(stage: str, alive: list[str], run_dir: Path) -> list[str]:
    """--replay-only: apply the gate to whatever's already on disk, no
    subprocess calls at all -- used to verify order-independence (see
    filter.md's Verification section) without touching any external tool.
    """
    gate_result = fl.GATES[stage](alive)
    survivors, eliminated = fl.apply_gate(alive, gate_result)
    fl.write_ids(run_dir / f"alive_after_{stage}.txt", survivors)
    append_audit(run_dir / "audit.csv", stage, eliminated)
    print(f"[{stage}] (replay) {len(survivors)}/{len(alive)} survived", flush=True)
    return survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default=",".join(STAGE_ORDER_DEFAULT),
                     help="Comma-separated stage codes, e.g. 03,04,05,01,02")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seq-file", help="Initial candidate subset (default: all of sequences_ga.csv)")
    ap.add_argument("--replay-only", action="store_true",
                     help="Apply gates to existing output files only, no subprocess calls")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    order = [s.strip() for s in args.order.split(",")]
    unknown = set(order) - set(fl.GATES)
    if unknown:
        raise SystemExit(f"unknown stage(s) in --order: {sorted(unknown)}")

    run_dir = STATE_ROOT / "-".join(order)
    run_dir.mkdir(parents=True, exist_ok=True)

    alive = fl.read_ids(Path(args.seq_file)) if args.seq_file else all_candidate_ids()
    print(f"Order: {'->'.join(order)}  |  Starting candidates: {len(alive)}\n")

    for stage in order:
        checkpoint = run_dir / f"alive_after_{stage}.txt"
        if args.resume and checkpoint.exists():
            alive = fl.read_ids(checkpoint)
            print(f"[{stage}] resumed from checkpoint: {len(alive)} alive", flush=True)
            continue
        if args.replay_only:
            alive = replay_stage(stage, alive, run_dir)
        elif stage == "04":
            alive = run_manual_stage_04(alive, run_dir)
        else:
            alive = run_auto_stage(stage, alive, run_dir, args.workers)

    sequences = list(csv.DictReader(open(fl.SEQ_CSV)))
    eliminated_at = {}
    audit_csv = run_dir / "audit.csv"
    if audit_csv.exists():
        for r in csv.DictReader(open(audit_csv)):
            eliminated_at[r["id"]] = r["eliminated_at_stage"]

    rows = fl.build_report_rows(sequences, eliminated_at=eliminated_at)
    out_csv = run_dir / "final_results.csv"
    fl.write_report(rows, out_csv)

    n_final = sum(r["final_pass"] for r in rows)
    print(f"\n{'='*50}")
    print(f"Order: {'->'.join(order)}")
    print(f"Alive at end of funnel : {len(alive)}")
    print(f"FINAL PASS (cross-check): {n_final} / {len(rows)}")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()
