"""
run_final_filter.py — Integrate all filter results and apply final criteria
(one-shot AND over every candidate at once, assuming 01-05 are already
fully computed for all of them).

Dependencies (must be pre-computed):
  01_pose_check/results.csv                     <- 01_pose_check/run_pose_check.py
  02_binding_energy/rosetta/rosetta_results.csv  <- 02_binding_energy/rosetta/run_rosetta_iface_parallel.py
  filter_tracker_ga.csv                          <- TANGO aggregation column (see filter_lib.gate_tango)
  05_toxicity/outputs/toxinpred3_raw_ga.csv      <- 05_toxicity/run_toxinpred3.py
  04_allergenicity/outputs/AllerCatPro2_prediction_*.csv  <- web upload (manual)

Final filter criteria (logic lives in filter_lib.py, shared with
run_pipeline.py's sequential-funnel version -- see filter_lib's module
docstring for why splitting this into per-stage gates is equivalent):
  Off-target selectivity (each receptor individually), using Rosetta's
  InterfaceAnalyzerMover dG_AB (REU, coordinate-constrained FastRelax
  first -- see 02_binding_energy/rosetta/run_rosetta_iface_parallel.py):
    1. CTR  : pose_pass = 0  OR  Rosetta dG_AB > amylin CTR
    2. CGRP : pose_pass = 0  OR  Rosetta dG_AB > amylin CGRP
    3. AM1R : pose_pass = 0  OR  Rosetta dG_AB > amylin AM1R
    4. AM2R : pose_pass = 0  OR  Rosetta dG_AB > amylin AM2R

  Safety (all three must pass):
    5. TANGO aggregation : no APR (>=5 consecutive residues > 5%)
    6. ToxinPred3        : Non-Toxin
    7. AllerCatPro2      : no evidence

  Note: AMY agonist criteria (pose + Rosetta score) not applied at this
        stage. AMY pose and Rosetta scores are reported for reference only.

  2026-08-28: stage 02 (binding_energy) is deferred / not run for this
        batch. `final_pass` still requires `offt_pass`, so it currently
        passes only GA_006. The report's `final_pass_no_energy` column
        (= amy_pose_pass AND safety_pass) records the 6 candidates that
        clear everything except the energy stage -- see filter_lib.py's
        docstring. Do NOT read the other 5 (GA_005/025/078/079/090) as
        rejected; they are simply not yet evaluated for binding energy.

Output: final_results.csv

Usage:
    python3 run_final_filter.py
"""

import csv

import filter_lib as fl

RECEPTORS = fl.RECEPTORS


def main():
    sequences = list(csv.DictReader(open(fl.SEQ_CSV)))
    ros = fl.load_rosetta()

    if fl.AMYLIN_ID not in ros:
        raise RuntimeError(f"amylin not found in Rosetta results — check {fl.ROSETTA_CSV}")
    amylin_ros = ros[fl.AMYLIN_ID]

    print("Amylin Rosetta dG_AB reference (REU):")
    for r in RECEPTORS:
        print(f"  {r:6s}: {amylin_ros.get(r, 'NA'):.2f}")
    print()

    out_rows = fl.build_report_rows(sequences)
    fl.write_report(out_rows, fl.HERE / "final_results.csv")

    n = len(out_rows)
    n_amy_pose = sum(1 for r in out_rows if r["amy_pose_pass"])
    n_offt     = sum(1 for r in out_rows if r["offt_pass"])
    n_safety   = sum(1 for r in out_rows if r["safety_pass"])
    n_final    = sum(1 for r in out_rows if r["final_pass"])
    n_final_ne = sum(1 for r in out_rows if r["final_pass_no_energy"])

    print(f"{'='*50}")
    print(f"Total sequences   : {n}")
    print(f"AMY pose pass     : {n_amy_pose}")
    print(f"Off-target pass   : {n_offt}")
    print(f"Safety pass       : {n_safety}")
    print(f"FINAL PASS        : {n_final}  (AMY pose + off-target + safety)")
    print(f"FINAL (no energy) : {n_final_ne}  (AMY pose + safety; stage 02 deferred 2026-08-28)")
    print(f"{'='*50}")

    finals = [r["id"] for r in out_rows if r["final_pass"]]
    if finals:
        print(f"\nFinal candidates (final_pass=1): {', '.join(finals)}")

    finals_ne = [r["id"] for r in out_rows if r["final_pass_no_energy"]]
    if finals_ne:
        print(f"Final candidates (final_pass_no_energy=1): {', '.join(finals_ne)}")

    print(f"\nOutput: {fl.HERE / 'final_results.csv'}")


if __name__ == "__main__":
    main()
