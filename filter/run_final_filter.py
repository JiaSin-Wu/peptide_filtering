"""
run_final_filter.py — Integrate all filter results and apply final criteria
(one-shot AND over every candidate at once, assuming 01, 03, 04 and 05 are
already fully computed for all of them).

Dependencies (must be pre-computed):
  01_pose_check/outputs/results_from_md_af3.csv  <- sequence-verified AMY1R/2R/3R structures
  03_aggregation/outputs/tango_results.csv       <- 03_aggregation/run_tango.py
  05_toxicity/outputs/toxinpred3_raw_ga.csv      <- 05_toxicity/run_toxinpred3.py
  04_allergenicity/outputs/AllerCatPro2_prediction_*.csv  <- web upload (manual)

Final filter criteria (logic lives in filter_lib.py, shared with
run_pipeline.py's sequential-funnel version -- see filter_lib's module
docstring for why splitting this into per-stage gates is equivalent):
  On-target pose (all three must dock): AMY1R, AMY2R, AMY3R
  Safety (all three must pass):
    1. TANGO aggregation : no APR (>=5 consecutive residues > 5%)
    2. ToxinPred3        : Non-Toxin
    3. AllerCatPro2      : no evidence

  final_pass = amy_pose_pass AND safety_pass

  Note: cd4_pass (CD4episcore, stage 06) is reported but not part of final_pass.

Output: final_results.csv

Usage:
    python3 run_final_filter.py
"""

import csv

import filter_lib as fl


def main():
    sequences = list(csv.DictReader(open(fl.SEQ_CSV)))

    out_rows = fl.build_report_rows(sequences)
    fl.write_report(out_rows, fl.HERE / "outputs" / "final_results.csv")

    n = len(out_rows)
    n_amy_pose = sum(1 for r in out_rows if r["amy_pose_pass"])
    n_safety   = sum(1 for r in out_rows if r["safety_pass"])
    n_final    = sum(1 for r in out_rows if r["final_pass"])

    print(f"{'='*50}")
    print(f"Total sequences   : {n}")
    print(f"AMY pose pass     : {n_amy_pose}")
    print(f"Safety pass       : {n_safety}")
    print(f"FINAL PASS        : {n_final}  (AMY pose + safety)")
    print(f"{'='*50}")

    finals = [r["id"] for r in out_rows if r["final_pass"]]
    if finals:
        print(f"\nFinal candidates (final_pass=1): {', '.join(finals)}")

    print(f"\nOutput: {fl.HERE / 'outputs' / 'final_results.csv'}")


if __name__ == "__main__":
    main()
