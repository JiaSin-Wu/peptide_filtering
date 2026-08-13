"""
run_final_filter.py — Integrate all filter results and apply final criteria.

Dependencies (must be pre-computed):
  filter/01_pose_check/results.csv         ← 01_pose_check/run_pose_check.py
  filter/02_binding_energy/foldx_results.csv ← 02_binding_energy/run_foldx.py
  filter/filter_tracker_ga.csv             ← run_ga_filters.py (TANGO)
  filter/05_toxicity/outputs/toxinpred3_raw_ga.csv  ← run_ga_filters.py or manual
  filter/04_allergenicity/outputs/AllerCatPro2_prediction_*.csv  ← web upload (manual)

Final filter criteria:
  Off-target selectivity (each receptor individually):
    1. CTR  : pose_pass = 0  OR  FoldX dG_AB > amylin CTR  (-38.14 kcal/mol)
    2. CGRP : pose_pass = 0  OR  FoldX dG_AB > amylin CGRP (-34.10 kcal/mol)
    3. AM1R : pose_pass = 0  OR  FoldX dG_AB > amylin AM1R (-30.70 kcal/mol)
    4. AM2R : pose_pass = 0  OR  FoldX dG_AB > amylin AM2R (-36.87 kcal/mol)

  Safety (all three must pass):
    5. TANGO aggregation : no APR (≥5 consecutive residues > 5%)
    6. ToxinPred3        : Non-Toxin
    7. AllerCatPro2      : no evidence

  Note: AMY agonist criteria (pose + FoldX score) not applied at this stage.
        AMY pose and FoldX scores are reported for reference only.

Output: filter/final_results.csv

Usage:
    python3 filter/run_final_filter.py
"""

import csv
import glob
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
FILTER  = ROOT / "filter"

POSE_CSV    = FILTER / "01_pose_check" / "results.csv"
FOLDX_CSV   = FILTER / "02_binding_energy" / "foldx_results.csv"
TRACKER_CSV = FILTER / "filter_tracker_ga.csv"
TOX3_CSV    = FILTER / "05_toxicity" / "outputs" / "toxinpred3_raw_ga.csv"
ALLER_GLOB  = str(FILTER / "04_allergenicity" / "outputs" / "AllerCatPro2_prediction_*.csv")
SEQ_CSV     = FILTER / "sequences_ga.csv"
IMM_CSV     = FILTER / "06_immunogenicity" / "outputs" / "netmhciipan_sb_summary.csv"
OUT_CSV     = FILTER / "final_results.csv"

RECEPTORS     = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]
AMY_RECS      = ["AMY1R", "AMY2R", "AMY3R"]
OFFT_RECS     = ["CTR", "CGRP", "AM1R", "AM2R"]
AMYLIN_ID     = "amylin"


def load_pose() -> defaultdict:
    d = defaultdict(dict)
    for r in csv.DictReader(open(POSE_CSV)):
        d[r["id"]][r["receptor"]] = int(r["pose_pass"])
    return d


def load_foldx() -> defaultdict:
    d = defaultdict(dict)
    for r in csv.DictReader(open(FOLDX_CSV)):
        d[r["id"]][r["receptor"]] = float(r["dG_AB"])
    return d


def load_tracker() -> dict:
    return {r["id"]: r for r in csv.DictReader(open(TRACKER_CSV))}


def load_tox3() -> dict:
    """ToxinPred3: Non-Toxin = PASS"""
    d = {}
    for r in csv.DictReader(open(TOX3_CSV)):
        sid = r.get("Subject") or r.get("id", "")
        d[sid] = r.get("Prediction", "") == "Non-Toxin"
    return d


def load_allercatpro() -> dict:
    """AllerCatPro2: 'no evidence' = PASS"""
    d = {}
    for path in glob.glob(ALLER_GLOB):
        for r in csv.DictReader(open(path)):
            sid = r.get("Protein", "")
            if sid:
                d[sid] = r.get("Result", "") == "no evidence"
    return d


def load_immunogenicity() -> dict:
    """netmhciipan_sb_summary.csv → seq_id: {n_sb_lt2pct, n_wb_le10pct, sb_vs_amylin}

    NetMHCIIpan 4.1 BA, 7-allele DRB panel, 15-mer scan of the full 37aa
    sequence. n_sb_lt2pct = Strong Binder count (percentile rank < 2%) --
    the metric that actually differentiates candidates; at the looser 10%
    Weak-Binder cutoff several final candidates are indistinguishable from
    each other since their sequence differences fall outside every scored
    15-mer window. See filter/06_immunogenicity/run_netmhciipan_sb.py.
    """
    d = {}
    if not IMM_CSV.exists():
        return d
    for r in csv.DictReader(open(IMM_CSV)):
        d[r["seq_id"]] = r
    return d


def main():
    # ── Load all data ────────────────────────────────────────────────────────
    sequences = list(csv.DictReader(open(SEQ_CSV)))
    pose      = load_pose()
    fx        = load_foldx()
    tracker   = load_tracker()
    tox3      = load_tox3()
    aller     = load_allercatpro()
    imm       = load_immunogenicity()

    if AMYLIN_ID not in fx:
        raise RuntimeError(f"amylin not found in FoldX results — check {FOLDX_CSV}")

    amylin_fx = fx[AMYLIN_ID]

    print("Amylin FoldX reference (kcal/mol):")
    for r in RECEPTORS:
        print(f"  {r:6s}: {amylin_fx.get(r, 'NA'):.2f}")
    print()

    # ── Apply criteria ───────────────────────────────────────────────────────
    out_rows = []

    for seq in sequences:
        sid = seq["id"]

        # Off-target selectivity criteria
        offt_results = {}
        for r in OFFT_RECS:
            bad_pose = pose[sid].get(r, 0) == 0
            weak_bind = fx[sid].get(r, 0.0) > amylin_fx[r]
            offt_results[r] = bad_pose or weak_bind
        offt_pass = all(offt_results.values())

        # Safety criteria
        t       = tracker.get(sid, {})
        tango_pass = str(t.get("aggregation", "")) == "1"
        tox3_pass  = tox3.get(sid, False)
        aller_pass = aller.get(sid, False)
        safety_pass = tango_pass and tox3_pass and aller_pass

        # AMY criteria not applied — scores reported for reference only
        amy_results = {r: (pose[sid].get(r, 0) == 1 and fx[sid].get(r, 0.0) < amylin_fx[r])
                       for r in AMY_RECS}
        amy_pass = all(amy_results.values())

        # AMY pose check: all three target receptors must have good pose
        amy_pose_pass = all(pose[sid].get(r, 0) == 1 for r in AMY_RECS)

        final_pass = offt_pass and safety_pass and amy_pose_pass

        row = {
            "id":         sid,
            "seq_24aa":   seq["seq_24aa"],
            "full_37aa":  seq["full_37aa"],
            # FoldX scores
            "fx_AMY1R":   f"{fx[sid].get('AMY1R', float('nan')):.2f}",
            "fx_AMY2R":   f"{fx[sid].get('AMY2R', float('nan')):.2f}",
            "fx_AMY3R":   f"{fx[sid].get('AMY3R', float('nan')):.2f}",
            "fx_CTR":     f"{fx[sid].get('CTR',   float('nan')):.2f}",
            "fx_CGRP":    f"{fx[sid].get('CGRP',  float('nan')):.2f}",
            "fx_AM1R":    f"{fx[sid].get('AM1R',  float('nan')):.2f}",
            "fx_AM2R":    f"{fx[sid].get('AM2R',  float('nan')):.2f}",
            "fx_amy_avg": f"{sum(fx[sid].get(r, 0) for r in AMY_RECS)/3:.2f}",
            # Pose pass
            "pose_AMY1R": pose[sid].get("AMY1R", 0),
            "pose_AMY2R": pose[sid].get("AMY2R", 0),
            "pose_AMY3R": pose[sid].get("AMY3R", 0),
            "pose_CTR":   pose[sid].get("CTR",   0),
            "pose_CGRP":  pose[sid].get("CGRP",  0),
            "pose_AM1R":  pose[sid].get("AM1R",  0),
            "pose_AM2R":  pose[sid].get("AM2R",  0),
            # Safety
            "tango_pass": int(tango_pass),
            "tox3_pass":  int(tox3_pass),
            "aller_pass": int(aller_pass),
            # Immunogenicity (NetMHCIIpan 4.1 BA, 7-allele, vs amylin baseline)
            "imm_n_sb_lt2pct":  imm.get(sid, {}).get("n_sb_lt2pct", ""),
            "imm_n_wb_le10pct": imm.get(sid, {}).get("n_wb_le10pct", ""),
            "imm_sb_vs_amylin": imm.get(sid, {}).get("sb_vs_amylin", ""),
            # Criteria results
            "amy_pose_pass": int(amy_pose_pass),
            "amy_pass":    int(amy_pass),
            "offt_pass":   int(offt_pass),
            "safety_pass": int(safety_pass),
            "final_pass":  int(final_pass),
        }
        out_rows.append(row)

    # ── Write output ─────────────────────────────────────────────────────────
    fields = list(out_rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(out_rows)
    n_amy_pose = sum(1 for r in out_rows if r["amy_pose_pass"])
    n_offt     = sum(1 for r in out_rows if r["offt_pass"])
    n_safety   = sum(1 for r in out_rows if r["safety_pass"])
    n_final    = sum(1 for r in out_rows if r["final_pass"])

    print(f"{'='*50}")
    print(f"Total sequences   : {n}")
    print(f"AMY pose pass     : {n_amy_pose}")
    print(f"Off-target pass   : {n_offt}")
    print(f"Safety pass       : {n_safety}")
    print(f"FINAL PASS        : {n_final}  (AMY pose + off-target + safety)")
    print(f"{'='*50}")

    finals = [r for r in out_rows if r["final_pass"]]
    if finals:
        finals.sort(key=lambda x: float(x["fx_amy_avg"]))
        print(f"\nFinal candidates (sorted by FoldX AMY avg):")
        print(f"  {'ID':12s}  {'AMY1R':>7s}  {'AMY2R':>7s}  {'AMY3R':>7s}  {'avg':>7s}  {'CTR':>7s}  {'CGRP':>7s}  {'AM1R':>7s}  {'AM2R':>7s}")
        print(f"  {'-'*80}")
        # amylin reference
        print(f"  {'amylin':12s}  {amylin_fx['AMY1R']:7.2f}  {amylin_fx['AMY2R']:7.2f}  {amylin_fx['AMY3R']:7.2f}  {sum(amylin_fx[r] for r in AMY_RECS)/3:7.2f}  {amylin_fx['CTR']:7.2f}  {amylin_fx['CGRP']:7.2f}  {amylin_fx['AM1R']:7.2f}  {amylin_fx['AM2R']:7.2f}")
        print()
        for r in finals:
            print(f"  {r['id']:12s}  {r['fx_AMY1R']:>7s}  {r['fx_AMY2R']:>7s}  {r['fx_AMY3R']:>7s}  {r['fx_amy_avg']:>7s}  {r['fx_CTR']:>7s}  {r['fx_CGRP']:>7s}  {r['fx_AM1R']:>7s}  {r['fx_AM2R']:>7s}")

    print(f"\nOutput: {OUT_CSV}")


if __name__ == "__main__":
    main()
