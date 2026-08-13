"""
integrate_cd4episcore_v2.py

Summarizes CD4episcore results for the 7 final candidates + amylin.

Unlike the original two-step pipeline (run_consensus_epitopes.py pre-filters
MHC-II binders via NetMHCIIpan, then integrate_cd4episcore.py joins those
binders against CD4episcore scores), this version skips the pre-filter step
entirely: the full 37aa sequences were submitted directly to the Legacy
CD4episcore website ("IEDB recommended (combined)" method), which does its
own MHC-II binding + immunogenicity scoring internally per 15-mer window.

Input:
  outputs/cd4episcore_results_v2.csv — CD4episcore output for 8 sequences
    (amylin + GA_051/074/085/086/087/088/104), 6 scored 15-mer windows each

High-risk cutoff: Combined Score >= 66 (Dhanda et al. 2018, Table 2 —
75% sensitivity / 37% specificity). Note: 66, not 43 — the paper's Table 2
gives 43 -> 59% sensitivity, 66 -> 75% sensitivity; an earlier version of
this pipeline mislabeled 43 as the 75%-sensitivity cutoff.

Positions 1-13 (KCNTATCATQRLA) are the fixed, unmutated construct prefix —
identical to native amylin at those positions, so the immune system is
already tolerized to that exact stretch (central tolerance). A window can
still score "high risk" while overlapping this region (its score is driven
by the mutated residues at position >=14 within the same 15-mer), so
n_high_risk / best_combined still count those windows. But the hotspot
map, which is meant to flag *novel* risk introduced by the GA mutations,
excludes positions 1-13 — only position >=14 is reported as a hotspot.

Output:
  outputs/immunogenicity_report_v2.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

HERE   = Path(__file__).parent
OUTDIR = HERE / "outputs"

CD4_CSV    = OUTDIR / "cd4episcore_results_v2.csv"
REPORT_CSV = OUTDIR / "immunogenicity_report_v2.csv"

HIGH_RISK_CUTOFF   = 66.0
FIXED_PREFIX_LEN   = 13  # KCNTATCATQRLA — identical to native amylin, pre-tolerized

AMYLIN_ID = "amylin"
FINAL_IDS = ["GA_051", "GA_074", "GA_085", "GA_086", "GA_087", "GA_088", "GA_104"]


def _compress_ranges(positions: list[int]) -> str:
    if not positions:
        return "-"
    positions = sorted(set(positions))
    ranges, start, prev = [], positions[0], positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def load_rows() -> dict[str, list[dict]]:
    """seq_id (Protein Description) -> list of window rows"""
    d = defaultdict(list)
    for r in csv.DictReader(open(CD4_CSV)):
        d[r["Protein Description"]].append(r)
    return d


def analyze_sequence(seq_id: str, rows: list[dict]) -> dict:
    windows = []
    for r in rows:
        combined = float(r["Combined Score"])
        imm      = float(r["Immunogenicity Score"])
        windows.append({
            "peptide":  r["Peptide"],
            "start":    int(r["Start"]),
            "end":      int(r["End"]),
            "combined": combined,
            "imm":      imm,
        })

    n_windows  = len(windows)
    high_risk  = [w for w in windows if w["combined"] >= HIGH_RISK_CUTOFF]
    n_high     = len(high_risk)
    best_combined = max((w["combined"] for w in windows), default=0.0)
    best_imm      = max((w["imm"] for w in windows), default=0.0)

    hotspot_positions = []
    for w in high_risk:
        hotspot_positions.extend(range(w["start"], w["end"] + 1))
    hotspot_str = _compress_ranges(hotspot_positions)

    top3 = sorted(windows, key=lambda w: -w["combined"])[:3]
    top_str = " | ".join(
        f"{w['peptide']}(combined={w['combined']:.1f},imm={w['imm']:.0f}%,pos{w['start']}-{w['end']})"
        for w in top3
    )

    return {
        "seq_id":          seq_id,
        "n_windows_scored": n_windows,
        "n_high_risk":      n_high,
        "best_combined":    f"{best_combined:.2f}",
        "best_imm_score":   f"{best_imm:.1f}",
        "hotspot_pos":      hotspot_str,
        "top_epitopes":     top_str,
    }


def main():
    rows_by_seq = load_rows()

    print(f"Sequences found in {CD4_CSV.name}: {len(rows_by_seq)}")
    print(f"High-risk cutoff: Combined Score >= {HIGH_RISK_CUTOFF}\n")

    order = [AMYLIN_ID] + FINAL_IDS
    missing = [sid for sid in order if sid not in rows_by_seq]
    if missing:
        print(f"[WARN] missing from {CD4_CSV.name}: {missing}")

    report_rows = [analyze_sequence(sid, rows_by_seq[sid]) for sid in order if sid in rows_by_seq]

    fields = ["seq_id", "n_windows_scored", "n_high_risk", "best_combined",
              "best_imm_score", "hotspot_pos", "top_epitopes"]
    with open(REPORT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report_rows)
    print(f"Report saved: {REPORT_CSV}\n")

    print(f"{'='*90}")
    print(f"  {'ID':10s}  {'windows':>7s}  {'high(>=66)':>10s}  {'best_combined':>13s}  {'best_imm':>8s}  hotspots")
    print(f"  {'-'*85}")
    for r in report_rows:
        marker = " <- ref" if r["seq_id"] == AMYLIN_ID else ""
        print(f"  {r['seq_id']:10s}  {r['n_windows_scored']:7d}  {r['n_high_risk']:10d}"
              f"  {r['best_combined']:>13s}  {r['best_imm_score']:>8s}  {r['hotspot_pos']}{marker}")


if __name__ == "__main__":
    main()
