"""
integrate_cd4episcore.py

Joins CD4episcore results with consensus MHC II binders to produce:
  1. Per-sequence immunogenicity summary
  2. Hotspot map (which positions are high-risk)
  3. Final immunogenicity report comparing all 16 candidates to amylin

Inputs:
  outputs/consensus_binders.csv   — all rank<10% binders (peptide → seq_id)
  outputs/cd4episcore_results.csv — CD4episcore immunogenicity scores

Output:
  outputs/immunogenicity_report.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

HERE   = Path(__file__).parent
OUTDIR = HERE / "outputs"

BINDERS_CSV    = OUTDIR / "consensus_binders.csv"
CD4_CSV        = OUTDIR / "cd4episcore_results.csv"
REPORT_CSV     = OUTDIR / "immunogenicity_report.csv"

FINAL_IDS = [
    "GA_022","GA_051","GA_074","GA_085","GA_086","GA_087",
    "GA_088","GA_102","GA_104","GA_108","GA_112","GA_117",
    "GA_121","GA_122","GA_132","GA_146",
]
AMYLIN_ID = "amylin"


def load_cd4scores() -> dict[str, dict]:
    """peptide → {immunogenicity_score, combined_score, median_rank}"""
    d = {}
    for r in csv.DictReader(open(CD4_CSV)):
        pep = r["Peptide"].strip()
        d[pep] = {
            "imm_score":    float(r["Immunogenicity Score"]),
            "combined":     float(r["Combined Score"]),
            "median_rank":  float(r["Median Percentile Rank (7-allele)"]),
        }
    return d


def load_binders() -> dict[str, list[dict]]:
    """seq_id → list of binder rows"""
    d = defaultdict(list)
    for r in csv.DictReader(open(BINDERS_CSV)):
        d[r["seq_id"]].append(r)
    return d


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


def analyze_sequence(seq_id: str, binders: list[dict], cd4: dict[str, dict]) -> dict:
    cd4_hits = []
    hotspot_positions = []

    for b in binders:
        pep = b["peptide"]
        if pep in cd4:
            combined = cd4[pep]["combined"]
            cd4_hits.append({
                "peptide":   pep,
                "start":     int(b["start"]),
                "end":       int(b["end"]),
                "mhc_rank":  float(b["rank"]),
                "imm_score": cd4[pep]["imm_score"],
                "combined":  combined,
            })
            # High-risk: Combined Score >= 43 (Dhanda et al. 2018, Table 2; sensitivity=75%)
            if combined >= 43:
                for p in range(int(b["start"]), int(b["end"]) + 1):
                    hotspot_positions.append(p)

    n_cd4          = len(cd4_hits)
    n_high         = sum(1 for h in cd4_hits if h["combined"] >= 43)
    best_imm       = max((h["imm_score"] for h in cd4_hits), default=0.0)
    best_combined  = max((h["combined"]  for h in cd4_hits), default=0.0)
    hotspot_str    = _compress_ranges(hotspot_positions)

    top_epitopes = sorted(cd4_hits, key=lambda x: -x["combined"])[:3]
    top_str = " | ".join(
        f"{h['peptide']}(combined={h['combined']:.1f},imm={h['imm_score']:.0f}%,pos{h['start']}-{h['end']})"
        for h in top_epitopes
    )

    return {
        "seq_id":        seq_id,
        "n_mhc_binders": len(binders),
        "n_cd4_hits":    n_cd4,
        "n_high_risk":   n_high,        # imm_score >= 80
        "best_imm_score": f"{best_imm:.1f}",
        "best_combined": f"{best_combined:.2f}",
        "hotspot_pos":   hotspot_str,
        "top_epitopes":  top_str,
    }


def main():
    cd4    = load_cd4scores()
    binders = load_binders()

    print(f"CD4episcore peptides: {len(cd4)}")
    print(f"Sequences with binders: {len(binders)}\n")

    order = [AMYLIN_ID] + FINAL_IDS
    rows  = []

    for seq_id in order:
        b_list = binders.get(seq_id, [])
        row = analyze_sequence(seq_id, b_list, cd4)
        rows.append(row)

    # Write report
    fields = ["seq_id","n_mhc_binders","n_cd4_hits","n_high_risk",
              "best_imm_score","best_combined","hotspot_pos","top_epitopes"]
    with open(REPORT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Report saved: {REPORT_CSV}\n")

    # Print summary table
    print(f"{'='*90}")
    print(f"  {'ID':12s}  {'MHC':>5s}  {'CD4':>5s}  {'high(≥43)':>10s}  {'best_imm':>8s}  hotspots")
    print(f"  {'-'*85}")
    for r in rows:
        marker = " ← ref" if r["seq_id"] == AMYLIN_ID else ""
        print(f"  {r['seq_id']:12s}  {r['n_mhc_binders']:5d}  {r['n_cd4_hits']:5d}"
              f"  {r['n_high_risk']:10d}  {r['best_imm_score']:>8s}  {r['hotspot_pos']}{marker}")

    print(f"\n  Top immunogenic epitopes per sequence:")
    print(f"  {'-'*85}")
    for r in rows:
        if r["top_epitopes"]:
            print(f"  {r['seq_id']:12s}  {r['top_epitopes']}")


if __name__ == "__main__":
    main()
