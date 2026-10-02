"""
integrate_cd4episcore.py

Step 2 of the CD4 immunogenicity pipeline: takes the MHC-II binder peptides
identified by run_netmhciipan_sb.py (rank < 10%, see its own docstring) and
applies CD4episcore's Combined Score to judge which candidates carry a real
immunogenicity risk.

CD4episcore status (2026-08-26): confirmed WORKING via the Legacy web form
(https://tools.iedb.org/CD4episcore/) -- contradicts the 2026-07-04 finding
in the former project notes that both the new API and the Legacy form were broken. Not
retested via the new API in this session; still manual/web-only either way
since neither this script nor run_netmhciipan_sb.py can submit to
CD4episcore programmatically.

Rule (Dhanda et al. 2018, Frontiers in Immunology, doi:10.3389/fimmu.2018.01369,
Table 2): Combined Score is a percentile-style score like MHC percentile
rank -- LOWER values mean HIGHER predicted immunogenicity risk, not the
other way round. This inverts the naive reading of "Combined Score" as a
plain 0-100 risk score, and inverts an earlier (wrong) analysis pass in
this project that flagged peptides with Combined Score >= 43 as high-risk.
Confirmed directly from the paper's methods: Imm_score = (1 - NN_output) x 100,
where NN_output itself is 0-1 with high = more immunogenic -- inverting it
puts Imm_score on the same "low = risky" scale as the HLA percentile-rank
component it's averaged with.

Threshold 43 is the paper's own recommended cutoff -- the one point in
Table 2 where sensitivity and specificity are equal (both 59%), i.e. the
balanced-accuracy choice, not an arbitrary pick:

  Threshold | Sensitivity | Specificity
  8         | 20%         | 91%
  18        | 31%         | 85%
  36        | 51%         | 65%
  43        | 59%         | 59%   <- balanced, used here
  66        | 75%         | 37%

Decision rule (per-peptide, then rolled up per-candidate):
  Combined Score < 43  -> that peptide is a predicted immunogenicity risk
  Combined Score >= 43 -> acceptable

  cd4_pass (per candidate) = 0 if ANY of its rank<10% peptides has
  Combined Score < 43, else 1. A candidate with zero rank<10% peptides
  (nothing for CD4episcore to flag) is trivially cd4_pass = 1.

Status: reference/ranking only, NOT wired into final_pass -- CD4episcore
coverage is manual and currently only exists for whichever candidates
already had MHC-II binder peptides submitted to the web tool (see
Input below), not for the full candidate roster. Promote to a hard gate
only once coverage is decided to be complete enough.

Input:
  06_immunogenicity/outputs/netmhciipan_sb_raw.csv   -- to map peptide -> candidate id
  06_immunogenicity/outputs/cd4episcore_results_v3.csv -- CD4episcore's own output
    (manually downloaded from the Legacy web form; re-run this script after
    adding more candidates' peptides and re-uploading, no code changes needed
    as long as the new CD4episcore export also lands at that same path, or
    edit CD4_CSV below to point at the new file)

Output:
  06_immunogenicity/outputs/cd4episcore_summary.csv -- id, n_peptides_scored,
    min_combined_score, n_below_43, status, cd4_pass. `status` is "tested"
    (every rank<10% peptide for that id was found in the CD4episcore export)
    or "incomplete" (some weren't submitted yet); `cd4_pass` is only ever
    0/1 when status is "tested" -- left blank for "incomplete" rather than
    defaulting to a false "pass".

Usage:
  python3 filter/06_immunogenicity/integrate_cd4episcore.py
"""

import csv
from collections import defaultdict
from pathlib import Path

HERE   = Path(__file__).parent
OUTDIR = HERE / "outputs"

RAW_CSV    = OUTDIR / "netmhciipan_sb_raw.csv"
CD4_CSV    = OUTDIR / "cd4episcore_results_v3.csv"
SUMMARY_CSV = OUTDIR / "cd4episcore_summary.csv"

WB_CUTOFF        = 10.0  # must match run_netmhciipan_sb.py's WB_CUTOFF
RISK_THRESHOLD   = 43.0  # Dhanda et al. 2018 Table 2 -- balanced sens=spec=59%


def load_peptide_to_ids() -> dict[str, set[str]]:
    peptide_to_ids: dict[str, set[str]] = defaultdict(set)
    for r in csv.DictReader(open(RAW_CSV)):
        pct = r.get("netmhciipan_ba_percentile")
        if pct in (None, "", "-"):
            continue
        if float(pct) < WB_CUTOFF:
            peptide_to_ids[r["peptide"]].add(r["seq_id"])
    return peptide_to_ids


def load_cd4_scores() -> dict[str, list[float]]:
    """peptide -> list of Combined Score values (usually one, but a peptide
    could in principle be scored more than once across separate submissions).
    """
    scores: dict[str, list[float]] = defaultdict(list)
    for r in csv.DictReader(open(CD4_CSV)):
        scores[r["Peptide"]].append(float(r["Combined Score"]))
    return scores


def main():
    peptide_to_ids = load_peptide_to_ids()
    cd4_scores = load_cd4_scores()

    all_ids = set()
    for ids in peptide_to_ids.values():
        all_ids.update(ids)

    rows = []
    for sid in sorted(all_ids):
        peptides = [p for p, ids in peptide_to_ids.items() if sid in ids]
        scored = [(p, s) for p in peptides for s in cd4_scores.get(p, [])]
        missing = [p for p in peptides if p not in cd4_scores]

        vals = [s for _, s in scored]
        n_below = sum(1 for v in vals if v < RISK_THRESHOLD)

        if missing:
            # Some binder peptides were never submitted to CD4episcore --
            # status is genuinely unknown, NOT a pass. Do not let an empty
            # `vals` (zero real scores) silently compute n_below=0 -> pass.
            print(f"  [WARN] {sid}: {len(missing)}/{len(peptides)} peptide(s) "
                  f"not found in {CD4_CSV.name} -- not yet submitted to CD4episcore")
            status = "incomplete"
            cd4_pass = ""
        else:
            status = "tested"
            cd4_pass = 1 if n_below == 0 else 0

        rows.append({
            "id": sid,
            "n_peptides_scored": len(vals),
            "min_combined_score": f"{min(vals):.2f}" if vals else "",
            "n_below_43": n_below,
            "status": status,
            "cd4_pass": cd4_pass,
        })

    fields = ["id", "n_peptides_scored", "min_combined_score", "n_below_43", "status", "cd4_pass"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'ID':10s}  {'n_scored':>8s}  {'min_score':>9s}  {'n<43':>5s}  {'status':>10s}  {'cd4_pass':>8s}")
    print("-" * 62)
    for r in rows:
        print(f"{r['id']:10s}  {r['n_peptides_scored']:>8d}  "
              f"{r['min_combined_score']:>9s}  {r['n_below_43']:>5d}  "
              f"{r['status']:>10s}  {str(r['cd4_pass']):>8s}")
    print(f"\nSummary saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
