"""
run_iedb.py

MHC Class II immunogenicity screening via IEDB REST API.

Method    : NetMHCIIpan-BA (binding affinity)
Panel     : 7-allele HLA-DRB1 panel (~90% global population)
Input     : full 37aa sequence; IEDB generates all 15-mers internally
Threshold : any 15-mer with IC50 < 1000 nM in any allele = epitope
Decision  : >=1 epitope -> FAIL;  0 epitopes -> PASS
"""

import csv
import time
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HERE    = Path(__file__).parent
INPUT   = HERE.parent / "sequences.csv"
OUTDIR  = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)
TRACKER = HERE.parent / "filter_tracker.csv"

IEDB_HOST  = "tools-cluster-interface.iedb.org"
IEDB_URL   = f"https://{IEDB_HOST}/tools_api/mhcii/"
# SSH port forward from Chromebook:
#   ssh -R 9443:tools-cluster-interface.iedb.org:443 -N jiasin@lab_193
# LOCAL_PORT routes traffic locally; --connect-to ensures correct SSL SNI
LOCAL_PORT = 9443

# 7-allele DRB1 panel — all submitted in one call (comma-separated)
ALLELES = ",".join([
    "HLA-DRB1*01:01",
    "HLA-DRB1*03:01",
    "HLA-DRB1*04:01",
    "HLA-DRB1*07:01",
    "HLA-DRB1*11:01",
    "HLA-DRB1*13:01",
    "HLA-DRB1*15:01",
])

IC50_THRESHOLD = 1000.0  # nM
PEPTIDE_LEN    = 15
SLEEP_SEC      = 1.0     # between API calls (IEDB rate limit)
MAX_RETRIES    = 3


def predict_sequence(seq_id: str, sequence: str) -> list[dict]:
    """Submit one sequence to IEDB MHC II API (all 7 alleles in one call).

    Connects via SSH port forward (LOCAL_PORT -> IEDB:443).
    urllib3 assert_hostname sets the correct SSL SNI without changing the URL.
    Prerequisite (run on Chromebook):
        ssh -R 9443:tools-cluster-interface.iedb.org:443 -N jiasin@lab_193
    """
    fasta = f">{seq_id}\n{sequence}\n"
    for attempt in range(MAX_RETRIES):
        try:
            pool = urllib3.HTTPSConnectionPool(
                "localhost",
                port=LOCAL_PORT,
                assert_hostname=IEDB_HOST,
                cert_reqs="CERT_NONE",
                timeout=urllib3.Timeout(connect=10, read=120),
            )
            r = pool.request("POST", "/tools_api/mhcii/", fields={
                "method":        "netmhciipan_ba",
                "sequence_text": fasta,
                "allele":        ALLELES,
                "length":        str(PEPTIDE_LEN),
            })
            if r.status != 200:
                raise Exception(f"HTTP {r.status}: {r.data[:200]}")
            lines = [l for l in r.data.decode().strip().split("\n") if l.strip()]
            if len(lines) < 2:
                return []
            header = lines[0].split("\t")
            return [dict(zip(header, line.split("\t"))) for line in lines[1:]]
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [WARN] {seq_id} attempt {attempt+1}: {e} — retry in {wait}s")
            time.sleep(wait)
    print(f"    [ERROR] {seq_id}: gave up after {MAX_RETRIES} attempts")
    return []



def find_ic50(row: dict) -> float | None:
    for col in ("ic50", "median_binding", "score", "binding"):
        if col in row:
            try:
                return float(row[col])
            except (ValueError, TypeError):
                continue
    return None


def find_rank(row: dict) -> float | None:
    for col in ("rank", "percentile_rank", "adjusted_rank"):
        if col in row:
            try:
                return float(row[col])
            except (ValueError, TypeError):
                continue
    return None


def score_sequence(seq_id: str, sequence: str) -> dict:
    """Collect IC50 and percentile rank for all 15-mers × 7 alleles. No threshold applied."""
    rows = predict_sequence(seq_id, sequence)
    ic50_values = []
    rank_values = []

    for row in rows:
        ic50 = find_ic50(row)
        rank = find_rank(row)
        if ic50 is not None:
            ic50_values.append(ic50)
        if rank is not None:
            rank_values.append(rank)

    best_ic50 = min(ic50_values, default=None)
    best_rank = min(rank_values, default=None)
    # n_ic50: binders at IC50 < 1000 nM (reference threshold, stored for reference)
    n_ic50 = sum(1 for v in ic50_values if v < IC50_THRESHOLD)
    # n_rank: binders at percentile rank <= 10% (IEDB top-10% recommendation)
    n_rank = sum(1 for v in rank_values if v <= 10.0)

    return {
        "id":       seq_id,
        "sequence": sequence,
        "n_ic50":   n_ic50,
        "best_ic50": f"{best_ic50:.1f}" if best_ic50 is not None else "none",
        "n_rank":   n_rank,
        "best_rank": f"{best_rank:.1f}" if best_rank is not None else "none",
        "total_15mers": len(ic50_values),
    }


RESULT_FIELDS = ["id", "sequence", "n_ic50", "best_ic50", "n_rank", "best_rank", "total_15mers"]


def update_tracker(results: list[dict]):
    """Store n_rank (count of top-10% binders) as raw score — no binary filter."""
    result_map = {r["id"]: r["n_rank"] for r in results}

    rows   = list(csv.DictReader(open(TRACKER)))
    fields = list(rows[0].keys()) if rows else ["id", "sequence"]

    if "immunogenicity" not in fields:
        fields.append("immunogenicity")
    for row in rows:
        row.setdefault("immunogenicity", "")
        if row["id"] in result_map:
            row["immunogenicity"] = result_map[row["id"]]

    with open(TRACKER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"filter_tracker.csv updated -> immunogenicity: n_rank stored for {len(result_map)} sequences")


def main():
    sequences = list(csv.DictReader(open(INPUT)))
    print(f"{len(sequences)} sequences to process")
    print(f"Panel: 7 HLA-DRB1 alleles | IC50 threshold: {IC50_THRESHOLD} nM | peptide length: {PEPTIDE_LEN}aa")
    print(f"Estimated time: ~{len(sequences) * SLEEP_SEC / 60:.0f} min\n")

    # Resume support: skip already-completed sequences
    result_path = OUTDIR / "iedb_results.csv"
    done: dict[str, dict] = {}
    if result_path.exists():
        for row in csv.DictReader(open(result_path)):
            done[row["id"]] = row
        print(f"[Resume] {len(done)} sequences already done, skipping\n")

    results = list(done.values())
    todo    = [r for r in sequences if r["id"] not in done]

    for i, row in enumerate(todo, 1):
        seq_id = row["id"]
        seq    = row["full_37aa"]
        print(f"[{i:3d}/{len(todo)}] {seq_id} ...", end=" ", flush=True)
        result = score_sequence(seq_id, seq)
        results.append(result)
        print(f"n_ic50={result['n_ic50']}  best_ic50={result['best_ic50']}  n_rank={result['n_rank']}  best_rank={result['best_rank']}")

        # Write after every sequence (crash-safe)
        with open(result_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            w.writeheader()
            w.writerows(results)

        time.sleep(SLEEP_SEC)

    update_tracker(results)

    print(f"\n{'='*60}")
    print(f"Done: {len(results)} sequences")
    print(f"Columns: n_ic50 (IC50<1000nM), n_rank (percentile<=10%), best_ic50, best_rank")
    print(f"Full results -> outputs/iedb_results.csv")


if __name__ == "__main__":
    main()
