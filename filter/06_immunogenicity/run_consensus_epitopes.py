"""
run_consensus_epitopes.py

Step 1 of CD4 immunogenicity pipeline:
  NetMHCIIpan-BA (7-allele DRB1) → filter rank < 10% binders → export for CD4episcore

Method  : netmhciipan_ba
Panel   : 7-allele DRB1 reference set (01:01/03:01/04:01/07:01/11:01/13:01/15:01)
Cutoff  : rank < 10%  (IEDB binder definition)
Input   : 16 final candidates + amylin
Outputs :
  outputs/consensus_binders.csv — all rank<10% binders across all sequences
  outputs/consensus_summary.csv — per-sequence summary
  outputs/cd4episcore_input.txt — unique peptides for CD4episcore web tool

SSH tunnel prerequisite:
  ssh -R 9443:tools-cluster-interface.iedb.org:443 -N jiasin@lab_193

Usage:
  python3 filter/04_immunogenicity/run_consensus_epitopes.py
"""

import csv
import io
import time
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HERE      = Path(__file__).parent
ROOT      = HERE.parent.parent
SEQ_CSV   = ROOT / "filter" / "inputs" / "sequences_ga.csv"
OUTDIR    = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

BINDERS_CSV  = OUTDIR / "consensus_binders.csv"
CD4_INPUT    = OUTDIR / "cd4episcore_input.txt"
SUMMARY_CSV  = OUTDIR / "consensus_summary.csv"

IEDB_HOST  = "tools-cluster-interface.iedb.org"
LOCAL_PORT = 9443

ALLELES = ",".join([
    "HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01",
    "HLA-DRB1*07:01", "HLA-DRB1*11:01", "HLA-DRB1*13:01", "HLA-DRB1*15:01",
])

RANK_CUTOFF   = 10.0   # IEDB binder definition
PEPTIDE_LEN   = 15
SLEEP_SEC     = 1.5
MAX_RETRIES   = 4

AMYLIN_ID  = "amylin"
AMYLIN_SEQ = "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNT"

FINAL_IDS = [
    "GA_022","GA_051","GA_074","GA_085","GA_086","GA_087",
    "GA_088","GA_102","GA_104","GA_108","GA_112","GA_117",
    "GA_121","GA_122","GA_132","GA_146",
]

BINDER_FIELDS = [
    "seq_id","sequence","allele","start","end","peptide","core_peptide",
    "ic50","rank",
]

SUMMARY_FIELDS = [
    "seq_id","n_binders","best_rank","n_alleles_hit","hotspot_positions",
]


def query_consensus(seq_id: str, sequence: str) -> list[dict]:
    fasta = f">{seq_id}\n{sequence}\n"
    for attempt in range(MAX_RETRIES):
        try:
            pool = urllib3.HTTPSConnectionPool(
                "localhost", port=LOCAL_PORT,
                assert_hostname=IEDB_HOST,
                cert_reqs="CERT_NONE",
                timeout=urllib3.Timeout(connect=60, read=120),
            )
            r = pool.request("POST", "/tools_api/mhcii/", fields={
                "method":        "netmhciipan_ba",
                "sequence_text": fasta,
                "allele":        ALLELES,
                "length":        str(PEPTIDE_LEN),
            })
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}")
            text = r.data.decode()
            lines = [l for l in text.splitlines() if not l.startswith("#") and l.strip()]
            if len(lines) < 2:
                return []
            rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"))
            return rows
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {seq_id}: {e} — wait {wait}s", flush=True)
            time.sleep(wait)
    print(f"    [FAILED] {seq_id}", flush=True)
    return []


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def process_sequence(seq_id: str, sequence: str) -> tuple[list[dict], dict]:
    rows = query_consensus(seq_id, sequence)
    if not rows:
        return [], {}

    # Filter binders (rank < RANK_CUTOFF)
    binders = []
    ranks = []
    alleles_hit = set()

    for row in rows:
        rank = safe_float(row.get("rank", ""))
        if rank is None:
            continue
        ranks.append(rank)
        if rank < RANK_CUTOFF:
            alleles_hit.add(row.get("allele", ""))
            binders.append({
                "seq_id":      seq_id,
                "allele":      row.get("allele", ""),
                "start":       row.get("start", ""),
                "end":         row.get("end", ""),
                "peptide":     row.get("peptide", ""),
                "core_peptide": row.get("core_peptide", ""),
                "ic50":        row.get("ic50", ""),
                "rank":        f"{rank:.2f}",
            })

    # Hotspot: positions covered by ≥2 allele-binder pairs
    from collections import Counter
    pos_counter = Counter()
    for b in binders:
        s = safe_float(b["start"])
        e = safe_float(b["end"])
        if s and e:
            for p in range(int(s), int(e) + 1):
                pos_counter[p] += 1
    hotspot_str = _compress_ranges(sorted(p for p, c in pos_counter.items() if c >= 2))

    best_rank = min(ranks) if ranks else None
    best_str  = f"{best_rank:.1f}" if best_rank is not None else "-"

    summary = {
        "seq_id":           seq_id,
        "n_binders":        len(binders),
        "best_rank":        best_str,
        "n_alleles_hit":    len(alleles_hit),
        "hotspot_positions": hotspot_str,
    }

    print(
        f"  {seq_id:12s}  binders={len(binders):3d}  best={best_str}"
        f"  alleles={len(alleles_hit)}  hotspots={hotspot_str}",
        flush=True,
    )
    return binders, summary


def _compress_ranges(positions: list[int]) -> str:
    if not positions:
        return "-"
    ranges, start, prev = [], positions[0], positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def main():
    # Build target list
    seq_lookup = {r["id"]: r["full_37aa"] for r in csv.DictReader(open(SEQ_CSV))}
    targets = [(AMYLIN_ID, AMYLIN_SEQ)] + [
        (sid, seq_lookup[sid]) for sid in FINAL_IDS if sid in seq_lookup
    ]

    print(f"Sequences: {len(targets)}  |  cutoff: rank < {RANK_CUTOFF}%")
    print(f"Method: netmhciipan_ba  |  Panel: 7-allele DRB1 (01:01/03:01/04:01/07:01/11:01/13:01/15:01)\n")

    all_binders   = []
    all_summaries = []

    for seq_id, sequence in targets:
        binders, summary = process_sequence(seq_id, sequence)
        all_binders.extend(binders)
        if summary:
            all_summaries.append(summary)
        time.sleep(SLEEP_SEC)

    # Write consensus_binders.csv
    if all_binders:
        with open(BINDERS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BINDER_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_binders)
        print(f"\nBinders saved: {BINDERS_CSV}  ({len(all_binders)} rows)")

    # Write consensus_summary.csv
    if all_summaries:
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(all_summaries)
        print(f"Summary saved: {SUMMARY_CSV}")

    # Write cd4episcore_input.txt — unique peptides only
    unique_peptides = sorted(set(b["peptide"] for b in all_binders))
    with open(CD4_INPUT, "w") as f:
        f.write("\n".join(unique_peptides) + "\n")
    print(f"CD4episcore input: {CD4_INPUT}  ({len(unique_peptides)} unique peptides)\n")

    # Summary table
    amy = next((s for s in all_summaries if s["seq_id"] == AMYLIN_ID), None)
    print(f"{'='*70}")
    print(f"  {'ID':12s}  {'binders':>7s}  {'best_rank':>9s}  {'alleles':>7s}  hotspots")
    print(f"  {'-'*65}")
    for s in all_summaries:
        marker = " ← ref" if s["seq_id"] == AMYLIN_ID else ""
        print(f"  {s['seq_id']:12s}  {s['n_binders']:7d}  {s['best_rank']:>9s}"
              f"  {s['n_alleles_hit']:7d}  {s['hotspot_positions']}{marker}")

    print(f"\nNext step: upload {CD4_INPUT} to https://tools.iedb.org/CD4episcore/")


if __name__ == "__main__":
    main()
