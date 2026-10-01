"""
run_netmhciipan_sb.py

Immunogenicity screen using NetMHCIIpan 4.1 BA (IEDB's currently-recommended
MHC-II binding predictor, per IEDB's 2023.09 tool recommendation) via the
IEDB Next-Generation Tools API.

Replaces the old two-tool pipeline (run_iedb.py / run_consensus_epitopes.py
+ manual CD4episcore web upload), which depended on:
  - tools-cluster-interface.iedb.org (legacy tools_api) -- dead, unreachable
  - CD4episcore immunogenicity predictor (new API) -- server-side bug,
    reproducibly fails with "Completion flag file missing" regardless of
    request format
  - Legacy CD4episcore website -- form submission silently no-ops

Method    : netmhciipan_ba via api-nextgen-tools.iedb.org (no SSH tunnel
            needed -- this host is directly reachable).

            2026-08-26: briefly switched to IEDB's **Consensus** method
            (median percentile rank across NN-align/SMM-align/Tepitope/
            Comblib) on the strength of Paul et al. 2013 (J Immunol Res,
            doi:10.1155/2013/467852), which found Consensus beats each of
            those individual (now-superseded) methods for protein-drug
            immunogenicity screening (AROC 0.89 vs 0.76-0.85). Reverted the
            same day after reading Reynisson et al. 2020 (Nucleic Acids
            Res 48:W449, doi:10.1093/nar/gkaa379) -- the actual NetMHCIIpan
            4.0/4.1 paper: it's a single neural-network method, but trained
            on 4.1M data points across 116 MHC-II molecules (binding
            affinity + mass-spec eluted ligand data via NNAlign_MA), a full
            generation newer than the four 2013-era constituent methods
            Consensus averages together, and it's IEDB's current default
            recommendation. Net effect: **use netmhciipan_ba, not
            Consensus** -- "newer, better-trained single method" beat "old
            methods averaged together" here, so don't re-derive Consensus
            without a specific reason.
Panel     : 7-allele DRB reference set
            HLA-DRB1*03:01, *07:01, *15:01, HLA-DRB3*01:01, *02:02,
            HLA-DRB4*01:01, HLA-DRB5*01:01
Window    : full 37aa sequence, 15-mer sliding window (standard for MHC-II,
            whose open-ended groove is conventionally scanned wider than the
            9aa binding core; NetMHCIIpan identifies the 9aa core internally)

Metric    : n_WB = count of (peptide x allele) pairs with percentile rank
            <= 10% (IEDB's general "binder" recommendation, Paul et al.
            2013 above). Used comparatively -- rank candidates against each
            other by n_WB, not against a fixed pass/fail cutoff (the paper
            doesn't give one; see its methodology).

            2026-08-26: dropped the earlier n_SB metric (percentile rank <
            2%, "Strong Binder"). It was originally added because n_WB
            alone made several candidates indistinguishable in an earlier,
            smaller batch. But n_SB itself turned out to be uninformative:
            with only ~161 (peptide x allele) combinations tested per
            candidate, a rank<2% cutoff has a ~2% background hit rate by
            construction, i.e. ~3 hits expected by chance alone -- so the
            0-1 values every candidate actually showed were consistent
            with pure noise, not a real signal, and gave no discriminating
            power. n_WB doesn't have this problem at the same scale (it's
            already showing real spread: 0 to 23 across current
            candidates) and matches what the cited paper actually
            recommends using.

Output:
  outputs/netmhciipan_sb_raw.csv     -- full per-window x allele results
  outputs/netmhciipan_sb_summary.csv -- n_WB per sequence

Usage:
  python3 filter/06_immunogenicity/run_netmhciipan_sb.py
  python3 filter/06_immunogenicity/run_netmhciipan_sb.py --seq GA_051,GA_074

(File/output names keep the "_sb" suffix for continuity even though the SB
metric itself is gone -- renaming would just churn every downstream
reference for no functional benefit.)
"""

import argparse
import csv
import json
import time
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HERE    = Path(__file__).parent
FILTER  = HERE.parent
SEQ_CSV = FILTER / "final_results.csv"
OUTDIR  = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_CSV     = OUTDIR / "netmhciipan_sb_raw.csv"
SUMMARY_CSV = OUTDIR / "netmhciipan_sb_summary.csv"

API_HOST = "api-nextgen-tools.iedb.org"
PIPELINE_URL = f"https://{API_HOST}/api/v1/pipeline"
RESULTS_URL  = f"https://{API_HOST}/api/v1/results"

ALLELES = ",".join([
    "HLA-DRB1*03:01", "HLA-DRB1*07:01", "HLA-DRB1*15:01",
    "HLA-DRB3*01:01", "HLA-DRB3*02:02", "HLA-DRB4*01:01", "HLA-DRB5*01:01",
])
METHOD = "netmhciipan_ba"     # NetMHCIIpan 4.1 BA -- IEDB-recommended (2023.09)
PEPTIDE_LEN = 15
WB_CUTOFF   = 10.0            # Weak Binder: percentile rank <= 10% (IEDB's general binder definition)

POLL_INTERVAL_S = 8
POLL_MAX_TRIES  = 30


def load_targets(seq_filter: set[str] | None) -> list[tuple[str, str]]:
    """Default (no --seq): whoever currently has final_pass=1. With --seq,
    query exactly those IDs regardless of final_pass -- e.g. to check
    immunogenicity for candidates that only made it through an earlier
    stage (03/04/05) but haven't been through 01/02 yet.
    """
    rows = list(csv.DictReader(open(SEQ_CSV)))
    if seq_filter:
        pool = rows
    else:
        pool = [r for r in rows if r.get("final_pass") == "1"]
    targets = []
    for r in pool:
        if seq_filter and r["id"] not in seq_filter:
            continue
        targets.append((r["id"], r["full_37aa"]))
    return targets


def build_fasta(targets: list[tuple[str, str]]) -> str:
    return "\n".join(f">{sid}\n{seq}" for sid, seq in targets)


def submit(fasta_text: str) -> str:
    http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    payload = {
        "pipeline_title": "netmhciipan_ba 7-allele SB screen",
        "run_stage_range": [1, 1],
        "stages": [{
            "stage_number": 1,
            "tool_group": "mhcii",
            "input_sequence_text": fasta_text,
            "input_parameters": {
                "alleles": ALLELES,
                "peptide_length_range": [PEPTIDE_LEN, PEPTIDE_LEN],
                "predictors": [{"type": "binding", "method": METHOD}],
            },
        }],
    }
    r = http.request(
        "POST", PIPELINE_URL,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    if r.status != 200:
        raise RuntimeError(f"submit failed: HTTP {r.status}: {r.data[:300]}")
    data = json.loads(r.data)
    if "result_id" not in data:
        raise RuntimeError(f"no result_id in response: {data}")
    return data["result_id"]


def poll(result_id: str) -> dict:
    http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    for i in range(POLL_MAX_TRIES):
        time.sleep(POLL_INTERVAL_S)
        r = http.request("GET", f"{RESULTS_URL}/{result_id}")
        d = json.loads(r.data)
        status = d.get("status") or d.get("data", {}).get("status")
        print(f"  [poll {i+1}/{POLL_MAX_TRIES}] status={status}", flush=True)
        if status == "done":
            return d
        if status == "error":
            raise RuntimeError(f"job errored: {d['data']['errors']}")
    raise TimeoutError(f"job {result_id} did not finish in time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", help="Comma-separated seq IDs (default: all final_pass candidates)")
    args = parser.parse_args()
    seq_filter = {s.strip() for s in args.seq.split(",")} if args.seq else None

    targets = load_targets(seq_filter)
    print(f"Sequences: {len(targets)} ({', '.join(t[0] for t in targets)})")
    print(f"Method: {METHOD} | Alleles: 7-allele DRB panel | Window: {PEPTIDE_LEN}-mer")
    print(f"WB cutoff: percentile rank <= {WB_CUTOFF}%\n")

    fasta_text = build_fasta(targets)
    result_id = submit(fasta_text)
    print(f"Submitted: result_id={result_id}")
    result = poll(result_id)

    res = result["data"]["results"][0]
    cols = [c["name"] for c in res["table_columns"]]
    rows = res["table_data"]
    idx = {c: i for i, c in enumerate(cols)}
    seq_names = {i + 1: sid for i, (sid, _) in enumerate(targets)}

    with open(RAW_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id"] + cols)
        for r in rows:
            sid = seq_names[r[idx["sequence_number"]]]
            w.writerow([sid] + r)
    print(f"\nRaw results saved: {RAW_CSV} ({len(rows)} rows)")

    n_wb = {sid: 0 for sid, _ in targets}
    for r in rows:
        sid = seq_names[r[idx["sequence_number"]]]
        pct = r[idx["netmhciipan_ba_percentile"]]
        if pct is None or pct == "-":
            continue
        pct = float(pct)
        if pct <= WB_CUTOFF:
            n_wb[sid] += 1

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "n_wb_lt10pct"])
        for sid, _ in targets:
            w.writerow([sid, n_wb[sid]])
    print(f"Summary saved: {SUMMARY_CSV}\n")

    ranked = sorted(targets, key=lambda t: n_wb[t[0]])
    print(f"{'ID':10s}  {'n_WB(<10%)':>11s}   (ranked ascending -- comparative, not pass/fail)")
    print(f"{'-'*45}")
    for sid, _ in ranked:
        print(f"{sid:10s}  {n_wb[sid]:11d}")


if __name__ == "__main__":
    main()
