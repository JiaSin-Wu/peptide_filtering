"""
run_netmhciipan_sb.py

Immunogenicity screen for final candidates using NetMHCIIpan 4.1 BA
(IEDB-recommended binding predictor, 2023.09) via the IEDB Next-Generation
Tools API.

Replaces the old two-tool pipeline (run_iedb.py / run_consensus_epitopes.py
+ manual CD4episcore web upload), which depended on:
  - tools-cluster-interface.iedb.org (legacy tools_api) -- dead, unreachable
  - CD4episcore immunogenicity predictor (new API) -- server-side bug,
    reproducibly fails with "Completion flag file missing" regardless of
    request format
  - Legacy CD4episcore website -- form submission silently no-ops

Method    : netmhciipan_ba (NetMHCIIpan 4.1 BA) via api-nextgen-tools.iedb.org
            No SSH tunnel needed -- this host is directly reachable.
Panel     : 7-allele DRB reference set
            HLA-DRB1*03:01, *07:01, *15:01, HLA-DRB3*01:01, *02:02,
            HLA-DRB4*01:01, HLA-DRB5*01:01
Window    : full 37aa sequence, 15-mer sliding window (standard for MHC-II,
            whose open-ended groove is conventionally scanned wider than the
            9aa binding core; NetMHCIIpan identifies the 9aa core internally)

Metric    : n_SB = count of (peptide x allele) pairs with percentile rank < 2%
            (Strong Binder -- stricter than the 10% Weak-Binder cutoff IEDB
            uses as its general epitope-flagging recommendation). Chosen
            because at the 10% cutoff, several final candidates turned out
            to be indistinguishable from each other (identical binder sets,
            since their sequence differences fall outside every 15-mer
            binder window) and gave only a fuzzy "all candidates lower than
            amylin" signal. At <2%, native amylin has 4 strong binders and
            all 7 GA candidates have 0 -- a clean, decisive result.

Output:
  outputs/netmhciipan_sb_raw.csv     -- full per-window x allele results
  outputs/netmhciipan_sb_summary.csv -- n_SB per sequence, vs amylin baseline

Usage:
  python3 filter/06_immunogenicity/run_netmhciipan_sb.py
  python3 filter/06_immunogenicity/run_netmhciipan_sb.py --seq GA_051,GA_074
"""

import argparse
import csv
import json
import time
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HERE    = Path(__file__).parent
ROOT    = HERE.parent.parent
FILTER  = ROOT / "filter"
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
SB_CUTOFF   = 2.0             # Strong Binder: percentile rank < 2%

AMYLIN_ID  = "amylin"
AMYLIN_SEQ = "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY"

POLL_INTERVAL_S = 8
POLL_MAX_TRIES  = 30


def load_targets(seq_filter: set[str] | None) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(open(SEQ_CSV)))
    finals = [r for r in rows if r.get("final_pass") == "1"]
    targets = [(AMYLIN_ID, AMYLIN_SEQ)]
    for r in finals:
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
    print(f"SB cutoff: percentile rank < {SB_CUTOFF}%\n")

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

    n_sb = {sid: 0 for sid, _ in targets}
    n_wb = {sid: 0 for sid, _ in targets}
    for r in rows:
        sid = seq_names[r[idx["sequence_number"]]]
        pct = r[idx["netmhciipan_ba_percentile"]]
        if pct is None:
            continue
        pct = float(pct)
        if pct < SB_CUTOFF:
            n_sb[sid] += 1
        if pct <= 10.0:
            n_wb[sid] += 1

    amylin_sb = n_sb[AMYLIN_ID]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "n_sb_lt2pct", "n_wb_lt10pct", "sb_vs_amylin"])
        for sid, _ in targets:
            w.writerow([sid, n_sb[sid], n_wb[sid], n_sb[sid] - amylin_sb])
    print(f"Summary saved: {SUMMARY_CSV}\n")

    print(f"{'ID':10s}  {'n_SB(<2%)':>10s}  {'n_WB(<10%)':>11s}")
    print(f"{'-'*35}")
    for sid, _ in targets:
        marker = " <- ref" if sid == AMYLIN_ID else ""
        print(f"{sid:10s}  {n_sb[sid]:10d}  {n_wb[sid]:11d}{marker}")


if __name__ == "__main__":
    main()
