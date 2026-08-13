"""
run_ga_filters.py

Run aggregation / toxicity / allergenicity filters on 154 GA qualifying sequences.
Input:  sequences_ga.csv  (built from GA run_003 library)
Output: filter_tracker_ga.csv

Usage:
    conda run -n raghava_tools python3 run_ga_filters.py
    (raghava_tools has toxinpred2; algpred2 is called via conda run internally)
"""

import csv
import subprocess
import tempfile
from pathlib import Path

HERE      = Path(__file__).resolve().parent
INPUT     = HERE / "sequences_ga.csv"
TRACKER   = HERE / "filter_tracker_ga.csv"

TANGO     = HERE / "01_aggregation" / "tools" / "tango_x86_64_release"
TANGO_OUT = HERE / "01_aggregation" / "outputs"
TANGO_OUT.mkdir(exist_ok=True)

TOX_OUT   = HERE / "06_toxicity" / "outputs"
TOX_OUT.mkdir(exist_ok=True)

ALG_OUT   = HERE / "05_allergenicity" / "outputs"
ALG_OUT.mkdir(exist_ok=True)

ALGPRED2_SCRIPT = HERE / "05_allergenicity" / "tools" / "algpred2" / "algpred2.py"
ALGPRED2_DIR    = ALGPRED2_SCRIPT.parent

# ── thresholds ──────────────────────────────────────────────────────────────
TANGO_APR_THRESHOLD  = 5.0
TANGO_APR_MIN_LEN    = 5
TOX_THRESHOLD        = 0.5   # ToxinPred2 default
ALG_THRESHOLD        = 0.384 # native amylin baseline


# ── TANGO ───────────────────────────────────────────────────────────────────

def run_tango_single(seq_id: str, seq: str) -> dict:
    cmd = [
        str(TANGO), seq_id,
        "ct=Y", "nt=N", "ph=7.4", "te=310", "io=0.1",
        f"seq={seq}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           cwd=str(TANGO_OUT))
        summary = {}
        parts = r.stdout.strip().split()
        for i in range(0, len(parts) - 1, 2):
            summary[parts[i]] = float(parts[i + 1])
        return summary
    except Exception as e:
        print(f"  [TANGO ERROR] {seq_id}: {e}")
        return {}


def parse_tango_per_res(seq_id: str) -> list:
    txt = TANGO_OUT / f"{seq_id}.txt"
    if not txt.exists():
        return []
    rows = []
    with open(txt) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    rows.append(float(parts[5]))
                except ValueError:
                    pass
    return rows


def run_tango_batch(sequences: list) -> list:
    results = []
    for i, row in enumerate(sequences, 1):
        seq_id = row["id"]
        seq    = row["full_37aa"]
        print(f"  [{i:3d}/{len(sequences)}] TANGO {seq_id}", end="  ", flush=True)
        summary = run_tango_single(seq_id, seq)
        per_res = parse_tango_per_res(seq_id)

        # detect APR
        has_apr = False
        run = 0
        for score in per_res:
            run = run + 1 if score > TANGO_APR_THRESHOLD else 0
            if run >= TANGO_APR_MIN_LEN:
                has_apr = True
                break

        status = "FAIL" if has_apr else "PASS"
        print(status)
        results.append({
            "id":            seq_id,
            "sequence":      seq,
            "tango_agg":     summary.get("Aggregation", ""),
            "tango_has_apr": int(has_apr),
            "tango_pass":    status,
        })
    return results


# ── ToxinPred2 ──────────────────────────────────────────────────────────────

def run_toxinpred2_batch(sequences: list) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as fa:
        for row in sequences:
            fa.write(f">{row['id']}\n{row['full_37aa']}\n")
        fa_path = fa.name

    raw_out = str(TOX_OUT / "toxinpred2_raw_ga.csv")
    cmd = ["toxinpred2", "-i", fa_path, "-o", raw_out,
           "-t", str(TOX_THRESHOLD), "-m", "1", "-d", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ToxinPred2 ERROR]\n{r.stderr}")
        return []
    return list(csv.DictReader(open(raw_out)))


# ── AlgPred2 ────────────────────────────────────────────────────────────────

def run_algpred2_batch(sequences: list) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as fa:
        for row in sequences:
            fa.write(f">{row['id']}\n{row['full_37aa']}\n")
        fa_path = fa.name

    raw_out = str(ALG_OUT / "algpred2_raw_ga.csv")
    cmd = [
        "conda", "run", "-n", "algpred2",
        "python3", str(ALGPRED2_SCRIPT),
        "-i", fa_path, "-o", raw_out,
        "-t", str(ALG_THRESHOLD), "-m", "1", "-d", "2",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(ALGPRED2_DIR))
    if r.returncode != 0:
        print(f"[AlgPred2 ERROR]\n{r.stderr}")
        return []
    return list(csv.DictReader(open(raw_out)))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    sequences = list(csv.DictReader(open(INPUT)))
    print(f"GA sequences: {len(sequences)}\n")

    # ── 01 TANGO aggregation ─────────────────────────────────────────────
    print("=== 01 Aggregation (TANGO) ===")
    tango_rows = run_tango_batch(sequences)
    tango_map  = {r["id"]: r for r in tango_rows}
    n_pass = sum(1 for r in tango_rows if r["tango_pass"] == "PASS")
    print(f"  PASS: {n_pass}/{len(tango_rows)}\n")

    # ── 06 Toxicity (ToxinPred2) ─────────────────────────────────────────
    print("=== 06 Toxicity (ToxinPred2) ===")
    tox_raw  = run_toxinpred2_batch(sequences)
    tox_map  = {}
    for r in tox_raw:
        sid    = r.get("ID") or r.get("id") or r.get("SeqID", "")
        score  = float(r.get("ML_Score") or r.get("Score", 0))
        pred   = r.get("Prediction", "")
        status = "PASS" if "Non-Toxin" in pred else "FAIL"
        tox_map[sid] = {"score": score, "pred": pred, "pass": status}
        print(f"  {sid}  score={score:.3f}  {pred}  → {status}")
    n_pass = sum(1 for v in tox_map.values() if v["pass"] == "PASS")
    print(f"  PASS: {n_pass}/{len(tox_map)}\n")

    # ── 05 Allergenicity (AlgPred2) ──────────────────────────────────────
    print("=== 05 Allergenicity (AlgPred2) ===")
    alg_raw = run_algpred2_batch(sequences)
    alg_map = {}
    for r in alg_raw:
        sid    = r.get("ID") or r.get("id") or r.get("SeqID", "")
        score  = float(r.get("ML_Score") or r.get("Score", 0))
        pred   = r.get("Prediction", "")
        status = "PASS" if "Non-Allergen" in pred else "FAIL"
        alg_map[sid] = {"score": score, "pred": pred, "pass": status}
        print(f"  {sid}  score={score:.3f}  {pred}  → {status}")
    n_pass = sum(1 for v in alg_map.values() if v["pass"] == "PASS")
    print(f"  PASS: {n_pass}/{len(alg_map)}\n")

    # ── write tracker ────────────────────────────────────────────────────
    fields = [
        "id", "seq_24aa", "full_37aa",
        "AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R",
        "aggregation", "tango_agg",
        "immunogenicity",
        "toxicity", "tox_score",
        "allergenicity", "alg_score",
        "all_pass",
    ]
    tracker_rows = []
    for row in sequences:
        sid = row["id"]
        t   = tango_map.get(sid, {})
        tx  = tox_map.get(sid, {})
        al  = alg_map.get(sid, {})

        agg_pass = 1 if t.get("tango_pass") == "PASS" else (0 if t else "")
        tox_pass = 1 if tx.get("pass") == "PASS" else (0 if tx else "")
        alg_pass = 1 if al.get("pass") == "PASS" else (0 if al else "")
        all_pass = 1 if (agg_pass == 1 and tox_pass == 1 and alg_pass == 1) else 0

        tracker_rows.append({
            "id":             sid,
            "seq_24aa":       row["seq_24aa"],
            "full_37aa":      row["full_37aa"],
            "AMY1R":          row["AMY1R"],
            "AMY2R":          row["AMY2R"],
            "AMY3R":          row["AMY3R"],
            "CTR":            row["CTR"],
            "CGRP":           row["CGRP"],
            "AM1R":           row["AM1R"],
            "AM2R":           row["AM2R"],
            "aggregation":    agg_pass,
            "tango_agg":      t.get("tango_agg", ""),
            "immunogenicity": "",   # filled later via IEDB
            "toxicity":       tox_pass,
            "tox_score":      f"{tx['score']:.3f}" if tx else "",
            "allergenicity":  alg_pass,
            "alg_score":      f"{al['score']:.3f}" if al else "",
            "all_pass":       all_pass,
        })

    with open(TRACKER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(tracker_rows)

    total_pass = sum(1 for r in tracker_rows if r["all_pass"] == 1)
    print(f"=== Summary ===")
    print(f"  Aggregation PASS : {sum(1 for r in tracker_rows if r['aggregation'] == 1)}/{len(tracker_rows)}")
    print(f"  Toxicity PASS    : {sum(1 for r in tracker_rows if r['toxicity'] == 1)}/{len(tracker_rows)}")
    print(f"  Allergenicity PASS: {sum(1 for r in tracker_rows if r['allergenicity'] == 1)}/{len(tracker_rows)}")
    print(f"  ALL PASS (3/3)   : {total_pass}/{len(tracker_rows)}")
    print(f"\nTracker: {TRACKER}")


if __name__ == "__main__":
    main()
