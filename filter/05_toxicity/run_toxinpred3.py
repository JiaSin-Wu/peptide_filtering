"""
run_toxinpred3.py

Toxicity screening via ToxinPred3 (Raghava group).

Method    : Extra Trees on AAC + DPC (Model 1, sklearn 1.0.2)
Threshold : ML_Score < 0.38 → Non-Toxin → PASS  (ToxinPred3 default)
            Native amylin baseline = 0.35 (passes at default threshold)
Input     : full 37aa sequence
Run with  : conda run -n raghava_tools python3 run_toxinpred3.py
            conda run -n raghava_tools python3 run_toxinpred3.py --seq GA_001,GA_002
            conda run -n raghava_tools python3 run_toxinpred3.py --seq-file alive.txt

Subset filtering (for run_pipeline.py's funnel): --seq / --seq-file narrow
the batch down; neither given runs every candidate in sequences_ga.csv.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import filter_lib as fl

INPUT     = HERE.parent / "sequences_ga.csv"
OUTDIR    = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)

TOOL_DIR  = HERE / "tools" / "toxinpred3"
THRESHOLD = 0.38
RESULT_FIELDS = ["id", "sequence", "ml_score", "ppv", "prediction", "toxin_pass"]


def run_toxinpred3_batch(sequences: list) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as fa:
        for row in sequences:
            fa.write(f">{row['id']}\n{row['full_37aa']}\n")
        fa_path = fa.name

    out_path = str(OUTDIR / "toxinpred3_raw_ga.csv")
    cmd = [
        sys.executable, str(TOOL_DIR / "toxinpred3.py"),  # absolute path fixes nf_path for Model 2
        "-i", fa_path,
        "-o", out_path,
        "-t", str(THRESHOLD),
        "-m", "2",
        "-d", "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TOOL_DIR))
    if result.returncode != 0:
        print(f"[ERROR] toxinpred3 failed:\n{result.stderr}")
        return []
    return list(csv.DictReader(open(out_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="Comma-separated seq IDs")
    ap.add_argument("--seq-file", help="Path to a file with one seq ID per line")
    args = ap.parse_args()

    all_sequences = list(csv.DictReader(open(INPUT)))
    wanted_ids = set(fl.resolve_ids([r["id"] for r in all_sequences], args.seq, args.seq_file))
    sequences = [r for r in all_sequences if r["id"] in wanted_ids]

    print(f"{len(sequences)} sequences")
    print(f"Tool: ToxinPred3 Model 1 (AAC+DPC Extra Trees) | Threshold: {THRESHOLD}")
    print(f"Native amylin baseline: 0.35 (Non-Toxin)\n")

    raw_rows = run_toxinpred3_batch(sequences)
    if not raw_rows:
        return

    seq_map = {r["id"]: r["full_37aa"] for r in sequences}
    results = []
    for row in raw_rows:
        # Model 2 uses "Subject" and "Hybrid Score"; Model 1 uses "ID" and "ML Score"
        seq_id     = row.get("Subject", row.get("ID", "?"))
        ml_score   = float(row.get("Hybrid Score", row.get("ML Score", 0)))
        prediction = row["Prediction"]
        ppv        = row.get("PPV", "")
        status     = "PASS" if prediction == "Non-Toxin" else "FAIL"
        results.append({
            "id":         seq_id,
            "sequence":   seq_map.get(seq_id, ""),
            "ml_score":   f"{ml_score:.3f}",
            "ppv":        ppv,
            "prediction": prediction,
            "toxin_pass": status,
        })
        print(f"  {seq_id}  score={ml_score:.3f}  {prediction}  -> {status}")

    with open(OUTDIR / "toxinpred3_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)

    passed = [r for r in results if r["toxin_pass"] == "PASS"]
    with open(OUTDIR / "toxinpred3_passed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(passed)

    print(f"\n{'='*60}")
    print(f"Result: {len(passed)}/{len(results)} PASS")
    print(f"Criterion: ML_Score < {THRESHOLD} (Non-Toxin)")
    print(f"Full results -> outputs/toxinpred3_results.csv")
    print(f"Passed       -> outputs/toxinpred3_passed.csv")


if __name__ == "__main__":
    main()
