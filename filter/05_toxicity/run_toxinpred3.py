"""
run_toxinpred3.py

Toxicity screening via ToxinPred3 (Raghava group).

Method    : Extra Trees on AAC + DPC (Model 1, sklearn 1.0.2)
Threshold : ML_Score < 0.38 → Non-Toxin → PASS  (ToxinPred3 default)
            Native amylin baseline = 0.35 (passes at default threshold)
Input     : full 37aa sequence
Run with  : conda run -n raghava_tools python3 run_toxinpred3.py
"""

import csv
import subprocess
import tempfile
from pathlib import Path

HERE      = Path(__file__).parent
INPUT     = HERE.parent / "sequences.csv"
OUTDIR    = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)
TRACKER   = HERE.parent / "filter_tracker.csv"

TOOL_DIR  = HERE / "tools" / "toxinpred3"
THRESHOLD = 0.38
RESULT_FIELDS = ["id", "sequence", "ml_score", "ppv", "prediction", "toxin_pass"]


def run_toxinpred3_batch(sequences: list) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as fa:
        for row in sequences:
            fa.write(f">{row['id']}\n{row['full_37aa']}\n")
        fa_path = fa.name

    out_path = str(OUTDIR / "toxinpred3_raw.csv")
    cmd = [
        "python3", str(TOOL_DIR / "toxinpred3.py"),  # absolute path fixes nf_path for Model 2
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


def update_tracker(results: list):
    result_map = {r["id"]: 1 if r["toxin_pass"] == "PASS" else 0 for r in results}

    rows   = list(csv.DictReader(open(TRACKER)))
    fields = list(rows[0].keys()) if rows else ["id", "sequence"]
    if "toxicity" not in fields:
        fields.append("toxicity")
    for row in rows:
        row.setdefault("toxicity", "")
        if row["id"] in result_map:
            row["toxicity"] = result_map[row["id"]]

    with open(TRACKER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(result_map.values())
    print(f"filter_tracker.csv updated -> toxicity: {n_pass} PASS / {len(result_map)} total")


def main():
    sequences = list(csv.DictReader(open(INPUT)))
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

    update_tracker(results)

    print(f"\n{'='*60}")
    print(f"Result: {len(passed)}/{len(results)} PASS")
    print(f"Criterion: ML_Score < {THRESHOLD} (Non-Toxin)")
    print(f"Full results -> outputs/toxinpred3_results.csv")
    print(f"Passed       -> outputs/toxinpred3_passed.csv")


if __name__ == "__main__":
    main()
