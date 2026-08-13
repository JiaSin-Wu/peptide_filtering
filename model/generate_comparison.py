"""
generate_comparison.py

Generate ml_comparison.html from results/*.json
Usage: python3 generate_comparison.py
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

RES_DIR  = Path("/home/jiasin/AMY123R_agonist_Design/model/results")
OUT      = Path("/home/jiasin/AMY123R_agonist_Design/model/ml_comparison.html")

UNIPROT_NAME = {
    "P30988": "CALCR_HUMAN",
    "P32214": "CALCR_RAT",
    "Q16602": "CALRL_HUMAN",
    "Q63118": "CALRL_RAT",
    "Q9R1W5": "CALRL_MOUSE",
}

# preferred model order for per-receptor analysis
PER_SAMPLE_PRIORITY = [
    "per_sample_gnn_gat_gat_h8_drop01.csv",
    "per_sample_gnn_gat_drop01.csv",
]


def shorten_feat(name):
    return name.replace("single_", "").replace("pair_", "")


def load_results():
    rows_main = []
    for f in sorted(RES_DIR.glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        has_oof      = "oof" in d
        has_held_out = "held_out_er04" in d
        if has_oof or (has_held_out and d.get("train_mode") == "all"):
            cfg = d.get("config", {})
            if cfg.get("model_type") in ("gat", "xattn"):
                features = f"single_emb(384d) + pair(128d) | cutoff={cfg.get('cutoff', 8.0):.0f}A"
                ffn_str  = " ffn=SwiGLU" if cfg.get("ffn") else ""
                config   = f"h={cfg.get('hidden',128)} layers={cfg.get('n_layers',2)} heads={cfg.get('n_heads',4)} dropout={cfg.get('dropout',0.4)} batch={cfg.get('batch',32)} epochs={cfg.get('epochs',50)}{ffn_str}"
                model_name = cfg["model_type"]
                n_folds    = "all" if d.get("train_mode") == "all" else cfg.get("n_folds", d.get("n_folds", "—"))
            elif cfg.get("model_type") == "egnn" or d.get("model_type") == "egnn":
                features   = f"single_emb(384d) + pair(128d) + pos(3d) | cutoff={cfg.get('cutoff', 8.0):.0f}A"
                config     = f"h={cfg.get('hidden',128)} layers={cfg.get('n_layers',2)} dropout={cfg.get('dropout',0.1)} batch={cfg.get('batch',32)} epochs={cfg.get('epochs',100)}"
                model_name = "egnn"
                n_folds    = "all" if d.get("train_mode") == "all" else "—"
            elif cfg.get("model_type") in ("mlp", "mlp_gated"):
                features   = ", ".join(shorten_feat(fn) for fn in d.get("features", []))
                config     = f"dropout={cfg.get('dropout',0.4)} lr={cfg.get('lr',1e-3)} epochs={cfg.get('epochs',100)}"
                model_name = d.get("model_type", "mlp")
                n_folds    = "all" if d.get("train_mode") == "all" else d.get("n_folds", "—")
            else:
                features   = ", ".join(shorten_feat(fn) for fn in d.get("features", [])) or d.get("tag", "")
                config     = "n_estimators=500"
                model_name = d.get("model_type", "lgbm")
                n_folds    = "all" if d.get("train_mode") == "all" else d.get("n_folds", "—")
            er04 = d.get("held_out_er04", {})
            nan  = float("nan")
            rows_main.append({
                "name":           f.stem,
                "model":          model_name,
                "features":       features,
                "config":         config,
                "er04_auroc":     er04.get("auroc",     nan),
                "er04_auprc":     er04.get("auprc",     nan),
                "er04_f1":        er04.get("f1",        nan),
                "er04_mcc":       er04.get("mcc",       nan),
                "er04_precision": er04.get("precision", nan),
                "er04_recall":    er04.get("recall",    nan),
                "er04_accuracy":  er04.get("accuracy",  nan),
            })
        elif "auroc_finetune" in d:
            nan = float("nan")
            er04_ft  = d.get("held_out_er04_finetune", {})
            er04_nft = d.get("held_out_er04_no_finetune", {})
            gated = "gated" in f.stem
            base_feat = "B_mean + AB_mean_8A"
            rows_main.append({
                "model":          "mlp_transfer" + ("_gated" if gated else ""),
                "features":       base_feat,
                "config":         "Stage1: E13, Stage2: fine-tune B1 head",
                "er04_auroc":     er04_ft.get("auroc",     d.get("auroc_finetune",    nan)),
                "er04_auprc":     er04_ft.get("auprc",     d.get("auprc_finetune",    nan)),
                "er04_f1":        er04_ft.get("f1",        d.get("f1_finetune",       nan)),
                "er04_mcc":       er04_ft.get("mcc",       d.get("mcc_finetune",      nan)),
                "er04_precision": er04_ft.get("precision", d.get("precision_finetune",nan)),
                "er04_recall":    er04_ft.get("recall",    d.get("recall_finetune",   nan)),
                "er04_accuracy":  er04_ft.get("accuracy",  nan),
            })
            if "auroc_no_finetune" in d:
                rows_main.append({
                    "model":          "mlp_no_finetune" + ("_gated" if gated else ""),
                    "features":       base_feat,
                    "config":         "Stage1: E13 only (no B1 fine-tune)",
                    "er04_auroc":     er04_nft.get("auroc",     d.get("auroc_no_finetune",    nan)),
                    "er04_auprc":     er04_nft.get("auprc",     d.get("auprc_no_finetune",    nan)),
                    "er04_f1":        er04_nft.get("f1",        d.get("f1_no_finetune",       nan)),
                    "er04_mcc":       er04_nft.get("mcc",       d.get("mcc_no_finetune",      nan)),
                    "er04_precision": er04_nft.get("precision", d.get("precision_no_finetune",nan)),
                    "er04_recall":    er04_nft.get("recall",    d.get("recall_no_finetune",   nan)),
                    "er04_accuracy":  er04_nft.get("accuracy",  nan),
                })
        else:
            continue
    rows_main.sort(key=lambda r: r["er04_mcc"] if r["er04_mcc"] == r["er04_mcc"] else -1, reverse=True)
    return rows_main


def fmt(v):
    return "—" if v != v else f"{v:.4f}"


COLS_MAIN = ("<th>model</th><th>features</th><th>config</th>"
             "<th>AUROC</th><th>AUPRC</th><th>F1</th><th>MCC</th><th>Precision</th><th>Recall</th><th>Accuracy</th>")

COLS_B1 = ("<th>name</th><th>model</th><th>features</th><th>config</th>"
           "<th>n_folds</th><th>AUROC</th><th>AUPRC</th><th>F1</th><th>MCC</th><th>Precision</th><th>Recall</th>")


def make_table_main(rows):
    if not rows:
        return "<p><em>no results</em></p>\n"
    best_er04_mcc = rows[0]["er04_mcc"]
    t = f"<table>\n<tr>{COLS_MAIN}</tr>\n"
    for r in rows:
        cls = ' class="best"' if r["er04_mcc"] == r["er04_mcc"] and abs(r["er04_mcc"] - best_er04_mcc) < 1e-6 else ""
        t += (
            f'<tr{cls}>'
            f'<td>{r["model"]}</td>'
            f'<td>{r["features"]}</td>'
            f'<td>{r["config"]}</td>'
            f'<td>{fmt(r["er04_auroc"])}</td>'
            f'<td>{fmt(r["er04_auprc"])}</td>'
            f'<td>{fmt(r["er04_f1"])}</td>'
            f'<td>{fmt(r["er04_mcc"])}</td>'
            f'<td>{fmt(r["er04_precision"])}</td>'
            f'<td>{fmt(r["er04_recall"])}</td>'
            f'<td>{fmt(r["er04_accuracy"])}</td>'
            f'</tr>\n'
        )
    t += "</table>\n"
    return t



def make_html(rows_main, per_sample_model=None, per_sample_rows=None):
    header = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ML Comparison</title>
<style>
  body { font-family: monospace; padding: 20px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 40px; }
  th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }
  th { background: #f0f0f0; }
  tr:hover { background: #f9f9f9; }
  .best { background: #d4edda; font-weight: bold; }
  .er04-header { background: #e8f4f8; }
  h2 { margin-top: 40px; }
</style>
</head><body>
"""
    legend = """
<div style="background:#f8f8f8;border:1px solid #ddd;padding:14px 20px;border-radius:6px;margin-bottom:20px;font-size:0.95em;line-height:1.8;">
  <strong>Feature notation</strong><br>
  <b>A</b> = Chain A &nbsp;(receptor / GPCR protein)<br>
  <b>B</b> = Chain B &nbsp;(ligand / small molecule)<br>
  <b>AB</b> = A×B pair &nbsp;(cross-chain interaction)<br>
  <br>
  <b>_mean / _max</b> &nbsp;— pooling over all residues of that chain<br>
  <b>_8A</b> &nbsp;— restricted to residue pairs within 8 Å contact distance<br>
  <br>
  Embeddings come from AlphaFold3 internal representations:<br>
  &nbsp;&nbsp;• <b>single_A / single_B</b>: per-residue single representation &nbsp;(384-dim)<br>
  &nbsp;&nbsp;• <b>pair_AB</b>: per-residue-pair interaction representation &nbsp;(128-dim)
</div>
<div style="background:#f8f8f8;border:1px solid #ddd;padding:14px 20px;border-radius:6px;margin-bottom:20px;font-size:0.95em;line-height:2.0;">
  <strong>Metric formulas</strong> &nbsp;(TP = true positive, TN = true negative, FP = false positive, FN = false negative)<br>
  <b>Precision</b> = TP / (TP + FP)<br>
  <b>Recall</b> = TP / (TP + FN)<br>
  <b>F1</b> = 2 × Precision × Recall / (Precision + Recall) = 2TP / (2TP + FP + FN)<br>
  <b>MCC</b> = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] &nbsp;∈ [−1, 1], accounts for class imbalance<br>
  <b>AUROC</b> = area under TPR vs FPR curve &nbsp;(threshold-free; 0.5 = random, 1.0 = perfect)<br>
  <b>AUPRC</b> = area under Precision vs Recall curve &nbsp;(more sensitive to class imbalance than AUROC)
</div>
<div style="background:#fff8e1;border:1px solid #ffe082;padding:14px 20px;border-radius:6px;margin-bottom:30px;font-size:0.95em;line-height:1.8;">
  <strong>Evaluation protocol</strong><br>
  <b>Training</b>: E13 series only (RAMP-free single-receptor AF3 structures). P30988 (CALCR_HUMAN) excluded so all CALCR-family receptors are unseen at test time.
  5619 samples, 163 receptors. Training includes cross-class decoy negatives (artificially generated negatives from cross-receptor labeling).<br>
  <b>Held-out test set</b>: 72 samples from AF3 structures that include a RAMP subunit as an additional chain —
  making them structurally distinct from the RAMP-free E13 structures used in training.
  All belong to the calcitonin receptor family (Class B1 GPCR). Cross-class decoys excluded.<br>
  <br>
  <b>Held-out test set composition</b> (82 samples total: <b>61 agonist / 21 non-agonist</b>):<br>
  Includes ER04 RAMP complexes (72) + CALCR_HUMAN E13 RAMP-free structures (10, excluded from training):<br>
  <table style="border-collapse:collapse;margin-top:8px;font-size:0.92em;">
    <tr style="background:#f0f0f0;">
      <th style="padding:4px 10px;border:1px solid #ccc;">Receptor (UniProt)</th>
      <th style="padding:4px 10px;border:1px solid #ccc;">RAMP (UniProt)</th>
      <th style="padding:4px 10px;border:1px solid #ccc;">Complex</th>
      <th style="padding:4px 10px;border:1px solid #ccc;">Agonist</th>
      <th style="padding:4px 10px;border:1px solid #ccc;">Non-ag</th>
      <th style="padding:4px 10px;border:1px solid #ccc;">Total</th>
    </tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALCR_HUMAN (P30988)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP1 (O60894)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AMY1</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">8</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">3</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">11</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALCR_HUMAN (P30988)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP2 (O60895)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AMY2</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">6</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">0</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">6</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALCR_HUMAN (P30988)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP3 (O60896)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AMY3</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">7</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">3</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">10</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALCR_RAT (P32214)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP2 (Q9JJ73)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AMY2</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">0</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALCR_RAT (P32214)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP3 (Q9JJ74)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AMY3</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">0</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_HUMAN (Q16602)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP1 (O60894)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>CGRP</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">5</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">6</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_HUMAN (Q16602)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP2 (O60895)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AM1</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">5</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">7</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_HUMAN (Q16602)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP3 (O60896)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AM2</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">5</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">7</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_RAT (Q63118)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP1 (Q9JHJ1)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>CGRP</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">4</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">6</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_RAT (Q63118)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP2 (Q9JJ73)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AM1</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">3</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">5</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_RAT (Q63118)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP3 (Q9JJ74)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AM2</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">3</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">4</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_MOUSE (Q9R1W5)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP1 (Q9WTJ5)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>CGRP</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">0</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_MOUSE (Q9R1W5)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP2 (Q9WUP0)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AM1</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">1</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALRL_MOUSE (Q9R1W5)</td><td style="padding:3px 10px;border:1px solid #ccc;">RAMP3 (Q9WUP1)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>AM2</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">4</td></tr>
    <tr style="background:#f0f0f0;font-weight:bold;"><td style="padding:3px 10px;border:1px solid #ccc;" colspan="3">ER04 subtotal</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">53</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">19</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">72</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #ccc;">CALCR_HUMAN (P30988)</td><td style="padding:3px 10px;border:1px solid #ccc;">— (RAMP-free)</td><td style="padding:3px 10px;border:1px solid #ccc;"><b>E13</b></td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">8</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">2</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">10</td></tr>
    <tr style="background:#f0f0f0;font-weight:bold;"><td style="padding:3px 10px;border:1px solid #ccc;" colspan="3">Total</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">61</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">21</td><td style="padding:3px 10px;border:1px solid #ccc;text-align:right;">82</td></tr>
  </table>
  <br>
  <br>
  <b>Data exclusion note</b>: P30988 (CALCR_HUMAN) has 36 samples in E13 (RAMP-free, single-receptor structures).
  These are excluded from training to ensure the model has never seen CALCR_HUMAN in any form —
  making the held-out evaluation fully receptor-unseen.
  The other 4 test receptors (P32214, Q16602, Q63118, Q9R1W5) have no E13 entries and are naturally absent from training.
  The 10 non-decoy P30988 E13 samples (8 agonist, 2 non-agonist) are included in the held-out test set alongside the 72 ER04 RAMP samples.
  <br><br>
  <b>Validation split</b>: In train_all mode, 10% of training receptors (receptor-stratified) are held out as a validation set for early stopping (max 300 epochs). Decoys are excluded from validation metrics.
  <br><br>
  <b>No-contact fallback</b>: Graph-based models (GNN, EGNN) require at least one pocket–ligand contact edge within the distance cutoff.
  If AF3 places the ligand &gt;8 Å from all receptor residues (no contact edges), the graph cannot be built.
  For the EGNN model, such samples are assigned P(agonist) = 0.0 (predicted nonagonist) by default,
  since AF3 failing to dock the ligand near the receptor is itself a strong signal of non-binding.
</div>
"""
    body  = legend
    body += "<h2>Held-out Test Set (RAMP complexes) — trained on E13 excluding CALCR_HUMAN (P30988)</h2>\n"
    body += make_table_main(rows_main)
    body += "<h2>Per-Receptor Accuracy (best model)</h2>\n"
    if per_sample_rows:
        body += make_per_receptor_table(per_sample_model, per_sample_rows)
    else:
        body += "<p><em>No per-sample predictions available. Run infer_gnn.py first.</em></p>\n"
    body += "<h2>Model Architecture — GATv2 Agonism Predictor</h2>\n"
    body += ARCH_SVG
    body += "</body></html>"
    return header + body


def load_per_sample():
    """Load best available per-sample CSV. Returns (model_name, rows) or (None, [])."""
    for fname in PER_SAMPLE_PRIORITY:
        p = RES_DIR / fname
        if p.exists():
            rows = list(csv.DictReader(open(p)))
            model_name = fname.replace("per_sample_", "").replace(".csv", "")
            return model_name, rows
    return None, []


def make_per_receptor_table(model_name, rows):
    if not rows:
        return ""
    cplx_stats = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"n":0})
    for r in rows:
        s = cplx_stats[r["complex"]]
        s["n"]   += 1
        yt, yp    = int(r["y_true"]), int(r["y_pred"])
        if   yt==1 and yp==1: s["tp"] += 1
        elif yt==0 and yp==0: s["tn"] += 1
        elif yt==1 and yp==0: s["fn"] += 1
        else:                  s["fp"] += 1

    total_correct = sum(int(r["correct"]) for r in rows)
    total_n       = len(rows)

    def sort_key(cname):
        order = ["CTR","AMY1","AMY2","AMY3","CGRP","AM1","AM2"]
        for i, k in enumerate(order):
            if cname.startswith(k): return i
        return 99

    ts = "padding:4px 10px;border:1px solid #ccc;"
    t  = f'<p style="margin-bottom:6px;"><em>Model: <strong>{model_name}</strong></em></p>\n'
    t += f'<table style="border-collapse:collapse;font-size:0.93em;width:auto;">\n'
    t += (f'  <tr style="background:#f0f0f0;">'
          f'<th style="{ts}">Complex</th>'
          f'<th style="{ts}">N</th><th style="{ts}">Agonist</th><th style="{ts}">Non-ag</th>'
          f'<th style="{ts}">Accuracy</th><th style="{ts}">TP</th><th style="{ts}">TN</th>'
          f'<th style="{ts}">FP</th><th style="{ts}">FN</th>'
          f'</tr>\n')

    for cname, s in sorted(cplx_stats.items(), key=lambda x: sort_key(x[0])):
        ag  = s["tp"] + s["fn"]
        nag = s["tn"] + s["fp"]
        acc = (s["tp"] + s["tn"]) / s["n"]
        bar = int(acc * 20)
        bar_html = f'<span style="display:inline-block;width:{bar*6}px;height:10px;background:#4caf50;vertical-align:middle;border-radius:2px;"></span> {acc:.3f}'
        t += (f'<tr>'
              f'<td style="{ts}">{cname}</td>'
              f'<td style="{ts};text-align:right;">{s["n"]}</td>'
              f'<td style="{ts};text-align:right;">{ag}</td>'
              f'<td style="{ts};text-align:right;">{nag}</td>'
              f'<td style="{ts}">{bar_html}</td>'
              f'<td style="{ts};text-align:right;">{s["tp"]}</td>'
              f'<td style="{ts};text-align:right;">{s["tn"]}</td>'
              f'<td style="{ts};text-align:right;">{s["fp"]}</td>'
              f'<td style="{ts};text-align:right;">{s["fn"]}</td>'
              f'</tr>\n')

    overall_acc = total_correct / total_n
    bar = int(overall_acc * 20)
    bar_html = f'<span style="display:inline-block;width:{bar*6}px;height:10px;background:#2196f3;vertical-align:middle;border-radius:2px;"></span> {overall_acc:.3f}'
    t += (f'<tr style="font-weight:bold;background:#f0f0f0;">'
          f'<td style="{ts}">Overall</td>'
          f'<td style="{ts};text-align:right;">{total_n}</td>'
          f'<td style="{ts};text-align:right;">{sum(s["tp"]+s["fn"] for s in cplx_stats.values())}</td>'
          f'<td style="{ts};text-align:right;">{sum(s["tn"]+s["fp"] for s in cplx_stats.values())}</td>'
          f'<td style="{ts}">{bar_html}</td>'
          f'<td style="{ts};text-align:right;">{sum(s["tp"] for s in cplx_stats.values())}</td>'
          f'<td style="{ts};text-align:right;">{sum(s["tn"] for s in cplx_stats.values())}</td>'
          f'<td style="{ts};text-align:right;">{sum(s["fp"] for s in cplx_stats.values())}</td>'
          f'<td style="{ts};text-align:right;">{sum(s["fn"] for s in cplx_stats.values())}</td>'
          f'</tr>\n')
    t += "</table>\n"
    return t


ARCH_SVG = """
<svg width="820" height="420" xmlns="http://www.w3.org/2000/svg" font-family="monospace" font-size="13">
  <!-- background -->
  <rect width="820" height="420" fill="#fafafa" stroke="#ddd" rx="8"/>

  <!-- Title -->
  <text x="410" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">GATv2 Agonism Predictor — Architecture</text>

  <!-- === Input boxes === -->
  <!-- AF3 structure -->
  <rect x="20" y="55" width="140" height="50" fill="#e3f2fd" stroke="#1976d2" rx="5"/>
  <text x="90" y="76" text-anchor="middle" font-weight="bold" fill="#1976d2">AF3 Output</text>
  <text x="90" y="93" text-anchor="middle" fill="#555" font-size="11">CIF + NPZ</text>

  <!-- Pocket nodes -->
  <rect x="200" y="45" width="160" height="34" fill="#fff3e0" stroke="#f57c00" rx="4"/>
  <text x="280" y="59" text-anchor="middle" font-weight="bold" fill="#f57c00">Pocket Nodes</text>
  <text x="280" y="73" text-anchor="middle" fill="#555" font-size="11">Receptor residues ×384d</text>

  <!-- Ligand nodes -->
  <rect x="200" y="95" width="160" height="34" fill="#e8f5e9" stroke="#388e3c" rx="4"/>
  <text x="280" y="109" text-anchor="middle" font-weight="bold" fill="#388e3c">Ligand Nodes</text>
  <text x="280" y="123" text-anchor="middle" fill="#555" font-size="11">Peptide residues ×384d</text>

  <!-- Edges -->
  <rect x="200" y="145" width="160" height="34" fill="#fce4ec" stroke="#c2185b" rx="4"/>
  <text x="280" y="159" text-anchor="middle" font-weight="bold" fill="#c2185b">Edges (8Å cutoff)</text>
  <text x="280" y="173" text-anchor="middle" fill="#555" font-size="11">pair embed attr ×128d</text>

  <!-- arrows from AF3 to inputs -->
  <line x1="160" y1="80" x2="198" y2="62" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="160" y1="80" x2="198" y2="112" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="160" y1="80" x2="198" y2="162" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- === Projection === -->
  <rect x="400" y="45" width="130" height="55" fill="#ede7f6" stroke="#6a1b9a" rx="4"/>
  <text x="465" y="63" text-anchor="middle" font-weight="bold" fill="#6a1b9a">Projection</text>
  <text x="465" y="79" text-anchor="middle" fill="#555" font-size="11">node: 384→128d</text>
  <text x="465" y="93" text-anchor="middle" fill="#555" font-size="11">edge: 128→128d</text>

  <!-- arrows input → proj -->
  <line x1="360" y1="62" x2="398" y2="68" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="360" y1="112" x2="398" y2="72" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="360" y1="162" x2="398" y2="96" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- === GATv2Conv layers === -->
  <!-- Layer 1 -->
  <rect x="400" y="160" width="130" height="70" fill="#e1f5fe" stroke="#0288d1" rx="4"/>
  <text x="465" y="180" text-anchor="middle" font-weight="bold" fill="#0288d1">GATv2Conv ×1</text>
  <text x="465" y="196" text-anchor="middle" fill="#555" font-size="11">pocket → ligand</text>
  <text x="465" y="210" text-anchor="middle" fill="#555" font-size="11">ligand → pocket</text>
  <text x="465" y="224" text-anchor="middle" fill="#555" font-size="11">+ LayerNorm + Residual</text>

  <!-- Layer 2 -->
  <rect x="400" y="250" width="130" height="70" fill="#e1f5fe" stroke="#0288d1" rx="4"/>
  <text x="465" y="270" text-anchor="middle" font-weight="bold" fill="#0288d1">GATv2Conv ×2</text>
  <text x="465" y="286" text-anchor="middle" fill="#555" font-size="11">pocket → ligand</text>
  <text x="465" y="300" text-anchor="middle" fill="#555" font-size="11">ligand → pocket</text>
  <text x="465" y="314" text-anchor="middle" fill="#555" font-size="11">+ LayerNorm + Residual</text>

  <!-- arrows proj → l1 → l2 -->
  <line x1="465" y1="100" x2="465" y2="158" stroke="#0288d1" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="465" y1="230" x2="465" y2="248" stroke="#0288d1" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- layer config annotation -->
  <text x="540" y="200" fill="#888" font-size="11">h=128, heads=8</text>
  <text x="540" y="213" fill="#888" font-size="11">dropout=0.1</text>

  <!-- === Attention Pool === -->
  <rect x="400" y="345" width="130" height="40" fill="#fff9c4" stroke="#f9a825" rx="4"/>
  <text x="465" y="361" text-anchor="middle" font-weight="bold" fill="#f9a825">Attention Pool</text>
  <text x="465" y="377" text-anchor="middle" fill="#555" font-size="11">over ligand nodes</text>
  <line x1="465" y1="320" x2="465" y2="343" stroke="#0288d1" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- === MLP === -->
  <rect x="580" y="335" width="120" height="60" fill="#fbe9e7" stroke="#d84315" rx="4"/>
  <text x="640" y="355" text-anchor="middle" font-weight="bold" fill="#d84315">MLP Classifier</text>
  <text x="640" y="371" text-anchor="middle" fill="#555" font-size="11">128 → 64 → 1</text>
  <text x="640" y="385" text-anchor="middle" fill="#555" font-size="11">ReLU + Dropout</text>
  <line x1="530" y1="365" x2="578" y2="365" stroke="#d84315" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- === Output === -->
  <rect x="720" y="345" width="80" height="40" fill="#e8f5e9" stroke="#2e7d32" rx="4"/>
  <text x="760" y="361" text-anchor="middle" font-weight="bold" fill="#2e7d32">P(agonist)</text>
  <text x="760" y="377" text-anchor="middle" fill="#555" font-size="11">sigmoid</text>
  <line x1="700" y1="365" x2="718" y2="365" stroke="#2e7d32" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- === Ensemble note === -->
  <rect x="580" y="50" width="220" height="60" fill="#f3e5f5" stroke="#7b1fa2" stroke-dasharray="4,3" rx="4"/>
  <text x="690" y="69" text-anchor="middle" font-weight="bold" fill="#7b1fa2">Ensemble (×5 models)</text>
  <text x="690" y="85" text-anchor="middle" fill="#555" font-size="11">5 independently trained seeds</text>
  <text x="690" y="99" text-anchor="middle" fill="#555" font-size="11">avg sigmoid(logits)</text>

  <!-- arrowhead marker -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
</svg>
"""


rows_main = load_results()
model_name, per_sample_rows = load_per_sample()
html = make_html(rows_main, model_name, per_sample_rows)
OUT.write_text(html)
print(f"Written {len(rows_main)} rows → {OUT}")
