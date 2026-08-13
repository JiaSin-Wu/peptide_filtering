"""
infer_gnn.py

Run inference on the held-out test set with a saved GNN model ensemble.
Outputs per-sample predictions to results/per_sample_<model_name>.csv

Usage:
  python3 infer_gnn.py models/gnn_gat_gat_h8_drop01/
  python3 infer_gnn.py models/gnn_gat_drop01/
"""

import argparse
import csv
import json
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import DataLoader

HOME      = Path("/home/jiasin/AMY123R_agonist_Design/model")
DATA_DIR  = HOME / "data"
RES_DIR   = HOME / "results"
LABEL_CSV = Path("/home/jiasin/outputs/ml_label_final.csv")

HELD_OUT_RECS = {"P30988"}

RAMP_NUMBER = {
    # human RAMPs
    "O60894": "1", "O60895": "2", "O60896": "3",
    # rat RAMPs
    "Q9JHJ1": "1", "Q9JJ73": "2", "Q9JJ74": "3",
    # mouse RAMPs
    "Q9WTJ5": "1", "Q9WUP0": "2", "Q9WUP1": "3",
}

# (receptor_uniprot, ramp_number) → complex name
COMPLEX_NAME = {
    ("P30988", "1"): "AMY1R (human)",
    ("P30988", "2"): "AMY2R (human)",
    ("P30988", "3"): "AMY3R (human)",
    ("P32214", "2"): "AMY2R (rat)",
    ("P32214", "3"): "AMY3R (rat)",
    ("Q16602", "1"): "CGRPR (human)",
    ("Q16602", "2"): "AM1R (human)",
    ("Q16602", "3"): "AM2R (human)",
    ("Q63118", "1"): "CGRPR (rat)",
    ("Q63118", "2"): "AM1R (rat)",
    ("Q63118", "3"): "AM2R (rat)",
    ("Q9R1W5", "1"): "CGRPR (mouse)",
    ("Q9R1W5", "2"): "AM1R (mouse)",
    ("Q9R1W5", "3"): "AM2R (mouse)",
}


def get_complex(name: str, kind: str) -> str:
    """Derive receptor complex name from folder name and kind."""
    parts = name.split("_")
    receptor = parts[0]
    if kind == "E13":
        return "CTR (CALCR_HUMAN, no RAMP)"
    # ER04: format is receptor_RAMP_LIG...
    ramp_uniprot = parts[1] if len(parts) > 1 else ""
    ramp_num = RAMP_NUMBER.get(ramp_uniprot, "?")
    return COMPLEX_NAME.get((receptor, ramp_num), f"{receptor}+RAMP{ramp_num}")


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def eval_loader(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu().numpy()
            labels = batch["ligand"].y.cpu().numpy() if hasattr(batch["ligand"], "y") else \
                     batch.y.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    if not all_probs:
        return np.array([]), np.array([])
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to saved model directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg       = json.load(open(model_dir / "config.json"))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_dir.name}  cfg={cfg}")

    # load model architecture
    model_type = cfg["model_type"]
    if model_type == "xattn":
        from gnn.model_xattn import CrossAttnAgonism
        def make_model():
            return CrossAttnAgonism(
                hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                use_ffn=cfg.get("ffn", False), ligand_only=cfg.get("ligand_only", False),
            ).to(device)
    else:
        from gnn.model import AgonismGNN
        def make_model():
            return AgonismGNN(
                hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"], dropout=cfg["dropout"],
            ).to(device)

    # load held-out rows
    src_map  = {int(r["id"]): r["source"]
                for r in csv.DictReader(open(LABEL_CSV))}
    meta_all = list(csv.DictReader(open(DATA_DIR / "meta.csv")))

    def is_decoy(row):
        return src_map.get(int(row["row_id"]), "") == "decoy_cross_class"

    held_rows = [r for r in meta_all
                 if (r["kind"] == "ER04" or
                     (r["kind"] == "E13" and r["name"].split("_")[0] in HELD_OUT_RECS))
                 and not is_decoy(r)]
    y_held = np.array([1 if r["label"] == "agonist" else 0 for r in held_rows])
    print(f"Held-out samples: {len(held_rows)}  agonist={y_held.sum()}")

    from gnn.dataset import AgonismDataset
    ds     = AgonismDataset(rows=held_rows)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=collate_skip_none, num_workers=0)
    valid_pos = ds.valid_positions

    # ensemble inference
    n_models   = cfg.get("n_models", 5)
    probs_sum  = np.zeros(len(held_rows), dtype=np.float64)

    for i in range(n_models):
        pt = model_dir / f"model_{i}.pt"
        if not pt.exists():
            print(f"  [SKIP] model_{i}.pt not found")
            continue
        model = make_model()
        model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
        model.eval()
        p, _ = eval_loader(model, loader, device)
        probs_sum[valid_pos] += p
        print(f"  model_{i}: done")

    probs_avg = probs_sum / n_models
    preds     = (probs_avg >= 0.5).astype(int)

    # per-sample output
    out_rows = []
    for i, row in enumerate(held_rows):
        complex_name = get_complex(row["name"], row["kind"])
        out_rows.append({
            "name":        row["name"],
            "complex":     complex_name,
            "kind":        row["kind"],
            "label":       row["label"],
            "y_true":      int(y_held[i]),
            "p_agonist":   float(probs_avg[i]),
            "y_pred":      int(preds[i]),
            "correct":     int(y_held[i] == preds[i]),
            "has_graph":   int(i in valid_pos),
        })

    out_csv = RES_DIR / f"per_sample_{model_dir.name}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    print(f"Saved per-sample predictions → {out_csv}")

    # per-complex summary
    from collections import defaultdict
    cplx_stats = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"n":0})
    for r in out_rows:
        s = cplx_stats[r["complex"]]
        s["n"]   += 1
        yt, yp    = r["y_true"], r["y_pred"]
        if   yt==1 and yp==1: s["tp"] += 1
        elif yt==0 and yp==0: s["tn"] += 1
        elif yt==1 and yp==0: s["fn"] += 1
        else:                  s["fp"] += 1

    # sort: CTR first, then AMY, then CGRP, AM
    def sort_key(cname):
        order = ["CTR","AMY1","AMY2","AMY3","CGRP","AM1","AM2"]
        for i, k in enumerate(order):
            if cname.startswith(k): return i
        return 99

    print(f"\n{'Complex':<26} {'N':>4} {'Ag/NAg':>8} {'Acc':>6} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-"*76)
    for cname, s in sorted(cplx_stats.items(), key=lambda x: sort_key(x[0])):
        ag  = s["tp"] + s["fn"]
        nag = s["tn"] + s["fp"]
        acc = (s["tp"]+s["tn"]) / s["n"]
        print(f"{cname:<26} {s['n']:>4} {ag:>4}/{nag:<4} {acc:>6.3f} {s['tp']:>4} {s['tn']:>4} {s['fp']:>4} {s['fn']:>4}")

    n_correct = sum(r["correct"] for r in out_rows)
    print(f"\nOverall: acc={n_correct/len(out_rows):.3f}  n={len(out_rows)}")


if __name__ == "__main__":
    main()
