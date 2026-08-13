"""
train.py

Train on E13 only (10-fold receptor-stratified CV).
  - Train fold : all samples including cross-class decoys
  - Val fold   : real labels only (IUPHAR + ChEMBL), decoys excluded from metrics
Held-out evaluation on ER04 (RAMP-included), cross-class decoys excluded.

Usage:
  python3 -m gnn.train
  python3 -m gnn.train --model_type xattn --dropout 0.1 --tag drop01
"""

import argparse, csv, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from torch.utils.data import DataLoader

from .model import AgonismGNN
from .model_xattn import CrossAttnAgonism
from .dataset import AgonismDataset, collate_skip_none

HOME      = Path("/home/jiasin/AMY123R_agonist_Design/model")
RES_DIR   = HOME / "results"
MODEL_DIR = HOME / "models"
LABEL_CSV = Path("/home/jiasin/outputs/ml_label_final.csv")
RES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        probs  = torch.sigmoid(model(batch)).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_probs), np.array(all_labels)


def compute_metrics(y_true, probs):
    if len(set(y_true)) < 2:
        return {"auroc": 0.5, "auprc": 0.0, "f1": 0.0, "mcc": 0.0,
                "precision": 0.0, "recall": 0.0}
    pred = (probs >= 0.5).astype(int)
    return {
        "auroc":     float(roc_auc_score(y_true, probs)),
        "auprc":     float(average_precision_score(y_true, probs)),
        "f1":        float(f1_score(y_true, pred)),
        "mcc":       float(matthews_corrcoef(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "accuracy":  float((pred == y_true).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds",          type=int,   default=10)
    parser.add_argument("--n_models",         type=int,   default=5)
    parser.add_argument("--epochs",           type=int,   default=200)
    parser.add_argument("--patience",         type=int,   default=50)
    parser.add_argument("--hidden",           type=int,   default=128)
    parser.add_argument("--n_heads",          type=int,   default=4)
    parser.add_argument("--n_layers",         type=int,   default=2)
    parser.add_argument("--dropout",          type=float, default=0.4)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--batch",            type=int,   default=32)
    parser.add_argument("--cutoff",           type=float, default=8.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--label_smoothing",  type=float, default=0.1)
    parser.add_argument("--model_type",       choices=["gat", "xattn"], default="gat")
    parser.add_argument("--tag",              type=str,   default=None)
    parser.add_argument("--graph_dir",        type=str,   default=None)
    parser.add_argument("--ffn",              action="store_true")
    parser.add_argument("--ligand_only",      action="store_true",
                        help="Ablation: skip receptor cross-attention, ligand embeddings only")
    parser.add_argument("--train_all",        action="store_true",
                        help="Train on full E13 (no CV), evaluate on held-out only")
    parser.add_argument("--save_model",       action="store_true",
                        help="Save trained model weights to models/ directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # source map for decoy detection
    src_map  = {int(r["id"]): r["source"]
                for r in csv.DictReader(open(LABEL_CSV))}
    meta_all = list(csv.DictReader(open(HOME / "data/meta.csv")))

    def is_decoy(row):
        return src_map.get(int(row["row_id"]), "") == "decoy_cross_class"

    HELD_OUT_RECS = {"P30988"}
    # ── Held-out test: ER04 + held-out receptor E13 samples (no decoy) ───────
    er04_rows = [r for r in meta_all
                 if (r["kind"] == "ER04" or
                     (r["kind"] == "E13" and r["name"].split("_")[0] in HELD_OUT_RECS))
                 and not is_decoy(r)]
    y_er04    = np.array([1 if r["label"] == "agonist" else 0 for r in er04_rows])
    print(f"ER04 held-out: {len(er04_rows)} samples "
          f"(agonist={int(y_er04.sum())}, nonagonist={int((y_er04==0).sum())})", flush=True)

    er04_ds     = AgonismDataset(rows=er04_rows, graph_dir=args.graph_dir)
    er04_loader = DataLoader(er04_ds, batch_size=args.batch, shuffle=False,
                             collate_fn=collate_skip_none,
                             num_workers=1, persistent_workers=True)

    # ── E13 training set (exclude RAMP-combining receptors kept for ER04) ────
    meta      = [r for r in meta_all if r["kind"] == "E13"
                 and r["name"].split("_")[0] not in HELD_OUT_RECS]
    N         = len(meta)
    y         = np.array([1 if r["label"] == "agonist" else 0 for r in meta])
    groups    = np.array([r["name"].split("_")[0] for r in meta])
    decoy_arr = np.array([is_decoy(r) for r in meta])
    print(f"E13 train: {N} samples  "
          f"(agonist={int(y.sum())}, nonagonist={int((y==0).sum())})", flush=True)
    print(f"  decoy={decoy_arr.sum()}  real={(~decoy_arr).sum()}", flush=True)

    er04_valid_pos = er04_ds.valid_positions  # rows with actual graph files

    if args.train_all:
        # ── Train-all: val split (fold 0) for early stopping ─────────────────
        skf_val = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
        tr_idx, val_idx = next(skf_val.split(np.zeros(N), y, groups))

        tr_ds  = AgonismDataset(rows=[meta[i] for i in tr_idx],  graph_dir=args.graph_dir)
        val_ds = AgonismDataset(rows=[meta[i] for i in val_idx], graph_dir=args.graph_dir)
        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                                collate_fn=collate_skip_none,
                                num_workers=1, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                collate_fn=collate_skip_none,
                                num_workers=1, persistent_workers=True)

        pos_rate   = y[tr_idx].mean()
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device)
        bce        = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        smooth     = args.label_smoothing

        def criterion(logits, labels):
            return bce(logits, labels * (1 - smooth) + smooth / 2)

        actual_vi  = np.array(tr_idx)[tr_ds.valid_positions] if len(tr_ds.valid_positions) else np.array([], dtype=int)
        actual_val = np.array(val_idx)[val_ds.valid_positions] if len(val_ds.valid_positions) else np.array([], dtype=int)
        val_real   = ~decoy_arr[actual_val] if len(actual_val) else np.array([], dtype=bool)

        er04_probs_sum = np.zeros(len(er04_rows), dtype=np.float64)
        saved_states   = []
        for m_idx in range(args.n_models):
            if args.model_type == "xattn":
                model = CrossAttnAgonism(
                    hidden_dim=args.hidden, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=args.dropout,
                    use_ffn=args.ffn, ligand_only=args.ligand_only,
                ).to(device)
            else:
                model = AgonismGNN(
                    hidden_dim=args.hidden, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=args.dropout,
                ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            best_auroc, best_state, no_improve = 0.0, None, 0
            for epoch in range(args.epochs):
                tr_loss = train_epoch(model, tr_loader, optimizer, criterion, device)
                scheduler.step()
                val_probs, val_labels = eval_loader(model, val_loader, device)
                if val_real.sum() > 1 and len(set(val_labels[val_real])) > 1:
                    auroc = roc_auc_score(val_labels[val_real], val_probs[val_real])
                else:
                    auroc = 0.0
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if (epoch + 1) % 10 == 0:
                    print(f"  model {m_idx} ep{epoch+1}: loss={tr_loss:.4f} val={auroc:.4f} best={best_auroc:.4f} p={no_improve}/{args.patience}", flush=True)
                if no_improve >= args.patience:
                    print(f"  model {m_idx} early stop ep{epoch+1}", flush=True)
                    break
            model.load_state_dict(best_state)
            saved_states.append(best_state)
            model.eval()
            with torch.no_grad():
                er04_p, _ = eval_loader(model, er04_loader, device)
            er04_probs_sum[er04_valid_pos] += er04_p

        # fallback: samples without graphs stay at 0.0 (nonagonist)
        n_fallback     = len(er04_rows) - len(er04_valid_pos)
        er04_probs_avg = er04_probs_sum / args.n_models
        er04_metrics   = compute_metrics(y_er04, er04_probs_avg)
        print(f"Held-out  AUROC={er04_metrics['auroc']:.4f}  AUPRC={er04_metrics['auprc']:.4f}  "
              f"F1={er04_metrics['f1']:.4f}  MCC={er04_metrics['mcc']:.4f}  "
              f"Prec={er04_metrics['precision']:.4f}  Rec={er04_metrics['recall']:.4f}  "
              f"(n={len(er04_rows)}, fallback={n_fallback})", flush=True)
        suffix   = f"_{args.tag}" if args.tag else ""
        out_path = RES_DIR / f"gnn_{args.model_type}{suffix}_trainall.json"
        json.dump({
            "config":        vars(args),
            "train_mode":    "all",
            "held_out_er04": {**er04_metrics, "n_samples": int(len(er04_rows)),
                              "n_fallback": int(n_fallback)},
        }, open(out_path, "w"), indent=2)
        print(f"Results saved to {out_path}")
        if args.save_model:
            model_name = f"gnn_{args.model_type}{suffix}"
            save_dir   = MODEL_DIR / model_name
            save_dir.mkdir(exist_ok=True)
            for i, state in enumerate(saved_states):
                torch.save(state, save_dir / f"model_{i}.pt")
            json.dump({
                "model_type": args.model_type,
                "hidden_dim": args.hidden,
                "n_heads":    args.n_heads,
                "n_layers":   args.n_layers,
                "dropout":    args.dropout,
                "ffn":        args.ffn,
                "ligand_only": args.ligand_only,
                "n_models":   args.n_models,
                "cutoff":     args.cutoff,
            }, open(save_dir / "config.json", "w"), indent=2)
            print(f"Model saved to {save_dir}/")
        return

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,
                               random_state=args.seed)

    fold_metrics   = []
    oof_probs      = np.full(N, np.nan, dtype=np.float32)
    er04_probs_sum = np.zeros(len(er04_rows), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(N), y, groups)):
        print(f"\n── Fold {fold} ──────────────────", flush=True)
        print(f"  train={len(tr_idx)}  val={len(val_idx)}", flush=True)

        tr_ds  = AgonismDataset(rows=[meta[i] for i in tr_idx],  graph_dir=args.graph_dir)
        val_ds = AgonismDataset(rows=[meta[i] for i in val_idx], graph_dir=args.graph_dir)
        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                                collate_fn=collate_skip_none,
                                num_workers=1, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                collate_fn=collate_skip_none,
                                num_workers=1, persistent_workers=True)

        pos_rate   = y[tr_idx].mean()
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device)
        bce        = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        smooth     = args.label_smoothing

        def criterion(logits, labels):
            return bce(logits, labels * (1 - smooth) + smooth / 2)

        fold_probs = []
        for m_idx in range(args.n_models):
            if args.model_type == "xattn":
                model = CrossAttnAgonism(
                    hidden_dim=args.hidden, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=args.dropout,
                    use_ffn=args.ffn,
                    ligand_only=args.ligand_only,
                ).to(device)
            else:
                model = AgonismGNN(
                    hidden_dim=args.hidden, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=args.dropout,
                ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            best_auroc, best_probs, best_state = 0.0, None, None
            for epoch in range(args.epochs):
                tr_loss          = train_epoch(model, tr_loader, optimizer, criterion, device)
                probs_v, labels_v = eval_loader(model, val_loader, device)
                scheduler.step()

                # track best by auroc on real val samples
                actual_vi = val_idx[val_ds.valid_positions]
                real_v    = ~decoy_arr[actual_vi]
                if real_v.sum() > 1 and len(set(labels_v[real_v])) > 1:
                    auroc = roc_auc_score(labels_v[real_v], probs_v[real_v])
                else:
                    auroc = 0.0
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_probs = probs_v.copy()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if (epoch + 1) % 10 == 0:
                    print(f"    model {m_idx} ep{epoch+1}: "
                          f"loss={tr_loss:.4f} auroc={auroc:.4f}", flush=True)

            fold_probs.append(best_probs)

            # accumulate ER04 predictions with best model state
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                er04_p, _ = eval_loader(model, er04_loader, device)
            er04_probs_sum[er04_valid_pos] += er04_p

        # ensemble over n_models
        ensemble_probs = np.mean(fold_probs, axis=0)
        actual_val_idx = val_idx[val_ds.valid_positions]
        oof_probs[actual_val_idx] = ensemble_probs

        # fold metrics on real val samples only
        real_v = ~decoy_arr[actual_val_idx]
        m      = compute_metrics(y[actual_val_idx][real_v], ensemble_probs[real_v])
        fold_metrics.append({"fold": fold, **m})
        print(f"  Fold {fold} ensemble: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
              f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  (real val={real_v.sum()})", flush=True)

    # ── 10-fold CV overall (real only) ────────────────────────────────────────
    real_mask  = ~decoy_arr & ~np.isnan(oof_probs)
    cv_metrics = compute_metrics(y[real_mask], oof_probs[real_mask])
    print(f"\n10-fold CV  AUROC={cv_metrics['auroc']:.4f}  AUPRC={cv_metrics['auprc']:.4f}  "
          f"F1={cv_metrics['f1']:.4f}  MCC={cv_metrics['mcc']:.4f}  "
          f"Prec={cv_metrics['precision']:.4f}  Rec={cv_metrics['recall']:.4f}  "
          f"(n={int(real_mask.sum())})", flush=True)

    # ── ER04 held-out ─────────────────────────────────────────────────────────
    # fallback: samples without graphs stay at 0.0 (nonagonist)
    n_accum    = args.n_folds * args.n_models
    n_fallback = len(er04_rows) - len(er04_valid_pos)
    er04_probs_avg = er04_probs_sum / n_accum
    er04_metrics = compute_metrics(y_er04, er04_probs_avg)
    print(f"ER04 held-out  AUROC={er04_metrics['auroc']:.4f}  AUPRC={er04_metrics['auprc']:.4f}  "
          f"F1={er04_metrics['f1']:.4f}  MCC={er04_metrics['mcc']:.4f}  "
          f"Prec={er04_metrics['precision']:.4f}  Rec={er04_metrics['recall']:.4f}  "
          f"(n={len(er04_rows)}, fallback={n_fallback})", flush=True)

    suffix   = f"_{args.tag}" if args.tag else ""
    out_path = RES_DIR / f"gnn_{args.model_type}{suffix}.json"
    json.dump({
        "config":        vars(args),
        "fold_metrics":  fold_metrics,
        "oof":           cv_metrics,
        "held_out_er04": {**er04_metrics, "n_samples": int(len(er04_rows)),
                          "n_fallback": int(n_fallback)},
    }, open(out_path, "w"), indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
