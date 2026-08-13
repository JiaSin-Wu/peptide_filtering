"""
train_mlp.py

Train on E13 only (10-fold receptor-stratified CV).
  - Train fold : all samples including cross-class decoys
  - Val fold   : real labels only (IUPHAR + ChEMBL), decoys excluded from metrics
Held-out evaluation on ER04 (RAMP-included), cross-class decoys excluded.

Usage:
  python3 train_mlp.py
  python3 train_mlp.py --gated
"""

import numpy as np, csv, json, argparse
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef,
                             precision_score, recall_score)
import torch
import torch.nn as nn

HOME      = Path("/home/jiasin/AMY123R_agonist_Design/model")
RAW_DIR   = HOME / "data/raw"
DATA_DIR  = HOME / "data"
RES_DIR   = HOME / "results"
LABEL_CSV = Path("/home/jiasin/outputs/ml_label_final.csv")


class GatedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate   = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.norm(self.linear(x) * torch.nn.functional.silu(self.gate(x))))


class MLP(nn.Module):
    def __init__(self, input_dim=512, dropout=0.4, gated=False):
        super().__init__()
        block = (lambda i, o: GatedBlock(i, o, dropout)) if gated else \
                (lambda i, o: nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o),
                                            nn.ReLU(), nn.Dropout(dropout)))
        self.encoder = nn.Sequential(
            block(input_dim, 256),
            block(256, 128),
            block(128, 64),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def train_epoch(model, X, y, optimizer, criterion, batch_size=64, device="cpu"):
    model.train()
    idx = torch.randperm(len(X))
    total_loss = 0.0
    for start in range(0, len(X), batch_size):
        b = idx[start:start+batch_size]
        loss = criterion(model(X[b].to(device)), y[b].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(b)
    return total_loss / len(X)


@torch.no_grad()
def predict(model, X, batch_size=256, device="cpu"):
    model.eval()
    probs = []
    for start in range(0, len(X), batch_size):
        probs.append(torch.sigmoid(model(X[start:start+batch_size].to(device))).cpu())
    return torch.cat(probs).numpy()


def make_criterion(y_tr, device):
    pos_rate   = y_tr.mean().item()
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_metrics(y_true, probs):
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
    parser.add_argument("--gated",    action="store_true")
    parser.add_argument("--n_folds",  type=int, default=10)
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--dropout",  type=float, default=0.4)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--features", nargs="+",
                        default=["single_B_mean", "pair_AB_mean_8A"])
    parser.add_argument("--tag",       type=str, default=None)
    parser.add_argument("--train_all", action="store_true",
                        help="Train on full E13 (no CV), evaluate on held-out only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  gated={args.gated}  features={args.features}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # source map for decoy detection
    src_map  = {int(r["id"]): r["source"]
                for r in csv.DictReader(open(LABEL_CSV))}
    meta_all = list(csv.DictReader(open(DATA_DIR / "meta.csv")))

    def is_decoy(row):
        return src_map.get(int(row["row_id"]), "") == "decoy_cross_class"

    X_all_np = np.concatenate([
        np.load(RAW_DIR / f"{feat}.npy") for feat in args.features
    ], axis=1).astype(np.float32)
    y_all_np = np.load(DATA_DIR / "y.npy").astype(np.float32)

    HELD_OUT_RECS = {"P30988"}

    # ── Held-out test set: ER04 + held-out receptor E13 samples (no decoy) ───
    er04_idx = np.array([i for i, r in enumerate(meta_all)
                         if (r["kind"] == "ER04" or
                             (r["kind"] == "E13" and r["name"].split("_")[0] in HELD_OUT_RECS))
                         and not is_decoy(r)])
    y_er04   = y_all_np[er04_idx]
    print(f"Held-out test: {len(er04_idx)} samples "
          f"(agonist={int(y_er04.sum())}, nonagonist={int((y_er04==0).sum())})")

    # ── E13 training set (exclude RAMP-combining receptors kept for ER04) ────
    e13_mask  = np.array([r["kind"] == "E13" and r["name"].split("_")[0] not in HELD_OUT_RECS
                          for r in meta_all])
    meta_e13  = [r for r, k in zip(meta_all, e13_mask) if k]
    X_np      = X_all_np[e13_mask]
    y_np      = y_all_np[e13_mask]
    groups    = np.array([r["name"].split("_")[0] for r in meta_e13])
    decoy_e13 = np.array([is_decoy(r) for r in meta_e13])
    print(f"E13 train: {len(y_np)} samples  "
          f"(agonist={int(y_np.sum())}, nonagonist={int((y_np==0).sum())})")
    print(f"  decoy={decoy_e13.sum()}  real={(~decoy_e13).sum()}")

    # normalise on E13 training data
    mean   = X_np.mean(0); std = X_np.std(0) + 1e-8
    X_np   = (X_np - mean) / std
    X_er04_np = (X_all_np[er04_idx] - mean) / std

    X      = torch.from_numpy(X_np)
    y      = torch.from_numpy(y_np)
    X_er04 = torch.from_numpy(X_er04_np)

    if args.train_all:
        # ── Train-all: val split for early stopping ───────────────────────────
        base = "mlp_gated" if args.gated else "mlp"
        tag  = (args.tag if args.tag else base) + "_trainall"

        skf_val = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
        tr_idx, val_idx = next(skf_val.split(X_np, y_np, groups))
        val_real = ~decoy_e13[val_idx]

        X_tr, y_tr   = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model     = MLP(input_dim=X_np.shape[1], dropout=args.dropout, gated=args.gated).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = make_criterion(y_tr, device)

        best_auroc, best_state, no_improve = 0.0, None, 0
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, X_tr, y_tr, optimizer, criterion, device=device)
            scheduler.step()
            val_probs = predict(model, X_val, device=device)
            if val_real.sum() > 1 and len(set(y_np[val_idx][val_real])) > 1:
                auroc = roc_auc_score(y_np[val_idx][val_real], val_probs[val_real])
            else:
                auroc = 0.0
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 20 == 0:
                print(f"  ep{epoch} val_auroc={auroc:.4f} best={best_auroc:.4f} patience={no_improve}/{args.patience}", flush=True)
            if no_improve >= args.patience:
                print(f"  Early stop at ep{epoch} (no improvement for {args.patience} epochs)", flush=True)
                break

        model.load_state_dict(best_state)
        er04_probs   = predict(model, X_er04, device=device)
        er04_metrics = compute_metrics(y_er04, er04_probs)
        print(f"Held-out  AUROC={er04_metrics['auroc']:.4f}  AUPRC={er04_metrics['auprc']:.4f}  "
              f"F1={er04_metrics['f1']:.4f}  MCC={er04_metrics['mcc']:.4f}  "
              f"Prec={er04_metrics['precision']:.4f}  Rec={er04_metrics['recall']:.4f}  "
              f"Acc={er04_metrics['accuracy']:.4f}  (n={len(er04_idx)})", flush=True)
        result = {
            "model_type":    base,
            "train_mode":    "all",
            "features":      args.features,
            "config":        vars(args),
            "held_out_er04": {**er04_metrics, "n_samples": int(len(er04_idx))},
        }
        out_path = RES_DIR / f"{tag}.json"
        json.dump(result, open(out_path, "w"), indent=2)
        print(f"Results saved to {out_path}")
        return

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_probs      = np.full(len(y), np.nan, dtype=np.float32)
    er04_probs_sum = np.zeros(len(er04_idx), dtype=np.float64)
    fold_metrics   = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np, groups)):
        model     = MLP(input_dim=X_np.shape[1], dropout=args.dropout, gated=args.gated).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = make_criterion(y[tr_idx], device)

        best_auroc, best_state = 0.0, None
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, X[tr_idx], y[tr_idx], optimizer, criterion, device=device)
            scheduler.step()
            if epoch % 20 == 0 or epoch == args.epochs:
                # evaluate on real val samples only
                real    = ~decoy_e13[val_idx]
                probs_v = predict(model, X[val_idx], device=device)
                auroc   = roc_auc_score(y_np[val_idx][real], probs_v[real])
                print(f"  fold {fold} ep{epoch}: auroc={auroc:.4f}", flush=True)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs_val = predict(model, X[val_idx], device=device)
        oof_probs[val_idx] = probs_val
        er04_probs_sum    += predict(model, X_er04, device=device)

        real = ~decoy_e13[val_idx]
        m    = compute_metrics(y_np[val_idx][real], probs_val[real])
        fold_metrics.append({"fold": fold, **m})
        print(f"  Fold {fold}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
              f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  (real val={real.sum()})", flush=True)

    # ── 10-fold CV overall (real only) ────────────────────────────────────────
    real_mask  = ~decoy_e13 & ~np.isnan(oof_probs)
    cv_metrics = compute_metrics(y_np[real_mask], oof_probs[real_mask])
    print(f"\n10-fold CV  AUROC={cv_metrics['auroc']:.4f}  AUPRC={cv_metrics['auprc']:.4f}  "
          f"F1={cv_metrics['f1']:.4f}  MCC={cv_metrics['mcc']:.4f}  "
          f"Prec={cv_metrics['precision']:.4f}  Rec={cv_metrics['recall']:.4f}  "
          f"(n={int(real_mask.sum())})", flush=True)

    # ── ER04 held-out ─────────────────────────────────────────────────────────
    er04_probs   = er04_probs_sum / args.n_folds
    er04_metrics = compute_metrics(y_er04, er04_probs)
    print(f"ER04 held-out  AUROC={er04_metrics['auroc']:.4f}  AUPRC={er04_metrics['auprc']:.4f}  "
          f"F1={er04_metrics['f1']:.4f}  MCC={er04_metrics['mcc']:.4f}  "
          f"Prec={er04_metrics['precision']:.4f}  Rec={er04_metrics['recall']:.4f}  "
          f"(n={len(er04_idx)})", flush=True)

    base     = "mlp_gated" if args.gated else "mlp"
    tag      = args.tag if args.tag else base
    result   = {
        "model_type":    base,
        "features":      args.features,
        "n_folds":       args.n_folds,
        "n_samples":     int((~decoy_e13).sum()),
        "config":        vars(args),
        "fold_metrics":  fold_metrics,
        "oof":           cv_metrics,
        "held_out_er04": {**er04_metrics, "n_samples": int(len(er04_idx))},
    }
    out_path = RES_DIR / f"{tag}.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
