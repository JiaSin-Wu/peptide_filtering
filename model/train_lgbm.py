"""
train_lgbm.py

Train on E13 only (10-fold receptor-stratified CV).
  - Train fold : all samples including cross-class decoys
  - Val fold   : real labels only (IUPHAR + ChEMBL), decoys excluded from metrics
Held-out evaluation on ER04 (RAMP-included), cross-class decoys excluded.

Usage:
  python3 train_lgbm.py
  python3 train_lgbm.py --features all_8A
"""

import argparse, csv, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef,
                             precision_score, recall_score)
import lightgbm as lgb
import joblib

HOME      = Path("/home/jiasin/AMY123R_agonist_Design/model")
RAW_DIR   = HOME / "data/raw"
DATA_DIR  = HOME / "data"
RES_DIR   = HOME / "results"
MODEL_DIR = HOME / "models"
LABEL_CSV = Path("/home/jiasin/outputs/ml_label_final.csv")
RES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_FILES = {
    "A_mean":     RAW_DIR / "single_A_mean.npy",
    "A_max":      RAW_DIR / "single_A_max.npy",
    "A_mean_8A":  RAW_DIR / "single_A_mean_8A.npy",
    "B_mean":     RAW_DIR / "single_B_mean.npy",
    "B_max":      RAW_DIR / "single_B_max.npy",
    "AB_mean":    RAW_DIR / "pair_AB_mean.npy",
    "AB_max":     RAW_DIR / "pair_AB_max.npy",
    "AB_mean_8A": RAW_DIR / "pair_AB_mean_8A.npy",
    "AB_max_8A":  RAW_DIR / "pair_AB_max_8A.npy",
}

PRESETS = {
    "default":      ["B_mean", "AB_mean"],
    "no_pair":      ["A_mean", "B_mean"],
    "all":          ["A_mean", "A_max", "B_mean", "B_max", "AB_mean", "AB_max"],
    "all_8A":       ["A_mean", "A_max", "B_mean", "B_max", "AB_mean_8A", "AB_max_8A"],
    "contact_only": ["B_mean", "AB_mean_8A"],
    "pocket_mean":  ["A_mean_8A", "B_mean", "AB_mean_8A"],
}


def load_features(keys):
    return np.concatenate([np.load(FEATURE_FILES[k]) for k in keys], axis=1).astype(np.float32)


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


def load_source_map():
    """row_id -> source string from ml_label_final.csv"""
    return {int(r["id"]): r["source"]
            for r in csv.DictReader(open(LABEL_CSV))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="default")
    parser.add_argument("--n_folds",    type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--split",      default="receptor",
                        choices=["random", "receptor"])
    parser.add_argument("--train_all",  action="store_true",
                        help="Train on full E13 (no CV), evaluate on held-out only")
    parser.add_argument("--save_model", action="store_true",
                        help="Save trained model to models/ directory")
    args = parser.parse_args()

    feat_keys = PRESETS.get(args.features,
                            [f.strip() for f in args.features.split(",")])
    print(f"Features: {feat_keys}  split={args.split}")

    X_all    = load_features(feat_keys)
    y_all    = np.load(DATA_DIR / "y.npy")
    meta_all = list(csv.DictReader(open(DATA_DIR / "meta.csv")))
    src_map  = load_source_map()

    def is_decoy(row):
        return src_map.get(int(row["row_id"]), "") == "decoy_cross_class"

    HELD_OUT_RECS = {"P30988"}

    # ── Held-out test: ER04 + held-out receptor E13 samples (no decoy) ───────
    er04_idx = np.array([i for i, r in enumerate(meta_all)
                         if (r["kind"] == "ER04" or
                             (r["kind"] == "E13" and r["name"].split("_")[0] in HELD_OUT_RECS))
                         and not is_decoy(r)])
    X_er04   = X_all[er04_idx]
    y_er04   = y_all[er04_idx]
    print(f"Held-out test: {len(er04_idx)} samples "
          f"(agonist={int(y_er04.sum())}, nonagonist={int((y_er04==0).sum())})")

    # ── E13 training set (exclude RAMP-combining receptors kept for ER04) ────
    e13_mask  = np.array([r["kind"] == "E13" and r["name"].split("_")[0] not in HELD_OUT_RECS
                          for r in meta_all])
    X         = X_all[e13_mask]
    y         = y_all[e13_mask]
    meta_e13  = [r for r, k in zip(meta_all, e13_mask) if k]
    groups    = np.array([r["name"].split("_")[0] for r in meta_e13])
    decoy_e13 = np.array([is_decoy(r) for r in meta_e13])
    print(f"E13 train: {len(y)} samples  "
          f"(agonist={int(y.sum())}, nonagonist={int((y==0).sum())})")
    print(f"  decoy={decoy_e13.sum()}  real={(~decoy_e13).sum()}")
    print(f"Unique receptors: {len(set(groups))}")

    if args.train_all:
        # ── Train-all: val split for early stopping ───────────────────────────
        skf_val = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
        tr_idx, val_idx = next(skf_val.split(X, y, groups))
        val_real = ~decoy_e13[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=args.seed, verbose=-1,
        )
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx][val_real], y[val_idx][val_real])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)],
        )
        best_n = model.best_iteration_
        print(f"Best n_estimators: {best_n}")
        er04_probs   = model.predict_proba(X_er04)[:, 1]
        er04_metrics = compute_metrics(y_er04, er04_probs)
        print(f"Held-out  AUROC={er04_metrics['auroc']:.4f}  AUPRC={er04_metrics['auprc']:.4f}  "
              f"F1={er04_metrics['f1']:.4f}  MCC={er04_metrics['mcc']:.4f}  "
              f"Prec={er04_metrics['precision']:.4f}  Rec={er04_metrics['recall']:.4f}  "
              f"Acc={er04_metrics['accuracy']:.4f}  (n={len(er04_idx)})")
        tag = f"{args.features.replace(',', '+')}_trainall"
        out = {
            "model_type":    "lgbm",
            "train_mode":    "all",
            "features":      feat_keys,
            "config":        {"best_n_estimators": best_n},
            "held_out_er04": {**er04_metrics, "n_samples": int(len(er04_idx))},
        }
        out_path = RES_DIR / f"lgbm_{tag}.json"
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"Results saved to {out_path}")
        if args.save_model:
            model_path = MODEL_DIR / f"lgbm_{args.features}.pkl"
            joblib.dump({"model": model, "feat_keys": feat_keys}, model_path)
            print(f"Model saved to {model_path}")
        return

    cv = (StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
          if args.split == "receptor"
          else StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed))
    split_iter = (cv.split(X, y, groups) if args.split == "receptor"
                  else cv.split(X, y))

    fold_metrics   = []
    oof_probs      = np.full(len(y), np.nan, dtype=np.float32)
    er04_probs_sum = np.zeros(len(er04_idx), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(split_iter):
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=args.seed, verbose=-1,
        )
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        prob = model.predict_proba(X[val_idx])[:, 1]
        oof_probs[val_idx] = prob
        er04_probs_sum += model.predict_proba(X_er04)[:, 1]

        # metrics on real (non-decoy) val samples only
        real = ~decoy_e13[val_idx]
        m = compute_metrics(y[val_idx][real], prob[real])
        fold_metrics.append({"fold": fold, **m})
        print(f"  Fold {fold}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
              f"F1={m['f1']:.4f}  (real val={real.sum()})")

    # ── 10-fold CV overall (real samples only) ────────────────────────────────
    real_mask  = ~decoy_e13 & ~np.isnan(oof_probs)
    cv_metrics = compute_metrics(y[real_mask], oof_probs[real_mask])
    print(f"\n10-fold CV  AUROC={cv_metrics['auroc']:.4f}  AUPRC={cv_metrics['auprc']:.4f}  "
          f"F1={cv_metrics['f1']:.4f}  MCC={cv_metrics['mcc']:.4f}  "
          f"Prec={cv_metrics['precision']:.4f}  Rec={cv_metrics['recall']:.4f}  "
          f"(n={int(real_mask.sum())})")

    # ── ER04 held-out ─────────────────────────────────────────────────────────
    er04_probs   = er04_probs_sum / args.n_folds
    er04_metrics = compute_metrics(y_er04, er04_probs)
    print(f"ER04 held-out  AUROC={er04_metrics['auroc']:.4f}  AUPRC={er04_metrics['auprc']:.4f}  "
          f"F1={er04_metrics['f1']:.4f}  MCC={er04_metrics['mcc']:.4f}  "
          f"Prec={er04_metrics['precision']:.4f}  Rec={er04_metrics['recall']:.4f}  "
          f"(n={len(er04_idx)})")

    tag = f"{args.features.replace(',', '+')}_{args.split}"
    out = {
        "features":      feat_keys,
        "split":         args.split,
        "n_folds":       args.n_folds,
        "n_samples":     int((~decoy_e13).sum()),
        "fold_metrics":  fold_metrics,
        "oof":           cv_metrics,
        "held_out_er04": {**er04_metrics, "n_samples": int(len(er04_idx))},
    }
    out_path = RES_DIR / f"lgbm_{tag}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
