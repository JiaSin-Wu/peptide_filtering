"""Evaluate LightGBM (contact_only) on Class B1 receptors — leave-one-receptor-out CV."""
import numpy as np, csv
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

RAW_DIR  = Path("ml/data/raw")
DATA_DIR = Path("ml/data")

X_all = np.concatenate([
    np.load(RAW_DIR / "single_B_mean.npy"),
    np.load(RAW_DIR / "pair_AB_mean_8A.npy"),
], axis=1)
y_all = np.load(DATA_DIR / "y.npy")
meta  = list(csv.DictReader(open(DATA_DIR / "meta.csv")))

B1 = {"P47871","P41587","P30988","P43220","Q03431","Q13324","P32241","P41586"}
idx_b1 = [i for i, r in enumerate(meta) if r["name"].split("_")[0] in B1]

X      = X_all[idx_b1]
y      = y_all[idx_b1]
groups = np.array([meta[i]["name"].split("_")[0] for i in idx_b1])

print(f"Class B1: {len(y)} samples, {len(set(groups))} receptors")
print(f"Agonist={int(y.sum())}  Nonagonist={int(len(y)-y.sum())}")
print()

logo = LeaveOneGroupOut()
all_probs, all_y = [], []

for tr_idx, val_idx in logo.split(X, y, groups):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    g = groups[val_idx[0]]

    if len(set(y_val)) < 2:
        print(f"  {g}: skip (single class)")
        continue

    pos_w = (1 - y_tr.mean()) / y_tr.mean()
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=5,
        scale_pos_weight=pos_w,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    prob  = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, prob)
    auprc = average_precision_score(y_val, prob)
    print(f"  {g}: n={len(y_val):3d}  agonist={int(y_val.sum()):2d}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    all_probs.extend(prob.tolist())
    all_y.extend(y_val.tolist())

print()
print(f"Class B1  LOGO  AUROC = {roc_auc_score(all_y, all_probs):.4f}")
print(f"Class B1  LOGO  AUPRC = {average_precision_score(all_y, all_probs):.4f}")
print(f"All-class 10-fold     = 0.8322  (reference, contact_only)")
