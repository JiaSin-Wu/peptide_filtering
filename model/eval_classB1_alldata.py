"""
比較兩種訓練策略在 Class B1 的 LOGO-CV 表現：
  A) B1-only:  訓練 = 其他 7 個 B1 受體
  B) All-data: 訓練 = 全部 1803 筆（含 Class A）去掉 held-out 受體
"""
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
y_all  = np.load(DATA_DIR / "y.npy")
meta   = list(csv.DictReader(open(DATA_DIR / "meta.csv")))
rec_all = np.array([r["name"].split("_")[0] for r in meta])

B1 = {"P47871","P41587","P30988","P43220","Q03431","Q13324","P32241","P41586"}

# both-labels filter (same as main training)
from collections import defaultdict
rec_labels = defaultdict(set)
for r in meta:
    rec_labels[r["name"].split("_")[0]].add(r["label"])
both_recs = {rec for rec, labs in rec_labels.items() if len(labs) == 2}
both_idx  = [i for i, r in enumerate(meta) if r["name"].split("_")[0] in both_recs]

idx_b1 = [i for i, r in enumerate(meta) if r["name"].split("_")[0] in B1]
X_b1   = X_all[idx_b1]
y_b1   = y_all[idx_b1]
g_b1   = rec_all[idx_b1]

def fit_lgbm(X_tr, y_tr):
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
    return clf

logo = LeaveOneGroupOut()
results = {}

for tr_b1, val_b1 in logo.split(X_b1, y_b1, g_b1):
    g = g_b1[val_b1[0]]
    X_val, y_val = X_b1[val_b1], y_b1[val_b1]
    if len(set(y_val)) < 2:
        continue

    # Strategy A: B1-only
    clf_a = fit_lgbm(X_b1[tr_b1], y_b1[tr_b1])
    prob_a = clf_a.predict_proba(X_val)[:, 1]
    auroc_a = roc_auc_score(y_val, prob_a)

    # Strategy B: All-data (both-labels, exclude held-out receptor)
    all_tr_idx = [i for i in both_idx if rec_all[i] != g]
    clf_b = fit_lgbm(X_all[all_tr_idx], y_all[all_tr_idx])
    prob_b = clf_b.predict_proba(X_val)[:, 1]
    auroc_b = roc_auc_score(y_val, prob_b)

    results[g] = (len(y_val), int(y_val.sum()), auroc_a, auroc_b)
    print(f"  {g}: n={len(y_val):3d}  B1-only={auroc_a:.4f}  All-data={auroc_b:.4f}  diff={auroc_b-auroc_a:+.4f}")

print()
# aggregate
probs_a, probs_b, ys = [], [], []
for tr_b1, val_b1 in logo.split(X_b1, y_b1, g_b1):
    g = g_b1[val_b1[0]]
    if g not in results:
        continue
    X_val, y_val = X_b1[val_b1], y_b1[val_b1]
    all_tr_idx = [i for i in both_idx if rec_all[i] != g]
    prob_a = fit_lgbm(X_b1[[i for i,gg in enumerate(g_b1) if gg != g]],
                      y_b1[[i for i,gg in enumerate(g_b1) if gg != g]]).predict_proba(X_val)[:,1]
    prob_b = fit_lgbm(X_all[all_tr_idx], y_all[all_tr_idx]).predict_proba(X_val)[:,1]
    probs_a.extend(prob_a); probs_b.extend(prob_b); ys.extend(y_val)

print(f"Class B1 LOGO  B1-only  AUROC = {roc_auc_score(ys, probs_a):.4f}")
print(f"Class B1 LOGO  All-data AUROC = {roc_auc_score(ys, probs_b):.4f}")
