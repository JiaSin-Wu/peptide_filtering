"""
train_mlp_transfer.py

Transfer learning MLP for Class B1 agonism prediction.

Stage 1: Train MLP on all 1803 both-labels samples
         512 → 256 → 128 → 64 → 1
Stage 2: Freeze layers 1-3, fine-tune last layer on Class B1 (LOGO-CV)

Usage:
  python3 -m ml.train_mlp_transfer
"""

import numpy as np, csv, json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score
import torch
import torch.nn as nn

HOME      = Path("/home/jiasin/AMY123R_agonist_Design/model")
RAW_DIR   = HOME / "data/raw"
DATA_DIR  = HOME / "data"
RES_DIR   = HOME / "results"
LABEL_CSV = Path("/home/jiasin/outputs/ml_label_final.csv")

B1 = {"P47871","P41587","P30988","P43220","Q03431","Q13324","P32241","P41586"}
HELD_OUT_RECS = {"P30988"}


class GatedBlock(nn.Module):
    """GLU-style gated block: output = Linear(x) * sigmoid(Gate(x))"""
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

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def train_epoch(model, X, y, optimizer, criterion, batch_size=64, device="cpu"):
    model.train()
    idx = torch.randperm(len(X))
    total_loss = 0.0
    for start in range(0, len(X), batch_size):
        b = idx[start:start+batch_size]
        xb = X[b].to(device)
        yb = y[b].to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(b)
    return total_loss / len(X)


@torch.no_grad()
def predict(model, X, batch_size=256, device="cpu"):
    model.eval()
    probs = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start+batch_size].to(device)
        probs.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(probs).numpy()


def make_criterion(y_tr, device):
    pos_rate = y_tr.mean().item()
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_metrics(y_true, probs):
    pred = (np.array(probs) >= 0.5).astype(int)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated",     action="store_true", help="Use GLU gating in MLP")
    parser.add_argument("--train_all", action="store_true",
                        help="Train on full E13+B1 (no LOGO), evaluate on held-out RAMP")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  gated={args.gated}  train_all={args.train_all}")

    src_map = {int(r["id"]): r["source"] for r in csv.DictReader(open(LABEL_CSV))}

    def is_decoy(row):
        return src_map.get(int(row["row_id"]), "") == "decoy_cross_class"

    # ── load features ────────────────────────────────────────────────────────
    X_np = np.concatenate([
        np.load(RAW_DIR / "single_B_mean.npy"),
        np.load(RAW_DIR / "pair_AB_mean_8A.npy"),
    ], axis=1).astype(np.float32)
    y_np   = np.load(DATA_DIR / "y.npy").astype(np.float32)
    meta   = list(csv.DictReader(open(DATA_DIR / "meta.csv")))
    rec_np = np.array([r["name"].split("_")[0] for r in meta])

    # ── train_all mode ───────────────────────────────────────────────────────
    if args.train_all:
        # Stage 1: all E13 (exclude HELD_OUT_RECS), decoys included for training
        e13_idx = np.array([i for i, r in enumerate(meta)
                            if r["kind"] == "E13"
                            and r["name"].split("_")[0] not in HELD_OUT_RECS])
        # ER04 held-out: no decoys
        er04_idx = np.array([i for i, r in enumerate(meta)
                             if r["kind"] == "ER04" and not is_decoy(r)])
        y_er04 = y_np[er04_idx]
        print(f"Stage 1 train: {len(e13_idx)} E13 samples")
        print(f"ER04 held-out: {len(er04_idx)} samples "
              f"(agonist={int(y_er04.sum())}, nonagonist={int((y_er04==0).sum())})")

        # normalise on E13 training data
        mean = X_np[e13_idx].mean(0); std = X_np[e13_idx].std(0) + 1e-8
        X_s1   = torch.from_numpy((X_np[e13_idx] - mean) / std)
        y_s1   = torch.from_numpy(y_np[e13_idx])
        X_er04 = torch.from_numpy((X_np[er04_idx] - mean) / std)

        torch.manual_seed(42)
        model = MLP(input_dim=X_np.shape[1], gated=args.gated).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = make_criterion(y_s1, device)
        for epoch in range(1, 101):
            train_epoch(model, X_s1, y_s1, optimizer, criterion, device=device)
            scheduler.step()
            if epoch % 20 == 0:
                print(f"  Stage1 ep{epoch} done", flush=True)

        # no fine-tune evaluation
        prob_noft = predict(model, X_er04, device=device)

        # Stage 2: fine-tune head on all B1 (including P30988 E13 samples)
        b1_e13_idx = np.array([i for i, r in enumerate(meta)
                               if r["kind"] == "E13" and r["name"].split("_")[0] in B1])
        X_b1 = torch.from_numpy((X_np[b1_e13_idx] - mean) / std)
        y_b1 = torch.from_numpy(y_np[b1_e13_idx])
        print(f"Stage 2 fine-tune: {len(b1_e13_idx)} B1 samples")

        model.freeze_encoder()
        opt_ft  = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-3)
        crit_ft = make_criterion(y_b1, device)
        for _ in range(50):
            train_epoch(model, X_b1, y_b1, opt_ft, crit_ft, device=device)

        prob_ft = predict(model, X_er04, device=device)

        m_noft = compute_metrics(y_er04, prob_noft)
        m_ft   = compute_metrics(y_er04, prob_ft)
        print(f"Held-out no-finetune  AUROC={m_noft['auroc']:.4f}  MCC={m_noft['mcc']:.4f}")
        print(f"Held-out fine-tuned   AUROC={m_ft['auroc']:.4f}  MCC={m_ft['mcc']:.4f}")

        tag = "mlp_transfer_b1_gated" if args.gated else "mlp_transfer_b1"
        result = {
            "train_mode": "all",
            "held_out_er04_no_finetune": {**m_noft, "n_samples": int(len(er04_idx))},
            "held_out_er04_finetune":    {**m_ft,   "n_samples": int(len(er04_idx))},
            # keep flat keys for generate_comparison.py compatibility
            "auroc_no_finetune": m_noft["auroc"], "auprc_no_finetune": m_noft["auprc"],
            "f1_no_finetune":    m_noft["f1"],    "mcc_no_finetune":   m_noft["mcc"],
            "precision_no_finetune": m_noft["precision"], "recall_no_finetune": m_noft["recall"],
            "auroc_finetune":    m_ft["auroc"],   "auprc_finetune":    m_ft["auprc"],
            "f1_finetune":       m_ft["f1"],      "mcc_finetune":      m_ft["mcc"],
            "precision_finetune": m_ft["precision"], "recall_finetune": m_ft["recall"],
        }
        json.dump(result, open(RES_DIR / f"{tag}_trainall.json", "w"), indent=2)
        print(f"Results saved to {RES_DIR / f'{tag}_trainall.json'}")
        return

    # normalise features (fit on all data, no leakage since these are fixed AF3 outputs)
    mean = X_np.mean(0); std = X_np.std(0) + 1e-8
    X_np = (X_np - mean) / std

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    # both-labels filter
    rec_labs = defaultdict(set)
    for r in meta:
        rec_labs[r["name"].split("_")[0]].add(r["label"])
    both_recs = {rec for rec, labs in rec_labs.items() if len(labs) == 2}
    both_idx  = np.array([i for i, r in enumerate(meta)
                          if r["name"].split("_")[0] in both_recs])

    X_both = X[both_idx]; y_both = y[both_idx]; rec_both = rec_np[both_idx]
    print(f"Stage 1 data: {len(both_idx)} samples, {len(both_recs)} receptors")

    # ── Nested LOGO-CV ────────────────────────────────────────────────────────
    # For each held-out B1 receptor:
    #   Stage 1: train on all both-labels EXCEPT held-out receptor
    #   Stage 2: fine-tune last layer on remaining B1 receptors
    #   Eval:    predict held-out receptor
    print("\n── Nested LOGO-CV (proper transfer learning evaluation) ──")

    idx_b1 = np.array([i for i, r in enumerate(meta)
                       if r["name"].split("_")[0] in B1])
    X_b1 = X[idx_b1]; y_b1 = y[idx_b1]; g_b1 = rec_np[idx_b1]

    logo = LeaveOneGroupOut()
    all_probs_noft, all_probs_ft, all_y = [], [], []

    for tr_b1_idx, val_idx in logo.split(X_b1, y_b1.numpy(), g_b1):
        g = g_b1[val_idx[0]]
        X_val = X_b1[val_idx]; y_val = y_b1[val_idx]
        if len(set(y_val.numpy().tolist())) < 2:
            continue

        # Stage 1 training set: all both-labels EXCEPT held-out receptor
        s1_idx = np.array([i for i in both_idx if rec_np[i] != g])
        X_s1 = X[s1_idx]; y_s1 = y[s1_idx]

        # Stage 2 fine-tune set: other B1 receptors (not held-out)
        X_b1_tr = X_b1[tr_b1_idx]; y_b1_tr = y_b1[tr_b1_idx]

        # ── Stage 1 ──
        torch.manual_seed(42)
        model = MLP(input_dim=X_np.shape[1], gated=args.gated).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = make_criterion(y_s1, device)

        best_loss, best_state = float("inf"), None
        for epoch in range(1, 101):
            loss = train_epoch(model, X_s1, y_s1, optimizer, criterion, device=device)
            scheduler.step()
            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        # no fine-tune: Stage 1 weights applied directly to B1
        prob_noft = predict(model, X_val, device=device)

        # ── Stage 2: fine-tune last layer on other B1 ──
        model.freeze_encoder()
        opt_ft   = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-3)
        crit_ft  = make_criterion(y_b1_tr, device)
        for _ in range(50):
            train_epoch(model, X_b1_tr, y_b1_tr, opt_ft, crit_ft, device=device)
        prob_ft = predict(model, X_val, device=device)

        auroc_noft = roc_auc_score(y_val.numpy(), prob_noft)
        auroc_ft   = roc_auc_score(y_val.numpy(), prob_ft)
        print(f"  {g}: n={len(y_val):3d}  s1_train={len(s1_idx)}  "
              f"no-ft={auroc_noft:.4f}  fine-tune={auroc_ft:.4f}  diff={auroc_ft-auroc_noft:+.4f}")

        all_probs_noft.extend(prob_noft.tolist())
        all_probs_ft.extend(prob_ft.tolist())
        all_y.extend(y_val.numpy().tolist())

    pred_noft  = [1 if p >= 0.5 else 0 for p in all_probs_noft]
    pred_ft    = [1 if p >= 0.5 else 0 for p in all_probs_ft]
    auroc_noft = roc_auc_score(all_y, all_probs_noft)
    auroc_ft   = roc_auc_score(all_y, all_probs_ft)
    auprc_noft = average_precision_score(all_y, all_probs_noft)
    auprc_ft   = average_precision_score(all_y, all_probs_ft)
    f1_noft    = f1_score(all_y, pred_noft)
    f1_ft      = f1_score(all_y, pred_ft)
    mcc_noft   = matthews_corrcoef(all_y, pred_noft)
    mcc_ft     = matthews_corrcoef(all_y, pred_ft)
    prec_noft  = precision_score(all_y, pred_noft, zero_division=0)
    prec_ft    = precision_score(all_y, pred_ft, zero_division=0)
    rec_noft   = recall_score(all_y, pred_noft, zero_division=0)
    rec_ft     = recall_score(all_y, pred_ft, zero_division=0)
    print(f"\nClass B1 nested-LOGO  no-finetune  AUROC={auroc_noft:.4f}  AUPRC={auprc_noft:.4f}  F1={f1_noft:.4f}  MCC={mcc_noft:.4f}  Prec={prec_noft:.4f}  Rec={rec_noft:.4f}")
    print(f"Class B1 nested-LOGO  fine-tuned   AUROC={auroc_ft:.4f}  AUPRC={auprc_ft:.4f}  F1={f1_ft:.4f}  MCC={mcc_ft:.4f}  Prec={prec_ft:.4f}  Rec={rec_ft:.4f}")

    tag = "mlp_transfer_b1_gated" if args.gated else "mlp_transfer_b1"
    json.dump({
        "auroc_no_finetune": auroc_noft, "auprc_no_finetune": auprc_noft,
        "f1_no_finetune":    f1_noft,    "mcc_no_finetune":   mcc_noft,
        "precision_no_finetune": prec_noft, "recall_no_finetune": rec_noft,
        "auroc_finetune":    auroc_ft,   "auprc_finetune":    auprc_ft,
        "f1_finetune":       f1_ft,      "mcc_finetune":      mcc_ft,
        "precision_finetune": prec_ft,   "recall_finetune":   rec_ft,
    }, open(RES_DIR / f"{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
