"""
extract_features.py

Extract features from AF3 inference outputs (outputs/train/).
Saves to ml/data/raw/:
  single_A_mean.npy, single_A_max.npy  (N, 384)
  single_B_mean.npy, single_B_max.npy  (N, 384)
  pair_AB_mean.npy,  pair_AB_max.npy   (N, 128)

Also saves:
  ml/data/y.npy      (N,)  1=agonist, 0=nonagonist
  ml/data/meta.csv         sample metadata
"""

import csv
import json
import numpy as np
from pathlib import Path
from gnn.graph import parse_cif_coords

HOME       = Path("/home/jiasin")
OUTPUT_DIR = HOME / "outputs"
RAW_DIR    = HOME / "AMY123R_agonist_Design/model/data/raw"
DATA_DIR   = HOME / "AMY123R_agonist_Design/model/data"


def parse_folder(name: str) -> dict | None:
    """Parse folder name → meta dict. Returns None for unrecognised formats."""
    if name.endswith("_train_E13"):
        kind = "E13"
        stem = name[:-10]
    elif name.endswith("_train_ER04"):
        kind = "ER04"
        stem = name[:-11]
    else:
        return None

    # last two fields are always {label}_{row_id}
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None
    label, row_id = parts[-2], parts[-1]
    if label not in ("agonist", "nonagonist"):
        return None

    return {"name": name, "kind": kind, "label": label, "row_id": row_id}


def chain_lengths(data_json: Path) -> dict:
    """Return {chain_id: length} from _data.json."""
    d = json.load(open(data_json))
    return {s["protein"]["id"]: len(s["protein"]["sequence"])
            for s in d["sequences"] if "protein" in s}


def main():
    # ── collect valid samples ────────────────────────────────────────────────
    seen_row_ids = set()
    samples = []

    for folder in sorted(OUTPUT_DIR.iterdir()):
        if not folder.is_dir() or folder.name == "logs":
            continue

        meta = parse_folder(folder.name)
        if not meta:
            continue

        # skip duplicates (old-format folders share row_id with new-format)
        if meta["row_id"] in seen_row_ids:
            continue

        emb_dir = folder / "seed-1_embeddings"
        npz_list = list(emb_dir.glob("*_embeddings.npz")) if emb_dir.exists() else []
        djson    = list(folder.glob("*_data.json"))
        cif_list = list((folder / "seed-1_sample-0").glob("*_model.cif"))

        if not npz_list or not djson:
            missing = []
            if not npz_list: missing.append("npz")
            if not djson:    missing.append("data.json")
            print(f"  [SKIP] {folder.name}: missing {', '.join(missing)}")
            continue

        seen_row_ids.add(meta["row_id"])
        samples.append({
            **meta,
            "npz": npz_list[0],
            "data_json": djson[0],
            "cif": cif_list[0] if cif_list else None,
        })

    N = len(samples)
    print(f"Found {N} completed samples")

    # ── pre-allocate ─────────────────────────────────────────────────────────
    single_A_mean      = np.zeros((N, 384), dtype=np.float32)
    single_A_max       = np.zeros((N, 384), dtype=np.float32)
    single_B_mean      = np.zeros((N, 384), dtype=np.float32)
    single_B_max       = np.zeros((N, 384), dtype=np.float32)
    pair_AB_mean       = np.zeros((N, 128), dtype=np.float32)
    pair_AB_max        = np.zeros((N, 128), dtype=np.float32)
    pair_AB_mean_8A    = np.zeros((N, 128), dtype=np.float32)
    pair_AB_max_8A     = np.zeros((N, 128), dtype=np.float32)
    y                  = np.zeros(N, dtype=np.int32)

    meta_rows = []

    # ── extract ──────────────────────────────────────────────────────────────
    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f"  {i}/{N}...")

        lens  = chain_lengths(s["data_json"])
        len_A = lens.get("A", 0)
        len_B = lens.get("B", 0)

        d      = np.load(s["npz"])
        single = d["single_embeddings"].astype(np.float32)          # (T, 384)

        sA  = single[0:len_A]                                        # (len_A, 384)
        sB  = single[len_A : len_A + len_B]                         # (len_B, 384)

        # load only the A×B slice of pair_embeddings to save memory
        pair = d["pair_embeddings"]                                  # float16, (T, T, 128)
        pAB  = pair[0:len_A, len_A:len_A + len_B, :].astype(np.float32)  # (len_A, len_B, 128)
        pAB_flat = pAB.reshape(-1, 128)                              # (len_A*len_B, 128)

        single_A_mean[i] = sA.mean(axis=0)
        single_A_max[i]  = sA.max(axis=0)
        single_B_mean[i] = sB.mean(axis=0)
        single_B_max[i]  = sB.max(axis=0)
        pair_AB_mean[i]  = pAB_flat.mean(axis=0)
        pair_AB_max[i]   = pAB_flat.max(axis=0)

        # distance-filtered pair features (8Å cutoff)
        if s["cif"] is not None:
            try:
                coords = parse_cif_coords(s["cif"])
                cA = coords.get("A")
                cB = coords.get("B")
                if cA is not None and cB is not None and len(cA) == len_A and len(cB) == len_B:
                    dist = np.linalg.norm(
                        cA[:, None, :] - cB[None, :, :], axis=2)      # (len_A, len_B)
                    mask = dist <= 8.0                                  # (len_A, len_B)
                    if mask.any():
                        pAB_contact = pAB[mask].astype(np.float32)     # (n_contact, 128)
                        pair_AB_mean_8A[i] = pAB_contact.mean(axis=0)
                        pair_AB_max_8A[i]  = pAB_contact.max(axis=0)
                    else:
                        pair_AB_mean_8A[i] = pAB_flat.mean(axis=0)
                        pair_AB_max_8A[i]  = pAB_flat.max(axis=0)
                else:
                    pair_AB_mean_8A[i] = pAB_flat.mean(axis=0)
                    pair_AB_max_8A[i]  = pAB_flat.max(axis=0)
            except Exception:
                pair_AB_mean_8A[i] = pAB_flat.mean(axis=0)
                pair_AB_max_8A[i]  = pAB_flat.max(axis=0)
        else:
            pair_AB_mean_8A[i] = pAB_flat.mean(axis=0)
            pair_AB_max_8A[i]  = pAB_flat.max(axis=0)

        y[i] = 1 if s["label"] == "agonist" else 0

        meta_rows.append({
            "idx":    i,
            "name":   s["name"],
            "kind":   s["kind"],
            "label":  s["label"],
            "row_id": s["row_id"],
        })

    # ── save ─────────────────────────────────────────────────────────────────
    print("Saving...")
    np.save(RAW_DIR / "single_A_mean.npy",   single_A_mean)
    np.save(RAW_DIR / "single_A_max.npy",    single_A_max)
    np.save(RAW_DIR / "single_B_mean.npy",   single_B_mean)
    np.save(RAW_DIR / "single_B_max.npy",    single_B_max)
    np.save(RAW_DIR / "pair_AB_mean.npy",    pair_AB_mean)
    np.save(RAW_DIR / "pair_AB_max.npy",     pair_AB_max)
    np.save(RAW_DIR / "pair_AB_mean_8A.npy", pair_AB_mean_8A)
    np.save(RAW_DIR / "pair_AB_max_8A.npy",  pair_AB_max_8A)
    np.save(DATA_DIR / "y.npy", y)

    with open(DATA_DIR / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "name", "kind", "label", "row_id"])
        w.writeheader()
        w.writerows(meta_rows)

    agonist_count = int(y.sum())
    print(f"\nDone. N={N}  agonist={agonist_count}  nonagonist={N - agonist_count}")
    print(f"Output: {RAW_DIR}")


if __name__ == "__main__":
    main()
