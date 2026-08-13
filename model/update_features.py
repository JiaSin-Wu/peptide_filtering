"""
update_features.py

Partially re-extract features for a specific set of row_ids and update
the corresponding rows in the existing .npy arrays.

Usage:
  python3 update_features.py --row_ids 512 531 538 ...
  python3 update_features.py --row_ids_file ids.txt
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from gnn.graph import parse_cif_coords

HOME       = Path("/home/jiasin")
OUTPUT_DIR = HOME / "outputs"
RAW_DIR    = HOME / "AMY123R_agonist_Design/model/data/raw"
DATA_DIR   = HOME / "AMY123R_agonist_Design/model/data"


def chain_lengths(data_json: Path) -> dict:
    d = json.load(open(data_json))
    return {s["protein"]["id"]: len(s["protein"]["sequence"])
            for s in d["sequences"] if "protein" in s}


def extract_one(folder: Path, len_A: int, len_B: int, npz_path: Path, cif: Path | None):
    d      = np.load(npz_path)
    single = d["single_embeddings"].astype(np.float32)
    sA     = single[0:len_A]
    sB     = single[len_A : len_A + len_B]

    pair     = d["pair_embeddings"]
    pAB      = pair[0:len_A, len_A:len_A + len_B, :].astype(np.float32)
    pAB_flat = pAB.reshape(-1, 128)

    sA_mean = sA.mean(axis=0)
    sA_max  = sA.max(axis=0)
    sB_mean = sB.mean(axis=0)
    sB_max  = sB.max(axis=0)
    pAB_mean = pAB_flat.mean(axis=0)
    pAB_max  = pAB_flat.max(axis=0)

    # 8A contact features
    pAB_mean_8A = pAB_flat.mean(axis=0)
    pAB_max_8A  = pAB_flat.max(axis=0)

    if cif is not None:
        try:
            coords = parse_cif_coords(cif)
            cA = coords.get("A")
            cB = coords.get("B")
            if cA is not None and cB is not None and len(cA) == len_A and len(cB) == len_B:
                dist = np.linalg.norm(cA[:, None, :] - cB[None, :, :], axis=2)
                mask = dist <= 8.0
                if mask.any():
                    contact = pAB[mask].astype(np.float32)
                    pAB_mean_8A = contact.mean(axis=0)
                    pAB_max_8A  = contact.max(axis=0)
        except Exception as e:
            print(f"    [WARN] CIF parse failed: {e}")

    return {
        "single_A_mean":   sA_mean,
        "single_A_max":    sA_max,
        "single_B_mean":   sB_mean,
        "single_B_max":    sB_max,
        "pair_AB_mean":    pAB_mean,
        "pair_AB_max":     pAB_max,
        "pair_AB_mean_8A": pAB_mean_8A,
        "pair_AB_max_8A":  pAB_max_8A,
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--row_ids", nargs="+", type=int)
    group.add_argument("--row_ids_file", type=str)
    args = parser.parse_args()

    if args.row_ids_file:
        target_ids = set(int(x.strip()) for x in open(args.row_ids_file) if x.strip())
    else:
        target_ids = set(args.row_ids)

    print(f"Updating {len(target_ids)} row_ids: {sorted(target_ids)}")

    # load meta to find idx → folder mapping
    meta = list(csv.DictReader(open(DATA_DIR / "meta.csv")))
    targets = {int(r["row_id"]): int(r["idx"])
               for r in meta if int(r["row_id"]) in target_ids}

    if len(targets) != len(target_ids):
        missing = target_ids - set(targets.keys())
        print(f"[WARN] Not found in meta.csv: {sorted(missing)}")

    # load all arrays (mmap for memory efficiency)
    arrays = {
        "single_A_mean":   np.load(RAW_DIR / "single_A_mean.npy"),
        "single_A_max":    np.load(RAW_DIR / "single_A_max.npy"),
        "single_B_mean":   np.load(RAW_DIR / "single_B_mean.npy"),
        "single_B_max":    np.load(RAW_DIR / "single_B_max.npy"),
        "pair_AB_mean":    np.load(RAW_DIR / "pair_AB_mean.npy"),
        "pair_AB_max":     np.load(RAW_DIR / "pair_AB_max.npy"),
        "pair_AB_mean_8A": np.load(RAW_DIR / "pair_AB_mean_8A.npy"),
        "pair_AB_max_8A":  np.load(RAW_DIR / "pair_AB_max_8A.npy"),
    }

    # build name → row_id lookup from meta
    name_to_rid = {r["name"]: int(r["row_id"]) for r in meta}

    updated = 0
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        rid = name_to_rid.get(folder.name)
        if rid not in targets:
            continue

        idx = targets[rid]
        print(f"  [{updated+1}/{len(targets)}] row_id={rid}  idx={idx}  folder={folder.name}")

        emb_dir  = folder / "seed-1_embeddings"
        npz_list = list(emb_dir.glob("*_embeddings.npz")) if emb_dir.exists() else []
        djson    = list(folder.glob("*_data.json"))
        cif_list = list((folder / "seed-1_sample-0").glob("*_model.cif"))

        if not npz_list or not djson:
            missing = []
            if not npz_list: missing.append("npz")
            if not djson:    missing.append("data.json")
            print(f"    [SKIP] missing {', '.join(missing)}")
            continue

        lens  = chain_lengths(djson[0])
        len_A = lens.get("A", 0)
        len_B = lens.get("B", 0)
        cif   = cif_list[0] if cif_list else None

        feats = extract_one(folder, len_A, len_B, npz_list[0], cif)
        for key, vec in feats.items():
            arrays[key][idx] = vec

        updated += 1

    print(f"\nUpdated {updated}/{len(targets)} samples. Saving...")
    for key, arr in arrays.items():
        np.save(RAW_DIR / f"{key}.npy", arr)
        print(f"  saved {key}.npy")

    print("Done.")


if __name__ == "__main__":
    main()
