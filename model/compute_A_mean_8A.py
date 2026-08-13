"""
compute_A_mean_8A.py

Compute single_A_mean_8A.npy: mean of pocket residues' single embeddings
(only chain-A residues within 8Å of any chain-B residue).

Usage:
  python3 -m ml.compute_A_mean_8A
"""

import json
import numpy as np
from pathlib import Path
from gnn.graph import parse_cif_coords, get_pocket_residue_indices

HOME       = Path("/home/jiasin")
OUTPUT_DIR = HOME / "outputs"
RAW_DIR    = HOME / "AMY123R_agonist_Design/model/data/raw"
DATA_DIR   = HOME / "AMY123R_agonist_Design/model/data"
CUTOFF     = 8.0


def main():
    import csv

    meta_path = DATA_DIR / "meta.csv"
    rows = list(csv.DictReader(open(meta_path)))
    N = len(rows)
    print(f"Samples: {N}")

    single_A_mean_8A = np.zeros((N, 384), dtype=np.float32)

    # fallback: existing A_mean (used when CIF/coords unavailable)
    A_mean_fallback = np.load(RAW_DIR / "single_A_mean.npy")

    for row in rows:
        i = int(row["idx"])
        if i % 200 == 0:
            print(f"  {i}/{N}...")

        folder = OUTPUT_DIR / row["name"]
        emb_dir  = folder / "seed-1_embeddings"
        npz_list = list(emb_dir.glob("*_embeddings.npz")) if emb_dir.exists() else []
        djson    = list(folder.glob("*_data.json"))
        cif_list = list((folder / "seed-1_sample-0").glob("*_model.cif"))

        if not npz_list:
            raise FileNotFoundError(f"Error: Missing npz embeddings in {folder}")
        if not djson:
            raise FileNotFoundError(f"Error: Missing JSON data in {folder}")

        d = json.load(open(djson[0]))
        seqs = {s["protein"]["id"]: s["protein"]["sequence"]
                for s in d["sequences"] if "protein" in s}
        len_A = len(seqs.get("A", ""))
        len_B = len(seqs.get("B", ""))

        if len_A == 0 or len_B == 0:
            raise ValueError(f"Error: Empty sequence for Chain A or B in {row['name']}")

        npz = np.load(npz_list[0])
        sA  = npz["single_embeddings"].astype(np.float32)[0:len_A]  # (len_A, 384)

        if cif_list:
            try:
                coords = parse_cif_coords(cif_list[0])
                cA = coords.get("A")
                cB = coords.get("B")
                if (cA is not None and cB is not None
                        and len(cA) == len_A and len(cB) == len_B):
                    pocket_idx = get_pocket_residue_indices(cA, cB, CUTOFF)
                    if len(pocket_idx) > 0:
                        single_A_mean_8A[i] = sA[pocket_idx].mean(axis=0)
                    else:
                        single_A_mean_8A[i] = sA.mean(axis=0)
                else:
                    single_A_mean_8A[i] = sA.mean(axis=0)
            except Exception:
                single_A_mean_8A[i] = sA.mean(axis=0)
        else:
            single_A_mean_8A[i] = sA.mean(axis=0)

    out = RAW_DIR / "single_A_mean_8A.npy"
    np.save(out, single_A_mean_8A)
    print(f"Saved → {out}  shape={single_A_mean_8A.shape}")


if __name__ == "__main__":
    main()
