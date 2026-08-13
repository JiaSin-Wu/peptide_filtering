"""
predict.py

Predict P(agonist) from one or more AF3 output directories.

Usage:
  python3 predict.py /path/to/af3_output_dir
  python3 predict.py /path/to/dir1 /path/to/dir2 ...
  python3 predict.py --model models/lgbm_default.pkl /path/to/dir
"""

import argparse
import json
import numpy as np
import joblib
from pathlib import Path


HOME      = Path("/home/jiasin/AMY123R_agonist_Design/model")
MODEL_DIR = HOME / "models"


FEATURE_EXTRACTORS = {
    "B_mean":     lambda sB, pAB, _cif_path: sB.mean(axis=0),
    "B_max":      lambda sB, pAB, _cif_path: sB.max(axis=0),
    "AB_mean":    lambda sB, pAB, _cif_path: pAB.reshape(-1, pAB.shape[-1]).mean(axis=0),
    "AB_max":     lambda sB, pAB, _cif_path: pAB.reshape(-1, pAB.shape[-1]).max(axis=0),
    "AB_mean_8A": None,   # handled separately (needs CIF)
    "AB_max_8A":  None,
}


def _parse_chain_lengths(data_json: Path) -> dict:
    d = json.load(open(data_json))
    return {s["protein"]["id"]: len(s["protein"]["sequence"])
            for s in d["sequences"] if "protein" in s}


def _contact_pair(pAB: np.ndarray, cif_path: Path, len_A: int, len_B: int,
                  cutoff: float = 8.0, reduce: str = "mean") -> np.ndarray:
    from gnn.graph import parse_cif_coords
    pAB_flat = pAB.reshape(-1, pAB.shape[-1])
    try:
        coords = parse_cif_coords(cif_path)
        cA, cB = coords.get("A"), coords.get("B")
        if cA is not None and cB is not None and len(cA) == len_A and len(cB) == len_B:
            dist = np.linalg.norm(cA[:, None, :] - cB[None, :, :], axis=2)
            mask = dist <= cutoff
            if mask.any():
                contact = pAB[mask].astype(np.float32)
                return contact.mean(axis=0) if reduce == "mean" else contact.max(axis=0)
    except Exception:
        pass
    return pAB_flat.mean(axis=0) if reduce == "mean" else pAB_flat.max(axis=0)


def extract_features(folder: Path, feat_keys: list[str]) -> np.ndarray | None:
    folder = Path(folder)
    emb_dir  = folder / "seed-1_embeddings"
    npz_list = list(emb_dir.glob("*_embeddings.npz")) if emb_dir.exists() else []
    djson    = list(folder.glob("*_data.json"))
    cif_list = list((folder / "seed-1_sample-0").glob("*_model.cif"))

    if not npz_list or not djson:
        missing = []
        if not npz_list: missing.append("embeddings.npz")
        if not djson:    missing.append("_data.json")
        print(f"  [SKIP] {folder.name}: missing {', '.join(missing)}")
        return None

    lens  = _parse_chain_lengths(djson[0])
    len_A = lens.get("A", 0)
    len_B = lens.get("B", 0)

    d      = np.load(npz_list[0])
    single = d["single_embeddings"].astype(np.float32)
    sB     = single[len_A : len_A + len_B]

    pair = d["pair_embeddings"]
    pAB  = pair[0:len_A, len_A:len_A + len_B, :].astype(np.float32)

    cif = cif_list[0] if cif_list else None
    parts = []
    for key in feat_keys:
        if key == "AB_mean_8A":
            parts.append(_contact_pair(pAB, cif, len_A, len_B, reduce="mean"))
        elif key == "AB_max_8A":
            parts.append(_contact_pair(pAB, cif, len_A, len_B, reduce="max"))
        else:
            parts.append(FEATURE_EXTRACTORS[key](sB, pAB, cif))

    return np.concatenate(parts).astype(np.float32)


def predict(folders: list, model_path: Path) -> list[dict]:
    bundle    = joblib.load(model_path)
    model     = bundle["model"]
    feat_keys = bundle["feat_keys"]
    print(f"Loaded model: {model_path.name}  features={feat_keys}")

    results = []
    for folder in folders:
        folder = Path(folder)
        feat = extract_features(folder, feat_keys)
        if feat is None:
            results.append({"folder": folder.name, "p_agonist": None})
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        results.append({"folder": folder.name, "p_agonist": prob})
        print(f"  {folder.name}: P(agonist) = {prob:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", help="AF3 output directory/directories")
    parser.add_argument("--model", default=str(MODEL_DIR / "lgbm_default.pkl"),
                        help="Path to saved model .pkl")
    parser.add_argument("--out", default=None, help="Save predictions to JSON file")
    args = parser.parse_args()

    results = predict(args.folders, Path(args.model))

    if args.out:
        import json
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
