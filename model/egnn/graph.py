"""
egnn/graph.py

Build PyTorch Geometric graphs with 3D coordinates (pos) for EGNN.
Same pocket/edge logic as gnn/graph.py but also stores Cα positions.

Graph structure:
  pocket.x   : AF3 single embeddings (384-dim)
  pocket.pos  : Cα coordinates (3-dim)
  ligand.x    : AF3 single embeddings (384-dim)
  ligand.pos  : Cα coordinates (3-dim)
  edge_attr   : AF3 pair embeddings (128-dim)
"""

import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path


def parse_cif_coords(cif_path: Path) -> dict[str, np.ndarray]:
    """Parse mmCIF atom_site records.
    Returns {chain_id: (n_residues, 3)} Cα coords (first atom if no CA).
    """
    chain_res: dict[str, dict[int, dict]] = {}
    with open(cif_path) as f:
        in_atom = False
        col: dict[str, int] = {}
        for line in f:
            line = line.rstrip()
            if line.startswith("_atom_site."):
                col[line.split(".")[1]] = len(col)
                in_atom = True
                continue
            if in_atom and (line.startswith("ATOM") or line.startswith("HETATM")):
                parts = line.split()
                atom_name  = parts[col["label_atom_id"]]
                chain      = parts[col["label_asym_id"]]
                try:
                    seq_id = int(parts[col["label_seq_id"]])
                except ValueError:
                    continue
                x = float(parts[col["Cartn_x"]])
                y = float(parts[col["Cartn_y"]])
                z = float(parts[col["Cartn_z"]])
                chain_res.setdefault(chain, {})
                res = chain_res[chain].setdefault(seq_id, {"ca": None, "first": None})
                if atom_name == "CA":
                    res["ca"] = [x, y, z]
                if res["first"] is None:
                    res["first"] = [x, y, z]

    result = {}
    for chain, res_dict in chain_res.items():
        coords = []
        for sid in sorted(res_dict.keys()):
            r = res_dict[sid]
            coords.append(r["ca"] if r["ca"] is not None else r["first"])
        result[chain] = np.array(coords, dtype=np.float32)
    return result


def build_graph(sample_dir: Path, cutoff: float = 8.0) -> HeteroData | None:
    emb_dir  = sample_dir / "seed-1_embeddings"
    npz_list = list(emb_dir.glob("*_embeddings.npz")) if emb_dir.exists() else []
    cif_list = list((sample_dir / "seed-1_sample-0").glob("*_model.cif"))
    djson    = list(sample_dir.glob("*_data.json"))

    if not npz_list or not cif_list or not djson:
        return None

    d = json.load(open(djson[0]))
    seqs = {s["protein"]["id"]: s["protein"]["sequence"]
            for s in d["sequences"] if "protein" in s}
    seq_A, seq_B = seqs.get("A", ""), seqs.get("B", "")
    len_A, len_B = len(seq_A), len(seq_B)
    if len_A == 0 or len_B == 0:
        return None

    coords = parse_cif_coords(cif_list[0])
    coords_A = coords.get("A")
    coords_B = coords.get("B")
    if coords_A is None or coords_B is None:
        return None
    if len(coords_A) != len_A or len(coords_B) != len_B:
        return None  # skip if coord/sequence length mismatch

    # pocket residues within cutoff of any ligand residue
    diff      = coords_A[:, None, :] - coords_B[None, :, :]   # (A, B, 3)
    dist_AB   = np.linalg.norm(diff, axis=2)                   # (A, B)
    pocket_idx = np.where(dist_AB.min(axis=1) <= cutoff)[0]
    if len(pocket_idx) == 0:
        pocket_idx = np.arange(min(len_A, 30))

    # edges: pocket-ligand pairs within cutoff
    dist_pl  = dist_AB[pocket_idx]                             # (K, B)
    edge_mask = dist_pl <= cutoff
    src_p, dst_l = np.where(edge_mask)
    if len(src_p) == 0:
        return None

    # embeddings
    npz    = np.load(npz_list[0])
    single = npz["single_embeddings"].astype(np.float32)       # (T, 384)
    pair   = npz["pair_embeddings"]                             # float16 (T,T,128)
    pAB    = pair[0:len_A, len_A:len_A + len_B, :]             # (A, B, 128)
    pAB_pocket = pAB[pocket_idx].astype(np.float32)            # (K, B, 128)
    e_attr = pAB_pocket[src_p, dst_l]                          # (E, 128)

    # remove isolated ligand nodes
    connected_lig = np.unique(dst_l)
    remap = np.full(len_B, -1, dtype=np.int64)
    remap[connected_lig] = np.arange(len(connected_lig))
    dst_l_new = remap[dst_l]

    # node features
    x_pocket = single[pocket_idx]
    x_ligand = single[len_A:len_A + len_B][connected_lig]

    # coordinates
    pos_pocket = coords_A[pocket_idx]                          # (K, 3)
    pos_ligand = coords_B[connected_lig]                       # (M, 3)

    data = HeteroData()
    data["pocket"].x   = torch.from_numpy(x_pocket)
    data["pocket"].pos = torch.from_numpy(pos_pocket)
    data["ligand"].x   = torch.from_numpy(x_ligand)
    data["ligand"].pos = torch.from_numpy(pos_ligand)
    data["pocket", "binds", "ligand"].edge_index = torch.from_numpy(
        np.stack([src_p, dst_l_new]).astype(np.int64))
    data["pocket", "binds", "ligand"].edge_attr  = torch.from_numpy(e_attr)
    data["ligand", "binds", "pocket"].edge_index = torch.from_numpy(
        np.stack([dst_l_new, src_p]).astype(np.int64))
    data["ligand", "binds", "pocket"].edge_attr  = torch.from_numpy(e_attr)
    data.n_pocket = len(pocket_idx)
    data.n_ligand = len(connected_lig)
    return data
