"""
graph.py

Build PyTorch Geometric graphs from AF3 inference outputs.

Graph structure:
  Nodes:
    - Receptor (A) residues within cutoff Å of any ligand residue → "pocket"
    - Ligand (B) residues that have ≥1 contact edge              → "ligand"
  Node features: AF3 single embeddings (384-dim)
  Edges:
    - pocket ↔ ligand (within cutoff): pair_embeddings[i,j] (128-dim)
    Only ligand residues connected to ≥1 pocket residue are kept.
"""

import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

# ── Amino acid one-hot encoding ───────────────────────────────────────────────
_AA_VOCAB = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
_UNK_IDX  = 20   # index for non-standard / unknown residue
NODE_DIM  = 21   # one-hot dim

def aa_onehot(seq: str) -> np.ndarray:
    """Return (L, 21) one-hot matrix for amino acid sequence."""
    x = np.zeros((len(seq), NODE_DIM), dtype=np.float32)
    for i, aa in enumerate(seq):
        x[i, _AA_VOCAB.get(aa, _UNK_IDX)] = 1.0
    return x


# ── CIF parsing ──────────────────────────────────────────────────────────────

def parse_cif_coords(cif_path: Path) -> dict[str, np.ndarray]:
    """Parse mmCIF atom_site records.

    Returns {chain_id: (n_residues, 3)} representative-atom coordinates,
    indexed by label_seq_id order.

    Mirrors AF3 logic: CA if present, else first atom of the residue.
    """
    # chain -> seq_id -> (ca_xyz | first_xyz)
    # stored as (ca_xyz, first_xyz) then resolved at the end
    chain_res: dict[str, dict[int, dict]] = {}

    with open(cif_path) as f:
        in_atom = False
        col: dict[str, int] = {}

        for line in f:
            line = line.rstrip()
            if line.startswith("_atom_site."):
                key = line.split(".")[1]
                col[key] = len(col)
                in_atom = True
                continue
            if in_atom and (line.startswith("ATOM") or line.startswith("HETATM")):
                parts = line.split()
                atom_name  = parts[col["label_atom_id"]]
                chain      = parts[col["label_asym_id"]]
                seq_id_str = parts[col["label_seq_id"]]
                try:
                    seq_id = int(seq_id_str)
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
        sorted_ids = sorted(res_dict.keys())
        coords = []
        for sid in sorted_ids:
            r = res_dict[sid]
            coords.append(r["ca"] if r["ca"] is not None else r["first"])
        result[chain] = np.array(coords, dtype=np.float32)
    return result


def get_pocket_residue_indices(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    cutoff: float = 8.0,
) -> np.ndarray:
    """Return indices of A residues with Cα within `cutoff` Å of any B Cα."""
    # pairwise distances: (len_A, len_B)
    diff = coords_A[:, None, :] - coords_B[None, :, :]   # (len_A, len_B, 3)
    dist = np.linalg.norm(diff, axis=2)                   # (len_A, len_B)
    min_dist = dist.min(axis=1)                           # (len_A,)
    return np.where(min_dist <= cutoff)[0]


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    sample_dir: Path,
    cutoff: float = 8.0,
) -> HeteroData | None:
    """Build a HeteroData graph for one AF3 output directory.

    Returns None if required files are missing.
    """
    # locate files
    emb_dir  = sample_dir / "seed-1_embeddings"
    npz_list = list(emb_dir.glob("*_embeddings.npz")) if emb_dir.exists() else []
    cif_list = list((sample_dir / "seed-1_sample-0").glob("*_model.cif"))
    djson    = list(sample_dir.glob("*_data.json"))

    if not npz_list or not cif_list or not djson:
        return None

    # chain sequences and lengths from data.json
    d = json.load(open(djson[0]))
    seqs = {s["protein"]["id"]: s["protein"]["sequence"]
            for s in d["sequences"] if "protein" in s}
    seq_A = seqs.get("A", "")
    seq_B = seqs.get("B", "")
    len_A, len_B = len(seq_A), len(seq_B)
    if len_A == 0 or len_B == 0:
        return None

    # Cα coordinates
    coords = parse_cif_coords(cif_list[0])
    coords_A = coords.get("A")
    coords_B = coords.get("B")
    if coords_A is None or coords_B is None:
        return None
    if len(coords_A) != len_A or len(coords_B) != len_B:
        pocket_idx = np.arange(min(len_A, 30))
        coords_A   = None
    else:
        pocket_idx = get_pocket_residue_indices(coords_A, coords_B, cutoff)

    if len(pocket_idx) == 0:
        pocket_idx = np.arange(min(len_A, 30))

    # embeddings
    npz    = np.load(npz_list[0])
    single = npz["single_embeddings"].astype(np.float32)  # (T, 384)
    pair   = npz["pair_embeddings"]                        # float16, (T, T, 128)
    pAB  = pair[0:len_A, len_A:len_A + len_B, :]          # (len_A, len_B, 128)
    pAB_pocket = pAB[pocket_idx].astype(np.float32)        # (K, len_B, 128)
    K = len(pocket_idx)

    # edges: pocket-ligand pairs within cutoff
    if coords_A is not None:
        dist_pl = np.linalg.norm(
            coords_A[pocket_idx][:, None, :] - coords_B[None, :, :], axis=2
        )  # (K, len_B)
        edge_mask = dist_pl <= cutoff
        src_p, dst_l = np.where(edge_mask)
        e_attr = pAB_pocket[src_p, dst_l]                  # (n_edges, 128)
    else:
        src_p  = np.repeat(np.arange(K), len_B)
        dst_l  = np.tile(np.arange(len_B), K)
        e_attr = pAB_pocket.reshape(-1, 128)

    if len(src_p) == 0:
        return None  # no contact edges at all

    # remove isolated ligand nodes (no contact edges)
    connected_lig = np.unique(dst_l)                       # sorted ligand indices with ≥1 edge
    remap = np.full(len_B, -1, dtype=np.int64)
    remap[connected_lig] = np.arange(len(connected_lig))
    dst_l_new = remap[dst_l]                               # remapped ligand indices

    # node features: AF3 single embeddings (384-dim)
    x_pocket = single[pocket_idx]                          # (K, 384)
    x_ligand = single[len_A:len_A + len_B][connected_lig]  # (M, 384)

    # assemble HeteroData
    data = HeteroData()
    data["pocket"].x = torch.from_numpy(x_pocket)
    data["ligand"].x = torch.from_numpy(x_ligand)
    data["pocket", "binds", "ligand"].edge_index = torch.from_numpy(
        np.stack([src_p, dst_l_new]).astype(np.int64))
    data["pocket", "binds", "ligand"].edge_attr  = torch.from_numpy(e_attr)
    data["ligand", "binds", "pocket"].edge_index = torch.from_numpy(
        np.stack([dst_l_new, src_p]).astype(np.int64))
    data["ligand", "binds", "pocket"].edge_attr  = torch.from_numpy(e_attr)

    data.n_pocket = K
    data.n_ligand = len(connected_lig)

    return data
