"""
graph.py

Build PyTorch Geometric graphs from AF3 inference outputs.

Graph structure:
  Nodes:
    - Receptor (A) residues within cutoff Å of any ligand residue → "pocket"
    - Ligand (L) residues that have ≥1 contact edge              → "ligand"
  Node features:
    - pocket: AF3 single embeddings (384-dim)
    - ligand: AF3 single embeddings + relative chain position (384+1=385-dim).
      Position = (1-based residue index) / (full ligand chain length), so the
      model can distinguish "whole ligand engaged" from "only one end is".
  Graph-level attrs: n_pocket, n_ligand (kept-node counts), n_ligand_total
    (full ligand chain length before the isolated-residue filter below).
  Edges:
    - pocket ↔ ligand (within cutoff): edge_attr = pair_embeddings[i,j] (128-dim),
      edge_dist = real Cα-Cα distance in Å (1-dim)
    Only ligand residues connected to ≥1 pocket residue are kept.

  If parse_cif_coords' residue count disagrees with the AF3 token count for
  either chain, the sample is rejected (build_graph returns None) rather than
  falling back to a fake fully-connected/zero-distance graph.
"""

import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

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


def get_chain_token_info(confidences_json_path: Path) -> tuple[dict[str, tuple[int, int]], list[int]]:
    """Return (chain_offsets, token_res_ids) computed from AF3's own
    per-token bookkeeping in *_confidences.json, instead of guessing chain
    boundaries from data.json's plain sequence length.

    Deriving offsets from `data.json`'s plain sequence length would assume
    1 residue == 1 token, which AF3 violates for non-standard/modified
    residues (flatten_non_standard_residues, on by default): such a residue
    is expanded into one token per heavy atom. This under/misaligns
    sequence-length-derived offsets for any chain containing such residues
    (in this project: only the ligand chain, "L", ever has modifications —
    receptor/RAMP/Gα are always unmodified).

    chain_offsets: {chain_id: (start, end)} token-index ranges — correct even
    when a chain's true token count differs from its sequence length.
    token_res_ids: per-token 1-based residue index *within its chain*,
    matching the `label_seq_id` convention used by parse_cif_coords — lets
    callers group multiple tokens belonging to the same real residue.
    """
    conf = json.load(open(confidences_json_path))
    chain_ids = conf["token_chain_ids"]
    res_ids   = conf["token_res_ids"]
    offsets: dict[str, list[int]] = {}
    for i, cid in enumerate(chain_ids):
        if cid not in offsets:
            offsets[cid] = [i, i + 1]
        else:
            offsets[cid][1] = i + 1
    return {k: tuple(v) for k, v in offsets.items()}, res_ids


def collapse_tokens_to_residues(x: np.ndarray, res_ids, axis: int = 0) -> np.ndarray:
    """Mean-pool `x` along `axis`, grouping entries that share the same
    (1-based) residue id in `res_ids`.

    AF3 expands non-standard/modified residues into one token per heavy atom
    (flatten_non_standard_residues, on by default) instead of one token per
    residue. This folds that per-atom expansion back down to one vector per
    real residue — ordered 1..max(res_ids) — matching the residue indexing
    used by parse_cif_coords/get_pocket_residue_indices. A no-op for chains
    with no expanded residues (each residue already has exactly one token).
    """
    res_ids = np.asarray(res_ids)
    n_res = int(res_ids.max())
    x = np.moveaxis(x, axis, 0)                          # (n_tokens, ...)
    out = np.zeros((n_res,) + x.shape[1:], dtype=np.float32)
    for r in range(1, n_res + 1):
        out[r - 1] = x[res_ids == r].mean(axis=0)
    return np.moveaxis(out, 0, axis)


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
    # NOTE: "*_confidences.json" also matches "*_summary_confidences.json" —
    # must exclude it explicitly.
    conf_list = [p for p in sample_dir.glob("*_confidences.json") if "summary" not in p.name]

    if not npz_list or not cif_list or not djson or not conf_list:
        return None

    # chain token-index ranges from AF3's own per-token bookkeeping (receptor=A,
    # ligand=L; order varies — G/B may sit between them). Correct even when
    # non-standard/modified ligand residues expand into multiple tokens each,
    # unlike data.json's plain sequence length — see get_chain_token_info docstring.
    offsets, token_res_ids = get_chain_token_info(conf_list[0])
    if "A" not in offsets or "L" not in offsets:
        return None
    a_start, a_end = offsets["A"]
    l_start, l_end = offsets["L"]
    len_A = a_end - a_start
    if len_A == 0 or l_end == l_start:
        return None
    res_ids_L = np.asarray(token_res_ids[l_start:l_end])  # 1-based residue id per L token
    len_L = int(res_ids_L.max())                          # TRUE residue count (matches coords_L)

    # Cα coordinates
    coords = parse_cif_coords(cif_list[0])
    coords_A = coords.get("A")
    coords_L = coords.get("L")
    if coords_A is None or coords_L is None:
        return None
    if len(coords_A) != len_A or len(coords_L) != len_L:
        print(f"[FAIL] {sample_dir.name}: coords/token length mismatch "
              f"(A: {len(coords_A)} vs {len_A}, L: {len(coords_L)} vs {len_L})", flush=True)
        return None
    pocket_idx = get_pocket_residue_indices(coords_A, coords_L, cutoff)
    if len(pocket_idx) == 0:
        return None  # no receptor residue within cutoff of any ligand residue

    # embeddings
    npz    = np.load(npz_list[0])
    single = npz["single_embeddings"].astype(np.float32)  # (T, 384)
    pair   = npz["pair_embeddings"]                        # float16, (T, T, 128)
    pAL_tokens = pair[a_start:a_end, l_start:l_end, :].astype(np.float32)  # (len_A, n_L_tokens, 128)
    # fold AF3's per-atom token expansion (non-standard ligand residues) back
    # down to one vector per real residue, matching coords_L's indexing
    pAL  = collapse_tokens_to_residues(pAL_tokens, res_ids_L, axis=1)      # (len_A, len_L, 128)
    pAL_pocket = pAL[pocket_idx]                            # (K, len_L, 128)
    K = len(pocket_idx)

    # edges: pocket-ligand pairs within cutoff
    dist_pl = np.linalg.norm(
        coords_A[pocket_idx][:, None, :] - coords_L[None, :, :], axis=2
    )  # (K, len_L)
    edge_mask = dist_pl <= cutoff
    src_p, dst_l = np.where(edge_mask)
    e_attr = pAL_pocket[src_p, dst_l]                  # (n_edges, 128)
    e_dist = dist_pl[src_p, dst_l]                     # (n_edges,) real Cα-Cα distance

    if len(src_p) == 0:
        return None  # no contact edges at all

    # remove isolated ligand nodes (no contact edges)
    connected_lig = np.unique(dst_l)                       # sorted ligand indices with ≥1 edge
    remap = np.full(len_L, -1, dtype=np.int64)
    remap[connected_lig] = np.arange(len(connected_lig))
    dst_l_new = remap[dst_l]                               # remapped ligand indices

    # node features: AF3 single embeddings (384-dim)
    x_pocket = single[a_start + pocket_idx]                # (K, 384)
    sL_res   = collapse_tokens_to_residues(single[l_start:l_end], res_ids_L, axis=0)  # (len_L, 384)
    x_ligand_emb = sL_res[connected_lig]                    # (M, 384)

    # per-node relative position along the full ligand chain (0,1], N-term→C-term.
    # Lets the model tell "6 of 6 residues engaged" from "6 of 29 engaged, rest
    # dangling outside the pocket" — a bipartite pocket-ligand graph otherwise
    # has no signal for how much of a (possibly long, e.g. Class B1 peptide)
    # ligand actually made contact.
    res_id_norm = ((connected_lig + 1) / len_L).astype(np.float32)  # (M,)
    x_ligand = np.concatenate([x_ligand_emb, res_id_norm[:, None]], axis=1)  # (M, 385)

    # assemble HeteroData
    data = HeteroData()
    data["pocket"].x = torch.from_numpy(x_pocket)
    data["ligand"].x = torch.from_numpy(x_ligand)
    e_dist_t = torch.from_numpy(e_dist.astype(np.float32)).unsqueeze(-1)  # (n_edges, 1)
    data["pocket", "binds", "ligand"].edge_index = torch.from_numpy(
        np.stack([src_p, dst_l_new]).astype(np.int64))
    data["pocket", "binds", "ligand"].edge_attr  = torch.from_numpy(e_attr)
    data["pocket", "binds", "ligand"].edge_dist  = e_dist_t
    data["ligand", "binds", "pocket"].edge_index = torch.from_numpy(
        np.stack([dst_l_new, src_p]).astype(np.int64))
    data["ligand", "binds", "pocket"].edge_attr  = torch.from_numpy(e_attr)
    data["ligand", "binds", "pocket"].edge_dist  = e_dist_t

    data.n_pocket = K
    data.n_ligand = len(connected_lig)
    data.n_ligand_total = len_L  # full chain length, incl. non-contacting residues

    return data
