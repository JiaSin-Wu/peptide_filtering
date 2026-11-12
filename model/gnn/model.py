"""
model.py

AgonismGNN: bipartite GATv2 model for agonist prediction.

Architecture:
  1. Input projection: pocket 384-dim → hidden_dim, ligand 385-dim → hidden_dim
     (ligand's extra dim is its relative position along the full chain, see
     graph.py — separate projections since the two node types are no longer
     the same width)
  2. Edge projection:  128-dim → hidden_dim
  3. N × GATv2Conv layers (pocket ↔ ligand cross-attention)
  4. Attention-weighted pool over ligand nodes, then concat full ligand-chain
     length (data.n_ligand_total) as a graph-level context feature
  5. MLP classifier → P(agonist)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import HeteroData


PAIR_DIM = 128

# Cα-Cα distance is one-hot binned rather than fed in as a raw scalar: a bare
# scalar only lets a downstream Linear learn a linear response to distance.
# Edges only exist within the 8Å pocket cutoff (graph.py), so these three
# buckets span the full range. Shared by model_xattn.py and model_gatdist.py.
DIST_BIN_EDGES = (2.0, 5.0)              # boundaries -> buckets [0,2) [2,5) [5,8]
N_DIST_BINS = len(DIST_BIN_EDGES) + 1


def dist_to_onehot(edge_dist: torch.Tensor) -> torch.Tensor:
    """(E, 1) raw Å distance -> (E, N_DIST_BINS) one-hot bucket encoding."""
    d = edge_dist.squeeze(-1)
    bin_idx = torch.zeros_like(d, dtype=torch.long)
    for edge in DIST_BIN_EDGES:
        bin_idx = bin_idx + (d >= edge).long()
    return F.one_hot(bin_idx, num_classes=N_DIST_BINS).float()


def build_head(hidden_dim: int, dropout: float, head: str = "mlp") -> nn.Module:
    """The classifier that sits on the pooled graph vector.

    `mlp` is the original H -> 64 -> 1. `linear` drops the hidden layer and
    keeps the Dropout, so the two differ by the hidden layer alone and a
    weaker source head can be ablated against a stronger one without also
    changing the regularisation.

    Both are nn.Sequential ending in the output Linear, so whatever
    extract_embeddings.py hooks still receives h_graph as its input.
    """
    if head == "linear":
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    if head != "mlp":
        raise ValueError(f"unknown head {head!r}: expected 'mlp' or 'linear'")
    return nn.Sequential(
        nn.Linear(hidden_dim, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),
    )


LIGAND_LEN_NORM = 30.0  # soft scale for data.n_ligand_total (typical peptide chain length)


class AgonismGNN(nn.Module):
    def __init__(
        self,
        node_dim:   int = 384,  # AF3 single embedding
        edge_dim:   int = 128,
        hidden_dim: int = 128,
        n_heads:    int = 4,
        n_layers:   int = 2,
        dropout:    float = 0.4,
        head:       str = "mlp",
    ):
        super().__init__()

        self.proj_pocket = nn.Linear(node_dim, hidden_dim)
        self.proj_ligand = nn.Linear(node_dim + 1, hidden_dim)  # +1: relative chain position
        self.proj_edge = nn.Linear(edge_dim, hidden_dim)
        self.dropout   = dropout

        # GATv2Conv layers (pocket ↔ ligand)
        self.p2l_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // n_heads,
                      heads=n_heads, edge_dim=hidden_dim,
                      dropout=dropout, concat=True, add_self_loops=False)
            for _ in range(n_layers)
        ])
        self.l2p_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // n_heads,
                      heads=n_heads, edge_dim=hidden_dim,
                      dropout=dropout, concat=True, add_self_loops=False)
            for _ in range(n_layers)
        ])
        self.norms_p = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norms_l = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # attention pool over ligand nodes
        self.attn_pool_w = nn.Linear(hidden_dim, 1)

        # classifier (+1: full ligand-chain length, concatenated after pooling)
        self.classifier = build_head(hidden_dim + 1, dropout, head)

    def forward(self, data: HeteroData) -> torch.Tensor:
        # node features
        hp = self.proj_pocket(data["pocket"].x)   # (N_pocket, H)
        hl = self.proj_ligand(data["ligand"].x)   # (N_ligand, H)

        # edge features
        ei_p2l = data["pocket", "binds", "ligand"].edge_index   # (2, E_pl)
        ea_p2l = self.proj_edge(
            data["pocket", "binds", "ligand"].edge_attr)        # (E_pl, H)
        ei_l2p = data["ligand", "binds", "pocket"].edge_index
        ea_l2p = self.proj_edge(
            data["ligand", "binds", "pocket"].edge_attr)

        # message passing
        for i, (p2l, l2p, norm_p, norm_l) in enumerate(
            zip(self.p2l_convs, self.l2p_convs, self.norms_p, self.norms_l)
        ):
            # pocket → ligand
            hl_new = p2l((hp, hl), ei_p2l, ea_p2l)
            hl = norm_l(hl + F.dropout(hl_new, p=self.dropout, training=self.training))

            # ligand → pocket
            hp_new = l2p((hl, hp), ei_l2p, ea_l2p)
            hp = norm_p(hp + F.dropout(hp_new, p=self.dropout, training=self.training))

        # batch-aware attention pool over ligand nodes
        batch_vec = data["ligand"].batch                       # (N_ligand,)
        attn_w = pyg_softmax(self.attn_pool_w(hl), batch_vec) # (N_ligand, 1)
        # scatter_add per graph
        num_graphs = int(batch_vec.max().item()) + 1
        h_graph = torch.zeros(num_graphs, hl.size(1), device=hl.device)
        h_graph.scatter_add_(0, batch_vec.unsqueeze(1).expand_as(hl),
                             attn_w * hl)                      # (B, H)

        # full ligand-chain length as graph-level context (same value for every
        # ligand node in a sample, so it belongs here, not duplicated per-node)
        # n_ligand_total may still be CPU-resident here: callers that run
        # .to(device) on a single un-batched Data before Batch.from_data_list
        # (e.g. ga.oracle.gnn_predict_folder) see it collated into a fresh
        # tensor *after* that .to(device) call, so it never gets moved.
        len_feat = (data.n_ligand_total.float() / LIGAND_LEN_NORM).unsqueeze(-1).to(h_graph.device)  # (B, 1)
        h_graph = torch.cat([h_graph, len_feat], dim=1)        # (B, H+1)

        # classify
        return self.classifier(h_graph).squeeze(-1)            # (B,)
