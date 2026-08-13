"""
model.py

AgonismGNN: bipartite GATv2 model for agonist prediction.

Architecture:
  1. Input projection: 384-dim → hidden_dim
  2. Edge projection:  128-dim → hidden_dim
  3. N × GATv2Conv layers (pocket ↔ ligand cross-attention)
  4. Attention-weighted pool over ligand nodes
  5. MLP classifier → P(agonist)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import HeteroData


class AgonismGNN(nn.Module):
    def __init__(
        self,
        node_dim:   int = 384,  # AF3 single embedding
        edge_dim:   int = 128,
        hidden_dim: int = 128,
        n_heads:    int = 4,
        n_layers:   int = 2,
        dropout:    float = 0.4,
    ):
        super().__init__()

        self.proj_node = nn.Linear(node_dim, hidden_dim)
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

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        # node features
        hp = self.proj_node(data["pocket"].x)   # (N_pocket, H)
        hl = self.proj_node(data["ligand"].x)   # (N_ligand, H)

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

        # classify
        return self.classifier(h_graph).squeeze(-1)            # (B,)


class AgonismEnsemble(nn.Module):
    """Ensemble of N AgonismGNN models (averaged logits)."""

    def __init__(self, n_models: int = 5, **gnn_kwargs):
        super().__init__()
        self.models = nn.ModuleList(
            [AgonismGNN(**gnn_kwargs) for _ in range(n_models)])

    def forward(self, data: HeteroData) -> torch.Tensor:
        logits = torch.stack([m(data) for m in self.models])
        return logits.mean(dim=0)
