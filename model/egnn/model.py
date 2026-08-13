"""
egnn/model.py

E(n)-Equivariant GNN for agonism prediction on bipartite pocket-ligand graphs.

Each EGNNLayer:
  1. Compute edge messages using node features, squared distance, and edge attr
  2. Update node positions equivariantly: pos += Σ (pos_dst - pos_src) * w_ij
  3. Aggregate messages and update node features

Architecture:
  - Project node (384→H) and edge (128→H) features
  - N × EGNNLayer (pocket ↔ ligand, alternating)
  - Attention pool over ligand nodes → MLP classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import HeteroData


class EGNNLayer(nn.Module):
    """One directional EGNN layer: src nodes send messages to dst nodes."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        # message MLP: [h_src, h_dst, d², edge_attr] → message
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # coordinate weight: message → scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        # node update MLP: [h_dst, agg] → new h_dst
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h_src:      torch.Tensor,   # (N_src, D)
        h_dst:      torch.Tensor,   # (N_dst, D)
        pos_src:    torch.Tensor,   # (N_src, 3)
        pos_dst:    torch.Tensor,   # (N_dst, 3)
        edge_index: torch.Tensor,   # (2, E)  row0=src, row1=dst
        edge_attr:  torch.Tensor,   # (E, edge_dim)
    ):
        src_idx, dst_idx = edge_index

        # squared distance
        rel_pos = pos_dst[dst_idx] - pos_src[src_idx]          # (E, 3)
        d_sq    = (rel_pos ** 2).sum(dim=-1, keepdim=True)     # (E, 1)

        # messages
        m_ij = self.msg_mlp(
            torch.cat([h_src[src_idx], h_dst[dst_idx], d_sq, edge_attr], dim=-1)
        )                                                        # (E, H)

        # equivariant coordinate update
        w_ij      = self.coord_mlp(m_ij)                        # (E, 1)
        pos_delta = torch.zeros_like(pos_dst)
        pos_delta.index_add_(0, dst_idx, rel_pos * w_ij)
        n_nbr = torch.zeros(pos_dst.shape[0], 1, device=pos_dst.device)
        n_nbr.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=pos_dst.device))
        pos_dst_new = pos_dst + pos_delta / n_nbr.clamp(min=1)

        # aggregate messages
        agg = torch.zeros(h_dst.shape[0], m_ij.shape[-1], device=h_dst.device)
        agg.index_add_(0, dst_idx, m_ij)

        # node update with residual
        h_dst_new = self.norm(h_dst + self.node_mlp(
            torch.cat([h_dst, agg], dim=-1)
        ))

        return h_dst_new, pos_dst_new


class AgonismEGNN(nn.Module):
    def __init__(
        self,
        node_dim:   int = 384,
        edge_dim:   int = 128,
        hidden_dim: int = 128,
        n_layers:   int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.proj_node = nn.Linear(node_dim, hidden_dim)
        self.proj_edge = nn.Linear(edge_dim, hidden_dim)
        self.dropout   = dropout

        self.p2l_layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.l2p_layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.attn_pool_w = nn.Linear(hidden_dim, 1)
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        hp = self.proj_node(data["pocket"].x)
        hl = self.proj_node(data["ligand"].x)

        pos_p = data["pocket"].pos.float()
        pos_l = data["ligand"].pos.float()

        ei_p2l = data["pocket", "binds", "ligand"].edge_index
        ea_p2l = self.proj_edge(data["pocket", "binds", "ligand"].edge_attr)
        ei_l2p = data["ligand", "binds", "pocket"].edge_index
        ea_l2p = self.proj_edge(data["ligand", "binds", "pocket"].edge_attr)

        for p2l, l2p in zip(self.p2l_layers, self.l2p_layers):
            # pocket → ligand
            hl_new, pos_l_new = p2l(hp, hl, pos_p, pos_l, ei_p2l, ea_p2l)
            hl    = F.dropout(hl_new, p=self.dropout, training=self.training)
            pos_l = pos_l_new

            # ligand → pocket
            hp_new, pos_p_new = l2p(hl, hp, pos_l, pos_p, ei_l2p, ea_l2p)
            hp    = F.dropout(hp_new, p=self.dropout, training=self.training)
            pos_p = pos_p_new

        # attention pool over ligand nodes
        batch_vec = data["ligand"].batch
        attn_w    = pyg_softmax(self.attn_pool_w(hl), batch_vec)
        num_graphs = int(batch_vec.max().item()) + 1
        h_graph    = torch.zeros(num_graphs, hl.size(1), device=hl.device)
        h_graph.scatter_add_(
            0, batch_vec.unsqueeze(1).expand_as(hl), attn_w * hl
        )

        return self.classifier(h_graph).squeeze(-1)
