"""
model_xattn.py

CrossAttnAgonism: pocket ↔ ligand cross-attention with AF3 pair_embedding
as additive per-head attention bias, matching PairFormer's internal design.

Architecture:
  1. Project: 384-dim → hidden_dim (pocket & ligand separately)
  2. N × bidirectional cross-attention layers
       pocket queries ligand  (pair_emb as bias)
       ligand queries pocket  (pair_emb as bias)
  3. Attention pool over ligand nodes per graph
  4. MLP → P(agonist)
"""

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import HeteroData

PAIR_DIM = 128
NODE_DIM = 384  # AF3 single embedding


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: x → SiLU(W1 x) * W2 x → W3, with residual + LN."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff, bias=False)
        self.w2   = nn.Linear(d_model, d_ff, bias=False)
        self.w3   = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.drop(self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))))


class CrossAttnLayer(nn.Module):
    """
    One cross-attention pass: query_nodes attend over kv_nodes.
    pair_emb[edge] is projected to n_heads scalars and added to attention
    scores before softmax — exactly how PairFormer uses pair representations.
    Optionally followed by a SwiGLU FFN (use_ffn=True).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_ffn: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj    = nn.Linear(d_model, d_model, bias=False)
        self.k_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj    = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.pair_bias = nn.Linear(PAIR_DIM, n_heads)
        self.drop      = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(d_model)
        self.ffn       = SwiGLUFFN(d_model, d_model * 2, dropout) if use_ffn else None

    def forward(
        self,
        h_q:        torch.Tensor,   # (N_q, d)
        h_kv:       torch.Tensor,   # (N_kv, d)
        edge_index: torch.Tensor,   # (2, E)  row0=query_idx, row1=kv_idx
        edge_attr:  torch.Tensor,   # (E, 128)
    ) -> torch.Tensor:              # (N_q, d)

        qi, kvi = edge_index
        H, dh = self.n_heads, self.d_head

        Q = self.q_proj(h_q).view(-1, H, dh)
        K = self.k_proj(h_kv).view(-1, H, dh)
        V = self.v_proj(h_kv).view(-1, H, dh)

        scores = (Q[qi] * K[kvi]).sum(-1) / (dh ** 0.5)  # (E, H)
        scores = scores + self.pair_bias(edge_attr)

        attn = pyg_softmax(scores, qi, num_nodes=h_q.size(0))
        attn = self.drop(attn)

        weighted = attn.unsqueeze(-1) * V[kvi]             # (E, H, dh)
        out = torch.zeros(h_q.size(0), H, dh, device=h_q.device)
        out.scatter_add_(0, qi.view(-1, 1, 1).expand_as(weighted), weighted)
        out = out.view(-1, H * dh)

        h = self.norm(h_q + self.out_proj(out))           # residual + LN
        return self.ffn(h) if self.ffn is not None else h


class CrossAttnAgonism(nn.Module):
    """
    Bidirectional cross-attention GNN for agonist/non-agonist prediction.
    Pair embeddings act as additive attention biases (AF3 / PairFormer style).
    """

    def __init__(
        self,
        hidden_dim:   int   = 128,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        dropout:      float = 0.1,
        use_ffn:      bool  = False,
        ligand_only:  bool  = False,
    ):
        super().__init__()
        self.ligand_only = ligand_only
        self.proj_pocket = nn.Linear(NODE_DIM, hidden_dim)
        self.proj_ligand = nn.Linear(NODE_DIM, hidden_dim)

        self.p2l_layers = nn.ModuleList([
            CrossAttnLayer(hidden_dim, n_heads, dropout, use_ffn) for _ in range(n_layers)
        ])
        self.l2p_layers = nn.ModuleList([
            CrossAttnLayer(hidden_dim, n_heads, dropout, use_ffn) for _ in range(n_layers)
        ])

        # graph-level readout: learned attention pool over ligand nodes
        self.attn_pool_w = nn.Linear(hidden_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: HeteroData, return_nodes: bool = False):
        hl = self.proj_ligand(data["ligand"].x)   # (N_ligand, d)

        if not self.ligand_only:
            hp = self.proj_pocket(data["pocket"].x)   # (N_pocket, d)
            ei_p2l = data["pocket", "binds", "ligand"].edge_index
            ea_p2l = data["pocket", "binds", "ligand"].edge_attr
            ei_l2p = data["ligand", "binds", "pocket"].edge_index
            ea_l2p = data["ligand", "binds", "pocket"].edge_attr

            for p2l, l2p in zip(self.p2l_layers, self.l2p_layers):
                hp = p2l(hp, hl, ei_p2l, ea_p2l)   # pocket queries ligand
                hl = l2p(hl, hp, ei_l2p, ea_l2p)   # ligand queries pocket

        if return_nodes:
            return hp, hl   # (N_pocket, d), (N_ligand, d)

        # graph-level: attention pool over ligand nodes
        batch = data["ligand"].batch
        num_graphs = int(batch.max().item()) + 1
        attn_w = pyg_softmax(self.attn_pool_w(hl), batch, num_nodes=num_graphs)
        h_graph = torch.zeros(num_graphs, hl.size(1), device=hl.device)
        h_graph.scatter_add_(0, batch.unsqueeze(1).expand_as(hl), attn_w * hl)

        return self.mlp(h_graph).squeeze(-1)
