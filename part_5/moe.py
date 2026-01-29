from __future__ import annotations
import torch, torch.nn as nn
from gating import TopKGate
from experts import ExpertMLP

class MoE(nn.Module):
    """Mixture-of-Experts layer (token-wise top-k routing).
    Implementation is single-GPU friendly (loops over experts for clarity).
    https://arxiv.org/pdf/2101.03961
    """
    def __init__(self, dim: int, n_expert: int, k: int = 1, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_expert = n_expert
        self.k = k
        self.gate = TopKGate(dim, n_expert, k=k)
        self.experts = nn.ModuleList([ExpertMLP(dim, mult=mult, swiglu=swiglu, dropout=dropout) for _ in range(n_expert)])