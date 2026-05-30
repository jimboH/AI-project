"""Per-modality input adapters for shared-codebook (Design A) RQ-VAE training.

Each modality (text / image / multimodal) gets its own small residual MLP that
maps the raw 2048-d Qwen3-VL embedding into a *shared* pre-encoder space, before
a single shared RQ-VAE encoder + codebooks tokenize it. See RQVAE_improvement.md
(§5 Design A, §6.1) for the rationale.

This module is additive: it does not modify any existing training code. The
original per-modality RQ-VAE pipeline (train_rqvae.py) is unaffected.
"""

import torch
from torch import nn
from torch import Tensor


class ModalityAdapter(nn.Module):
    """Residual MLP mapping one modality's embedding into the shared space.

    Initialised as (close to) the identity: the residual branch is scaled by a
    learnable scalar ``alpha`` starting at 0, so at step 0 the adapter output
    equals its input. This keeps k-means initialisation of the shared codebook
    well-conditioned (it sees the raw embedding distribution) and lets the
    adapters specialise gradually as the alignment loss kicks in.
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
        )
        # Start as identity (alpha = 0) so kmeans-init sees raw embeddings.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.alpha * self.net(x)
