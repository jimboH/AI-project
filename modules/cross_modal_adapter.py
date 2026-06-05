"""Cross-Modal Adapter for Semantic ID Alignment.

Fixes the semantic-ID collapse that occurs when embeddings from one modality
(e.g. image) are passed through an RQ-VAE trained on a different modality
(e.g. text).

Root cause
----------
The RQ-VAE encoder (a deep ReLU MLP) was trained exclusively on text
embeddings.  When image embeddings arrive at inference time they are
out-of-distribution: the encoder maps them to a tiny, clustered region of the
32-dim latent space, so almost all items share the same few codebook entries
(observed: 0.1–0.6 % unique IDs vs 90 % for the native modality).

Fix
---
A lightweight residual MLP (the adapter) is trained to map source-modality
embeddings into the region of the input space that the frozen target-modality
encoder maps to a well-spread latent distribution.

Training objective
------------------
For each paired (src_emb_i, tgt_emb_i) belonging to the same item:

    z_src  = rqvae.encoder(adapter(src_emb_i))      # trainable
    z_tgt  = rqvae.encoder(tgt_emb_i).detach()      # frozen target

    L_align = MSE(z_src, z_tgt)                     # latent-space alignment

Optionally a small input-space reconstruction term is added:

    L_input = MSE(adapter(src_emb_i), tgt_emb_i)    # input-space pull

Total loss = L_align + input_weight * L_input

Architecture
------------
adapter(x) = x + MLP(LayerNorm(x))

The MLP's last linear layer is zero-initialised so the adapter starts as the
identity transformation and learns corrections incrementally.

Usage
-----
Training (train_cross_modal_adapter.py):
    adapter = CrossModalAdapter(source_dim=2048)
    adapter.train(rqvae, src_embeddings, tgt_embeddings, ...)

Inference (evaluate_grid.py):
    adapter = CrossModalAdapter.from_checkpoint(path)
    aligned_emb = adapter(image_emb)        # same shape as image_emb
    sem_ids = rqvae.get_semantic_ids(aligned_emb)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class AdapterLosses(NamedTuple):
    loss: Tensor           # total loss (scalar)
    loss_align: Tensor     # latent-space MSE
    loss_input: Tensor     # input-space MSE (0 if input_weight=0)
    z_src_norm: Tensor     # mean encoder-output norm for source (diagnostic)
    z_tgt_norm: Tensor     # mean encoder-output norm for target (diagnostic)


# ---------------------------------------------------------------------------
# Adapter module
# ---------------------------------------------------------------------------

class CrossModalAdapter(nn.Module):
    """Residual MLP that aligns source-modality embeddings to a frozen
    target-modality RQ-VAE's expected input distribution.

    Parameters
    ----------
    source_dim : int
        Dimensionality of the source (image) embeddings.
    hidden_dims : list[int]
        Hidden layer widths of the residual MLP.
    dropout : float
        Dropout probability (applied after each hidden activation).
    """

    def __init__(
        self,
        source_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.source_dim = source_dim
        self.hidden_dims = hidden_dims or [source_dim // 4, source_dim // 4]

        # Pre-normalise the input so the MLP sees unit-scale features
        self.norm = nn.LayerNorm(source_dim)

        # Build the residual MLP
        layers: list[nn.Module] = []
        in_dim = source_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=True))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        # Final projection back to source_dim
        out_layer = nn.Linear(in_dim, source_dim, bias=True)
        # Zero-init → adapter starts as identity, learns corrections gradually
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Map source embeddings to the target-RQ-VAE's expected space.

        Parameters
        ----------
        x : Tensor  shape (B, source_dim)

        Returns
        -------
        Tensor  shape (B, source_dim)
        """
        return x + self.mlp(self.norm(x))

    # ------------------------------------------------------------------
    # Loss computation (used during training)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        src_emb: Tensor,
        tgt_emb: Tensor,
        rqvae_encoder: nn.Module,
        input_weight: float = 0.0,
    ) -> AdapterLosses:
        """Compute alignment loss for a batch of paired embeddings.

        Parameters
        ----------
        src_emb : Tensor  (B, source_dim)  — source modality (e.g. image)
        tgt_emb : Tensor  (B, source_dim)  — target modality (e.g. text)
        rqvae_encoder : nn.Module          — frozen encoder from the target RQ-VAE
        input_weight : float               — weight for the input-space auxiliary loss

        Returns
        -------
        AdapterLosses
        """
        aligned = self.forward(src_emb)                   # (B, D)

        # Latent-space alignment: match encoder outputs
        z_src = rqvae_encoder(aligned)                    # (B, embed_dim=32)
        with torch.no_grad():
            z_tgt = rqvae_encoder(tgt_emb)                # (B, embed_dim=32)

        loss_align = F.mse_loss(z_src, z_tgt)

        # Optional input-space reconstruction pull
        if input_weight > 0.0:
            loss_input = F.mse_loss(aligned, tgt_emb.detach())
        else:
            loss_input = torch.zeros(1, device=src_emb.device)

        loss = loss_align + input_weight * loss_input

        with torch.no_grad():
            z_src_norm = z_src.norm(dim=-1).mean()
            z_tgt_norm = z_tgt.norm(dim=-1).mean()

        return AdapterLosses(
            loss=loss,
            loss_align=loss_align,
            loss_input=loss_input,
            z_src_norm=z_src_norm,
            z_tgt_norm=z_tgt_norm,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save adapter weights + config to a single .pt file."""
        torch.save(
            {
                "source_dim": self.source_dim,
                "hidden_dims": self.hidden_dims,
                "state_dict": self.state_dict(),
            },
            path,
        )
        print(f"[CrossModalAdapter] saved → {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "CrossModalAdapter":
        """Load an adapter from a .pt checkpoint."""
        data = torch.load(path, map_location=device, weights_only=False)
        adapter = cls(
            source_dim=data["source_dim"],
            hidden_dims=data["hidden_dims"],
        )
        adapter.load_state_dict(data["state_dict"])
        return adapter


# ---------------------------------------------------------------------------
# Evaluation helpers (used in training loop and diagnostics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_p_unique(
    adapter: Optional[CrossModalAdapter],
    rqvae,
    embeddings: Tensor,
    batch_size: int = 512,
    device: str = "cpu",
) -> float:
    """Compute the fraction of items with a unique semantic-ID tuple.

    Parameters
    ----------
    adapter : CrossModalAdapter or None
        If None, embeddings are passed directly to the RQ-VAE (baseline).
    rqvae : RqVae
    embeddings : Tensor  (N, D)
    """
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(embeddings), batch_size=batch_size, shuffle=False
    )
    if adapter is not None:
        adapter = adapter.to(device)
    all_ids = []
    for (batch,) in loader:
        batch = batch.to(device)
        if adapter is not None:
            batch = adapter(batch)
        out = rqvae.get_semantic_ids(batch, gumbel_t=0.001)
        all_ids.append(out.sem_ids.cpu())          # (B, n_layers)

    sem_ids = torch.cat(all_ids, dim=0)            # (N, n_layers)
    tuples = [tuple(r.tolist()) for r in sem_ids]
    return len(set(tuples)) / len(tuples)


@torch.no_grad()
def compute_codebook_utilisation(
    adapter: Optional[CrossModalAdapter],
    rqvae,
    embeddings: Tensor,
    batch_size: int = 512,
    device: str = "cpu",
) -> list[float]:
    """Return the fraction of codebook entries used per RQ-VAE layer."""
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(embeddings), batch_size=batch_size, shuffle=False
    )
    if adapter is not None:
        adapter = adapter.to(device)
    all_ids = []
    for (batch,) in loader:
        batch = batch.to(device)
        if adapter is not None:
            batch = adapter(batch)
        out = rqvae.get_semantic_ids(batch, gumbel_t=0.001)
        all_ids.append(out.sem_ids.cpu())

    sem_ids = torch.cat(all_ids, dim=0)
    utils = []
    for layer in range(sem_ids.shape[1]):
        used = sem_ids[:, layer].unique().numel()
        utils.append(used / rqvae.codebook_size)
    return utils
