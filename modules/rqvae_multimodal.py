"""Shared-codebook multimodal RQ-VAE (Design A).

Wraps the existing ``RqVae`` (modules/rqvae.py) WITHOUT modifying it. One shared
Rq-VAE (encoder + 3x256 codebooks + decoder) is fed by one ``ModalityAdapter``
per modality. All modalities therefore share a single semantic-ID space, and a
cross-modal alignment loss ties the same item's per-modality latents together so
that ``id(text_i) ~= id(image_i)``.

See RQVAE_improvement.md §6.2. This file is additive — the per-modality pipeline
in train_rqvae.py keeps using ``RqVae`` directly and is untouched.

Reconstruction target
----------------------
Mirroring ``RqVae.forward``, the shared decoder reconstructs the **encoder
input** — here that is the adapter output ``A_m(x)`` (the item's representation in
the shared space), not the raw modality embedding. This keeps a single shared
decoder coherent with cross-modal alignment (all modalities are reconstructed in
the same shared space) and reuses ``RqVae``'s reconstruction loss verbatim.
"""

import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn
from torch import Tensor
from typing import Dict, List, NamedTuple

from modules.adapter import ModalityAdapter
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode


class MultiModalRqVaeOutput(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    align_loss: Tensor
    # Per-modality fraction of unique code tuples within the batch.
    p_unique_ids: Dict[str, Tensor]
    # Mean pairwise cross-modal ID agreement within the batch (same item, same id).
    id_agreement: Tensor


class MultiModalRqVae(nn.Module):
    def __init__(
        self,
        modalities: List[str],
        input_dim: int = 2048,
        embed_dim: int = 32,
        hidden_dims: List[int] = (768, 512, 256),
        codebook_size: int = 256,
        n_layers: int = 3,
        n_cat_features: int = 0,
        commitment_weight: float = 0.25,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        codebook_distance_l2_normalize: bool = False,
        codebook_use_ema: bool = False,
        codebook_ema_decay: float = 0.99,
        codebook_ema_threshold: float = 1.0,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.0,
    ) -> None:
        self._config = {
            "modalities": list(modalities),
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "hidden_dims": list(hidden_dims),
            "codebook_size": codebook_size,
            "n_layers": n_layers,
            "n_cat_features": n_cat_features,
            "commitment_weight": commitment_weight,
            "codebook_normalize": codebook_normalize,
            "codebook_sim_vq": codebook_sim_vq,
            "codebook_distance_l2_normalize": codebook_distance_l2_normalize,
            "adapter_hidden": adapter_hidden,
            "adapter_dropout": adapter_dropout,
        }
        super().__init__()

        self.modalities = list(modalities)
        self.n_layers = n_layers
        self.codebook_size = codebook_size

        self.adapters = nn.ModuleDict(
            {
                m: ModalityAdapter(
                    dim=input_dim, hidden=adapter_hidden, dropout=adapter_dropout
                )
                for m in self.modalities
            }
        )

        # One shared RQ-VAE (encoder + codebooks + decoder).
        self.rqvae = RqVae(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dims=list(hidden_dims),
            codebook_size=codebook_size,
            codebook_kmeans_init=codebook_kmeans_init,
            codebook_normalize=codebook_normalize,
            codebook_sim_vq=codebook_sim_vq,
            codebook_mode=codebook_mode,
            codebook_distance_l2_normalize=codebook_distance_l2_normalize,
            codebook_use_ema=codebook_use_ema,
            codebook_ema_decay=codebook_ema_decay,
            codebook_ema_threshold=codebook_ema_threshold,
            n_layers=n_layers,
            n_cat_features=n_cat_features,
            commitment_weight=commitment_weight,
        )

    @property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.rqvae.encoder.parameters()).device

    # ---- core ops (reuse RqVae public API) --------------------------------

    def adapt(self, x: Tensor, modality: str) -> Tensor:
        return self.adapters[modality](x)

    def latent(self, x: Tensor, modality: str) -> Tensor:
        """Pre-quantization encoder latent (used by the alignment loss)."""
        return self.rqvae.encode(self.adapt(x, modality))

    @torch.no_grad()
    def codes(self, x: Tensor, modality: str, gumbel_t: float = 0.001) -> Tensor:
        """[B, n_layers] semantic-ID tuple for one modality's embeddings."""
        adapted = self.adapt(x, modality)
        return self.rqvae.get_semantic_ids(adapted, gumbel_t).sem_ids

    # ---- kmeans init over the union of all modalities ---------------------

    @torch.no_grad()
    def kmeans_init(self, x_by_mod: Dict[str, Tensor], gumbel_t: float) -> None:
        """Trigger the shared codebook's k-means init on the union of modalities.

        Adapters start at identity (alpha=0), so the codebook is seeded from the
        raw union distribution — covering every modality's cloud at once, which
        is exactly what prevents the cross-modal collapse documented in
        cross_modal.md.
        """
        adapted = torch.cat(
            [self.adapt(x_by_mod[m], m) for m in self.modalities if m in x_by_mod],
            dim=0,
        )
        # A single pass triggers per-layer _kmeans_init inside each Quantize layer.
        self.rqvae.get_semantic_ids(adapted, gumbel_t)

    # ---- training forward -------------------------------------------------

    def forward(
        self,
        x_by_mod: Dict[str, Tensor],
        gumbel_t: float,
        lam_align: float = 1.0,
    ) -> MultiModalRqVaeOutput:
        recon_terms, quant_terms = [], []
        latents: Dict[str, Tensor] = {}
        codes: Dict[str, Tensor] = {}
        p_unique: Dict[str, Tensor] = {}

        for m, x in x_by_mod.items():
            adapted = self.adapt(x, m)
            quantized = self.rqvae.get_semantic_ids(adapted, gumbel_t)

            # Reconstruct the encoder input (adapter output), mirroring RqVae.forward.
            x_hat = self.rqvae.decode(quantized.embeddings.sum(axis=-1))
            if self.rqvae.n_cat_feats > 0:
                from modules.normalize import l2norm

                x_hat = torch.cat(
                    [
                        l2norm(x_hat[..., : -self.rqvae.n_cat_feats]),
                        x_hat[..., -self.rqvae.n_cat_feats :],
                    ],
                    axis=-1,
                )

            recon = self.rqvae.reconstruction_loss(x_hat, adapted).mean()
            quant = quantized.quantize_loss.mean()
            recon_terms.append(recon)
            quant_terms.append(quant)

            # Pre-quant latent for alignment (one extra encode pass; clear + simple).
            latents[m] = self.rqvae.encode(adapted)
            codes[m] = quantized.sem_ids

            with torch.no_grad():
                sem = quantized.sem_ids
                p_unique[m] = (
                    ~torch.triu(
                        (
                            rearrange(sem, "b d -> b 1 d")
                            == rearrange(sem, "b d -> 1 b d")
                        ).all(axis=-1),
                        diagonal=1,
                    )
                ).all(axis=1).sum() / sem.shape[0]

        recon_loss = torch.stack(recon_terms).mean()
        quant_loss = torch.stack(quant_terms).mean()

        # Cross-modal alignment: same item index across modalities -> same latent.
        mods = list(latents.keys())
        align_pairs, agree_pairs = [], []
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a, b = latents[mods[i]], latents[mods[j]]
                align_pairs.append((1.0 - F.cosine_similarity(a, b, dim=-1)).mean())
                with torch.no_grad():
                    agree = (codes[mods[i]] == codes[mods[j]]).all(dim=-1).float().mean()
                    agree_pairs.append(agree)

        if align_pairs:
            align_loss = torch.stack(align_pairs).mean()
            id_agreement = torch.stack(agree_pairs).mean()
        else:
            zero = recon_loss.new_zeros(())
            align_loss, id_agreement = zero, zero

        loss = recon_loss + quant_loss + lam_align * align_loss

        return MultiModalRqVaeOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            rqvae_loss=quant_loss,
            align_loss=align_loss,
            p_unique_ids=p_unique,
            id_agreement=id_agreement,
        )
