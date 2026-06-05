import torch

from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from typing import NamedTuple
from torch import nn
from torch import Tensor

torch.set_float32_matmul_precision("high")


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class RqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        codebook_distance_l2_normalize: bool = False,
        codebook_use_ema: bool = False,
        codebook_ema_decay: float = 0.99,
        codebook_ema_threshold: float = 1.0,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 0,
    ) -> None:
        self._config = locals()

        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        self.layers = nn.ModuleList(
            modules=[
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    forward_mode=codebook_mode,
                    do_kmeans_init=codebook_kmeans_init,
                    codebook_normalize=i == 0 and codebook_normalize,
                    sim_vq=codebook_sim_vq,
                    commitment_weight=commitment_weight,
                    distance_l2_normalize=codebook_distance_l2_normalize,
                    use_ema=codebook_use_ema,
                    ema_decay=codebook_ema_decay,
                    ema_threshold=codebook_ema_threshold,
                )
                for i in range(n_layers)
            ]
        )

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=False,
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features)
            if n_cat_features != 0
            else ReconstructionLoss()
        )

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        # EMA buffers (ema_cluster_size, ema_embed_sum) are only used during
        # training; ignore them if the checkpoint has them but this instance
        # was created without use_ema=True.
        self.load_state_dict(state["model"], strict=False)
        print(f"---Loaded RQVAE Iter {state['iter']}---")

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(self, x: Tensor, gumbel_t: float = 0.001) -> RqVaeOutput:
        x = x.to(next(self.encoder.parameters()).dtype)
        res = self.encode(x)

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.layers:
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss,
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decode(embs.sum(axis=-1))

        if self.n_cat_feats > 0:
            x_hat = torch.cat(
                [l2norm(x_hat[..., : -self.n_cat_feats]), x_hat[..., -self.n_cat_feats :]],
                axis=-1,
            )

        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        loss = (reconstuction_loss + rqvae_loss).mean()

        with torch.no_grad():
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )

    def forward_pair(
        self, x_src: Tensor, x_tgt: Tensor, gumbel_t: float
    ) -> RqVaeComputedLosses:
        """Cross-modal forward pass for alignment-pair training.

        Encodes ``x_src``, passes through the residual codebook stack, then
        decodes and computes the reconstruction loss against ``x_tgt``.  This
        enables all four alignment-pair types:

            text  → text   (x_src = text_emb,  x_tgt = text_emb)
            image → image  (x_src = image_emb, x_tgt = image_emb)
            text  → image  (x_src = text_emb,  x_tgt = image_emb)
            image → text   (x_src = image_emb, x_tgt = text_emb)

        Because SigLIP places text and image embeddings in the same 1152-dim
        space, the same encoder, codebook, and decoder handle all pair types
        without any architectural change.

        Parameters
        ----------
        x_src : Tensor  (B, input_dim)  — source modality embedding (encoder input)
        x_tgt : Tensor  (B, input_dim)  — target modality embedding (decoder target)
        gumbel_t : float                — Gumbel-softmax temperature

        Returns
        -------
        RqVaeComputedLosses
        """
        x_src = x_src.to(next(self.encoder.parameters()).dtype)
        x_tgt = x_tgt.to(next(self.encoder.parameters()).dtype)

        quantized = self.get_semantic_ids(x_src, gumbel_t)
        embs = quantized.embeddings  # (n_layers, embed_dim, B)
        x_hat = self.decode(embs.sum(axis=-1))  # (B, input_dim)

        # Reconstruct against the TARGET modality, not the source
        reconstruction_loss = self.reconstruction_loss(x_hat, x_tgt)
        rqvae_loss = quantized.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        with torch.no_grad():
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
