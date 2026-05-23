import gin
import torch

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """4.2 in https://arxiv.org/abs/2410.06424"""
    e = rearrange(e, "b d -> b 1 d")
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    ).squeeze()


class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        distance_l2_normalize: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_threshold: float = 1.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.distance_l2_normalize = distance_l2_normalize
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_threshold = ema_threshold

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity(),
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)

        if use_ema:
            # Initialise cluster sizes at the threshold so that k-means-initialised
            # entries survive the first batch (entries assigned ≥1 sample will
            # immediately rise above threshold; truly unassigned ones will fall
            # below it and be restarted).
            self.register_buffer(
                "ema_cluster_size", torch.ones(n_embed) * ema_threshold
            )
            self.register_buffer("ema_embed_sum", torch.zeros(n_embed, embed_dim))

        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @torch.no_grad()
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))

    @torch.no_grad()
    def _ema_update(self, ids: Tensor, x: Tensor) -> None:
        """Update codebook via EMA and restart dead entries from encoder outputs.

        Uses pure tensor ops (no .item() / Python branching on tensor values) so
        that the call is compatible with torch.compile.
        """
        x_f = x.detach().float()
        encodings = F.one_hot(ids, self.n_embed).float()  # (B, n_embed)

        new_cluster_size = encodings.sum(0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            new_cluster_size, alpha=1.0 - self.ema_decay
        )

        new_embed_sum = encodings.T @ x_f  # (n_embed, embed_dim)
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            new_embed_sum, alpha=1.0 - self.ema_decay
        )

        # Laplace-smoothed centroid estimate
        n = self.ema_cluster_size.sum()
        smooth_size = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.n_embed * 1e-5)
            * n
        )
        updated_weight = self.ema_embed_sum / smooth_size.unsqueeze(1)

        # Random restart: for each dead entry, replace it with a random encoder
        # output from the current batch.  We use soft masking so the whole
        # operation stays in-graph for torch.compile.
        dead = (self.ema_cluster_size < self.ema_threshold).float()  # (n_embed,)
        rand_idx = torch.randint(0, x_f.shape[0], (self.n_embed,), device=x_f.device)
        random_vecs = x_f[rand_idx]  # (n_embed, embed_dim)

        updated_weight = (
            dead.unsqueeze(1) * random_vecs
            + (1.0 - dead).unsqueeze(1) * updated_weight
        )
        self.ema_cluster_size.data.copy_(
            dead * self.ema_threshold + (1.0 - dead) * self.ema_cluster_size
        )
        self.ema_embed_sum.data.copy_(
            dead.unsqueeze(1) * random_vecs * self.ema_threshold
            + (1.0 - dead).unsqueeze(1) * self.ema_embed_sum
        )

        self.embedding.weight.data.copy_(updated_weight)

    def forward(self, x, temperature) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)

        # Project both inputs and codebook onto the unit hypersphere before
        # computing distances, so that nearest-neighbour search is purely
        # directional and invariant to encoder output scale.
        if self.distance_l2_normalize:
            x_q = F.normalize(x, p=2, dim=-1)
            codebook_q = F.normalize(codebook, p=2, dim=-1)
        else:
            x_q = x
            codebook_q = codebook

        if self.distance_mode == QuantizeDistance.L2:
            dist = (
                (x_q**2).sum(axis=1, keepdim=True)
                + (codebook_q.T**2).sum(axis=0, keepdim=True)
                - 2 * x_q @ codebook_q.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                x_q
                / x_q.norm(dim=1, keepdim=True)
                @ (codebook_q.T)
                / codebook_q.T.norm(dim=0, keepdim=True)
            )
        else:
            raise Exception("Unsupported Quantize distance mode.")

        _, ids = (dist.detach()).min(axis=1)

        if self.use_ema and self.training:
            self._ema_update(ids, x)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = weights @ codebook
                emb_out = emb
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x,
                )
                emb_out = (
                    emb_out
                    * (
                        torch.norm(emb, dim=1, keepdim=True)
                        / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                    ).detach()
                )
            else:
                raise Exception("Unsupported Quantize forward mode.")

            loss = self.quantize_loss(query=x, value=emb, ema_update=self.use_ema)

        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out, ema_update=self.use_ema)

        return QuantizeOutput(embeddings=emb_out, ids=ids, loss=loss)
