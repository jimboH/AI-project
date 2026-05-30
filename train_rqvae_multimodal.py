#!/usr/bin/env python3
"""Joint shared-codebook RQ-VAE training with per-modality adapters (Design A).

This is the training entry point for the cross-modal improvement described in
RQVAE_improvement.md §5-6. It is a NEW, self-contained script: it does not modify
``train_rqvae.py`` or any existing function. The original per-modality training
remains the default path (run ``train_rqvae.py`` as before).

What it does
------------
- Loads the SAME item indices across every requested modality (all
  ``*_embeddings.pt`` files share the ``asins.json`` ordering, so item ``i`` is the
  same product in every modality — alignment is exact and free).
- Trains ONE shared RQ-VAE (encoder + 3x256 codebooks + decoder) fed by one
  ``ModalityAdapter`` per modality (``modules/rqvae_multimodal.py``).
- Optimises  ``recon + commitment + lam_align * (1 - cos(z_a, z_b))``  summed over
  modalities and averaged over cross-modal pairs, with an alignment warm-up.

Outputs (to ``{save_dir_root}/{category}/multimodal_joint/``)
-------------------------------------------------------------
- ``checkpoint_*.pt`` / ``checkpoint_best.pt`` — ``{model, mm_config, optimizer, ...}``
- ``train.log``     — per-step loss + per-modality batch unique-ID fraction
- ``metrics.jsonl`` — full-corpus per-modality unique ratio + cross-modal ID
                      agreement (the metrics that quantify the improvement).

Usage
-----
  python3 train_rqvae_multimodal.py configs/rqvae_multimodal_joint_all_beauty.gin
"""

import gin
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from modules.quantize import QuantizeForwardMode
from modules.rqvae_multimodal import MultiModalRqVae
from modules.utils import parse_config

EMBEDDING_BASE = Path("/work/u1304848/AI/project/outputs/embeddings")


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("rqvae_mm_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_embeddings(category: str, modality: str) -> torch.Tensor:
    path = EMBEDDING_BASE / category / f"{modality}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. Run precompute_embeddings.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False).float()


@torch.no_grad()
def compute_corpus_codes(
    model: MultiModalRqVae,
    emb: torch.Tensor,
    modality: str,
    device: str,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Return [N, n_layers] semantic IDs for every corpus item of one modality."""
    model.eval()
    out = []
    for start in range(0, emb.shape[0], batch_size):
        x = emb[start : start + batch_size].to(device)
        out.append(model.codes(x, modality, gumbel_t=0.001).cpu())
    return torch.cat(out, dim=0)


@gin.configurable
def train_multimodal(
    # Data
    category: str = "All_Beauty",
    modalities=("text", "image", "multimodal"),
    # Training
    iterations: int = 30000,
    batch_size: int = 640,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    # Alignment
    lam_align: float = 1.0,
    align_warmup_iters: int = 2000,
    # RQ-VAE architecture
    vae_input_dim: int = 2048,
    vae_embed_dim: int = 32,
    vae_hidden_dims=(768, 512, 256),
    vae_codebook_size: int = 256,
    vae_n_layers: int = 3,
    vae_n_cat_feats: int = 0,
    vae_codebook_normalize: bool = False,
    vae_sim_vq: bool = False,
    vae_codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
    commitment_weight: float = 0.25,
    use_kmeans_init: bool = True,
    vae_distance_l2_normalize: bool = False,
    vae_use_ema: bool = False,
    vae_ema_decay: float = 0.99,
    vae_ema_threshold: float = 1.0,
    # Adapter
    adapter_hidden: int = 1024,
    adapter_dropout: float = 0.0,
    # Checkpointing / logging
    save_dir_root: str = "out/rqvae/",
    save_model_every: int = 10000,
    eval_every: int = 10000,
    log_every: int = 1000,
    device: str = None,
):
    modalities = list(modalities)
    vae_hidden_dims = list(vae_hidden_dims)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load aligned embeddings -----------------------------------------
    emb_by_mod = {}
    for m in modalities:
        emb_by_mod[m] = load_embeddings(category, m)
    n_items = next(iter(emb_by_mod.values())).shape[0]
    for m, e in emb_by_mod.items():
        assert e.shape[0] == n_items, (
            f"Modality {m} has {e.shape[0]} items, expected {n_items}; "
            "all modalities must share the asins.json ordering."
        )
    actual_dim = next(iter(emb_by_mod.values())).shape[1]
    if actual_dim != vae_input_dim:
        print(f"Note: embedding dim {actual_dim} != vae_input_dim {vae_input_dim}; using {actual_dim}.")
        vae_input_dim = actual_dim

    # ---- model / optimizer -----------------------------------------------
    model = MultiModalRqVae(
        modalities=modalities,
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
        codebook_kmeans_init=use_kmeans_init,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        codebook_distance_l2_normalize=vae_distance_l2_normalize,
        codebook_use_ema=vae_use_ema,
        codebook_ema_decay=vae_ema_decay,
        codebook_ema_threshold=vae_ema_threshold,
        adapter_hidden=adapter_hidden,
        adapter_dropout=adapter_dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    save_dir = os.path.join(save_dir_root, f"{category}/multimodal_joint/")
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(os.path.join(save_dir, "train.log"))
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    logger.info(
        "Joint multimodal RQ-VAE — category=%s  modalities=%s  iters=%d  n_items=%d  dim=%d",
        category, modalities, iterations, n_items, vae_input_dim,
    )

    def sample_batch():
        idx = torch.randint(0, n_items, (batch_size,))
        return {m: emb_by_mod[m][idx].to(device) for m in modalities}

    # ---- k-means init on the union of modalities -------------------------
    if use_kmeans_init:
        n_init = min(20000, n_items)
        idx = torch.randperm(n_items)[:n_init]
        init_batch = {m: emb_by_mod[m][idx].to(device) for m in modalities}
        model.train()
        model.kmeans_init(init_batch, gumbel_t=0.2)
        logger.info("k-means init done on %d items (union of modalities).", n_init)

    # ---- training loop ----------------------------------------------------
    roll = {k: [] for k in ["loss", "recon", "quant", "align", "agree"]}
    roll_uniq = {m: [] for m in modalities}

    for it in range(iterations):
        model.train()
        lam = lam_align * min(1.0, (it + 1) / max(1, align_warmup_iters))

        x_by_mod = sample_batch()
        out = model(x_by_mod, gumbel_t=0.2, lam_align=lam)

        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        roll["loss"].append(out.loss.item())
        roll["recon"].append(out.reconstruction_loss.item())
        roll["quant"].append(out.rqvae_loss.item())
        roll["align"].append(out.align_loss.item())
        roll["agree"].append(out.id_agreement.item())
        for m in modalities:
            roll_uniq[m].append(out.p_unique_ids[m].item())
        for d in roll.values():
            del d[:-1000]
        for m in modalities:
            roll_uniq[m] = roll_uniq[m][-1000:]

        if it % log_every == 0:
            uniq_str = "  ".join(f"uniq[{m}]={np.mean(roll_uniq[m]):.3f}" for m in modalities)
            logger.info(
                "iter=%d  loss=%.4f  recon=%.4f  quant=%.4f  align=%.4f  agree=%.3f  lam=%.3f  %s",
                it, np.mean(roll["loss"]), np.mean(roll["recon"]), np.mean(roll["quant"]),
                np.mean(roll["align"]), np.mean(roll["agree"]), lam, uniq_str,
            )

        # ---- checkpoint + full-corpus metrics ----------------------------
        is_last = (it + 1 == iterations)
        if (it + 1) % save_model_every == 0 or is_last:
            state = {
                "iter": it,
                "model": model.state_dict(),
                "mm_config": model.config,
                "optimizer": optimizer.state_dict(),
                "category": category,
                "modalities": modalities,
            }
            torch.save(state, os.path.join(save_dir, f"checkpoint_{it}.pt"))
            if is_last:
                torch.save(state, os.path.join(save_dir, "checkpoint_best.pt"))

        if (it + 1) % eval_every == 0 or is_last:
            # Full-corpus codes per modality through the SHARED codebook.
            codes_by_mod = {
                m: compute_corpus_codes(model, emb_by_mod[m], m, device)
                for m in modalities
            }
            rec = {"iter": it}
            for m in modalities:
                c = codes_by_mod[m]
                rec[f"unique_ratio_{m}"] = torch.unique(c, dim=0).shape[0] / c.shape[0]
            # Cross-modal per-item ID agreement (the target metric).
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    a, b = codes_by_mod[modalities[i]], codes_by_mod[modalities[j]]
                    key = f"id_agree_{modalities[i]}_vs_{modalities[j]}"
                    rec[key] = (a == b).all(dim=1).float().mean().item()
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info("iter=%d  [corpus]  %s", it,
                        "  ".join(f"{k}={v:.3f}" for k, v in rec.items() if k != "iter"))

    logger.info("Done. Checkpoints in %s", save_dir)


if __name__ == "__main__":
    parse_config()
    train_multimodal()
