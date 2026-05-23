#!/usr/bin/env python3
"""Train Residual-Quantized Variational AutoEncoder (RQ-VAE) on item embeddings.

The RQ-VAE compresses item embeddings into discrete semantic IDs (codebook tuples),
which are then used as item tokens for sequential recommendation.

All embeddings are used for training — no train/eval split.

Modalities
----------
All three modalities use Qwen/Qwen3-VL-Embedding-2B (1536-dim):
- text       : text-only inputs via the VL chat template
- image      : image-only inputs via the VL chat template
- multimodal : combined text + image inputs via the VL chat template

Training quality monitoring
---------------------------
Two output files are written to the checkpoint directory:
- train.log     : human-readable per-step log with loss + p_unique_ids
- metrics.jsonl : machine-readable JSON lines emitted every eval_every steps,
                  containing loss, per-layer codebook utilisation, corpus entropy,
                  and max collision rate — suitable for plotting.

Usage
-----
  python3 train_rqvae.py <config_path>

  # Example:
  python3 train_rqvae.py configs/rqvae_text_all_beauty.gin
"""

import gin
import json
import logging
import os
import numpy as np
import torch

from accelerate import Accelerator
from data.amazon2023 import ItemEmbeddingDataset
from data.utils import batch_to, cycle, next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from tqdm import tqdm
from pathlib import Path

EMBEDDING_BASE = Path("/work/u1304848/AI/project/outputs/embeddings")


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("rqvae_train")
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
            f"Embeddings not found at {path}. "
            "Run precompute_embeddings.py first."
        )
    emb = torch.load(path, map_location="cpu", weights_only=False)
    return emb.float()


@gin.configurable
def train(
    # Data
    category: str = "All_Beauty",
    modality: str = "text",          # "text" | "image" | "multimodal"
    # Training
    iterations: int = 200000,
    batch_size: int = 640,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_accumulate_every: int = 1,
    # RQ-VAE architecture (dims depend on encoder output)
    vae_input_dim: int = 1024,
    vae_embed_dim: int = 32,
    vae_hidden_dims=(512, 256, 128),
    vae_codebook_size: int = 256,
    vae_codebook_normalize: bool = False,
    vae_codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
    vae_sim_vq: bool = False,
    vae_n_layers: int = 3,
    vae_n_cat_feats: int = 0,
    commitment_weight: float = 0.25,
    use_kmeans_init: bool = True,
    vae_distance_l2_normalize: bool = False,
    vae_use_ema: bool = False,
    vae_ema_decay: float = 0.99,
    vae_ema_threshold: float = 1.0,
    # Checkpointing
    save_dir_root: str = "out/rqvae/",
    pretrained_rqvae_path: str = None,
    save_model_every: int = 10000,
    eval_every: int = 10000,   # interval for full-corpus codebook quality metrics
    # Accelerate
    split_batches: bool = True,
    amp: bool = False,
    mixed_precision_type: str = "fp16",
    # Logging
    log_every: int = 10000,
    wandb_logging: bool = False,
):
    vae_hidden_dims = list(vae_hidden_dims)

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    # Load precomputed embeddings
    if accelerator.is_main_process:
        print(f"Loading {modality} embeddings for {category}...")
    all_embeddings = load_embeddings(category, modality)

    # Check input dimension matches config
    actual_dim = all_embeddings.shape[1]
    if actual_dim != vae_input_dim:
        if accelerator.is_main_process:
            print(
                f"Warning: embedding dim {actual_dim} != vae_input_dim {vae_input_dim}. "
                "Updating vae_input_dim."
            )
        vae_input_dim = actual_dim

    # Use ALL embeddings for training — no train/eval split
    train_dataset = ItemEmbeddingDataset(all_embeddings, split="all")
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    train_dataloader = cycle(train_dataloader)

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        codebook_distance_l2_normalize=vae_distance_l2_normalize,
        codebook_use_ema=vae_use_ema,
        codebook_ema_decay=vae_ema_decay,
        codebook_ema_threshold=vae_ema_threshold,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
    )

    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    save_dir = os.path.join(save_dir_root, f"{category}/{modality}/")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logger = setup_logger(os.path.join(save_dir, "train.log"))
        metrics_path = os.path.join(save_dir, "metrics.jsonl")
        logger.info(
            "Starting training — category=%s  modality=%s  iterations=%d  n_items=%d",
            category, modality, iterations, len(train_dataset),
        )
    else:
        logger = None
        metrics_path = None

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(
            pretrained_rqvae_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
        rqvae_distance_l2_normalize=vae_distance_l2_normalize,
    )
    tokenizer.rq_vae = model

    # Rolling windows: [total_loss, recon_loss, rqvae_loss, p_unique_ids]
    # p_unique_ids: fraction of unique code-tuples within each training batch (cheap proxy for ID collision)
    print_loss = print_rec_loss = print_vae_loss = print_p_unique = 0.0

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        losses = [[], [], [], []]  # total, recon, rqvae, p_unique_ids
        for iter in range(start_iter, start_iter + iterations):
            model.train()
            total_loss = 0
            t = 0.2

            if iter == 0 and use_kmeans_init:
                kmeans_init_data = batch_to(
                    train_dataset[torch.arange(min(20000, len(train_dataset)))], device
                )
                with accelerator.autocast():
                    model(kmeans_init_data, t)

            optimizer.zero_grad()

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[3].append(model_output.p_unique_ids.cpu().item())
            for i in range(4):
                losses[i] = losses[i][-1000:]

            if iter % log_every == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])
                print_p_unique = np.mean(losses[3])
                if accelerator.is_main_process:
                    logger.info(
                        "iter=%d  loss=%.4f  recon=%.4f  rqvae=%.4f  p_unique_ids=%.4f",
                        iter, print_loss, print_rec_loss, print_vae_loss, print_p_unique,
                    )

            pbar.set_description(
                f"loss:{print_loss:.4f} rl:{print_rec_loss:.4f} vl:{print_vae_loss:.4f} uniq:{print_p_unique:.3f}"
            )

            accelerator.wait_for_everyone()
            optimizer.step()
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                state = {
                    "iter": iter,
                    "model": model.state_dict(),
                    "model_config": model.config,
                    "optimizer": optimizer.state_dict(),
                    "category": category,
                    "modality": modality,
                }
                os.makedirs(save_dir, exist_ok=True)

                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    torch.save(state, os.path.join(save_dir, f"checkpoint_{iter}.pt"))

                if iter + 1 == iterations:
                    torch.save(state, os.path.join(save_dir, "checkpoint_best.pt"))

                # Full-corpus codebook quality metrics
                if (iter + 1) % eval_every == 0 or iter + 1 == iterations:
                    tokenizer.reset()
                    model.eval()

                    corpus_ids = tokenizer.precompute_corpus_ids(train_dataset)

                    # Fraction of corpus items sharing the most-duplicated code-tuple
                    max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]

                    # Entropy of the code-tuple distribution (higher = more uniform)
                    _, counts = torch.unique(corpus_ids[:, :-1], dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p * torch.log(p)).sum()

                    # Per-layer codebook utilisation: fraction of the 256 codes actually assigned
                    codebook_usage = {}
                    for cid in range(vae_n_layers):
                        used = len(torch.unique(corpus_ids[:, cid]))
                        codebook_usage[f"codebook_usage_{cid}"] = used / vae_codebook_size

                    metrics_record = {
                        "iter": iter,
                        "train_loss": float(print_loss),
                        "recon_loss": float(print_rec_loss),
                        "rqvae_loss": float(print_vae_loss),
                        "p_unique_ids_batch": float(print_p_unique),
                        "entropy": float(rqvae_entropy.cpu().item()),
                        "max_duplicates_frac": float(max_duplicates.cpu().item()),
                        **{k: float(v) for k, v in codebook_usage.items()},
                    }

                    usage_str = "  ".join(f"{k}={v:.3f}" for k, v in codebook_usage.items())
                    logger.info(
                        "iter=%d  [corpus]  entropy=%.4f  max_dup_frac=%.4f  %s",
                        iter,
                        metrics_record["entropy"],
                        metrics_record["max_duplicates_frac"],
                        usage_str,
                    )

                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(metrics_record) + "\n")

            pbar.update(1)


if __name__ == "__main__":
    parse_config()
    train()
