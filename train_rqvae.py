#!/usr/bin/env python3
"""Train Residual-Quantized Variational AutoEncoder (RQ-VAE) on item embeddings.

The RQ-VAE compresses item embeddings into discrete semantic IDs (codebook tuples),
which are then used as item tokens for sequential recommendation.

All embeddings are used for training — no train/eval split.

Modalities
----------
All modalities use google/siglip-so400m-patch14-384 (1152-dim):
- text        : text-only SigLIP text-tower embeddings
- image       : image-only SigLIP vision-tower embeddings
- multimodal  : L2-normalised average of text + image embeddings
- cross_modal : all four alignment-pair types trained jointly:
                  text  → text   (pair_type 0)
                  image → image  (pair_type 1)
                  text  → image  (pair_type 2)
                  image → text   (pair_type 3)
                Items without a valid image contribute only text→text pairs.
                All present pair types are sampled with equal probability.

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
  python3 train_rqvae.py configs/rqvae_cross_modal_all_beauty.gin
"""

import gin
import json
import logging
import os
import numpy as np
import torch

from accelerate import Accelerator
from data.amazon2023 import ItemEmbeddingDataset
from data.schemas import PairBatch
from data.utils import batch_to, cycle, next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from pathlib import Path

EMBEDDING_BASE = Path(__file__).resolve().parent / "outputs" / "embeddings"

# Integer codes for the four pair types
PAIR_TEXT_TEXT   = 0
PAIR_IMAGE_IMAGE = 1
PAIR_TEXT_IMAGE  = 2
PAIR_IMAGE_TEXT  = 3


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


# ---------------------------------------------------------------------------
# Cross-modal pair dataset
# ---------------------------------------------------------------------------

class CrossModalPairDataset(Dataset):
    """Dataset that enumerates all valid alignment pairs for every item.

    For each item ``i`` the following pairs are produced:
      - (text_emb[i],  text_emb[i])  — pair_type 0  (only if has_text[i])
      - (image_emb[i], image_emb[i]) — pair_type 1  (only if has_image[i])
      - (text_emb[i],  image_emb[i]) — pair_type 2  (only if has_text[i] AND has_image[i])
      - (image_emb[i], text_emb[i])  — pair_type 3  (only if has_text[i] AND has_image[i])

    Items are detected as having a valid modality when their embedding's L2
    norm exceeds 1e-6.  Zero-vector rows indicate a masked/missing modality
    (as written by apply_modality_mask_to_embeddings.py for the limited
    dataset, or the original zero-image convention for the full dataset).

    Single-modality items contribute only their available same-modality pair
    (text→text or image→image).  Cross-modal pairs (text→image, image→text)
    are emitted only when both modalities are present, so the model is never
    trained to bridge a genuinely absent modality.

    Each dataset entry is a flat (x_src, x_tgt, pair_type) triple.  A
    shuffled DataLoader over this dataset naturally interleaves all pair
    types, producing equal-probability sampling across the batch.

    Parameters
    ----------
    text_emb  : (N, D) float32 tensor — zero vectors where text is missing
    image_emb : (N, D) float32 tensor — zero vectors where image is missing
    """

    def __init__(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> None:
        super().__init__()
        assert text_emb.shape == image_emb.shape

        self.text_emb  = text_emb.float()
        self.image_emb = image_emb.float()

        # Detect items that have a real (non-zero) embedding for each modality
        has_text  = text_emb.norm(dim=-1)  > 1e-6  # (N,)
        has_image = image_emb.norm(dim=-1) > 1e-6  # (N,)

        # Build the list of (item_idx, pair_type) entries
        records = []
        for i in range(len(text_emb)):
            if has_text[i]:
                records.append((i, PAIR_TEXT_TEXT))
            if has_image[i]:
                records.append((i, PAIR_IMAGE_IMAGE))
            # Cross-modal pairs only when BOTH modalities are present
            if has_text[i] and has_image[i]:
                records.append((i, PAIR_TEXT_IMAGE))
                records.append((i, PAIR_IMAGE_TEXT))

        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        item_i, pair_type = self._records[idx]

        if pair_type == PAIR_TEXT_TEXT:
            x_src = self.text_emb[item_i]
            x_tgt = self.text_emb[item_i]
        elif pair_type == PAIR_IMAGE_IMAGE:
            x_src = self.image_emb[item_i]
            x_tgt = self.image_emb[item_i]
        elif pair_type == PAIR_TEXT_IMAGE:
            x_src = self.text_emb[item_i]
            x_tgt = self.image_emb[item_i]
        else:  # PAIR_IMAGE_TEXT
            x_src = self.image_emb[item_i]
            x_tgt = self.text_emb[item_i]

        return x_src, x_tgt, torch.tensor(pair_type, dtype=torch.long)


def collate_pairs(batch):
    x_srcs, x_tgts, ptypes = zip(*batch)
    return PairBatch(
        x_src=torch.stack(x_srcs),
        x_tgt=torch.stack(x_tgts),
        pair_type=torch.stack(ptypes),
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

@gin.configurable
def train(
    # Data
    category: str = "All_Beauty",
    modality: str = "text",          # "text" | "image" | "multimodal" | "cross_modal"
    modality_mask_path: str = None,  # path to limited_modality_mask.json; when set,
                                     # loads limited_{text,image}_embeddings.pt and
                                     # restricts cross-modal pairs per item's status
    # Training
    iterations: int = 200000,
    batch_size: int = 640,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_accumulate_every: int = 1,
    # RQ-VAE architecture (dims depend on encoder output)
    vae_input_dim: int = 1152,
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

    # -----------------------------------------------------------------------
    # Load precomputed embeddings
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Loading {modality} embeddings for {category}...")

    if modality == "cross_modal":
        # When a modality mask is provided, use the limited embedding variants
        # (produced by apply_modality_mask_to_embeddings.py) which have zero
        # rows for items whose modality is suppressed.  Otherwise fall back to
        # the standard full embeddings.
        if modality_mask_path is not None:
            text_emb_name  = "limited_text"
            image_emb_name = "limited_image"
            if accelerator.is_main_process:
                print(f"  modality_mask_path={modality_mask_path} — using limited embeddings")
        else:
            text_emb_name  = "text"
            image_emb_name = "image"

        text_embeddings  = load_embeddings(category, text_emb_name)
        image_embeddings = load_embeddings(category, image_emb_name)

        # Verify dimensions match
        assert text_embeddings.shape == image_embeddings.shape, (
            f"text and image embeddings must have the same shape; "
            f"got {text_embeddings.shape} vs {image_embeddings.shape}"
        )
        actual_dim = text_embeddings.shape[1]

        # Build cross-modal pair dataset (handles missing modalities via zero-norm detection)
        pair_dataset = CrossModalPairDataset(text_embeddings, image_embeddings)

        n_text_only = sum(1 for _, pt in pair_dataset._records if pt == PAIR_TEXT_TEXT)
        n_image_img = sum(1 for _, pt in pair_dataset._records if pt == PAIR_IMAGE_IMAGE)
        n_tex_img   = sum(1 for _, pt in pair_dataset._records if pt == PAIR_TEXT_IMAGE)
        n_img_tex   = sum(1 for _, pt in pair_dataset._records if pt == PAIR_IMAGE_TEXT)

        if accelerator.is_main_process:
            has_text_count  = text_embeddings.norm(dim=-1).gt(1e-6).sum().item()
            has_image_count = image_embeddings.norm(dim=-1).gt(1e-6).sum().item()
            n_items = len(text_embeddings)
            print(
                f"  Items: {n_items:,}  "
                f"(has_text: {int(has_text_count):,}  "
                f"has_image: {int(has_image_count):,}  "
                f"both: {int((text_embeddings.norm(dim=-1).gt(1e-6) & image_embeddings.norm(dim=-1).gt(1e-6)).sum().item()):,})"
            )
            print(
                f"  Pairs: text→text={n_text_only:,}  "
                f"image→image={n_image_img:,}  "
                f"text→image={n_tex_img:,}  "
                f"image→text={n_img_tex:,}  "
                f"total={len(pair_dataset):,}"
            )

        # Use a plain DataLoader with shuffle=True — CrossModalPairDataset
        # __getitem__ expects a single integer index, not a batch list, so we
        # cannot use the BatchSampler(batch_size=None) pattern used elsewhere.
        train_dataloader = DataLoader(
            pair_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_pairs,
            drop_last=True,
        )
        train_dataloader = cycle(train_dataloader)

        # For k-means init and eval, build a composite corpus embedding: use the
        # text embedding when it is non-zero (text_only or both items), and fall
        # back to the image embedding for image_only items (zero text row).
        # This ensures every item contributes a meaningful vector regardless of
        # which modality is available.
        text_norms  = text_embeddings.norm(dim=-1, keepdim=True)   # (N, 1)
        composite   = torch.where(text_norms > 1e-6, text_embeddings, image_embeddings)
        all_embeddings = composite
        train_dataset  = ItemEmbeddingDataset(all_embeddings, split="all")

    else:
        all_embeddings = load_embeddings(category, modality)
        actual_dim = all_embeddings.shape[1]
        train_dataset = ItemEmbeddingDataset(all_embeddings, split="all")
        train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,
            collate_fn=lambda batch: batch,
        )
        train_dataloader = cycle(train_dataloader)

    # Check input dimension matches config
    if actual_dim != vae_input_dim:
        if accelerator.is_main_process:
            print(
                f"Warning: embedding dim {actual_dim} != vae_input_dim {vae_input_dim}. "
                "Updating vae_input_dim."
            )
        vae_input_dim = actual_dim

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
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

    # Use a distinct subdirectory for limited-modality runs so checkpoints do
    # not overwrite the full-data cross_modal run.
    modality_subdir = f"{modality}_limited" if modality_mask_path is not None else modality
    save_dir = os.path.join(save_dir_root, f"{category}/{modality_subdir}/")
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

            # K-means codebook initialisation on first step
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
                    if modality == "cross_modal":
                        # data is a PairBatch — use forward_pair for cross-modal loss
                        model_output = model.forward_pair(
                            data.x_src, data.x_tgt, gumbel_t=t
                        )
                    else:
                        # data is a SeqBatch — standard single-modality reconstruction
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
                print_loss     = np.mean(losses[0])
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

                # Full-corpus codebook quality metrics (evaluated on text embeddings)
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

                    # Per-layer codebook utilisation
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
