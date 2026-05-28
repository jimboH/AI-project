#!/usr/bin/env python3
"""Train the sequential recommendation model (Qwen/Qwen3.5-0.8B).

The model encodes user interaction history (represented as semantic-ID
sequences produced by a pre-trained RQ-VAE) with Qwen3.5-0.8B and uses
the hidden state at the last valid position to predict the next item's
semantic IDs via three linear classification heads.

Dataset splits follow the leave-one-out protocol (see data/amazon2023.py):
  - Train : history = [i_1,...,i_{n-3}], target = i_{n-2}
  - Valid : history = [i_1,...,i_{n-2}], target = i_{n-1}
  - Test  : history = [i_1,...,i_{n-1}], target = i_n

Validation (hit@k, NDCG@k, eval_loss) runs every eval_every steps for all
modalities (text, image, multimodal).  The best checkpoint is saved whenever
hit@1 improves.

Training quality monitoring
---------------------------
Two output files are written to the checkpoint directory:
- train.log     : human-readable log; training loss every log_every steps,
                  validation results every eval_every steps.
- metrics.jsonl : machine-readable JSON lines emitted every eval_every steps,
                  containing eval_loss, hit@k, and ndcg@k.

The best checkpoint is saved whenever NDCG@10 (valid) improves.

Usage
-----
  python3 train_decoder.py <config_path>

  # Example:
  python3 train_decoder.py configs/decoder_text_all_beauty.gin
"""

import gin
import json
import logging
import os
import torch
import wandb

from accelerate import Accelerator
from data.amazon2023 import (
    ItemEmbeddingDataset,
    SequentialRecommendationDataset,
    build_asin_index,
    load_metadata,
    load_sequential_data,
    METADATA_FILES,
)
from data.utils import batch_to, cycle, next_batch
from evaluate.metrics import TopKAccumulator, NDCGAccumulator
from modules.model import QwenRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics, parse_config
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

EMBEDDING_BASE = Path("/work/u1304848/AI/project/outputs/embeddings")
HF_CACHE = str(Path("/work/u1304848/AI/project/datasets/hf_cache"))


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("decoder_train")
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


def load_asin_index(category: str):
    path = EMBEDDING_BASE / category / "asins.json"
    with open(path) as f:
        asins = json.load(f)
    asin2idx = {a: i for i, a in enumerate(asins)}
    return asins, asin2idx


@gin.configurable
def train(
    # Data
    category: str = "All_Beauty",
    modality: str = "text",          # "text" | "image" | "multimodal"
    max_seq_len: int = 20,
    # Training
    iterations: int = 100000,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_accumulate_every: int = 1,
    max_grad_norm: float = None,
    train_data_subsample: bool = True,
    # RQ-VAE (must match the trained checkpoint)
    vae_input_dim: int = 1536,
    vae_embed_dim: int = 32,
    vae_hidden_dims=(768, 512, 256),
    vae_codebook_size: int = 256,
    vae_codebook_normalize: bool = False,
    vae_sim_vq: bool = False,
    vae_n_cat_feats: int = 0,
    vae_n_layers: int = 3,
    pretrained_rqvae_path: str = None,
    # Qwen3.5-0.8B model
    qwen_model_name: str = "Qwen/Qwen3.5-0.8B",
    freeze_encoder: bool = False,
    top_k_for_generation: int = 10,
    should_add_sep_token: bool = True,
    num_user_bins: int = None,
    warmup_steps: int = 10000,
    # Checkpointing
    save_dir_root: str = "out/decoder/",
    run_name: str = None,
    pretrained_decoder_path: str = None,
    save_model_every: int = 2000,
    eval_every: int = 2000,
    top_k_eval_list=(1, 5, 10),
    # Accelerate
    split_batches: bool = True,
    amp: bool = False,
    mixed_precision_type: str = "fp16",
    # Logging
    log_every: int = 100,
    wandb_logging: bool = False,
    force_seq_reload: bool = False,
):
    if wandb_logging:
        params = locals()

    vae_hidden_dims = list(vae_hidden_dims)
    top_k_eval_list = list(top_k_eval_list)

    # Validation runs for all modalities
    should_validate = True

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    # Load ASIN index and embeddings
    asins, asin2idx = load_asin_index(category)
    all_embeddings = load_embeddings(category, modality)

    actual_dim = all_embeddings.shape[1]
    if actual_dim != vae_input_dim:
        if accelerator.is_main_process:
            print(f"Updating vae_input_dim: {vae_input_dim} → {actual_dim}")
        vae_input_dim = actual_dim

    # Load sequential data (leave-one-out splits)
    if accelerator.is_main_process:
        print(f"Loading sequential data for {category}...")
    seq_splits = load_sequential_data(
        category=category,
        asin2idx=asin2idx,
        cache_dir=HF_CACHE,
    )

    train_dataset = SequentialRecommendationDataset(
        embeddings=all_embeddings,
        split_data=seq_splits["train"],
        max_seq_len=max_seq_len,
        subsample=train_data_subsample,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)

    eval_dataset = SequentialRecommendationDataset(
        embeddings=all_embeddings,
        split_data=seq_splits["valid"],
        max_seq_len=max_seq_len,
        subsample=False,
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # Item dataset for corpus ID precomputation
    item_dataset = ItemEmbeddingDataset(all_embeddings, split="all")

    # Tokenizer: wraps frozen RQ-VAE to convert embeddings → semantic IDs
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
    )
    tokenizer = accelerator.prepare(tokenizer)

    if accelerator.is_main_process:
        print("Precomputing corpus semantic IDs...")
    tokenizer.precompute_corpus_ids(item_dataset)

    codebooks = tokenizer.cached_ids[:, :vae_n_layers].cpu()

    model = QwenRetrievalModel(
        codebooks=codebooks,
        num_hierarchies=vae_n_layers,
        num_embeddings_per_hierarchy=vae_codebook_size,
        qwen_model_name=qwen_model_name,
        freeze_encoder=freeze_encoder,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )

    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=warmup_steps)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    metrics_accumulator = TopKAccumulator(ks=top_k_eval_list)
    ndcg_accumulator = NDCGAccumulator(ks=top_k_eval_list)
    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"Device: {device}, Num parameters: {num_params:,}")

    best_ndcg10 = -1.0

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project="gen-retrieval-decoder-training", config=params)

    dir_name = run_name if run_name else modality
    save_dir = os.path.join(save_dir_root, f"{category}/{dir_name}/")

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logger = setup_logger(os.path.join(save_dir, "train.log"))
        metrics_path = os.path.join(save_dir, "metrics.jsonl")
        logger.info(
            "Starting training — category=%s  modality=%s  iterations=%d  "
            "n_train=%d  n_val=%d",
            category, modality, iterations, len(train_dataset), len(eval_dataset),
        )
    else:
        logger = None
        metrics_path = None

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        for iter in range(start_iter, start_iter + iterations):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            train_debug_metrics = {}

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every

                total_loss += loss.detach().item()

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data)

                accelerator.backward(loss)

            pbar.set_description(f"loss: {total_loss:.4f}")

            accelerator.wait_for_everyone()

            if max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            # Log training loss every log_every steps
            if accelerator.is_main_process and (iter + 1) % log_every == 0:
                logger.info(
                    "iter=%d  train_loss=%.4f  lr=%.2e",
                    iter + 1, total_loss, optimizer.param_groups[0]["lr"],
                )

            # Validation every eval_every steps — all modalities
            if should_validate and (iter + 1) % eval_every == 0:
                model.eval()
                eval_loss = 0.0
                n_eval = 0

                with tqdm(
                    eval_dataloader,
                    desc=f"Eval {iter + 1}",
                    disable=not accelerator.is_main_process,
                ) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        with torch.no_grad():
                            model_out = model(tokenized_data)
                            eval_loss += model_out.loss.item()
                            n_eval += 1

                            generated = model.generate_next_sem_id(
                                tokenized_data, top_k=True, temperature=1
                            )

                        actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
                        metrics_accumulator.accumulate(
                            actual=actual, top_k=generated.sem_ids
                        )
                        ndcg_accumulator.accumulate(
                            actual=actual, top_k=generated.sem_ids
                        )

                eval_loss = eval_loss / n_eval if n_eval > 0 else float("inf")
                eval_metrics = {
                    **metrics_accumulator.reduce(),
                    **ndcg_accumulator.reduce(),
                }
                metrics_accumulator.reset()
                ndcg_accumulator.reset()

                ndcg10 = float(eval_metrics.get("ndcg@10", 0.0))

                if accelerator.is_main_process:
                    metrics_str = "  ".join(
                        f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items())
                    )
                    logger.info(
                        "iter=%d  [eval]  loss=%.4f  %s",
                        iter + 1, eval_loss, metrics_str,
                    )

                    record = {
                        "iter": iter + 1,
                        "eval_loss": float(eval_loss),
                        **{k: float(v) for k, v in eval_metrics.items()},
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    if ndcg10 > best_ndcg10:
                        best_ndcg10 = ndcg10
                        state = {
                            "iter": iter,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict(),
                            "category": category,
                            "modality": modality,
                        }
                        torch.save(state, os.path.join(save_dir, "checkpoint_best.pt"))
                        logger.info(
                            "iter=%d  [new best]  ndcg@10=%.4f", iter + 1, ndcg10
                        )

                    if wandb_logging:
                        wandb.log({"eval_loss": eval_loss, "iter": iter, **eval_metrics})

            if accelerator.is_main_process:
                state = {
                    "iter": iter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "category": category,
                    "modality": modality,
                }
                os.makedirs(save_dir, exist_ok=True)

                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    torch.save(state, os.path.join(save_dir, f"checkpoint_{iter}.pt"))

                if wandb_logging:
                    wandb.log(
                        {
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": total_loss,
                            **train_debug_metrics,
                        }
                    )

            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
