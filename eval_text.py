#!/usr/bin/env python3
"""Evaluate all decoder checkpoints in out/decoder/All_Beauty/text/ on both
validation and test splits, and save per-checkpoint metrics.

The tokenizer (RQ-VAE) and dataloaders are built once; only the decoder
weights are swapped per checkpoint to keep runtime efficient.

Usage
-----
  python3 eval_text.py

  # Override defaults:
  python3 eval_text.py \\
      --checkpoint_dir out/decoder/All_Beauty/text \\
      --rqvae_dir      out/rqvae/ \\
      --category       All_Beauty \\
      --output_file    out/eval_text_results.json \\
      --device         cuda
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.amazon2023 import (
    ItemEmbeddingDataset,
    SequentialRecommendationDataset,
    load_sequential_data,
)
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator, NDCGAccumulator
from modules.model import QwenRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer

EMBEDDING_BASE = Path("/work/u1304848/AI/project/outputs/embeddings")
HF_CACHE = str(Path("/work/u1304848/AI/project/datasets/hf_cache"))

TOP_K_LIST = [1, 5, 10]
BATCH_SIZE = 32
MAX_SEQ_LEN = 20


def _sorted_checkpoints(ckpt_dir: Path) -> list[Path]:
    return sorted(
        (p for p in ckpt_dir.glob("checkpoint_*.pt") if p.stem.split("_")[-1].isdigit()),
        key=lambda p: int(p.stem.split("_")[-1]),
    )


def _load_embeddings(category: str, modality: str) -> torch.Tensor:
    path = EMBEDDING_BASE / category / f"{modality}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}. Run precompute_embeddings.py first.")
    return torch.load(path, map_location="cpu", weights_only=False).float()


def _load_asin_index(category: str):
    path = EMBEDDING_BASE / category / "asins.json"
    with open(path) as f:
        asins = json.load(f)
    return asins, {a: i for i, a in enumerate(asins)}


def _find_rqvae_checkpoint(rqvae_dir: Path, category: str, modality: str) -> Path:
    ckpt_dir = rqvae_dir / category / modality
    best = ckpt_dir / "checkpoint_best.pt"
    if best.exists():
        return best
    candidates = sorted(
        (p for p in ckpt_dir.glob("checkpoint_*.pt") if p.stem.split("_")[-1].isdigit()),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No RQ-VAE checkpoint found in {ckpt_dir}")
    return candidates[-1]


def evaluate_split(model, tokenizer, dataloader, device: str, vae_n_layers: int) -> dict:
    topk_acc = TopKAccumulator(ks=TOP_K_LIST)
    ndcg_acc = NDCGAccumulator(ks=TOP_K_LIST)

    with tqdm(dataloader, desc="  eval", leave=False) as pbar:
        for batch in pbar:
            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)
            with torch.no_grad():
                generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
            actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
            topk_acc.accumulate(actual=actual, top_k=generated.sem_ids)
            ndcg_acc.accumulate(actual=actual, top_k=generated.sem_ids)

    return {**topk_acc.reduce(), **ndcg_acc.reduce()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate all text decoder checkpoints.")
    parser.add_argument("--checkpoint_dir", default="out/decoder/All_Beauty/text")
    parser.add_argument("--rqvae_dir", default="out/rqvae/")
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--modality", default="text")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--device", default=None)
    # Model hyperparams — must match training config
    parser.add_argument("--vae_input_dim", type=int, default=1536)
    parser.add_argument("--vae_embed_dim", type=int, default=32)
    parser.add_argument("--vae_hidden_dims", type=int, nargs="+", default=[768, 512, 256])
    parser.add_argument("--vae_codebook_size", type=int, default=256)
    parser.add_argument("--vae_n_layers", type=int, default=3)
    parser.add_argument("--vae_n_cat_feats", type=int, default=0)
    parser.add_argument("--qwen_model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--top_k_for_generation", type=int, default=20)
    parser.add_argument("--no_sep_token", action="store_true", default=False)
    parser.add_argument("--num_user_bins", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    rqvae_dir = Path(args.rqvae_dir)
    output_file = Path(args.output_file) if args.output_file else checkpoint_dir / "eval_results.json"

    checkpoints = _sorted_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s): {[p.name for p in checkpoints]}")

    # -------------------------------------------------------------------------
    # Load embeddings and ASIN index once
    # -------------------------------------------------------------------------
    asins, asin2idx = _load_asin_index(args.category)
    embeddings = _load_embeddings(args.category, args.modality)

    actual_dim = embeddings.shape[1]
    vae_input_dim = actual_dim  # use actual dim; override arg if they differ
    if actual_dim != args.vae_input_dim:
        print(f"  Note: actual embedding dim {actual_dim} overrides --vae_input_dim {args.vae_input_dim}")

    # -------------------------------------------------------------------------
    # Build tokenizer once (shared across all checkpoints)
    # -------------------------------------------------------------------------
    rqvae_path = _find_rqvae_checkpoint(rqvae_dir, args.category, args.modality)
    print(f"Using RQ-VAE checkpoint: {rqvae_path}")

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=args.vae_hidden_dims,
        output_dim=args.vae_embed_dim,
        codebook_size=args.vae_codebook_size,
        n_layers=args.vae_n_layers,
        n_cat_feats=args.vae_n_cat_feats,
        rqvae_weights_path=str(rqvae_path),
    )
    tokenizer.eval()
    tokenizer = tokenizer.to(device)

    print("Precomputing corpus semantic IDs...")
    item_dataset = ItemEmbeddingDataset(embeddings, split="all")
    corpus_ids = tokenizer.precompute_corpus_ids(item_dataset)
    codebooks = corpus_ids[:, : args.vae_n_layers].cpu()

    # -------------------------------------------------------------------------
    # Build dataloaders once
    # -------------------------------------------------------------------------
    seq_splits = load_sequential_data(category=args.category, asin2idx=asin2idx, cache_dir=HF_CACHE)

    val_dataset = SequentialRecommendationDataset(
        embeddings=embeddings,
        split_data=seq_splits["valid"],
        max_seq_len=MAX_SEQ_LEN,
        subsample=False,
    )
    test_dataset = SequentialRecommendationDataset(
        embeddings=embeddings,
        split_data=seq_splits.get("test", seq_splits.get("valid")),
        max_seq_len=MAX_SEQ_LEN,
        subsample=False,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # -------------------------------------------------------------------------
    # Evaluate each checkpoint
    # -------------------------------------------------------------------------
    should_add_sep_token = not args.no_sep_token
    results = {}

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split("_")[-1])
        print(f"\n[{ckpt_path.name}] step={step}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = QwenRetrievalModel(
            codebooks=codebooks,
            num_hierarchies=args.vae_n_layers,
            num_embeddings_per_hierarchy=args.vae_codebook_size,
            qwen_model_name=args.qwen_model_name,
            freeze_encoder=args.freeze_encoder,
            top_k_for_generation=args.top_k_for_generation,
            should_add_sep_token=should_add_sep_token,
            num_user_bins=args.num_user_bins,
        )
        model.load_state_dict(state["model"])
        model.eval()
        model = model.to(device)

        print("  Evaluating validation split...")
        val_metrics = evaluate_split(model, tokenizer, val_dataloader, device, args.vae_n_layers)
        print(f"  val: {val_metrics}")

        print("  Evaluating test split...")
        test_metrics = evaluate_split(model, tokenizer, test_dataloader, device, args.vae_n_layers)
        print(f"  test: {test_metrics}")

        results[step] = {
            "checkpoint": ckpt_path.name,
            "step": step,
            "val": val_metrics,
            "test": test_metrics,
        }

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    metric_keys = [f"h@{k}" for k in TOP_K_LIST] + [f"ndcg@{k}" for k in TOP_K_LIST]
    col = 10
    header = f"{'step':>8}  {'split':<5}  " + "  ".join(f"{m:>{col}}" for m in metric_keys)
    print(f"\n{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for step in sorted(results.keys()):
        for split in ("val", "test"):
            m = results[step][split]
            row = f"{step:>8}  {split:<5}  " + "  ".join(f"{m.get(k, float('nan')):>{col}.4f}" for k in metric_keys)
            print(row)
        print()


if __name__ == "__main__":
    main()
