#!/usr/bin/env python3
"""3×3 Evaluation Grid for Generative Recommendation — Validation Split.

Evaluates all 9 combinations of training modality × test modality:

  Rows (training modality): text | image | multimodal
  Columns (test modality) : text | image | multimodal

Dataset splits follow the leave-one-out protocol:
  - Valid : second-to-last item as target, all preceding items as input.
  - Test  : absolute last item as target (held out for final evaluation).

This script evaluates on the VALIDATION split (second-to-last item as
target).  For test-split evaluation use evaluate_grid.py instead.

For each cell (train_mod, test_mod):
  1. Load the QwenRetrievalModel decoder trained on train_mod.
  2. Load the test-modality item embeddings.
  3. Re-tokenize the validation sequences using test_mod embeddings through
     the train_mod RQ-VAE (cross-modal codebook lookup).
  4. Run beam-search generation and accumulate hit@k / NDCG@k metrics.

Outputs a JSON result file and a printed table.

Usage
-----
  python3 evaluate_grid_val.py \\
      --category All_Beauty \\
      --rqvae_dir  out/rqvae/ \\
      --decoder_dir out/decoder/ \\
      --output_dir  out/grid_results_val/

The script expects checkpoint files at:
  {rqvae_dir}/{category}/{modality}/checkpoint_best.pt   (or latest)
  {decoder_dir}/{category}/{modality}/checkpoint_best.pt (or latest)
"""

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Dict, Optional

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

MODALITIES = ["text", "image", "multimodal"]
TOP_K_LIST = [1, 5, 10]
BATCH_SIZE = 32
MAX_SEQ_LEN = 20


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return the path to the highest-numbered checkpoint in a directory."""
    candidates = sorted(
        (
            p for p in ckpt_dir.glob("checkpoint_*.pt")
            if p.stem.split("_")[-1].isdigit()
        ),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return candidates[-1] if candidates else None


def _load_checkpoint(ckpt_dir: Path) -> dict:
    best = ckpt_dir / "checkpoint_best.pt"
    if best.exists():
        return torch.load(best, map_location="cpu", weights_only=False)
    latest = _find_latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return torch.load(latest, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_embeddings(category: str, modality: str) -> torch.Tensor:
    path = EMBEDDING_BASE / category / f"{modality}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {path}. Run precompute_embeddings.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False).float()


def load_asin_index(category: str):
    path = EMBEDDING_BASE / category / "asins.json"
    with open(path) as f:
        asins = json.load(f)
    return asins, {a: i for i, a in enumerate(asins)}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_cell(
    train_modality: str,
    test_modality: str,
    category: str,
    rqvae_dir: Path,
    decoder_dir: Path,
    device: str,
    qwen_model_name: str = "Qwen/Qwen3.5-0.8B",
    freeze_encoder: bool = False,
    top_k_for_generation: int = 20,
    should_add_sep_token: bool = True,
    num_user_bins: int = None,
) -> Dict[str, float]:
    """Evaluate one (train_mod, test_mod) cell on the validation split."""
    print(
        f"\n  [train={train_modality}, test={test_modality}] "
        f"category={category}"
    )

    # Load ASIN index and test-modality embeddings
    asins, asin2idx = load_asin_index(category)
    test_embeddings = load_embeddings(category, test_modality)
    train_embeddings = load_embeddings(category, train_modality)

    actual_dim = train_embeddings.shape[1]
    test_dim = test_embeddings.shape[1]

    # Load RQ-VAE checkpoint (trained on train_modality)
    rqvae_ckpt_dir = rqvae_dir / category / train_modality
    rqvae_state = _load_checkpoint(rqvae_ckpt_dir)
    model_cfg = rqvae_state.get("model_config", {})

    vae_input_dim = model_cfg.get("input_dim", actual_dim)
    vae_embed_dim = model_cfg.get("embed_dim", 32)
    vae_hidden_dims = model_cfg.get("hidden_dims", [512, 256, 128])
    vae_n_layers = model_cfg.get("n_layers", 3)
    vae_codebook_size = model_cfg.get("codebook_size", 256)
    vae_n_cat_feats = model_cfg.get("n_cat_features", 0)

    # Build tokenizer with the train-modality RQ-VAE weights
    _best = rqvae_ckpt_dir / "checkpoint_best.pt"
    rqvae_path = _best if _best.exists() else _find_latest_checkpoint(rqvae_ckpt_dir)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=str(rqvae_path),
    )
    tokenizer.eval()
    tokenizer = tokenizer.to(device)

    # Precompute corpus IDs using TEST-modality embeddings (cross-modal lookup)
    # We feed test_embeddings through the TRAIN-modality RQ-VAE
    print(f"    Precomputing corpus IDs (test emb → train RQ-VAE)...")
    if test_dim != vae_input_dim:
        print(
            f"    Warning: test embedding dim ({test_dim}) ≠ RQ-VAE input dim ({vae_input_dim}). "
            "Items will be encoded with zero residual for mismatched dimensions."
        )
    # Project test embeddings to the RQ-VAE input dim if needed
    if test_dim != vae_input_dim:
        # Simple linear projection (no learned weights — zero-pad or truncate)
        if test_dim < vae_input_dim:
            pad = torch.zeros(test_embeddings.shape[0], vae_input_dim - test_dim)
            test_emb_proj = torch.cat([test_embeddings, pad], dim=1)
        else:
            test_emb_proj = test_embeddings[:, :vae_input_dim]
    else:
        test_emb_proj = test_embeddings

    item_dataset = ItemEmbeddingDataset(test_emb_proj, split="all")
    corpus_ids = tokenizer.precompute_corpus_ids(item_dataset)

    # Load sequential data and use the VALIDATION split
    seq_splits = load_sequential_data(
        category=category,
        asin2idx=asin2idx,
        cache_dir=HF_CACHE,
    )
    val_split = seq_splits["valid"]

    val_dataset = SequentialRecommendationDataset(
        embeddings=test_emb_proj,
        split_data=val_split,
        max_seq_len=MAX_SEQ_LEN,
        subsample=False,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load decoder checkpoint (trained on train_modality)
    decoder_ckpt_dir = decoder_dir / category / train_modality
    decoder_state = _load_checkpoint(decoder_ckpt_dir)

    codebooks = corpus_ids[:, :vae_n_layers].cpu()

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
    model.load_state_dict(decoder_state["model"])
    model.eval()
    model = model.to(device)

    topk_acc = TopKAccumulator(ks=TOP_K_LIST)
    ndcg_acc = NDCGAccumulator(ks=TOP_K_LIST)

    with tqdm(val_dataloader, desc="    Evaluating", leave=False) as pbar:
        for batch in pbar:
            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)

            with torch.no_grad():
                generated = model.generate_next_sem_id(
                    tokenized_data, top_k=True, temperature=1
                )

            actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
            topk_acc.accumulate(actual=actual, top_k=generated.sem_ids)
            ndcg_acc.accumulate(actual=actual, top_k=generated.sem_ids)

    metrics = {**topk_acc.reduce(), **ndcg_acc.reduce()}
    print(f"    Results: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_grid(
    category: str,
    rqvae_dir: Path,
    decoder_dir: Path,
    output_dir: Path,
    device: str,
    modalities: list = None,
    qwen_model_name: str = "Qwen/Qwen3.5-0.8B",
    freeze_encoder: bool = False,
    top_k_for_generation: int = 20,
    should_add_sep_token: bool = True,
    num_user_bins: int = None,
) -> dict:
    grid_results = {}
    mods = modalities if modalities is not None else MODALITIES

    for train_mod, test_mod in product(mods, mods):
        cell_key = f"train={train_mod}_test={test_mod}"
        rqvae_ckpt_dir = rqvae_dir / category / train_mod
        decoder_ckpt_dir = decoder_dir / category / train_mod

        if not rqvae_ckpt_dir.exists():
            print(f"Skipping {cell_key}: no RQ-VAE checkpoint at {rqvae_ckpt_dir}")
            continue
        if not decoder_ckpt_dir.exists():
            print(f"Skipping {cell_key}: no decoder checkpoint at {decoder_ckpt_dir}")
            continue

        try:
            metrics = evaluate_cell(
                train_modality=train_mod,
                test_modality=test_mod,
                category=category,
                rqvae_dir=rqvae_dir,
                decoder_dir=decoder_dir,
                device=device,
                qwen_model_name=qwen_model_name,
                freeze_encoder=freeze_encoder,
                top_k_for_generation=top_k_for_generation,
                should_add_sep_token=should_add_sep_token,
                num_user_bins=num_user_bins,
            )
            grid_results[cell_key] = metrics
        except Exception as e:
            print(f"  ERROR in {cell_key}: {e}")
            grid_results[cell_key] = {"error": str(e)}

    return grid_results


def print_grid_table(grid_results: dict, category: str, modalities: list = None) -> None:
    """Print a human-readable table of the 3×3 results."""
    mods = modalities if modalities is not None else MODALITIES
    metric_keys = [f"h@{k}" for k in TOP_K_LIST] + [f"ndcg@{k}" for k in TOP_K_LIST]

    print(f"\n{'='*80}")
    print(f"  3x3 Evaluation Grid (Validation Split) -- Category: {category}")
    print(f"{'='*80}")

    col_width = 14
    train_test_label = "train/test"
    header = f"{train_test_label:<16}" + "".join(f"{m:>{col_width}}" for m in mods)
    print(header)
    print("-" * (16 + col_width * len(mods)))

    for train_mod in mods:
        # Print one line per metric
        for i, metric in enumerate(metric_keys):
            prefix = f"{train_mod:<16}" if i == 0 else f"{'':16}"
            row = prefix
            for test_mod in mods:
                key = f"train={train_mod}_test={test_mod}"
                if key in grid_results and "error" not in grid_results[key]:
                    val = grid_results[key].get(metric, float("nan"))
                    row += f"{val:>{col_width}.4f}"
                else:
                    row += f"{'N/A':>{col_width}}"
            row += f"  ({metric})"
            print(row)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="3×3 evaluation grid on the validation split."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="All_Beauty",
        help="Dataset category to evaluate.",
    )
    parser.add_argument(
        "--rqvae_dir",
        type=str,
        default="out/rqvae/",
        help="Root directory of RQ-VAE checkpoints.",
    )
    parser.add_argument(
        "--decoder_dir",
        type=str,
        default="out/decoder/",
        help="Root directory of decoder checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/grid_results_val/",
        help="Directory to save JSON results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda if available).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=None,
        choices=MODALITIES,
        help="Restrict evaluation to specific modalities.",
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model name for the Qwen backbone (must match training).",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Whether the Qwen encoder was frozen during training.",
    )
    parser.add_argument(
        "--top_k_for_generation",
        type=int,
        default=20,
        help="Number of top-k candidates to return during beam search (must match training).",
    )
    parser.add_argument(
        "--should_add_sep_token",
        action="store_true",
        default=True,
        help="Disable the separator token between items (pass if training used should_add_sep_token=False).",
    )
    parser.add_argument(
        "--num_user_bins",
        type=int,
        default=None,
        help="Number of user embedding bins (must match training; omit if not used).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rqvae_dir = Path(args.rqvae_dir)
    decoder_dir = Path(args.decoder_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities_to_use = args.modalities if args.modalities else MODALITIES

    print(f"\nRunning 3×3 grid (validation split) for category: {args.category}")
    results = run_grid(
        category=args.category,
        rqvae_dir=rqvae_dir,
        decoder_dir=decoder_dir,
        output_dir=output_dir,
        device=device,
        modalities=modalities_to_use,
        qwen_model_name=args.qwen_model_name,
        freeze_encoder=args.freeze_encoder,
        top_k_for_generation=args.top_k_for_generation,
        should_add_sep_token=args.should_add_sep_token,
        num_user_bins=args.num_user_bins,
    )

    # Save results
    out_path = output_dir / f"{args.category}_grid_results_val.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print table
    print_grid_table(results, category=args.category, modalities=modalities_to_use)


if __name__ == "__main__":
    main()
