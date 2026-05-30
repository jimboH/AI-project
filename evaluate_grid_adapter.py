#!/usr/bin/env python3
"""Evaluation for the shared-codebook + per-modality-adapter RQ-VAE (Design A).

This is a NEW file (parallel to evaluate_grid.py); it does not modify any
existing code. It loads the single shared multimodal RQ-VAE trained by
``train_rqvae_multimodal.py`` and evaluates the cross-modal improvement.

Why this is different from evaluate_grid.py
-------------------------------------------
``evaluate_grid.py`` uses one *independent* RQ-VAE per modality, so the
off-diagonal cells push modality-A embeddings through modality-B's codebook —
the cause of the ~3% unique-ID collapse documented in cross_modal.md.

Design A uses ONE shared codebook fed by per-modality adapters. There is a
single semantic-ID space, so:
  - the headline metric is the **unique-ID ratio** and the **cross-modal ID
    agreement** (same item, same ID across modalities) — these need only the
    RQ-VAE and directly quantify the improvement (RQVAE_improvement.md §3, §7);
  - if decoders trained on the shared IDs are supplied, a hit@k / NDCG@k grid is
    also produced.

Usage
-----
  # RQ-VAE metrics only (no decoder needed):
  python3 evaluate_grid_adapter.py --category All_Beauty \
      --mm_rqvae_dir out/rqvae/

  # also run the decoder hit@k grid (decoders must be trained on the SHARED IDs):
  python3 evaluate_grid_adapter.py --category All_Beauty \
      --mm_rqvae_dir out/rqvae/ --decoder_dir out/decoder/
"""

import argparse
import json
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
from modules.quantize import QuantizeForwardMode
from modules.rqvae_multimodal import MultiModalRqVae
from modules.tokenizer.semids import SemanticIdTokenizer

EMBEDDING_BASE = Path("/work/u1304848/AI/project/outputs/embeddings")
HF_CACHE = str(Path("/work/u1304848/AI/project/datasets/hf_cache"))

MODALITIES = ["text", "image", "multimodal"]
TOP_K_LIST = [1, 5, 10]
BATCH_SIZE = 32
MAX_SEQ_LEN = 20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings(category: str, modality: str) -> torch.Tensor:
    path = EMBEDDING_BASE / category / f"{modality}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False).float()


def load_asin_index(category: str):
    with open(EMBEDDING_BASE / category / "asins.json") as f:
        asins = json.load(f)
    return asins, {a: i for i, a in enumerate(asins)}


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    cands = sorted(
        (p for p in ckpt_dir.glob("checkpoint_*.pt") if p.stem.split("_")[-1].isdigit()),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return cands[-1] if cands else None


def _load_checkpoint(ckpt_dir: Path) -> dict:
    best = ckpt_dir / "checkpoint_best.pt"
    if best.exists():
        return torch.load(best, map_location="cpu", weights_only=False)
    latest = _find_latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return torch.load(latest, map_location="cpu", weights_only=False)


def load_shared_model(mm_rqvae_dir: Path, category: str, device: str) -> MultiModalRqVae:
    """Rebuild the trained shared MultiModalRqVae from its checkpoint."""
    ckpt_dir = mm_rqvae_dir / category / "multimodal_joint"
    state = _load_checkpoint(ckpt_dir)
    cfg = state["mm_config"]
    model = MultiModalRqVae(
        modalities=cfg["modalities"],
        input_dim=cfg["input_dim"],
        embed_dim=cfg["embed_dim"],
        hidden_dims=cfg["hidden_dims"],
        codebook_size=cfg["codebook_size"],
        n_layers=cfg["n_layers"],
        n_cat_features=cfg["n_cat_features"],
        commitment_weight=cfg["commitment_weight"],
        codebook_kmeans_init=False,          # weights come from the checkpoint
        codebook_normalize=cfg["codebook_normalize"],
        codebook_sim_vq=cfg["codebook_sim_vq"],
        codebook_mode=QuantizeForwardMode.STE,
        codebook_distance_l2_normalize=cfg["codebook_distance_l2_normalize"],
        adapter_hidden=cfg["adapter_hidden"],
        adapter_dropout=cfg["adapter_dropout"],
    )
    # strict=False: EMA buffers (ema_cluster_size / ema_embed_sum) are training-only
    # and absent here (codebook_use_ema not needed for inference). This mirrors
    # RqVae.load_pretrained's strict=False convention.
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    unexpected = [k for k in unexpected if "ema_" not in k]
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Adapted-embedding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def adapt_embeddings(
    model: MultiModalRqVae, emb: torch.Tensor, modality: str, device: str,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Apply the modality adapter to a full embedding matrix (-> shared space)."""
    out = []
    for s in range(0, emb.shape[0], batch_size):
        out.append(model.adapt(emb[s : s + batch_size].to(device), modality).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def corpus_codes(
    model: MultiModalRqVae, emb: torch.Tensor, modality: str, device: str,
    batch_size: int = 1024,
) -> torch.Tensor:
    out = []
    for s in range(0, emb.shape[0], batch_size):
        out.append(model.codes(emb[s : s + batch_size].to(device), modality).cpu())
    return torch.cat(out, dim=0)


def build_shared_tokenizer(model: MultiModalRqVae, device: str) -> SemanticIdTokenizer:
    """A SemanticIdTokenizer whose RQ-VAE IS the shared one. Feed it embeddings
    that have ALREADY been passed through the relevant modality adapter."""
    cfg = model.config
    tok = SemanticIdTokenizer(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        output_dim=cfg["embed_dim"],
        codebook_size=cfg["codebook_size"],
        n_layers=cfg["n_layers"],
        n_cat_feats=cfg["n_cat_features"],
        rqvae_weights_path=None,
    )
    tok.rq_vae = model.rqvae      # share the trained encoder + codebooks
    tok.eval()
    tok = tok.to(device)
    return tok


# ---------------------------------------------------------------------------
# RQ-VAE-level metrics (the Design A headline)
# ---------------------------------------------------------------------------

def rqvae_metrics(model, category, modalities, device) -> dict:
    codes = {m: corpus_codes(model, load_embeddings(category, m), m, device)
             for m in modalities}
    out = {"unique_ratio": {}, "id_agreement": {}}
    for m in modalities:
        c = codes[m]
        out["unique_ratio"][m] = torch.unique(c, dim=0).shape[0] / c.shape[0]
    for a, b in product(modalities, modalities):
        if a < b:
            agree = (codes[a] == codes[b]).all(dim=1).float().mean().item()
            out["id_agreement"][f"{a}_vs_{b}"] = agree
    return out


def print_rqvae_metrics(m: dict) -> None:
    print("\n" + "=" * 60)
    print("  Design A — shared-codebook RQ-VAE metrics")
    print("=" * 60)
    print("  Unique-ID ratio (per modality, shared codebook):")
    for k, v in m["unique_ratio"].items():
        print(f"    {k:12s}: {100*v:5.1f}%")
    print("  Cross-modal ID agreement (same item -> same ID):")
    for k, v in m["id_agreement"].items():
        print(f"    {k:24s}: {100*v:5.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Optional decoder hit@k grid (requires decoders trained on the SHARED IDs)
# ---------------------------------------------------------------------------

def evaluate_decoder_cell(
    model: MultiModalRqVae,
    train_modality: str,
    test_modality: str,
    category: str,
    decoder_dir: Path,
    device: str,
    qwen_model_name: str,
    freeze_encoder: bool,
    top_k_for_generation: int,
    should_add_sep_token: bool,
    num_user_bins: Optional[int],
) -> Dict[str, float]:
    cfg = model.config
    n_layers = cfg["n_layers"]

    asins, asin2idx = load_asin_index(category)
    raw_test = load_embeddings(category, test_modality)
    # Everything downstream uses ADAPTED (shared-space) embeddings.
    test_emb = adapt_embeddings(model, raw_test, test_modality, device)

    tokenizer = build_shared_tokenizer(model, device)
    corpus_ids = tokenizer.precompute_corpus_ids(ItemEmbeddingDataset(test_emb, split="all"))

    seq_splits = load_sequential_data(category=category, asin2idx=asin2idx, cache_dir=HF_CACHE)
    test_split = seq_splits.get("test", seq_splits.get("valid"))
    test_dataset = SequentialRecommendationDataset(
        embeddings=test_emb, split_data=test_split, max_seq_len=MAX_SEQ_LEN, subsample=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    decoder_state = _load_checkpoint(decoder_dir / category / train_modality)

    # Same pristine-clone discipline as evaluate_grid.py: load_state_dict copies
    # the checkpoint's codebooks buffer in place, so hand the constructor its own
    # throwaway clone and restore the real (shared-space) IDs afterwards.
    shared_codebooks = corpus_ids[:, :n_layers].clone().cpu()
    decoder = QwenRetrievalModel(
        codebooks=shared_codebooks.clone(),
        num_hierarchies=n_layers,
        num_embeddings_per_hierarchy=cfg["codebook_size"],
        qwen_model_name=qwen_model_name,
        freeze_encoder=freeze_encoder,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )
    decoder.load_state_dict(decoder_state["model"])
    decoder.codebooks = shared_codebooks.to(device)
    decoder.eval().to(device)

    topk_acc = TopKAccumulator(ks=TOP_K_LIST)
    ndcg_acc = NDCGAccumulator(ks=TOP_K_LIST)
    with tqdm(test_dataloader, desc=f"    {train_modality}->{test_modality}", leave=False) as pbar:
        for batch in pbar:
            data = batch_to(batch, device)
            tok = tokenizer(data)
            with torch.no_grad():
                generated = decoder.generate_next_sem_id(tok, top_k=True, temperature=1)
            actual = tok.sem_ids_fut[:, :n_layers]
            topk_acc.accumulate(actual=actual, top_k=generated.sem_ids)
            ndcg_acc.accumulate(actual=actual, top_k=generated.sem_ids)

    return {**topk_acc.reduce(), **ndcg_acc.reduce()}


def run_decoder_grid(model, category, decoder_dir, device, modalities, **kw) -> dict:
    results = {}
    for train_mod, test_mod in product(modalities, modalities):
        key = f"train={train_mod}_test={test_mod}"
        if not (decoder_dir / category / train_mod).exists():
            print(f"  Skipping {key}: no decoder at {decoder_dir / category / train_mod}")
            continue
        try:
            results[key] = evaluate_decoder_cell(
                model, train_mod, test_mod, category, decoder_dir, device, **kw
            )
            print(f"  {key}: {results[key]}")
        except Exception as e:
            print(f"  ERROR in {key}: {e}")
            results[key] = {"error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Design A (adapter) evaluation.")
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--mm_rqvae_dir", default="out/rqvae/",
                    help="Root holding {category}/multimodal_joint/ shared RQ-VAE.")
    ap.add_argument("--decoder_dir", default=None,
                    help="Optional: root of decoders trained on the SHARED IDs.")
    ap.add_argument("--output_dir", default="out/grid_results/")
    ap.add_argument("--device", default=None)
    ap.add_argument("--modalities", nargs="+", default=None, choices=MODALITIES)
    ap.add_argument("--qwen_model_name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--freeze_encoder", action="store_true", default=False)
    ap.add_argument("--top_k_for_generation", type=int, default=20)
    ap.add_argument("--should_add_sep_token", action="store_true", default=True)
    ap.add_argument("--num_user_bins", type=int, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_shared_model(Path(args.mm_rqvae_dir), args.category, device)
    modalities = args.modalities or model.modalities
    print(f"Shared RQ-VAE modalities: {model.modalities}; evaluating: {modalities}")

    out = {"rqvae_metrics": rqvae_metrics(model, args.category, modalities, device)}
    print_rqvae_metrics(out["rqvae_metrics"])

    if args.decoder_dir is not None:
        print("\nRunning decoder hit@k grid (shared semantic-ID space)...")
        print("NOTE: for meaningful hits, decoders must be trained on the shared "
              "RQ-VAE's IDs (tokenize histories via the shared codebook + adapter).")
        out["decoder_grid"] = run_decoder_grid(
            model, args.category, Path(args.decoder_dir), device, modalities,
            qwen_model_name=args.qwen_model_name,
            freeze_encoder=args.freeze_encoder,
            top_k_for_generation=args.top_k_for_generation,
            should_add_sep_token=args.should_add_sep_token,
            num_user_bins=args.num_user_bins,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.category}_adapter_grid_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
