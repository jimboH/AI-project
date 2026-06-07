#!/usr/bin/env python3
"""Precompute and cache item embeddings for all modalities and categories.

Run this script once before training. All three modalities use the same
backbone — google/siglip-so400m-patch14-384 — producing 1152-dim embeddings:
  - text       : text-only inputs via SigLIP's text tower (max 64 tokens)
  - image      : image-only inputs via SigLIP's vision tower (384×384 px)
  - multimodal : L2-normalised average of text + image embeddings;
                 falls back to text-only for items with no image

Embeddings are saved as .pt files in:
  outputs/embeddings/{category}/
    text_embeddings.pt          -- (N, 1152) float32
    image_embeddings.pt         -- (N, 1152) float32  (zero vector if no image)
    multimodal_embeddings.pt    -- (N, 1152) float32
    asins.json                  -- list of N parent_asin strings (index → asin)

For cross_modal RQVAE training, both text_embeddings.pt and
image_embeddings.pt are required (run with --modality all).

Usage:
  python3 precompute_embeddings.py --category All_Beauty [--modality text]
  python3 precompute_embeddings.py --category Musical_Instruments --modality all
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

from data.amazon2023 import (
    IMAGE_DIR,
    METADATA_FILES,
    build_asin_index,
    build_text_prompt,
    load_metadata,
)
from embeddings.image_encoder import ImageEncoder
from embeddings.multimodal_encoder import MultimodalEncoder
from embeddings.text_encoder import TextEncoder

# Mapping from dataset category name to ICLGR data subdirectory name
CATEGORY_TO_ICLGR_DIR = {
    "All_Beauty": "amazon_all_beauty",
    "Musical_Instruments": "amazon_musical_instruments",
}

OUTPUT_BASE = Path(__file__).resolve().parent / "outputs" / "embeddings"

SUPPORTED_CATEGORIES = list(METADATA_FILES.keys())


def get_output_dir(category: str) -> Path:
    out = OUTPUT_BASE / category
    out.mkdir(parents=True, exist_ok=True)
    return out


def precompute_text(
    category: str,
    asins: list,
    metadata: dict,
    device: str,
    force: bool = False,
) -> None:
    out_dir = get_output_dir(category)
    out_path = out_dir / "text_embeddings.pt"
    if out_path.exists() and not force:
        print(f"[text] {category}: cache exists, skipping. Use --force to recompute.")
        return

    print(f"[text] {category}: encoding {len(asins)} items...")
    texts = [build_text_prompt(metadata[a]) for a in asins]

    encoder = TextEncoder(device=device)
    embeddings = encoder.encode(texts)

    torch.save(embeddings, out_path)
    print(f"[text] {category}: saved {embeddings.shape} to {out_path}")


def precompute_image(
    category: str,
    asins: list,
    device: str,
    force: bool = False,
) -> None:
    out_dir = get_output_dir(category)
    out_path = out_dir / "image_embeddings.pt"
    if out_path.exists() and not force:
        print(f"[image] {category}: cache exists, skipping. Use --force to recompute.")
        return

    print(f"[image] {category}: encoding {len(asins)} items...")
    encoder = ImageEncoder(image_dir=IMAGE_DIR, category=category, device=device)
    embeddings = encoder.encode(asins)

    torch.save(embeddings, out_path)
    print(f"[image] {category}: saved {embeddings.shape} to {out_path}")


def precompute_multimodal(
    category: str,
    asins: list,
    metadata: dict,
    device: str,
    force: bool = False,
) -> None:
    out_dir = get_output_dir(category)
    out_path = out_dir / "multimodal_embeddings.pt"
    if out_path.exists() and not force:
        print(f"[multimodal] {category}: cache exists, skipping. Use --force to recompute.")
        return

    print(f"[multimodal] {category}: encoding {len(asins)} items...")
    texts = [build_text_prompt(metadata[a]) for a in asins]

    encoder = MultimodalEncoder(image_dir=IMAGE_DIR, category=category, device=device)
    embeddings = encoder.encode(asins, texts)

    torch.save(embeddings, out_path)
    print(f"[multimodal] {category}: saved {embeddings.shape} to {out_path}")


def precompute_pseudo_queries(
    category: str,
    asin2idx: Dict[str, int],
    device: str,
    force: bool = False,
) -> None:
    """Encode pseudo-query texts and save embeddings + target item indices.

    Reads pseudo_queries.jsonl from the ICLGR data directory for the category,
    encodes every query string with TextEncoder, and saves:
      outputs/embeddings/{category}/pseudo_query_embeddings.pt  — (N_pq, D) float32
      outputs/embeddings/{category}/pseudo_query_targets.json   — list of N_pq ints
    """
    out_dir = get_output_dir(category)
    emb_path = out_dir / "pseudo_query_embeddings.pt"
    tgt_path = out_dir / "pseudo_query_targets.json"

    if emb_path.exists() and tgt_path.exists() and not force:
        print(f"[pseudo_queries] {category}: cache exists, skipping. Use --force to recompute.")
        return

    iclgr_subdir = CATEGORY_TO_ICLGR_DIR.get(category)
    if iclgr_subdir is None:
        print(f"[pseudo_queries] {category}: no ICLGR dir mapping, skipping.")
        return

    project_root = Path(__file__).resolve().parent
    pq_path = project_root / "data" / iclgr_subdir / "pseudo_queries.jsonl"
    train_path = project_root / "data" / iclgr_subdir / "train.jsonl"

    if not pq_path.exists():
        print(f"[pseudo_queries] {category}: {pq_path} not found, skipping.")
        return

    # Build doc_id → asin from the ICLGR train.jsonl (indexing rows only)
    doc_id_to_asin: Dict[str, str] = {}
    with open(train_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("operation") == "indexing" and "asin" in row:
                doc_id_to_asin[row["doc_id"]] = row["asin"]

    # Collect (query_text, target_item_idx) pairs
    texts: List[str] = []
    targets: List[int] = []
    with open(pq_path) as f:
        for line in f:
            row = json.loads(line)
            asin = doc_id_to_asin.get(row["doc_id"])
            if asin is None:
                continue
            item_idx = asin2idx.get(asin, -1)
            if item_idx == -1:
                continue
            for pq in row.get("pseudo_queries", []):
                pq = pq.strip()
                if pq:
                    texts.append(pq)
                    targets.append(item_idx)

    print(f"[pseudo_queries] {category}: encoding {len(texts)} pseudo queries...")
    encoder = TextEncoder(device=device)
    embeddings = encoder.encode(texts)
    del encoder  # free GPU memory before saving

    torch.save(embeddings, emb_path)
    with open(tgt_path, "w") as f:
        json.dump(targets, f)
    print(f"[pseudo_queries] {category}: saved {embeddings.shape} → {emb_path}")


def save_asin_index(category: str, asins: list) -> None:
    out_dir = get_output_dir(category)
    out_path = out_dir / "asins.json"
    with open(out_path, "w") as f:
        json.dump(asins, f)
    print(f"[index] {category}: saved {len(asins)} ASINs to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute item embeddings.")
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=["All_Beauty", "Musical_Instruments"],
        choices=SUPPORTED_CATEGORIES,
        help="Dataset category (or categories) to process.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="all",
        choices=["text", "image", "multimodal", "all"],
        help="Which modality to precompute.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if cache already exists.",
    )
    parser.add_argument(
        "--pseudo_queries",
        action="store_true",
        help="Also encode pseudo-query texts from the ICLGR data directory.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for category in args.category:
        print(f"\n=== Processing category: {category} ===")
        metadata = load_metadata(category)
        asins, asin2idx = build_asin_index(metadata)

        # Always save the ASIN index so downstream code can load it
        save_asin_index(category, asins)

        if args.modality in ("text", "all"):
            precompute_text(category, asins, metadata, device, force=args.force)

        if args.modality in ("image", "all"):
            precompute_image(category, asins, device, force=args.force)

        if args.modality in ("multimodal", "all"):
            precompute_multimodal(category, asins, metadata, device, force=args.force)

        if args.pseudo_queries:
            precompute_pseudo_queries(category, asin2idx, device, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
