#!/usr/bin/env python3
"""Precompute and cache item embeddings for all modalities and categories.

Run this script once before training. All three modalities use the same
backbone — Qwen/Qwen3-VL-Embedding-2B — producing 1536-dim embeddings:
  - text       : text-only inputs via the VL chat template
  - image      : image-only inputs via the VL chat template
  - multimodal : combined text + image inputs via the VL chat template

Embeddings are saved as .pt files in:
  /work/u1304848/AI/project/outputs/embeddings/{category}/
    text_embeddings.pt          -- (N, 1536) float32
    image_embeddings.pt         -- (N, 1536) float32
    multimodal_embeddings.pt    -- (N, 1536) float32
    asins.json                  -- list of N parent_asin strings (index → asin)

Usage:
  python3 precompute_embeddings.py --category All_Beauty [--modality text]
  python3 precompute_embeddings.py --category Musical_Instruments --modality all
"""

import argparse
import json
import os
from pathlib import Path

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

OUTPUT_BASE = Path("/work/u1304848/AI/project/outputs/embeddings")

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
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for category in args.category:
        print(f"\n=== Processing category: {category} ===")
        metadata = load_metadata(category)
        asins, _ = build_asin_index(metadata)

        # Always save the ASIN index so downstream code can load it
        save_asin_index(category, asins)

        if args.modality in ("text", "all"):
            precompute_text(category, asins, metadata, device, force=args.force)

        if args.modality in ("image", "all"):
            precompute_image(category, asins, device, force=args.force)

        if args.modality in ("multimodal", "all"):
            precompute_multimodal(category, asins, metadata, device, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
