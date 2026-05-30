#!/usr/bin/env python3
"""Upload trained model checkpoints to a HuggingFace Model repository.

Usage:
    python3 upload_checkpoints_to_hf.py --repo jimboH/amazon2023-genrec

This script uploads the following 6 checkpoint files:
    - out/rqvae/All_Beauty/text/checkpoint_best.pt
    - out/rqvae/All_Beauty/image/checkpoint_best.pt
    - out/rqvae/All_Beauty/multimodal/checkpoint_best.pt
    - out/decoder/All_Beauty/text/checkpoint_best.pt
    - out/decoder/All_Beauty/image/checkpoint_best.pt
    - out/decoder/All_Beauty/multimodal/checkpoint_best.pt

It uses upload_large_folder which supports resumption — safe to re-run
if interrupted.
"""

import argparse
from pathlib import Path
import huggingface_hub

OUT_DIR = Path("/work/u1304848/AI/project/out")

CHECKPOINTS = [
    "rqvae/All_Beauty/text/checkpoint_best.pt",
    "rqvae/All_Beauty/image/checkpoint_best.pt",
    "rqvae/All_Beauty/multimodal/checkpoint_best.pt",
    "decoder/All_Beauty/text/checkpoint_best.pt",
    "decoder/All_Beauty/image/checkpoint_best.pt",
    "decoder/All_Beauty/multimodal/checkpoint_best.pt",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace model repo id, e.g. jimboH/amazon2023-genrec",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF write token (optional if already logged in via huggingface-cli login)",
    )
    args = parser.parse_args()

    api = huggingface_hub.HfApi(token=args.token)

    print(f"Uploading to existing model repo: https://huggingface.co/{args.repo}")

    present = [p for p in CHECKPOINTS if (OUT_DIR / p).exists()]
    missing = [p for p in CHECKPOINTS if not (OUT_DIR / p).exists()]
    for p in missing:
        print(f"  Skipping (not found): {OUT_DIR / p}")
    if not present:
        print("Nothing to upload.")
        return

    size_gb = sum((OUT_DIR / p).stat().st_size for p in present) / 1e9
    print(f"\nUploading {len(present)} checkpoint(s) ({size_gb:.1f} GB)...")
    for p in present:
        print(f"  {p}")
    if size_gb > 1:
        print("Safe to interrupt and re-run; upload_large_folder resumes automatically.\n")

    api.upload_large_folder(
        folder_path=str(OUT_DIR),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=present,
        num_workers=4,
    )

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
