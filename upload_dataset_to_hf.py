#!/usr/bin/env python3
"""Upload images and metadata to a HuggingFace Dataset repository.

Usage:
    python3 upload_dataset_to_hf.py --repo jimboH/amazon2023-multimodal

This script uploads:
    - datasets/images/          → images/
    - datasets/meta_All_Beauty.jsonl
    - datasets/meta_Musical_Instruments.jsonl

It uses upload_large_folder which supports resumption — safe to re-run
if interrupted.
"""

import argparse
from pathlib import Path
import huggingface_hub

DATASETS_DIR = Path("/work/u1304848/AI/project/datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace dataset repo id, e.g. jimboH/amazon2023-multimodal",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF write token (optional if already logged in via huggingface-cli login)",
    )
    args = parser.parse_args()

    api = huggingface_hub.HfApi(token=args.token)

    print(f"Uploading to existing dataset repo: https://huggingface.co/datasets/{args.repo}")

    # Upload metadata files first (smaller, good smoke test)
    for fname in ["meta_All_Beauty.jsonl", "meta_Musical_Instruments.jsonl"]:
        src = DATASETS_DIR / fname
        if src.exists():
            print(f"\nUploading {fname} ({src.stat().st_size / 1e6:.0f} MB)...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"metadata/{fname}",
                repo_id=args.repo,
                repo_type="dataset",
            )
            print(f"  Done: metadata/{fname}")
        else:
            print(f"  Skipping {fname} (not found at {src})")

    # Upload images folder (large — uses chunked multipart upload with resumption)
    images_dir = DATASETS_DIR / "images"
    if images_dir.exists():
        print(f"\nUploading images/All_Beauty and images/Musical_Instruments — this will take a while.")
        print("Safe to interrupt and re-run; upload_large_folder resumes automatically.\n")
        api.upload_large_folder(
            folder_path=str(DATASETS_DIR),
            repo_id=args.repo,
            repo_type="dataset",
            allow_patterns=[
                "images/All_Beauty/**",
                "images/Musical_Instruments/**",
            ],
            num_workers=4,
        )
        print("\nImages upload complete.")
    else:
        print(f"images/ directory not found at {images_dir}")


if __name__ == "__main__":
    main()
