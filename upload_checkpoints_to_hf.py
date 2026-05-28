#!/usr/bin/env python3
"""Upload trained model checkpoints to a HuggingFace Model repository.

Usage:
    python3 upload_checkpoints_to_hf.py --repo jimboH/amazon2023-genrec

This script uploads:
    - out/rqvae/     → rqvae/       (~1.4 GB)
    - out/decoder/   → decoder/     (~63 GB)
    - out/grid_results/ → grid_results/

It uses upload_large_folder which supports resumption — safe to re-run
if interrupted.
"""

import argparse
from pathlib import Path
import huggingface_hub

OUT_DIR = Path("/work/u1304848/AI/project/out")


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
    parser.add_argument(
        "--subdir",
        default=None,
        choices=["rqvae", "decoder", "grid_results"],
        help="Upload only one subdirectory (omit to upload all)",
    )
    args = parser.parse_args()

    api = huggingface_hub.HfApi(token=args.token)

    print(f"Uploading to existing model repo: https://huggingface.co/{args.repo}")

    subdirs = [args.subdir] if args.subdir else ["rqvae", "grid_results", "decoder"]

    for subdir in subdirs:
        src = OUT_DIR / subdir
        if not src.exists():
            print(f"\nSkipping {subdir}/ (not found at {src})")
            continue

        size_gb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e9
        print(f"\nUploading {subdir}/ ({size_gb:.1f} GB)...")
        if size_gb > 1:
            print("Safe to interrupt and re-run; upload_large_folder resumes automatically.\n")

        api.upload_large_folder(
            folder_path=str(src),
            path_in_repo=subdir,
            repo_id=args.repo,
            repo_type="model",
            num_workers=4,
        )
        print(f"  Done: {subdir}/")

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
