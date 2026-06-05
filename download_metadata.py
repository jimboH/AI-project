#!/usr/bin/env python3
"""Download Amazon Reviews 2023 metadata JSONL files from HuggingFace.

Downloads meta_All_Beauty.jsonl and meta_Musical_Instruments.jsonl into
the datasets/ directory, which is required before running any training steps.

Usage:
    python3 download_metadata.py
    python3 download_metadata.py --category All_Beauty
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

CATEGORY_TO_META_FILE = {
    "All_Beauty": "meta_All_Beauty.jsonl",
    "Musical_Instruments": "meta_Musical_Instruments.jsonl",
}

HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"


def download_metadata(category: str, force: bool = False) -> None:
    out_path = DATASETS_DIR / CATEGORY_TO_META_FILE[category]

    if out_path.exists() and not force:
        print(f"[{category}] Already exists: {out_path}  (use --force to re-download)")
        return

    print(f"[{category}] Downloading metadata from HuggingFace...")
    print(f"  Dataset : {HF_DATASET}")
    print(f"  Config  : raw_meta_{category}")
    print(f"  Output  : {out_path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: 'huggingface-hub' not found. Run: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    hf_cache = str(DATASETS_DIR / "hf_cache")

    # The raw JSONL is hosted directly in the repo — download it without the
    # dataset loading script (which newer datasets versions no longer support).
    cached = hf_hub_download(
        repo_id=HF_DATASET,
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        repo_type="dataset",
        cache_dir=hf_cache,
    )

    import shutil
    print(f"[{category}] Copying cached JSONL to {out_path}...")
    shutil.copy2(cached, out_path)

    print(f"[{category}] Done. {out_path.stat().st_size / 1e6:.1f} MB written.")


def main():
    parser = argparse.ArgumentParser(description="Download Amazon 2023 metadata JSONL files.")
    parser.add_argument(
        "--category",
        nargs="+",
        default=list(CATEGORY_TO_META_FILE.keys()),
        choices=list(CATEGORY_TO_META_FILE.keys()),
        help="Which categories to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()

    for cat in args.category:
        download_metadata(cat, force=args.force)

    print("\nAll metadata downloads complete.")
    print("Next step: python3 precompute_embeddings.py --category All_Beauty --modality text")


if __name__ == "__main__":
    main()
