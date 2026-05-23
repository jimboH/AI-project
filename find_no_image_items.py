#!/usr/bin/env python3
"""Find metadata items that have no associated image file.

Uses the same image-discovery logic as ImageEncoder so the reported missing
items exactly match what that encoder would fall back to zero-vectors for.

Output (per category):
  outputs/missing_images/{category}_no_image_asins.json
    {
      "category": str,
      "total_items": int,
      "missing_count": int,
      "missing_fraction": float,
      "parent_asins": [str, ...]
    }

Usage:
  python3 find_no_image_items.py
  python3 find_no_image_items.py --category All_Beauty
  python3 find_no_image_items.py --category All_Beauty Musical_Instruments
"""

import argparse
import json
from pathlib import Path

from data.amazon2023 import IMAGE_DIR, METADATA_FILES, build_asin_index, load_metadata

VARIANT_PRIORITY = ["MAIN", "PT01", "PT02", "PT03"]
OUTPUT_DIR = Path("/work/u1304848/AI/project/outputs/missing_images")


def find_image_path(image_dir: Path, asin: str) -> Path | None:
    """Mirrors ImageEncoder._find_image_path exactly."""
    for variant in VARIANT_PRIORITY:
        candidate = image_dir / f"{asin}_{variant}.jpg"
        if candidate.exists():
            return candidate
    for f in image_dir.glob(f"{asin}_*.jpg"):
        return f
    return None


def find_no_image_asins(category: str) -> list[str]:
    metadata = load_metadata(category)
    asins, _ = build_asin_index(metadata)
    image_dir = IMAGE_DIR / category

    missing = [asin for asin in asins if find_image_path(image_dir, asin) is None]
    return asins, missing


def save_results(category: str, total: int, missing: list[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{category}_no_image_asins.json"
    result = {
        "category": category,
        "total_items": total,
        "missing_count": len(missing),
        "missing_fraction": round(len(missing) / total, 6) if total else 0.0,
        "parent_asins": missing,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Find items with no image.")
    parser.add_argument(
        "--category",
        nargs="+",
        default=list(METADATA_FILES.keys()),
        choices=list(METADATA_FILES.keys()),
    )
    args = parser.parse_args()

    for category in args.category:
        print(f"\n[{category}] Loading metadata...")
        asins, missing = find_no_image_asins(category)
        out_path = save_results(category, len(asins), missing)
        print(
            f"[{category}] {len(missing):,} / {len(asins):,} items have no image "
            f"({100 * len(missing) / len(asins):.2f}%)"
        )
        print(f"[{category}] Saved to {out_path}")


if __name__ == "__main__":
    main()
