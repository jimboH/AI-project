#!/usr/bin/env python3
"""Push a pure-image dataset for Amazon All_Beauty to HuggingFace.

Filtered: only ASINs from metadata that have at least a MAIN image on disk.

Dataset schema (one row per ASIN):
  parent_asin : string
  images      : Sequence[Image]  — all available variants in order:
                MAIN, PT01–PT06; only images that exist are included.
                No null/empty image structs are ever stored.

The root cause of the ValueError (both 'path' and 'bytes' are None) is
that HF's Image decoder rejects null structs.  Using Sequence[Image] and
only appending images that successfully load avoids this entirely.

Usage
-----
  python3 push_images_only_to_hf.py --token <HF_TOKEN>
  python3 push_images_only_to_hf.py --token <HF_TOKEN> --shard_size 500
  python3 push_images_only_to_hf.py --token <HF_TOKEN> --dry_run
"""

import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from PIL import Image as PILImage
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")
META_PATH    = DATASETS_DIR / "meta_All_Beauty.jsonl"
IMAGE_DIR    = DATASETS_DIR / "images" / "All_Beauty"
SHARD_DIR    = Path("/mnt/storage/temp/images_only_shards")

REPO_ID      = "theblackcat102/amazon-all-beauty-images"
SHARD_SIZE   = 1000   # items per parquet shard

# Variants to include, in priority order.
# Only files that exist on disk are added — no nulls.
VARIANT_ORDER = ["MAIN", "PT01", "PT02", "PT03", "PT04", "PT05", "PT06"]

# ---------------------------------------------------------------------------
# PyArrow schema
# Sequence(Image()) is stored as list<struct<bytes: binary, path: utf8>>
# ---------------------------------------------------------------------------

IMG_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCHEMA = pa.schema([
    ("parent_asin", pa.string()),
    ("images",      pa.list_(IMG_STRUCT)),
])

# dataset_info features block (written to the repo so HF knows the types)
DATASET_INFO = {
    "features": {
        "parent_asin": {"dtype": "string", "_type": "Value"},
        "images": {
            "feature": {"_type": "Image"},
            "_type":   "Sequence",
        },
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_asins() -> list[str]:
    """Return sorted unique list of parent_asin values from metadata."""
    asins: set[str] = set()
    with open(META_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            asin = item.get("parent_asin")
            if asin:
                asins.add(asin)
    return sorted(asins)


def image_to_bytes(path: Path) -> bytes | None:
    """Load a JPEG and return its raw bytes, or None on failure."""
    try:
        img = PILImage.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return None


def collect_images(asin: str) -> list[dict]:
    """Return list of {bytes, path} structs for all existing variants.

    Only variants that exist on disk AND load successfully are included.
    The list is never empty for a filtered ASIN (MAIN is guaranteed).
    """
    structs = []
    for variant in VARIANT_ORDER:
        p = IMAGE_DIR / f"{asin}_{variant}.jpg"
        if not p.exists():
            continue
        b = image_to_bytes(p)
        if b is not None:
            structs.append({"bytes": b, "path": None})
    return structs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push All_Beauty pure-image dataset to HuggingFace."
    )
    parser.add_argument("--token",      default=None, help="HuggingFace write token")
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE,
                        help=f"Items per parquet shard (default: {SHARD_SIZE})")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Build shards locally but do not upload")
    args = parser.parse_args()

    api = HfApi(token=args.token) if not args.dry_run else None
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect filtered ASINs (must have MAIN image)
    # ------------------------------------------------------------------
    print("Loading ASINs from metadata ...")
    all_asins = load_asins()
    print(f"  Metadata ASINs : {len(all_asins):,}")

    filtered = [a for a in all_asins if (IMAGE_DIR / f"{a}_MAIN.jpg").exists()]
    print(f"  With MAIN image: {len(filtered):,}  (filtered out {len(all_asins)-len(filtered):,})")

    n_shards = (len(filtered) + args.shard_size - 1) // args.shard_size
    print(f"  Shards         : {n_shards} × {args.shard_size} items\n")

    # ------------------------------------------------------------------
    # 2. Write + upload shards
    # ------------------------------------------------------------------
    for shard_idx in tqdm(range(n_shards), desc="Shards"):
        shard_path = SHARD_DIR / f"train-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        repo_path  = f"data/train-{shard_idx:05d}-of-{n_shards:05d}.parquet"

        if shard_path.exists():
            tqdm.write(f"  [{shard_idx+1}/{n_shards}] Already on disk — skipping build")
        else:
            batch = filtered[shard_idx * args.shard_size : (shard_idx + 1) * args.shard_size]

            asin_col:   list[str]       = []
            images_col: list[list[dict]] = []

            for asin in tqdm(batch, desc=f"  Shard {shard_idx+1}", leave=False):
                imgs = collect_images(asin)
                if not imgs:          # should not happen for filtered ASINs
                    continue
                asin_col.append(asin)
                images_col.append(imgs)

            # Build pyarrow arrays — images as list<struct>
            images_pa = pa.array(images_col, type=pa.list_(IMG_STRUCT))

            table = pa.table(
                {"parent_asin": pa.array(asin_col, type=pa.string()),
                 "images":      images_pa},
                schema=SCHEMA,
            )
            pq.write_table(table, shard_path, compression="snappy")
            mb = shard_path.stat().st_size / 1e6
            tqdm.write(f"  [{shard_idx+1}/{n_shards}] Wrote {shard_path.name}  ({mb:.0f} MB)")

        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            tqdm.write(f"  [{shard_idx+1}/{n_shards}] Uploaded → {repo_path}")

    # ------------------------------------------------------------------
    # 3. Upload dataset_info.json so HF recognises Sequence(Image())
    # ------------------------------------------------------------------
    info_path = SHARD_DIR / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(DATASET_INFO, f, indent=2)

    if not args.dry_run:
        api.upload_file(
            path_or_fileobj=str(info_path),
            path_in_repo="data/dataset_info.json",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print("\nUploaded dataset_info.json")

    print(f"\nDone.  {'(dry run — no uploads)' if args.dry_run else f'Repo: https://huggingface.co/datasets/{REPO_ID}'}")


if __name__ == "__main__":
    main()
