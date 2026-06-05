#!/usr/bin/env python3
"""Build and push the amazon-all-beauty HuggingFace dataset.

Two configs are created:
  - items        : one row per item (112,590) with metadata, images, semantic IDs
  - interactions : train / valid / test splits with user sequences

Usage
-----
  python3 push_to_hf.py --token <HF_TOKEN>
  python3 push_to_hf.py --token <HF_TOKEN> --skip_items   # interactions only
  python3 push_to_hf.py --token <HF_TOKEN> --skip_interactions
"""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR   = Path("datasets")
META_PATH      = DATASETS_DIR / "meta_All_Beauty.jsonl"
IMAGE_DIR      = DATASETS_DIR / "images" / "All_Beauty"
HF_CACHE       = str(DATASETS_DIR / "hf_cache")
SEMID_TOKENS   = Path("out/rqvae/All_Beauty/cross_modal/semantic_id_tokens.json")
SEMID_CODES    = Path("out/rqvae/All_Beauty/cross_modal/semantic_ids.json")
TRAIN_JSONL    = Path("data/amazon_all_beauty/train.jsonl")
VALID_JSONL    = Path("data/amazon_all_beauty/valid.jsonl")
TEST_JSONL     = Path("data/amazon_all_beauty/test.jsonl")

REPO_ID        = "theblackcat102/amazon-all-beauty"
CATEGORY       = "All_Beauty"
KCORE          = 0

TEXT_FIELDS = [
    "main_category", "title", "average_rating", "rating_number",
    "features", "description", "price", "categories", "details", "bought_together",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "N/A"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items()) or "N/A"
    return str(value)


def build_text_prompt(item: dict) -> str:
    return " | ".join(f"{f}: {_fmt(item.get(f))}" for f in TEXT_FIELDS)


def load_metadata() -> dict:
    print("Loading metadata ...")
    items = {}
    with open(META_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            asin = item.get("parent_asin")
            if asin:
                items[asin] = item
    print(f"  {len(items):,} items loaded")
    return items


def load_semids() -> tuple[dict, dict]:
    print("Loading semantic IDs ...")
    with open(SEMID_TOKENS) as f:
        tokens = json.load(f)     # {asin: "<|d0_X|> <|d1_Y|> <|d2_Z|>"}
    with open(SEMID_CODES) as f:
        codes = json.load(f)      # {asin: [c0, c1, c2, dedup]}
    print(f"  {len(tokens):,} token strings, {len(codes):,} code lists")
    return tokens, codes


def load_image(path: Path):
    """Return PIL Image or None if file missing."""
    if not path.exists():
        return None
    try:
        return PILImage.open(path).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1: Reconstruct history per user from the 0core CSV
# ---------------------------------------------------------------------------

def reconstruct_user_sequences(corpus_asins: set) -> dict:
    """Return {user_id: [asin_0, ..., asin_n]} in timestamp order.

    Uses the cached 0core/last_out_w_his test CSV which contains the full
    interaction history per user (same logic as build_dataset.py).
    """
    print(f"\nReconstructing user sequences from {KCORE}core CSV ...")
    csv_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"benchmark/{KCORE}core/last_out_w_his/{CATEGORY}.test.csv",
        repo_type="dataset",
        cache_dir=HF_CACHE,
    )
    df = pd.read_csv(csv_path)
    print(f"  CSV rows: {len(df):,}")

    user_sequences = {}
    dropped = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Building sequences"):
        uid = str(row["user_id"])
        target_asin = str(row.get("parent_asin") or row.get("asin") or "")
        history_raw = row.get("history", "")
        history_asins = (
            history_raw.split()
            if isinstance(history_raw, str) and pd.notna(history_raw)
            else []
        )

        if target_asin not in corpus_asins:
            dropped["target_not_in_corpus"] += 1
            continue

        hist_filtered = [a for a in history_asins if a in corpus_asins]

        if not hist_filtered:
            dropped["empty_history"] += 1
            continue

        full_seq = hist_filtered + [target_asin]

        # Dedup keeping last occurrence
        seen = {}
        for i, asin in enumerate(full_seq):
            seen[asin] = i
        full_seq = [a for a, _ in sorted(seen.items(), key=lambda x: x[1])]

        if len(full_seq) < 3:
            dropped["too_short"] += 1
            continue

        user_sequences[uid] = full_seq

    print(f"  Users kept : {len(user_sequences):,}")
    print(f"  Dropped    : {dict(dropped)}")
    return user_sequences


# ---------------------------------------------------------------------------
# Step 2: Build interactions dataset
# ---------------------------------------------------------------------------

def load_query_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("operation") == "query":
                rows.append(obj)
    return rows


def build_interactions(
    user_sequences: dict,
    semid_tokens: dict,
    semid_codes: dict,
) -> DatasetDict:
    """Build train/valid/test DatasetDict for the interactions config."""
    print("\nBuilding interactions dataset ...")

    splits_jsonl = {
        "train": TRAIN_JSONL,
        "valid": VALID_JSONL,
        "test":  TEST_JSONL,
    }

    split_datasets = {}

    for split_name, jsonl_path in splits_jsonl.items():
        query_rows = load_query_rows(jsonl_path)
        print(f"  {split_name}: {len(query_rows):,} query rows")

        records = []
        skipped = 0

        for row in tqdm(query_rows, desc=f"  Processing {split_name}", leave=False):
            uid      = row["user_id"]
            t_asin   = row["asin"]
            t_text   = row["text"]
            t_semid  = semid_tokens.get(t_asin, "")
            t_codes  = semid_codes.get(t_asin, [])

            # Reconstruct ordered history = full_seq[:-1] for test/valid,
            # or full_seq[:k] for train subsequences.
            # We know the full sequence per user; find the position of target.
            full_seq = user_sequences.get(uid)
            if full_seq is None:
                skipped += 1
                continue

            if t_asin not in full_seq:
                skipped += 1
                continue

            target_pos = full_seq.index(t_asin)
            history_asins = full_seq[:target_pos]  # everything before the target

            # Map history ASINs → semantic ID token strings
            history_semids = [
                semid_tokens[a] for a in history_asins if a in semid_tokens
            ]

            records.append({
                "user_id":               uid,
                "target_asin":           t_asin,
                "target_text":           t_text,
                "target_semantic_id":    t_semid,
                "target_semantic_codes": t_codes,
                "history_semantic_ids":  history_semids,
                "split":                 split_name,
            })

        if skipped:
            print(f"    Skipped {skipped:,} rows (user not in sequences or target not found)")

        split_datasets[split_name] = Dataset.from_list(records, features=Features({
            "user_id":               Value("string"),
            "target_asin":           Value("string"),
            "target_text":           Value("string"),
            "target_semantic_id":    Value("string"),
            "target_semantic_codes": Sequence(Value("int32")),
            "history_semantic_ids":  Sequence(Value("string")),
            "split":                 Value("string"),
        }))
        print(f"  {split_name} dataset: {len(split_datasets[split_name]):,} rows")

    return DatasetDict(split_datasets)


# ---------------------------------------------------------------------------
# Step 3: Build items dataset
# ---------------------------------------------------------------------------

ITEMS_FEATURES = Features({
    "parent_asin":    Value("string"),
    "title":          Value("string"),
    "main_category":  Value("string"),
    "average_rating": Value("float64"),
    "rating_number":  Value("int64"),
    "price":          Value("float64"),
    "store":          Value("string"),
    "features":       Sequence(Value("string")),
    "description":    Sequence(Value("string")),
    "details":        Value("string"),
    "image_urls":     Value("string"),
    "text_prompt":    Value("string"),
    "semantic_id":    Value("string"),
    "semantic_codes": Sequence(Value("int32")),
    "image_main":     Image(),
    "image_pt01":     Image(),
    "image_pt02":     Image(),
})


def items_generator(metadata: dict, semid_tokens: dict, semid_codes: dict):
    """Yield one item row at a time — avoids loading all images into RAM."""
    for asin, item in metadata.items():
        yield {
            "parent_asin":    asin,
            "title":          item.get("title") or "",
            "main_category":  item.get("main_category") or "",
            "average_rating": float(item.get("average_rating") or 0.0),
            "rating_number":  int(item.get("rating_number") or 0),
            "price":          float(item["price"]) if item.get("price") not in (None, "None", "") else None,
            "store":          item.get("store") or None,
            "features":       [str(x) for x in (item.get("features") or [])],
            "description":    [str(x) for x in (item.get("description") or [])],
            "details":        json.dumps(item.get("details") or {}),
            "image_urls":     json.dumps(item.get("images") or []),
            "text_prompt":    build_text_prompt(item),
            "semantic_id":    semid_tokens.get(asin, ""),
            "semantic_codes": semid_codes.get(asin, []),
            "image_main":     load_image(IMAGE_DIR / f"{asin}_MAIN.jpg"),
            "image_pt01":     load_image(IMAGE_DIR / f"{asin}_PT01.jpg"),
            "image_pt02":     load_image(IMAGE_DIR / f"{asin}_PT02.jpg"),
        }


SHARD_DIR = Path("/mnt/storage/temp/items_shards")
SHARD_SIZE = 2000  # items per shard


def build_and_push_items_sharded(
    metadata: dict,
    semid_tokens: dict,
    semid_codes: dict,
    repo_id: str,
    token: str | None,
):
    """Write Parquet shards locally and upload each one immediately.

    Skips shards whose Parquet file already exists on disk — safe to resume
    if the process is interrupted.
    """
    from huggingface_hub import HfApi
    import pyarrow as pa
    import pyarrow.parquet as pq

    api = HfApi(token=token)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    asins = list(metadata.keys())
    n_shards = (len(asins) + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"\nBuilding items config: {len(asins):,} items → {n_shards} shards of {SHARD_SIZE}")

    # Build pyarrow schema from HF Features
    pa_schema = ITEMS_FEATURES.arrow_schema

    for shard_idx in range(n_shards):
        shard_path = SHARD_DIR / f"items-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        repo_path  = f"data/items/items-{shard_idx:05d}-of-{n_shards:05d}.parquet"

        # Check if already uploaded (resumability)
        if shard_path.exists():
            print(f"  Shard {shard_idx+1}/{n_shards}: {shard_path.name} already on disk, uploading ...")
        else:
            batch_asins = asins[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]
            rows = list(items_generator(
                {a: metadata[a] for a in batch_asins}, semid_tokens, semid_codes
            ))

            # Convert PIL Images to bytes for pyarrow
            for row in rows:
                for img_col in ("image_main", "image_pt01", "image_pt02"):
                    img = row[img_col]
                    if img is not None:
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG")
                        row[img_col] = {"bytes": buf.getvalue(), "path": None}
                    else:
                        row[img_col] = {"bytes": None, "path": None}

            # Build columnar dict for pyarrow
            cols = {col: [r[col] for r in rows] for col in rows[0]}

            # Handle Image struct columns manually
            for img_col in ("image_main", "image_pt01", "image_pt02"):
                cols[img_col] = pa.array(
                    cols[img_col],
                    type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                )

            table = pa.table(
                {k: v if isinstance(v, pa.Array) else pa.array(v)
                 for k, v in cols.items()},
                schema=pa_schema,
            )
            pq.write_table(table, shard_path, compression="snappy")
            mb = shard_path.stat().st_size / 1e6
            print(f"  Shard {shard_idx+1}/{n_shards}: wrote {shard_path.name} ({mb:.0f} MB)", end=" ... ")

        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("uploaded.")

    # Write dataset_info card entry for the items config
    print("  All shards uploaded.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="HuggingFace write token")
    parser.add_argument("--skip_items",        action="store_true")
    parser.add_argument("--skip_interactions", action="store_true")
    args = parser.parse_args()

    # --- Load shared resources ---
    metadata      = load_metadata()
    semid_tokens, semid_codes = load_semids()
    corpus_asins  = set(metadata.keys()) & set(semid_tokens.keys())

    # --- Step 1: Reconstruct history per user ---
    user_sequences = reconstruct_user_sequences(corpus_asins)

    # --- Step 2: Interactions config ---
    if not args.skip_interactions:
        interactions = build_interactions(user_sequences, semid_tokens, semid_codes)
        print(f"\nInteractions DatasetDict:")
        print(interactions)
        print("\nSample train row:")
        print(interactions["train"][0])

    # --- Step 3: Items config ---
    # --- Step 4: Push to Hub ---
    print(f"\nPushing to {REPO_ID} ...")

    if not args.skip_interactions:
        print("  Pushing interactions config ...")
        interactions.push_to_hub(
            REPO_ID,
            config_name="interactions",
            token=args.token,
        )
        print("  interactions config pushed.")

    if not args.skip_items:
        build_and_push_items_sharded(
            metadata, semid_tokens, semid_codes,
            repo_id=REPO_ID,
            token=args.token,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
