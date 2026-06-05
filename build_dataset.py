#!/usr/bin/env python3
"""Build train/val/test JSONL splits from McAuley-Lab/Amazon-Reviews-2023.

Uses the pre-processed ``benchmark/{kcore}/last_out_w_his`` CSV files from
HuggingFace (default: ``0core`` — no interaction-count filter).

The CSV test split contains one row per user with:
  - user_id       : user identifier
  - parent_asin   : the user's last (test) item
  - history       : space-separated ordered interaction history

From this we reconstruct the full per-user sequence and apply leave-one-out:
  - test  : last item
  - valid : second-to-last item
  - train : all earlier (history, target) pairs

Items not present in local metadata (or without a semantic ID) are skipped.
Users whose sequence drops below 3 items after filtering are excluded.

JSONL output format
--------------------
Indexing row (one per corpus item in train.jsonl):
  { "operation": "indexing",
    "text":   "<item description>",
    "doc_id": "<|d0_X|> <|d1_Y|> <|d2_Z|>",
    "item":   "<|d0_X|> <|d1_Y|> <|d2_Z|>",
    "asin":   "<parent_asin>" }

Query row (train / valid / test):
  { "operation": "query",
    "text":    "<target item description>",
    "doc_id":  "<target semantic id>",
    "item":    "<target semantic id>",
    "asin":    "<target parent_asin>",
    "user_id": "<user_id>" }

Usage
-----
  python3 build_dataset.py \\
      --category All_Beauty \\
      --out_dir  data/amazon_all_beauty \\
      --kcore    0 \\
      --min_seq_len 3
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Text prompt builder (mirrors data/amazon2023.py)
# ---------------------------------------------------------------------------

TEXT_FIELDS = [
    "main_category", "title", "average_rating", "rating_number",
    "features", "description", "price", "categories", "details", "bought_together",
]


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


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(category: str, datasets_dir: Path) -> dict:
    path = datasets_dir / f"meta_{category}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    items = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            asin = item.get("parent_asin")
            if asin:
                items[asin] = item
    print(f"Loaded {len(items):,} items from metadata")
    return items


def load_semantic_id_tokens(semid_path: Path) -> dict:
    if not semid_path.exists():
        raise FileNotFoundError(
            f"Semantic ID tokens not found at {semid_path}. "
            "Run encode_semantic_ids.py first."
        )
    with open(semid_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load benchmark CSVs from HuggingFace
# ---------------------------------------------------------------------------

def download_csv(category: str, kcore: int, split: str, hf_cache: str) -> Path:
    filename = f"benchmark/{kcore}core/last_out_w_his/{category}.{split}.csv"
    print(f"  Downloading {filename} ...")
    path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=filename,
        repo_type="dataset",
        cache_dir=hf_cache,
    )
    return Path(path)


def load_user_sequences(
    category: str,
    kcore: int,
    corpus_asins: set,
    hf_cache: str,
    min_seq_len: int,
) -> dict:
    """Build per-user sequences from the test CSV (which has the full history).

    Returns
    -------
    {user_id: [asin_0, asin_1, ..., asin_n]}  in timestamp order
    """
    test_csv = download_csv(category, kcore, "test", hf_cache)
    df = pd.read_csv(test_csv)
    print(f"  Loaded {len(df):,} rows from test CSV")

    user_sequences = {}
    dropped_target = dropped_hist = dropped_short = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building sequences"):
        uid = str(row["user_id"])
        target_asin = str(row.get("parent_asin") or row.get("asin") or "")
        history_raw = row.get("history", "")
        history_asins = history_raw.split() if isinstance(history_raw, str) and history_raw == history_raw else []

        # Filter target
        if target_asin not in corpus_asins:
            dropped_target += 1
            continue

        # Filter history to corpus items
        hist_filtered = [a for a in history_asins if a in corpus_asins]

        if not hist_filtered:
            dropped_hist += 1
            continue

        full_seq = hist_filtered + [target_asin]

        # Remove duplicates, keeping last occurrence (most recent)
        seen = {}
        for i, asin in enumerate(full_seq):
            seen[asin] = i
        full_seq = [asin for asin, _ in sorted(seen.items(), key=lambda x: x[1])]

        if len(full_seq) < min_seq_len:
            dropped_short += 1
            continue

        user_sequences[uid] = full_seq

    print(
        f"  Users kept: {len(user_sequences):,} | "
        f"Dropped — target not in corpus: {dropped_target:,}  "
        f"empty history: {dropped_hist:,}  "
        f"seq too short: {dropped_short:,}"
    )
    return user_sequences


# ---------------------------------------------------------------------------
# Leave-one-out splitting
# ---------------------------------------------------------------------------

def build_splits(user_sequences: dict) -> tuple[list, list, list]:
    train_rows, val_rows, test_rows = [], [], []

    for uid, seq in user_sequences.items():
        n = len(seq)
        # guaranteed n >= min_seq_len (≥3) by load_user_sequences

        # Test: last item
        test_rows.append({"user_id": uid, "target_asin": seq[-1], "history": seq[:-1]})

        # Valid: second-to-last item
        if n >= 3:
            val_rows.append({"user_id": uid, "target_asin": seq[-2], "history": seq[:-2]})

        # Train: all subsequences target = seq[k], history = seq[:k], k in {1..n-2}
        for k in range(1, n - 1):
            train_rows.append({"user_id": uid, "target_asin": seq[k], "history": seq[:k]})

    print(
        f"Splits — train: {len(train_rows):,} query rows  "
        f"val: {len(val_rows):,}  test: {len(test_rows):,}"
    )
    return train_rows, val_rows, test_rows


# ---------------------------------------------------------------------------
# JSONL builders
# ---------------------------------------------------------------------------

def make_indexing_row(asin: str, metadata: dict, semid_tokens: dict) -> dict:
    return {
        "operation": "indexing",
        "text": build_text_prompt(metadata[asin]),
        "doc_id": semid_tokens[asin],
        "item": semid_tokens[asin],
        "asin": asin,
    }


def make_query_row(user_id: str, target_asin: str, metadata: dict, semid_tokens: dict) -> dict:
    return {
        "operation": "query",
        "text": build_text_prompt(metadata[target_asin]),
        "doc_id": semid_tokens[target_asin],
        "item": semid_tokens[target_asin],
        "asin": target_asin,
        "user_id": user_id,
    }


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def build_train_jsonl(
    train_rows: list,
    metadata: dict,
    semid_tokens: dict,
    corpus_asins: set,
) -> list[dict]:
    """Indexing rows for every corpus item + query rows for all training samples."""
    out = []
    # All corpus items as indexing rows
    for asin in sorted(corpus_asins):
        out.append(make_indexing_row(asin, metadata, semid_tokens))
    # Training query rows
    for row in tqdm(train_rows, desc="Building train query rows"):
        out.append(make_query_row(row["user_id"], row["target_asin"], metadata, semid_tokens))
    return out


def build_eval_jsonl(
    eval_rows: list,
    metadata: dict,
    semid_tokens: dict,
) -> list[dict]:
    out = []
    for row in eval_rows:
        out.append(make_query_row(row["user_id"], row["target_asin"], metadata, semid_tokens))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Amazon2023 JSONL dataset splits.")
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--out_dir", default="data/amazon_all_beauty")
    parser.add_argument("--datasets_dir", default="datasets")
    parser.add_argument("--semid_path",
                        default="out/rqvae/All_Beauty/cross_modal/semantic_id_tokens.json")
    parser.add_argument("--hf_cache", default="datasets/hf_cache")
    parser.add_argument("--kcore", type=int, default=0,
                        help="k-core benchmark version to use (0 = all users, 5 = 5-core)")
    parser.add_argument("--min_seq_len", type=int, default=3,
                        help="Minimum sequence length after corpus filtering (default 3)")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.out_dir)
    semid_path = Path(args.semid_path)

    # 1. Load metadata + semantic IDs
    metadata = load_metadata(args.category, datasets_dir)
    semid_tokens = load_semantic_id_tokens(semid_path)
    corpus_asins = set(metadata.keys()) & set(semid_tokens.keys())
    print(f"Corpus size: {len(corpus_asins):,} items\n")

    # 2. Load & filter user sequences from the HF benchmark CSV
    print(f"Loading {args.kcore}core benchmark from HuggingFace ...")
    user_sequences = load_user_sequences(
        args.category, args.kcore, corpus_asins, args.hf_cache, args.min_seq_len
    )
    print(f"\nFinal unique users: {len(user_sequences):,}\n")

    # 3. Leave-one-out splits
    train_rows, val_rows, test_rows = build_splits(user_sequences)

    # 4. Write JSONL files
    print(f"\nWriting JSONL files to {out_dir} ...")

    train_jsonl = build_train_jsonl(train_rows, metadata, semid_tokens, corpus_asins)
    write_jsonl(out_dir / "train.jsonl", train_jsonl)

    val_jsonl = build_eval_jsonl(val_rows, metadata, semid_tokens)
    write_jsonl(out_dir / "valid.jsonl", val_jsonl)

    test_jsonl = build_eval_jsonl(test_rows, metadata, semid_tokens)
    write_jsonl(out_dir / "test.jsonl", test_jsonl)

    # Update test.jsonl → replaces old file (test.jsonl already exists)
    # (valid.jsonl is new — existing codebase uses test.jsonl for evaluation)

    n_idx = sum(1 for r in train_jsonl if r["operation"] == "indexing")
    n_qry = sum(1 for r in train_jsonl if r["operation"] == "query")
    print(f"\nSummary")
    print(f"  train.jsonl : {n_idx:,} indexing + {n_qry:,} query = {len(train_jsonl):,} rows")
    print(f"  valid.jsonl : {len(val_jsonl):,} rows")
    print(f"  test.jsonl  : {len(test_jsonl):,} rows")
    print(f"  Users       : {len(user_sequences):,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
