"""Amazon 2023 dataset loading for generative recommendation.

Supports multiple categories and three input modalities:
  - text   : Qwen3-VL-Embedding-2B embeddings of item metadata
  - image  : Qwen3-VL-Embedding-2B embeddings of product images
  - multimodal: Qwen3-VL-Embedding-2B embeddings combining text and image

Dataset splits follow the leave-one-out protocol based on review timestamps:
  - Test  : last interacted item as target; all preceding items as input.
  - Valid : second-to-last item as target; preceding items as input.
  - Train : history up to the third-to-last item as input, predicting the
            second-to-last item.  subsample=True augments by random slicing.

The class handles:
  - Metadata from local JSONL files in DATASETS_DIR
  - Sequential recommendation data downloaded from HuggingFace
  - Missing images (skipped gracefully)
  - Caching of computed embeddings to avoid recomputation
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

from data.schemas import FUT_SUFFIX, SeqBatch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("/work/u1304848/AI/project/datasets")
IMAGE_DIR = DATASETS_DIR / "images"
CACHE_DIR = Path("/work/u1304848/AI/project/outputs/embeddings")

# HuggingFace config names for sequential data
# Pattern: 5core_last_out_w_his_{Category}
CATEGORY_TO_HF_CONFIG = {
    "All_Beauty": "5core_last_out_w_his_All_Beauty",
    "Musical_Instruments": "5core_last_out_w_his_Musical_Instruments",
    # Add more categories here as needed
}

METADATA_FILES = {
    "All_Beauty": DATASETS_DIR / "meta_All_Beauty.jsonl",
    "Musical_Instruments": DATASETS_DIR / "meta_Musical_Instruments.jsonl",
    # Add more categories here as needed
}

MAX_SEQ_LEN = 20

TEXT_FIELDS = [
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "categories",
    "details",
    "bought_together",
]


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _format_field(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "N/A"
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items()]
        return "; ".join(parts) if parts else "N/A"
    return str(value)


def build_text_prompt(item: dict) -> str:
    parts = []
    for field in TEXT_FIELDS:
        val = item.get(field)
        parts.append(f"{field}: {_format_field(val)}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(category: str) -> Dict[str, dict]:
    """Return a dict mapping parent_asin → item metadata dict."""
    path = METADATA_FILES[category]
    items = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            asin = item.get("parent_asin")
            if asin:
                items[asin] = item
    return items


def build_asin_index(metadata: Dict[str, dict]) -> Tuple[List[str], Dict[str, int]]:
    """Create a stable sorted list of ASINs and an ASIN→integer mapping."""
    asins = sorted(metadata.keys())
    asin2idx = {asin: idx for idx, asin in enumerate(asins)}
    return asins, asin2idx


# ---------------------------------------------------------------------------
# Sequential data loading from HuggingFace
# ---------------------------------------------------------------------------

def load_sequential_data(
    category: str,
    asin2idx: Dict[str, int],
    cache_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """Build leave-one-out train/valid/test splits from HuggingFace data.

    The test split of the HuggingFace ``5core_last_out_w_his`` config
    provides each user's timestamp-ordered interaction history together
    with the absolute last item.  This function reconstructs the full
    sequence per user and applies the leave-one-out protocol:

      - Test  : history = [i_1, ..., i_{n-1}],  target = i_n
      - Valid : history = [i_1, ..., i_{n-2}],  target = i_{n-1}
      - Train : history = [i_1, ..., i_{n-3}],  target = i_{n-2}
                (requires n >= 4; used with subsample=True for augmentation)

    Items not present in ``asin2idx`` are skipped; users with fewer than
    3 metadata-filtered items are excluded.

    Returns a dict with keys 'train', 'valid', 'test', each mapping to:
      {
        'user_ids':    List[int],
        'history':     List[List[int]],   # integer-indexed item history
        'target':      List[int],         # target item integer index
      }
    """
    config_name = CATEGORY_TO_HF_CONFIG[category]
    hf_cache = cache_dir or str(DATASETS_DIR / "hf_cache")

    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        trust_remote_code=True,
        cache_dir=hf_cache,
    )

    if "test" not in dataset:
        raise KeyError(
            f"No 'test' split found in HuggingFace dataset for category '{category}'. "
            "Cannot reconstruct full user sequences for leave-one-out splitting."
        )

    train_data: Dict[str, list] = {"user_ids": [], "history": [], "target": []}
    valid_data: Dict[str, list] = {"user_ids": [], "history": [], "target": []}
    test_data: Dict[str, list] = {"user_ids": [], "history": [], "target": []}

    for uid, row in enumerate(dataset["test"]):
        target_asin = row.get("parent_asin") or row.get("asin")
        history_raw = row.get("history", "")
        history_asins = history_raw.split() if isinstance(history_raw, str) else history_raw

        target_idx = asin2idx.get(target_asin, -1)
        hist_idx = [asin2idx[a] for a in history_asins if a in asin2idx]

        if target_idx == -1 or len(hist_idx) == 0:
            continue

        # Full timestamp-ordered sequence: history (sorted by HF) + last item
        full_seq = hist_idx + [target_idx]
        n = len(full_seq)

        if n < 3:
            continue

        # Test: last item as target, all preceding items as input
        test_data["user_ids"].append(uid)
        test_data["history"].append(full_seq[:-1])
        test_data["target"].append(full_seq[-1])

        # Valid: second-to-last as target, preceding items as input
        valid_data["user_ids"].append(uid)
        valid_data["history"].append(full_seq[:-2])
        valid_data["target"].append(full_seq[-2])

        # Train: all valid subsequences (history=[i_1,...,i_{k-1}], target=i_k)
        # for k in {2,...,n-2}, preserving valid and test as leave-one-out.
        for k in range(2, n - 1):
            train_data["user_ids"].append(uid)
            train_data["history"].append(full_seq[:k - 1])   # [i_1, ..., i_{k-1}]
            train_data["target"].append(full_seq[k - 1])     # i_k
    return {"train": train_data, "valid": valid_data, "test": test_data}


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class ItemEmbeddingDataset(Dataset):
    """Dataset of item embeddings for RQ-VAE training.

    Parameters
    ----------
    embeddings : Tensor of shape (N, D)
    split : 'train' | 'eval' | 'all'
        Uses a 95/5 random split (seed=42) for train/eval.
    """

    def __init__(
        self,
        embeddings: Tensor,
        split: str = "all",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.split = split

        gen = torch.Generator()
        gen.manual_seed(seed)
        is_train = torch.rand(embeddings.shape[0], generator=gen) > 0.05

        if split == "train":
            self.embeddings = embeddings[is_train]
        elif split == "eval":
            self.embeddings = embeddings[~is_train]
        else:
            self.embeddings = embeddings

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        item_ids = (
            torch.tensor(idx).unsqueeze(0)
            if not isinstance(idx, torch.Tensor)
            else idx
        )
        x = self.embeddings[idx]
        dummy = torch.zeros_like(item_ids.squeeze(0))
        return SeqBatch(
            user_ids=dummy,
            ids=item_ids,
            ids_fut=dummy,
            x=x,
            x_fut=torch.zeros_like(x),
            seq_mask=torch.ones_like(item_ids, dtype=torch.bool),
        )


class SequentialRecommendationDataset(Dataset):
    """Dataset of user interaction sequences for decoder training/eval.

    Parameters
    ----------
    embeddings : Tensor of shape (N, D)
        Item embeddings aligned with asin2idx ordering.
    split_data : dict
        Output of load_sequential_data for a specific split.
    max_seq_len : int
        Maximum history length to use.
    subsample : bool
        If True, randomly subsample subsequences from history (for training).
    """

    def __init__(
        self,
        embeddings: Tensor,
        split_data: dict,
        max_seq_len: int = MAX_SEQ_LEN,
        subsample: bool = False,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.user_ids = split_data["user_ids"]
        self.histories = split_data["history"]
        self.targets = split_data["target"]
        self.max_seq_len = max_seq_len
        self.subsample = subsample

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> SeqBatch:
        user_id = self.user_ids[idx]
        history = self.histories[idx]
        target = self.targets[idx]

        if self.subsample and len(history) > 2:
            start = random.randint(0, max(0, len(history) - 2))
            end = random.randint(start + 1, min(len(history), start + self.max_seq_len))
            history = history[start:end]

        # Truncate and pad history
        history = history[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(history)
        item_ids = torch.tensor(history + [-1] * pad_len, dtype=torch.long)
        item_ids_fut = torch.tensor([target], dtype=torch.long)

        x = self.embeddings[item_ids.clamp(min=0)]
        x[item_ids == -1] = 0.0

        x_fut = self.embeddings[item_ids_fut.clamp(min=0)]

        return SeqBatch(
            user_ids=torch.tensor([user_id], dtype=torch.long),
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0),
        )
