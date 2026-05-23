#!/work/u1304848/.conda/envs/AI1/bin/python3
"""
Multimodal embedding generation for Amazon Beauty items.

Uses Qwen/Qwen3-VL-Embedding-2B as the multimodal backbone to produce
dense embeddings from each item's title, categories, description, price,
and product images. Embeddings are extracted via last-token pooling of the
final hidden layer and L2-normalised.

Qwen/Qwen3-VL-Embedding-2B is a dedicated multimodal visual-language
embedding model that jointly encodes image and text inputs.

Input:
  /work/u1304848/AI/project/datasets/meta_All_Beauty.jsonl
  /work/u1304848/AI/project/datasets/images/All_Beauty/<asin>_<variant>.jpg

Output:
  /work/u1304848/AI/project/outputs/multimodal_embeddings.npy  -- (N, 2048) float32
  /work/u1304848/AI/project/outputs/item_ids.json              -- list of N parent_asin strings

Usage:
  python3 generate_multimodal_embeddings.py
  # or with the conda env explicitly:
  /work/u1304848/.conda/envs/AI1/bin/python3 generate_multimodal_embeddings.py
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Qwen/Qwen3-VL-Embedding-2B is the target model.
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

IMAGE_DIR  = Path("/work/u1304848/AI/project/datasets/images/All_Beauty")
DATA_FILE  = Path("/work/u1304848/AI/project/datasets/meta_All_Beauty.jsonl")
OUTPUT_DIR = Path("/work/u1304848/AI/project/outputs")

MAX_IMAGES   = 3     # max product images per item
MAX_DESC_LEN = 400   # truncate long description strings before tokenisation
# NOTE: do not truncate at the processor level — image tokens occupy many
# positions and truncation causes a text/input_ids token count mismatch.
# Text is already bounded by format_text() so sequences stay manageable.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Instruction prefix that steers the last-token representation toward retrieval
EMBED_INSTRUCTION = (
    "Represent this beauty product for dense retrieval based on its "
    "appearance and attributes: "
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_paths(parent_asin: str, image_meta: list) -> list:
    """Return existing local image paths for an item (up to MAX_IMAGES)."""
    paths = []
    for img in image_meta or []:
        variant = img.get("variant", "MAIN")
        p = IMAGE_DIR / f"{parent_asin}_{variant}.jpg"
        if p.exists():
            paths.append(p)
        if len(paths) >= MAX_IMAGES:
            break
    return paths


def format_text(item: dict) -> str:
    """Build a structured text description from item metadata."""
    parts = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")

    cats = item.get("categories") or []
    if isinstance(cats, list):
        cats = ", ".join(str(c) for c in cats if c)
    if cats:
        parts.append(f"Categories: {cats}")

    desc = item.get("description") or []
    if isinstance(desc, list):
        desc = " ".join(str(d) for d in desc if d)
    if desc:
        parts.append(f"Description: {str(desc)[:MAX_DESC_LEN]}")

    price = item.get("price")
    if price is not None:
        parts.append(f"Price: ${price}")

    return "\n".join(parts) if parts else item.get("parent_asin", "")


def build_messages(item: dict, image_paths: list) -> list:
    """Build Qwen-chat-style messages list for one item."""
    content = []
    for p in image_paths:
        # Pass file path as string; processor loads and pre-processes images
        content.append({"type": "image", "image": str(p)})
    text = EMBED_INSTRUCTION + format_text(item)
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the hidden state of the last real (non-pad) token per sample."""
    # attention_mask: (B, L)  hidden: (B, L, D)
    seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, seq_lens]                # (B, D)


@torch.no_grad()
def encode_item(
    item: dict,
    model: AutoModel,
    processor: AutoProcessor,
) -> np.ndarray | None:
    """
    Encode a single item.  Returns a (1, hidden_size) float32 numpy array,
    or None if processing fails.
    """
    parent_asin = item.get("parent_asin", "")
    image_paths = get_image_paths(parent_asin, item.get("images") or [])

    try:
        messages  = build_messages(item, image_paths)
        text_str  = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Load PIL images; skip corrupted files silently
        pil_images = []
        for p in image_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass

        inputs = processor(
            text=[text_str],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state                        # (1, L, D)
        emb = last_token_pool(last_hidden, inputs["attention_mask"])   # (1, D)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().float().numpy()

    except Exception as exc:
        print(f"  Warning: skipping {parent_asin}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = OUTPUT_DIR / "multimodal_embeddings.npy"
    ids_path = OUTPUT_DIR / "item_ids.json"

    if emb_path.exists() and ids_path.exists():
        embs = np.load(emb_path)
        with open(ids_path) as f:
            ids = json.load(f)
        print(f"Embeddings already exist: {len(ids)} items, shape {embs.shape}. Skipping.")
        return

    # Load dataset
    print(f"Loading dataset from {DATA_FILE} ...")
    items = []
    with open(DATA_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"  {len(items):,} items loaded")

    # Load model
    print(f"Loading {MODEL_NAME} ...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print(f"  Model on {DEVICE}, dtype float16")

    # Generate embeddings
    print("Generating multimodal embeddings ...")
    all_embeddings: list[np.ndarray] = []
    all_ids: list[str] = []

    for item in tqdm(items, desc="Embedding items"):
        emb = encode_item(item, model, processor)
        if emb is not None:
            all_embeddings.append(emb)
            all_ids.append(item.get("parent_asin", ""))

    # Save
    embeddings = np.concatenate(all_embeddings, axis=0)   # (N, 2048)
    np.save(emb_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(all_ids, f)

    print(f"\nDone.")
    print(f"  Saved {len(all_ids):,} embeddings → {emb_path}  shape={embeddings.shape}")
    print(f"  Item IDs                         → {ids_path}")


if __name__ == "__main__":
    main()
