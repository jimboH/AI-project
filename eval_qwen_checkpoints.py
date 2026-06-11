#!/usr/bin/env python3
"""Evaluate checkpoints from outputs/qwen_v2_semantic_id on the interactions
test or validation split of theblackcat102/amazon-all-beauty-filtered.

Uses trie-constrained beam search: a prefix trie is built from every valid
item semantic ID in the corpus so the model can only generate sequences that
correspond to real items, eliminating invalid level-mixing (e.g. d0→d3→d3).

For each checkpoint-* directory found, the script:
  1. Loads the model and tokenizer directly (no Unsloth needed at inference).
  2. Builds a prefix trie from all item semantic IDs in the items dataset.
  3. Builds the same semantic-ID prompt used during training.
  4. Runs trie-constrained beam search to produce `--top_k` candidate IDs.
  5. Accumulates Hit@k / NDCG@k and prints per-checkpoint results.
  6. Saves all results to JSON.

Usage
-----
    python eval_qwen_checkpoints.py \\
        --checkpoint_dir outputs/qwen_v2_semantic_id \\
        --dataset theblackcat102/amazon-all-beauty-filtered \\
        --split test \\
        --batch_size 4 \\
        --top_k 10 \\
        --output_dir outputs/eval_results

If the root of --checkpoint_dir contains a model (config.json + model.safetensors),
it is evaluated as a "final" entry in addition to sub-checkpoints.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from datasets import Image as HFImage
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessor

# Project root on path so evaluate/ is importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate.metrics import CombinedMetricsAccumulator  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

# Regex for extracting <|d{level}_{code}|> tokens from decoded text
_SEMID_RE = re.compile(r"<\|d(\d+)_(\d+)\|>")

# The Amazon All-Beauty dataset uses 3 hierarchical levels (d0, d1, d2).
# d3 exists in the vocabulary but the 4th code is always 0 and is omitted
# from target_semantic_id strings, so we evaluate on 3 levels.
N_LEVELS = 3

# k values to report — only those <= --top_k are emitted
_ALL_K = [1, 5, 10]


# ── Semantic ID utilities ─────────────────────────────────────────────────────

def parse_semid_codes(text: str, n_levels: int = N_LEVELS) -> list[int] | None:
    """Extract [c0, c1, …, c_{n-1}] from a decoded generation string.

    Returns None when fewer than n_levels distinct level tokens are found.
    First occurrence per level wins (matches generation order).
    """
    found: dict[int, int] = {}
    for level_s, code_s in _SEMID_RE.findall(text):
        level = int(level_s)
        if level < n_levels and level not in found:
            found[level] = int(code_s)
        if len(found) == n_levels:
            break
    if len(found) < n_levels:
        return None
    return [found[lvl] for lvl in range(n_levels)]


# ── Trie-constrained decoding ─────────────────────────────────────────────────

# Token IDs for semantic level tokens (added during training via build_semantic_tokens()):
#   <|d0_i|>  → 248077 + i          (i = 0..255)
#   <|d1_i|>  → 248077 + 256 + i
#   <|d2_i|>  → 248077 + 512 + i
#   <|d3_i|>  → 248077 + 768 + i    ← we STOP before these when building the trie
_D_TOKEN_BASE = 248077
_D3_TOKEN_START = _D_TOKEN_BASE + 3 * 256   # 248845
_D3_TOKEN_END   = _D3_TOKEN_START + 255     # 249100


class TrieNode:
    """Single node in the token-level prefix trie."""
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.is_end: bool = False


def build_semid_trie(tokenizer, item_semids: list[str]) -> TrieNode:
    """Build a prefix trie whose paths are tokenized semantic ID sequences.

    Every path is truncated to exactly 3 semantic levels (d0, d1, d2) so
    the trie always drives the model to produce a 3-token output and force
    EOS after d2.  The items dataset contains ~86 % 3-token entries and
    ~14 % 4-token entries (with a non-zero d3 code); 4-token items are
    included via their (d0, d1, d2) prefix — their d3 tokens are dropped
    before insertion.  Interaction targets are also always evaluated on the
    first 3 codes, so this keeps trie structure and metric space consistent.

    Token-ID layout for the added semantic tokens:
        <|d0_i|>  → _D_TOKEN_BASE + 0*256 + i
        <|d1_i|>  → _D_TOKEN_BASE + 1*256 + i
        <|d2_i|>  → _D_TOKEN_BASE + 2*256 + i
        <|d3_i|>  → _D_TOKEN_BASE + 3*256 + i  ← truncated before insertion
    Spaces (token 220) between level tokens are kept so the trie paths
    match the model's generation output exactly.

    Parameters
    ----------
    tokenizer:
        Inner tokenizer loaded from the checkpoint.
    item_semids:
        List of raw semantic-ID strings from the items dataset.
    """
    root = TrieNode()
    n_inserted = 0
    n_truncated = 0
    for semid in item_semids:
        semid = semid.strip()
        if not semid:
            continue
        token_ids = tokenizer.encode(semid, add_special_tokens=False)
        # Drop any d3 (or higher) tokens.  _D3_TOKEN_START..+255 is the d3 band.
        truncated: list[int] = []
        hit_d3 = False
        for tid in token_ids:
            if _D3_TOKEN_START <= tid <= _D3_TOKEN_END:
                hit_d3 = True
                break               # also stops the trailing space before d3
            truncated.append(tid)
        # Strip a trailing space that was the separator before d3
        while truncated and truncated[-1] == 220:   # 220 = space token
            truncated.pop()
        if not truncated:
            continue
        if hit_d3:
            n_truncated += 1
        node = root
        for tid in truncated:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True
        n_inserted += 1
    print(f"  [Trie] {n_inserted} paths inserted "
          f"({n_truncated} truncated from 4→3 levels), "
          f"root has {len(root.children)} distinct d0 tokens")
    return root


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    """Restricts beam-search token choices to valid continuations in the trie.

    After each generated token the processor walks the trie using all tokens
    generated so far (after the prompt) and sets every out-of-trie token's
    logit to -inf.  When the trie signals end-of-sequence (is_end=True) only
    EOS tokens are allowed.

    Usage pattern
    -------------
        processor = TrieConstrainedLogitsProcessor(root, eos_ids)
        # Before every model.generate() call:
        processor.prompt_len = enc["input_ids"].shape[1]
        outputs = model.generate(..., logits_processor=[processor])

    Cache note
    ----------
    ``self._cache`` maps generated-token tuples to their valid-next frozensets.
    It survives across batches (many samples share the same trie prefixes).
    It is intentionally NOT cleared between checkpoints because the trie is
    checkpoint-independent — only the logits themselves change.
    """

    def __init__(self, trie_root: TrieNode, eos_token_ids: list[int]):
        self.root = trie_root
        self.eos_ids: frozenset[int] = frozenset(eos_token_ids)
        self.prompt_len: int = 0
        # Trie lookup cache: generated_prefix_tuple → frozenset of valid next IDs
        # None means the prefix is off-trie.
        self._cache: dict[tuple[int, ...], frozenset[int] | None] = {}

    def _valid_next(self, generated: tuple[int, ...]) -> frozenset[int] | None:
        """Return the set of valid next token IDs, or None if off-trie."""
        if generated in self._cache:
            return self._cache[generated]

        node = self.root
        for tid in generated:
            child = node.children.get(tid)
            if child is None:
                self._cache[generated] = None
                return None
            node = child

        # Empty frozenset signals "sequence complete → allow EOS"
        result: frozenset[int] = (
            frozenset() if node.is_end else frozenset(node.children)
        )
        self._cache[generated] = result
        return result

    def __call__(
        self,
        input_ids: torch.LongTensor,   # [B * num_beams, seq_len]
        scores: torch.FloatTensor,     # [B * num_beams, vocab_size]
    ) -> torch.FloatTensor:
        V = scores.shape[1]

        for i in range(input_ids.shape[0]):
            generated = tuple(input_ids[i, self.prompt_len:].tolist())
            valid = self._valid_next(generated)

            if valid is None:
                # Off-trie (shouldn't happen; allow EOS as safety valve)
                allowed = self.eos_ids
            elif len(valid) == 0:
                # Trie says the sequence is complete → force EOS
                allowed = self.eos_ids
            else:
                allowed = valid

            # Build new scores: keep original logit for allowed tokens, -inf elsewhere
            new_row = torch.full((V,), float("-inf"), dtype=scores.dtype,
                                 device=scores.device)
            for tid in allowed:
                if tid < V:
                    new_row[tid] = scores[i, tid]
            scores[i] = new_row

        return scores


# ── Image-mode helpers ────────────────────────────────────────────────────────

def load_items_for_image_eval(dataset: str):
    """Load the items dataset and build a semantic_id → row-index map.

    Only items with a non-null image_main are indexed (same filter as training).
    Returns (items_dataset projected to [semantic_id, image_main], semid_to_index).
    """
    print(f"Loading items from {dataset} for image-mode eval …")
    items = load_dataset(dataset, "items", split="train")
    items = items.select_columns(["semantic_id", "image_main"])
    # Build index without decoding images (decode=False → raw bytes)
    items_raw = items.cast_column("image_main", HFImage(decode=False))
    semid_to_index: dict[str, int] = {}
    for idx, row in enumerate(items_raw):
        sid = (row.get("semantic_id") or "").strip()
        if sid and row.get("image_main") is not None:
            semid_to_index[sid] = idx
    print(f"  {len(semid_to_index)} items indexed with image_main")
    return items, semid_to_index


def _build_image_messages(sample: dict, items_dataset, semid_to_index: dict,
                           image_size: int = 224,
                           max_history_items: int | None = None) -> list[dict]:
    """Build a single-sample message list with PIL images in the history.

    Mirrors LazyTask2Dataset.__getitem__ (image mode) from data_utils.py exactly:
      - Iterates history_semantic_ids (most-recent max_history_items kept)
      - For each sid that has an image_main, appends:
          {"type": "text", "text": f"{n}."}
          {"type": "image", "image": <PIL image>}
          {"type": "text", "text": "<|im_end|>\\n"}
      - Ends with {"type": "text", "text": "Predict the next item's semantic ID:"}
    """
    history_sids = sample["history_semantic_ids"]
    if max_history_items is not None:
        history_sids = history_sids[-max_history_items:]

    content: list[dict] = [{"type": "text", "text": "User interaction history:\n"}]
    count = 0
    for sid in history_sids:
        item_idx = semid_to_index.get(sid)
        if item_idx is None:
            continue
        item = items_dataset[item_idx]
        img: Image.Image | None = item.get("image_main")
        if img is None:
            continue
        if image_size > 0:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        count += 1
        content.append({"type": "text", "text": f"{count}."})
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": "<|im_end|>\n"})

    content.append({"type": "text", "text": "Predict the next item's semantic ID:"})
    return [{"role": "user", "content": content}]


def _extract_images(messages: list[dict]) -> list[Image.Image]:
    """Return all PIL images from message content dicts, in order."""
    imgs = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                img = item.get("image")
                if img is not None:
                    imgs.append(img)
    return imgs


def _strip_oldest_history_item(messages: list[dict]) -> bool:
    """Remove the oldest (first) history item from the message content in-place.

    Content layout produced by build_visual_messages:
      [0] header text  "User interaction history:\\n"
      [1..k] oldest item: number-text + image(s) + optional text + "<|im_end|>\\n" text
      [k+1..] remaining items ...
      [-1] footer text "Predict the next item's semantic ID:"

    Returns True if an item was removed, False if the history is already empty.
    """
    content = messages[0]["content"]
    for i in range(1, len(content)):
        block = content[i]
        if block.get("type") == "text" and "<|im_end|>" in block.get("text", ""):
            del content[1 : i + 1]
            return True
    return False


def _encode_single(processor, messages: list[dict], max_length: int) -> dict:
    """Encode one multimodal sample, truncating by dropping oldest history items
    before falling back to hard token-level truncation.

    Dropping whole items avoids cutting mid-image-token-span, which would cause
    the model's RoPE index calculation to see a shape mismatch at generation time.

    Returns a dict of tensors with the batch dimension squeezed out so samples
    can be left-padded and stacked by _pad_and_stack().
    """
    while True:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = _extract_images(messages)
        kwargs: dict = dict(text=text, return_tensors="pt")
        if images:
            kwargs["images"] = images
        enc = processor(**kwargs)

        if enc["input_ids"].shape[1] <= max_length:
            break  # fits within budget

        if not _strip_oldest_history_item(messages):
            break  # no history left to drop; fall through to hard truncation

    # Hard-truncation fallback: only reached when even zero history items still
    # exceed max_length (e.g. a single enormous image or very long footer text).
    if enc["input_ids"].shape[1] > max_length:
        enc["input_ids"] = enc["input_ids"][:, :max_length]
        enc["attention_mask"] = enc["attention_mask"][:, :max_length]
        if "mm_token_type_ids" in enc:
            enc["mm_token_type_ids"] = enc["mm_token_type_ids"][:, :max_length]
        if "image_grid_thw" in enc:
            vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            n_images_kept = int((enc["input_ids"][0] == vision_start_id).sum().item())
            thw = enc["image_grid_thw"]
            if n_images_kept < thw.shape[0]:
                tokens_per_img = (thw[:, 0] * thw[:, 1] * thw[:, 2]).tolist()
                keep_pv_rows = int(sum(tokens_per_img[:n_images_kept]))
                enc["pixel_values"]   = enc["pixel_values"][:keep_pv_rows]
                enc["image_grid_thw"] = thw[:n_images_kept]

    # squeeze the batch dim the processor always adds
    out = {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
    }
    if "pixel_values" in enc:
        out["pixel_values"]   = enc["pixel_values"]        # (n_patches, C*m²)
        out["image_grid_thw"] = enc["image_grid_thw"]      # (n_images, 3)
    if "mm_token_type_ids" in enc:
        out["mm_token_type_ids"] = enc["mm_token_type_ids"].squeeze(0)
    return out


def _pad_and_stack(encoded_list: list[dict], pad_token_id: int, device: str) -> dict:
    """Left-pad a list of individually-encoded samples into a batch.

    image_grid_thw and pixel_values are concatenated along dim-0 (the Qwen3-VL
    model scatters patches into the sequence using the cumulative grid sizes, so
    all per-image tensors must appear in declaration order).

    mm_token_type_ids is left-padded with 0 (the text token-type code) since
    padding positions are not image or video content.
    """
    max_len = max(e["input_ids"].shape[0] for e in encoded_list)

    padded_ids  = []
    padded_mask = []
    padded_mm   = []

    for e in encoded_list:
        seq_len = e["input_ids"].shape[0]
        pad_len = max_len - seq_len
        padded_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                       e["input_ids"]])
        )
        padded_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long),
                       e["attention_mask"]])
        )
        if "mm_token_type_ids" in e:
            padded_mm.append(
                torch.cat([torch.zeros(pad_len, dtype=e["mm_token_type_ids"].dtype),
                           e["mm_token_type_ids"]])
            )

    batch: dict = {
        "input_ids":      torch.stack(padded_ids).to(device),
        "attention_mask": torch.stack(padded_mask).to(device),
    }

    pv_list  = [e["pixel_values"]   for e in encoded_list if "pixel_values"   in e]
    thw_list = [e["image_grid_thw"] for e in encoded_list if "image_grid_thw" in e]
    if pv_list:
        batch["pixel_values"]   = torch.cat(pv_list,  dim=0).to(device)
        batch["image_grid_thw"] = torch.cat(thw_list, dim=0).to(device)

    if padded_mm:
        batch["mm_token_type_ids"] = torch.stack(padded_mm).to(device)

    return batch


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_item_semids(dataset: str) -> list[str]:
    """Return all unique semantic_id strings from the items train split."""
    print(f"Loading items from {dataset} to build trie …")
    items = load_dataset(dataset, "items", split="train")
    semids = [
        row["semantic_id"].strip()
        for row in items.select_columns(["semantic_id"])
        if row["semantic_id"]
    ]
    unique = list(dict.fromkeys(semids))  # deduplicate preserving order
    print(f"  {len(semids)} rows → {len(unique)} unique semantic IDs")
    return unique


def build_semid_to_text(dataset: str) -> dict[str, str]:
    """Return {semantic_id: "Title. Details"} from the items train split.

    Mirrors data_utils.build_semid_to_text() exactly so text-mode eval prompts
    are consistent with the training prompt format in train_qwen_full.py.
    Only text columns are scanned — no image decoding occurs.
    """
    print(f"Loading items from {dataset} to build semantic-ID → text lookup …")
    items = load_dataset(dataset, "items", split="train")
    items_text = items.select_columns(["semantic_id", "title", "details"])
    mapping: dict[str, str] = {}
    for sample in items_text:
        sid     = (sample.get("semantic_id") or "").strip()
        title   = sample.get("title", "") or ""
        details = sample.get("details", "") or ""
        if sid:
            text = title
            if details and details != "{}":
                text = f"{title}. {details}"
            mapping[sid] = text.strip()
    print(f"  {len(mapping)} entries in lookup")
    return mapping


def find_checkpoints(checkpoint_dir: str) -> list[tuple[str, str]]:
    """Return (label, path) pairs for every checkpoint-* sub-directory.

    Sorted by step number ascending.  If the root directory itself holds a
    saved model (config.json + model.safetensors), it is appended as "final".
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    sub_dirs = sorted(
        [
            p for p in glob.glob(pattern)
            if os.path.isdir(p) and p.rsplit("-", 1)[-1].isdigit()
        ],
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    entries = [(Path(p).name, p) for p in sub_dirs]

    root_cfg = os.path.join(checkpoint_dir, "config.json")
    root_model = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(root_cfg) and os.path.isfile(root_model):
        entries.append(("final", checkpoint_dir))

    return entries


def build_prompt_text(sample: dict) -> str:
    """Build the user-turn prompt for one interaction sample.

    Mirrors the semantic_id mode in train_qwen_full.py exactly:
        User interaction history:
        1.<|d0_X|> <|d1_Y|> <|d2_Z|><|im_end|>
        2....
        Predict the next item's semantic ID:
    """
    history_lines = "".join(
        f"{i + 1}.{sid}<|im_end|>\n"
        for i, sid in enumerate(sample["history_semantic_ids"])
    )
    return (
        f"User interaction history:\n{history_lines}"
        "Predict the next item's semantic ID:"
    )


def build_chat_prompts(tokenizer, samples: list[dict]) -> list[str]:
    """Apply the chat template (user turn + generation prompt) to each sample."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt_text(s)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in samples
    ]


def build_prompt_text_history(
    sample: dict,
    semid_to_text: dict[str, str],
    max_history_items: int | None = None,
    max_text_chars_per_item: int = 512,
) -> str:
    """Build the user-turn prompt for one sample using item text (title + details).

    Mirrors the text mode in train_qwen_full.py exactly:
        User interaction history:
        1.<title>. <details><|im_end|>
        2....
        Predict the next item's semantic ID:

    Parameters
    ----------
    sample:
        Interaction dict containing ``history_semantic_ids``.
    semid_to_text:
        Mapping from semantic_id string to "Title. Details" built by
        build_semid_to_text().  Items not found in the map are silently skipped,
        matching training behaviour.
    max_history_items:
        If set, only the most-recent N history items are used (tail of list).
        None = use all (matches training default).
    max_text_chars_per_item:
        Hard clip per item text before insertion into the prompt.
        Training code clips at 256 chars; the default here is 512 so the eval
        flag controls it explicitly.  Pass 256 to reproduce training exactly.
    """
    history_sids = sample["history_semantic_ids"]
    if max_history_items is not None:
        history_sids = history_sids[-max_history_items:]

    history_entries: list[str] = []
    for sid in history_sids:
        text = semid_to_text.get(sid)
        if text:
            history_entries.append(text[:max_text_chars_per_item])

    # Training code further slices to [-8:] after per-item clipping.
    history_lines = "".join(
        f"{i + 1}.{text}<|im_end|>\n"
        for i, text in enumerate(history_entries[-8:])
    )
    return (
        f"User interaction history:\n{history_lines}"
        "Predict the next item's semantic ID:"
    )


def build_chat_prompts_text(
    tokenizer,
    samples: list[dict],
    semid_to_text: dict[str, str],
    max_history_items: int | None = None,
    max_text_chars_per_item: int = 512,
) -> list[str]:
    """Apply the chat template in text mode (item title+details as history).

    Wraps build_prompt_text_history() with apply_chat_template so the output
    is identical to what the model saw during training in text mode.
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt_text_history(
                s, semid_to_text,
                max_history_items=max_history_items,
                max_text_chars_per_item=max_text_chars_per_item,
            )}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in samples
    ]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load model + tokenizer from a checkpoint directory.

    Uses the vision-language conditional-generation auto class because
    train_qwen_full.py saves Qwen3_5ForConditionalGeneration checkpoints, even
    for semantic_id/text-history runs.  AutoModelForCausalLM resolves the same
    config to the text-only Qwen3_5ForCausalLM class, whose state-dict keys do
    not match the saved full vision model.

    No Unsloth is required at inference.
    Tokenizer is configured with left-padding so batched generation aligns
    all sequences at the generation start position.
    """
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    print(f"  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    # Left-pad: ensures all sequences in a batch end at the same position,
    # which is required for correct generation with variable-length prompts.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loading model …")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model ready ({n_params:.0f}M params)")
    return model, tokenizer


# ── Image-mode model loading ──────────────────────────────────────────────────

def load_model_and_processor(checkpoint_path: str, device: str):
    """Load model + AutoProcessor for image-mode evaluation.

    AutoProcessor wraps both the tokenizer and the image processor (Qwen3-VL).
    Left-padding is set on the inner tokenizer for batched generation.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    print("  Loading processor …")
    processor = AutoProcessor.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print("  Loading model …")
    # Must use Qwen3_5ForConditionalGeneration (the VL class), not AutoModelForCausalLM
    # which resolves qwen3_5 to the text-only Qwen3_5ForCausalLM and rejects
    # pixel_values / image_grid_thw / mm_token_type_ids in _validate_model_kwargs.
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model ready ({n_params:.0f}M params)")
    return model, processor


# ── Core evaluation (image mode) ──────────────────────────────────────────────

def evaluate_checkpoint_image(
    checkpoint_path: str,
    samples: list[dict],
    items_dataset,
    semid_to_index: dict,
    trie_processor: "TrieConstrainedLogitsProcessor",
    top_k: int,
    batch_size: int,
    device: str,
    image_size: int = 224,
    max_history_items: int | None = None,
    max_new_tokens: int = 24,
    max_length: int = 2048,
) -> dict:
    """Evaluate one checkpoint with image-history prompts.

    For each sample the user's interaction history is represented as a sequence
    of product thumbnail images (one per history item that has image_main),
    matching the training format used by LazyTask2Dataset in image mode.

    Parameters mirror evaluate_checkpoint(); see that function's docstring.
    """
    model, processor = load_model_and_processor(checkpoint_path, device)
    pad_token_id = processor.tokenizer.pad_token_id

    ks = [k for k in _ALL_K if k <= top_k]
    acc = CombinedMetricsAccumulator(ks=ks)

    # EOS ids from generation config
    gen_cfg_eos = model.generation_config.eos_token_id
    if isinstance(gen_cfg_eos, int):
        eos_ids = [gen_cfg_eos]
    else:
        eos_ids = list(gen_cfg_eos) if gen_cfg_eos is not None else []
    if processor.tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(processor.tokenizer.eos_token_id)

    # Ground-truth codes
    gt_codes: list[list[int]] = []
    for s in samples:
        codes = parse_semid_codes(s["target_semantic_id"])
        if codes is None:
            raw = list(s["target_semantic_codes"])[:N_LEVELS]
            codes = raw if len(raw) == N_LEVELS else [0] * N_LEVELS
        gt_codes.append(codes)

    n = len(samples)
    for start in tqdm(range(0, n, batch_size), desc="  batches", leave=False):
        end = min(start + batch_size, n)
        batch_samples = samples[start:end]
        batch_gt = gt_codes[start:end]
        B = end - start

        # Build one multimodal message list per sample
        batch_messages = [
            _build_image_messages(s, items_dataset, semid_to_index,
                                  image_size=image_size,
                                  max_history_items=max_history_items)
            for s in batch_samples
        ]

        # Encode each sample independently (different image counts → varying lengths)
        encoded_list = [
            _encode_single(processor, msgs, max_length=max_length)
            for msgs in batch_messages
        ]

        # Left-pad and stack into a batch tensor
        enc = _pad_and_stack(encoded_list, pad_token_id=pad_token_id, device=device)

        prompt_len = enc["input_ids"].shape[1]
        trie_processor.prompt_len = prompt_len

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=top_k,
                num_return_sequences=top_k,
                do_sample=False,
                early_stopping=True,
                pad_token_id=pad_token_id,
                logits_processor=[trie_processor],
            )

        new_tokens = out[:, prompt_len:]
        decoded_all = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)

        preds: list[list[list[int]]] = []
        for b in range(B):
            beam_preds: list[list[int]] = []
            for k_idx in range(top_k):
                text = decoded_all[b * top_k + k_idx]
                codes = parse_semid_codes(text)
                if codes is None:
                    codes = [-1] * N_LEVELS
                beam_preds.append(codes)
            preds.append(beam_preds)

        actual  = torch.tensor(batch_gt, dtype=torch.long)
        top_k_t = torch.tensor(preds,    dtype=torch.long)
        acc.accumulate(actual=actual, top_k=top_k_t)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return zero_fill_metrics(acc.reduce(), ks)


# ── Metric helpers ────────────────────────────────────────────────────────────

def zero_fill_metrics(metrics: dict, ks: list[int]) -> dict:
    """Ensure all expected h@k and ndcg@k keys are present (fill missing with 0).

    NDCGAccumulator.reduce() omits ndcg@k keys when there are zero hits
    (it only iterates self.metrics.items() which is empty). This helper
    guarantees a complete dict for display and JSON serialisation.
    """
    for k in ks:
        metrics.setdefault(f"h@{k}", 0.0)
        metrics.setdefault(f"ndcg@{k}", 0.0)
    return metrics


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    samples: list[dict],
    trie_processor: TrieConstrainedLogitsProcessor,
    top_k: int,
    batch_size: int,
    device: str,
    max_new_tokens: int = 24,
    chat_prompts: list[str] | None = None,
) -> dict:
    """Evaluate one checkpoint and return a metrics dict.

    Parameters
    ----------
    checkpoint_path:
        HuggingFace model directory.
    samples:
        Interaction dicts from the HF dataset split.
    trie_processor:
        Pre-built TrieConstrainedLogitsProcessor (shared across checkpoints).
        Its prompt_len is updated per batch before each generate() call.
    top_k:
        Number of beam-search candidates to generate per sample.
    batch_size:
        Samples per GPU batch.
    device:
        Torch device string ("cuda" / "cpu").
    max_new_tokens:
        Token budget per beam.  A 3-level semantic ID with spaces is exactly
        5 tokens; 24 gives plenty of headroom for EOS and edge cases.
    chat_prompts:
        Optional pre-built list of chat-templated prompt strings (one per
        sample).  When supplied the function skips calling build_chat_prompts()
        — useful for text mode where prompts are built from item text rather
        than semantic ID tokens.  When None (default), prompts are built
        internally using build_chat_prompts() (semantic_id mode).
    """
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)

    ks = [k for k in _ALL_K if k <= top_k]
    acc = CombinedMetricsAccumulator(ks=ks)

    # Resolve EOS ids from generation config (may be int or list)
    gen_cfg_eos = model.generation_config.eos_token_id
    if isinstance(gen_cfg_eos, int):
        eos_ids = [gen_cfg_eos]
    else:
        eos_ids = list(gen_cfg_eos) if gen_cfg_eos is not None else []
    if tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)

    # Pre-build all chat prompts and ground-truth code tensors.
    # When pre-built prompts are supplied (text mode), use them directly;
    # otherwise fall back to the default semantic_id prompt builder.
    if chat_prompts is None:
        chat_prompts = build_chat_prompts(tokenizer, samples)
    gt_codes: list[list[int]] = []
    for s in samples:
        codes = parse_semid_codes(s["target_semantic_id"])
        if codes is None:
            raw = list(s["target_semantic_codes"])[:N_LEVELS]
            codes = raw if len(raw) == N_LEVELS else [0] * N_LEVELS
        gt_codes.append(codes)

    n = len(samples)
    for start in tqdm(range(0, n, batch_size), desc="  batches", leave=False):
        end = min(start + batch_size, n)
        batch_prompts = chat_prompts[start:end]
        batch_gt = gt_codes[start:end]
        B = end - start

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        prompt_len = enc["input_ids"].shape[1]
        # Tell the trie processor where generation starts for this batch.
        # With left-padding every row has the same total length, so a single
        # prompt_len applies to all rows in this batch.
        trie_processor.prompt_len = prompt_len

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=top_k,
                num_return_sequences=top_k,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=[trie_processor],
            )

        # out: (B * top_k, prompt_len + new_tokens)
        new_tokens = out[:, prompt_len:]
        decoded_all = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)

        # Build prediction tensor [B, K, N_LEVELS]
        preds: list[list[list[int]]] = []
        for b in range(B):
            beam_preds: list[list[int]] = []
            for k_idx in range(top_k):
                text = decoded_all[b * top_k + k_idx]
                codes = parse_semid_codes(text)
                # -1 as sentinel: can never equal a valid code (0–255)
                if codes is None:
                    codes = [-1] * N_LEVELS
                beam_preds.append(codes)
            preds.append(beam_preds)

        actual = torch.tensor(batch_gt, dtype=torch.long)    # [B, D]
        top_k_t = torch.tensor(preds, dtype=torch.long)      # [B, K, D]
        acc.accumulate(actual=actual, top_k=top_k_t)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return zero_fill_metrics(acc.reduce(), ks)


# ── Result display ─────────────────────────────────────────────────────────────

def print_results_table(all_results: dict[str, dict], ks: list[int]) -> None:
    """Print a human-readable metrics table."""
    if not all_results:
        print("No results to display.")
        return

    # Ordered metric columns: h@1 h@5 h@10 ndcg@1 ndcg@5 ndcg@10
    metric_keys = (
        [f"h@{k}" for k in ks] +
        [f"ndcg@{k}" for k in ks]
    )

    label_w = max(len(label) for label in all_results) + 2
    col_w = 12
    sep_w = label_w + col_w * len(metric_keys)

    header = f"{'checkpoint':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in metric_keys)
    print("\n" + "=" * sep_w)
    print(header)
    print("-" * sep_w)
    for label, metrics in all_results.items():
        row = f"{label:<{label_w}}"
        for m in metric_keys:
            val = metrics.get(m, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * sep_w)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen checkpoints (trie-constrained) on interactions test/valid split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/qwen_v2_semantic_id",
        help="Directory containing checkpoint-* sub-dirs (and optionally a "
             "root-level final model).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="theblackcat102/amazon-all-beauty-filtered",
        help="HuggingFace dataset repo ID.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "valid"],
        help="Interaction split to evaluate on.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Samples per GPU batch.  Reduce if OOM.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Beam width (= num_beams = num_return_sequences).  "
             "Hit@k and NDCG@k are reported for k in {1,5,10} that are <= this.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=24,
        help="Token budget per beam.  A 3-level semantic ID is 5 tokens; "
             "24 gives a generous safety margin.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval_results",
        help="Directory where JSON results are written.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap evaluation samples (None = full split).  Useful for smoke-tests.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate only this specific checkpoint path instead of discovering "
             "all checkpoints under --checkpoint_dir.",
    )
    p.add_argument(
        "--historical_inputs",
        type=str,
        default="semantic_id",
        choices=["semantic_id", "text", "image"],
        help="Prompt format for user history: "
             "'semantic_id' (raw semantic token IDs, default), "
             "'text' (product title + details looked up from the items dataset, "
             "matching the text-mode training run), or "
             "'image' (product thumbnail images, matching the qwen_v2_image_input "
             "training run).  Image mode requires AutoProcessor and loads item images "
             "from the items dataset.",
    )
    p.add_argument(
        "--max_text_chars_per_item",
        type=int,
        default=512,
        help="Maximum characters per history item in text mode.  Each item's "
             "'Title. Details' string is hard-clipped to this length before being "
             "inserted into the prompt.  Training used 256; the eval default is 512 "
             "so the flag can be set explicitly to reproduce training exactly.  "
             "Only used when --historical_inputs text.",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Resize history images to this square size before encoding.  "
             "Pass 0 to skip pre-resize (native resolution, more tokens, slower).  "
             "Only used when --historical_inputs image.",
    )
    p.add_argument(
        "--max_history_items",
        type=int,
        default=None,
        help="Cap on the number of history items included in the prompt.  "
             "None = use all (not recommended for image mode).  "
             "Strongly recommended to set 3–5 for image mode to stay within max_length.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset split ─────────────────────────────────────────────────────
    print(f"\nLoading interactions/{args.split} from {args.dataset} …")
    hf_dataset = load_dataset(args.dataset, "interactions", split=args.split)
    samples = list(hf_dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Evaluation samples: {len(samples)}")

    # ── Build trie (once, shared across all checkpoints) ──────────────────────
    # The trie is tokenizer-dependent, but the special token IDs are the same
    # across all checkpoints (they were added consistently during training).
    # We load the tokenizer from the first checkpoint to tokenise the semids.
    if args.checkpoint:
        first_ckpt_path = os.path.abspath(args.checkpoint)
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print(
                f"No checkpoints found in '{args.checkpoint_dir}'. "
                "Pass --checkpoint <path> to specify one explicitly."
            )
            sys.exit(1)
        first_ckpt_path = checkpoints[0][1]

    print(f"\nBuilding corpus trie from {args.dataset}/items …")
    from transformers import AutoTokenizer
    _tok_for_trie = AutoTokenizer.from_pretrained(first_ckpt_path, trust_remote_code=True)
    item_semids = load_item_semids(args.dataset)
    trie_root = build_semid_trie(_tok_for_trie, item_semids)
    del _tok_for_trie  # free — each checkpoint loads its own tokenizer

    # Resolve EOS ids using the generation_config of the first checkpoint
    import json as _json
    gen_cfg_path = os.path.join(first_ckpt_path, "generation_config.json")
    if os.path.isfile(gen_cfg_path):
        with open(gen_cfg_path) as fh:
            gen_cfg = _json.load(fh)
        raw_eos = gen_cfg.get("eos_token_id", [])
    else:
        raw_eos = []
    eos_ids = list(raw_eos) if isinstance(raw_eos, list) else [raw_eos]
    print(f"  EOS token IDs: {eos_ids}")

    trie_processor = TrieConstrainedLogitsProcessor(
        trie_root=trie_root,
        eos_token_ids=eos_ids,
    )

    # ── Discover checkpoints ────────────────────────────────────────────────────
    if args.checkpoint:
        checkpoints = [(Path(args.checkpoint).name, os.path.abspath(args.checkpoint))]
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir)

    print(f"\nCheckpoints to evaluate ({len(checkpoints)}):")
    for label, path in checkpoints:
        print(f"  {label}: {path}")

    ks = [k for k in _ALL_K if k <= args.top_k]

    # ── Load items dataset for image / text modes (once, shared across checkpoints) ──
    items_dataset = None
    semid_to_index: dict = {}
    semid_to_text: dict = {}
    if args.historical_inputs == "image":
        items_dataset, semid_to_index = load_items_for_image_eval(args.dataset)
    elif args.historical_inputs == "text":
        semid_to_text = build_semid_to_text(args.dataset)

    # ── Pre-build text-mode chat prompts (once, tokenizer-independent strings) ─
    # Text-mode prompts depend only on the items text lookup and the sample
    # history — not on the tokenizer — so they can be built once here and
    # reused across every checkpoint.  The tokenizer is applied inside
    # build_chat_prompts_text() via apply_chat_template which IS
    # tokenizer-specific, so we defer that call into evaluate_checkpoint via
    # the chat_prompts parameter and build them per-checkpoint there.
    # However, because apply_chat_template output is deterministic given the
    # same template (all checkpoints share the same chat template from the
    # same base model), we pre-build once using the first checkpoint's
    # tokenizer and pass the result to all evaluate_checkpoint calls.
    prebuilt_chat_prompts: list[str] | None = None
    if args.historical_inputs == "text":
        print(f"\nPre-building text-mode chat prompts from {args.dataset}/items …")
        from transformers import AutoTokenizer as _AutoTok
        _tok_for_prompts = _AutoTok.from_pretrained(first_ckpt_path, trust_remote_code=True)
        prebuilt_chat_prompts = build_chat_prompts_text(
            tokenizer=_tok_for_prompts,
            samples=samples,
            semid_to_text=semid_to_text,
            max_history_items=args.max_history_items,
            max_text_chars_per_item=args.max_text_chars_per_item,
        )
        del _tok_for_prompts
        print(f"  {len(prebuilt_chat_prompts)} prompts built")

    # ── Evaluate each checkpoint ───────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    for label, ckpt_path in checkpoints:
        print(f"\n{'─' * 60}")
        print(f"Checkpoint: {label}")
        if args.historical_inputs == "image":
            metrics = evaluate_checkpoint_image(
                checkpoint_path=ckpt_path,
                samples=samples,
                items_dataset=items_dataset,
                semid_to_index=semid_to_index,
                trie_processor=trie_processor,
                top_k=args.top_k,
                batch_size=args.batch_size,
                device=device,
                image_size=args.image_size,
                max_history_items=args.max_history_items,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            # Covers both "semantic_id" (chat_prompts=None → built internally)
            # and "text" (prebuilt_chat_prompts passed in).
            metrics = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                samples=samples,
                trie_processor=trie_processor,
                top_k=args.top_k,
                batch_size=args.batch_size,
                device=device,
                max_new_tokens=args.max_new_tokens,
                chat_prompts=prebuilt_chat_prompts,
            )
        all_results[label] = metrics
        print(f"  {metrics}")

    # ── Display and save results ───────────────────────────────────────────────
    print_results_table(all_results, ks)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir_name = (
        Path(args.checkpoint).name if args.checkpoint
        else Path(args.checkpoint_dir).name
    )
    out_path = os.path.join(
        args.output_dir,
        f"qwen_{args.split}_{ckpt_dir_name}.json",
    )
    with open(out_path, "w") as fh:
        json.dump(
            {
                "checkpoint_dir": args.checkpoint_dir,
                "dataset": args.dataset,
                "split": args.split,
                "top_k": args.top_k,
                "n_samples": len(samples),
                "trie_size": len(item_semids),
                "results": all_results,
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
