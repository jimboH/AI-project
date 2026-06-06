"""Dataset & collation machinery for full-weight Qwen finetuning.

This module is deliberately Unsloth-free.  Every class/function here is pickled
into DataLoader worker processes when ``num_workers > 0`` with the ``spawn``
start-method.  Under ``spawn`` each worker re-imports the *defining module* of
whatever object it unpickles — so if these classes lived in ``train_qwen_full``
(which does ``from unsloth import FastVisionModel`` at module top level), every
worker would re-run the full Unsloth + Torch patching machinery at startup.

Keeping them in this lean module — importing only ``torch``, ``PIL`` and
``datasets`` — means a spawned worker imports those three packages instead of
Unsloth, cutting per-worker startup from tens of seconds to ~1s.  ``wandb`` is
imported lazily inside ``TimedCollator`` so workers never pull it in either.
"""

import time as _time

import torch
from datasets import Image as HFImage
from PIL import Image

# Per-worker processor cache.  Under ``spawn`` each worker is a fresh process
# so the cache starts empty; the first ``__getitem__`` call loads the processor
# from disk and stores it here for the lifetime of the worker.
_WORKER_PROCESSOR_CACHE: dict = {}


def compute_max_images(max_length: int, image_size: int) -> int:
    """Derive a safe upper bound on images per sample to avoid mid-image truncation.

    Qwen3.5/Qwen3-VL uses patch_size=16, merge_size=2, smart_resize factor=32.
    For a square image of side S the token count is:
        H_bar = round(S / 32) * 32
        tokens = (H_bar // 16) ** 2 // 4  =  H_bar^2 / 1024

    At native resolution (image_size=0) product images are typically 800–1600 px
    on a side, yielding 625–2500 tokens each (observed: ~2200 average in practice).
    We use 2500 as a worst-case estimate so the guard is tight enough to actually
    prevent overflow.  We reserve 35 % of the budget for text + special tokens.
    """
    if image_size > 0:
        factor = 32  # patch_size=16 * merge_size=2
        h_bar = round(image_size / factor) * factor
        tokens_per_image = max(1, (h_bar // 16) ** 2 // 4)
    else:
        tokens_per_image = 2500  # worst-case for native-resolution product images
    visual_budget = int(max_length * 0.65)
    return max(1, visual_budget // tokens_per_image)


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """Resize a PIL image to (size x size).

    Pass size=0 to skip resizing entirely and let the vision processor handle it.

    Filter choice: BILINEAR is used instead of LANCZOS.  Benchmarks show LANCZOS
    costs ~2.4ms/image vs ~1.0ms for BILINEAR — slower than the decode step
    itself.  At bf16 precision the sub-pixel quality difference is indistinguishable
    to the model and irrelevant for training convergence.
    """
    if size <= 0:
        return img
    return img.resize((size, size), Image.BILINEAR)


# ── Parallel-collation helpers ─────────────────────────────────────────────────

def _clean_none_keys_messages(messages: list) -> list:
    """Remove None-valued keys from message content items.

    HuggingFace Arrow serialisation sometimes adds keys whose value is None
    (e.g. ``{"type": "image", "image": ..., "video": None}``).
    UnslothVisionDataCollator strips these in the main thread; we replicate
    the same cleanup here so workers produce identical dicts.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    for k in [k for k, v in list(item.items()) if v is None]:
                        del item[k]
    return messages


def _extract_images_from_messages(messages: list) -> list:
    """Return PIL images found in message content, in declaration order."""
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    if img is not None:
                        images.append(img)
    return images


class PreTokenizingDataset(torch.utils.data.Dataset):
    """Wraps any message-format dataset and tokenises each sample inside the worker.

    With PyTorch's default ``fork`` start-method on Linux the processor is
    inherited by workers at zero copy cost — no pickling needed.  Each worker
    calls ``apply_chat_template`` and ``processor(...)`` for its own samples in
    parallel, so the main thread's collate_fn only pads and stacks pre-built
    tensors instead of doing the full tokenisation serially.

    Buffering comparison
    --------------------
    Old path (UnslothVisionDataCollator in main):
        workers × prefetch_factor × batch_size samples sit in RAM as raw PIL
        dicts before collation.  At batch_size=32 and 16 workers this is
        1 024 partially-decoded samples → OOM / startup hang.

    New path (PreTokenizingDataset + parallel_collate_fn):
        workers × prefetch_factor × 1 sample worth of *tensors* sit in the
        prefetch queue.  Tensors are compact; PIL images are freed immediately
        after encoding.  4 workers × 2 prefetch = 8 pre-tokenised samples in
        RAM regardless of batch_size.

    Pickling note
    -------------
    Unsloth monkey-patches ``Qwen3VLProcessor`` at the class level, making the
    live processor object unpicklable under the ``spawn`` multiprocessing context
    (Python's pickle verifies ``type(obj)`` is the same object as the class
    retrieved via its qualified name from ``sys.modules`` — the patched class
    fails this check).

    To work around this, the live processor is **not** stored in pickle state.
    Instead, ``processor_path`` (a directory written by
    ``tokenizer.save_pretrained()``) is pickled.  Spawned workers load the
    processor from disk on their first ``__getitem__`` call and cache it in the
    module-level ``_WORKER_PROCESSOR_CACHE`` dict for the lifetime of that
    worker process.
    """

    def __init__(
        self,
        dataset,
        processor,
        max_seq_length: int | None = None,
        processor_path: str | None = None,
    ):
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.processor_path = processor_path
        # _processor is used directly in the main process.  Spawned workers
        # receive None here (via __getstate__) and reload from processor_path.
        self._processor = processor

    # ── Pickling support for spawn workers ────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Drop the live (Unsloth-patched) processor — it cannot be pickled.
        # Workers will reload it from processor_path via the lazy property below.
        state["_processor"] = None
        return state

    # __setstate__ is intentionally omitted; the default dict-update suffices.

    # ── Lazy processor accessor ────────────────────────────────────────────────

    @property
    def processor(self):
        """Return the processor, loading it from disk on first worker access."""
        if self._processor is None:
            if self.processor_path is None:
                raise RuntimeError(
                    "PreTokenizingDataset: processor_path must be set when "
                    "num_workers > 0 so spawned workers can reload the processor "
                    "from disk (the live processor cannot be pickled after Unsloth "
                    "patches it).  Pass processor_path=<dir> where the processor "
                    "has been saved with tokenizer.save_pretrained(<dir>)."
                )
            global _WORKER_PROCESSOR_CACHE
            if self.processor_path not in _WORKER_PROCESSOR_CACHE:
                from transformers import AutoProcessor
                _WORKER_PROCESSOR_CACHE[self.processor_path] = (
                    AutoProcessor.from_pretrained(self.processor_path)
                )
            self._processor = _WORKER_PROCESSOR_CACHE[self.processor_path]
        return self._processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]
        messages = example.get("messages", example)
        messages = _clean_none_keys_messages(messages)
        images = _extract_images_from_messages(messages)

        proc = self.processor  # triggers lazy load in spawned workers

        text = proc.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        kwargs: dict = dict(
            text=text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=self.max_seq_length is not None,
            max_length=self.max_seq_length,
        )
        if images:
            kwargs["images"] = images

        encoded = proc(**kwargs)

        out: dict = {
            # squeeze the batch dim the processor always adds for single samples
            "input_ids":      encoded["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (seq_len,)
        }
        if "pixel_values" in encoded:
            # pixel_values: (total_patches_for_this_sample, C*merge²)
            # image_grid_thw: (n_images_for_this_sample, 3)
            out["pixel_values"]   = encoded["pixel_values"]
            out["image_grid_thw"] = encoded["image_grid_thw"]
        # mm_token_type_ids is required by Qwen3.5 for multimodal M-RoPE.
        # The processor always returns it when multimodal inputs are present.
        if "mm_token_type_ids" in encoded:
            out["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)
        return out


class ParallelCollator:
    """Lightweight, picklable collate_fn for use with PreTokenizingDataset.

    Implemented as a top-level class (not a closure) so it can be pickled into
    spawned DataLoader workers.  A nested function returned from a factory is a
    *local* object that ``pickle`` cannot serialise under the ``spawn`` start
    method — instances of a module-level class pickle by reference and work
    fine.

    Workers have already done apply_chat_template + processor encoding.
    This collator only needs to:
      1. Pad ``input_ids`` / ``attention_mask`` to the batch-maximum length.
      2. Concatenate ``pixel_values`` and ``image_grid_thw`` across samples.
      3. Build ``labels`` by masking padding positions with ``ignore_index``.

    Parameters
    ----------
    pad_token_id:
        Token ID used to fill right-padded positions in ``input_ids``.
    ignore_index:
        Value written into ``labels`` at padding positions (default -100,
        matching PyTorch cross-entropy's default ignore index).
    pad_to_multiple_of:
        Round the padded sequence length up to a multiple of this value.
        Pass 8 to align to Tensor Core boundaries for bf16 training.
    pixel_dtype:
        Cast ``pixel_values`` to this dtype after concatenation (e.g.
        ``torch.bfloat16`` to match the model's parameter dtype).
    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int = -100,
        pad_to_multiple_of: int | None = None,
        pixel_dtype=None,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pixel_dtype = pixel_dtype

    def __call__(self, batch: list[dict]) -> dict:
        pad_token_id = self.pad_token_id
        ignore_index = self.ignore_index
        pad_to_multiple_of = self.pad_to_multiple_of
        pixel_dtype = self.pixel_dtype

        seq_lens = [x["input_ids"].size(0) for x in batch]
        max_len = max(seq_lens)
        if pad_to_multiple_of:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        B = len(batch)
        input_ids      = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)

        for i, (sample, L) in enumerate(zip(batch, seq_lens)):
            input_ids[i, :L]      = sample["input_ids"]
            attention_mask[i, :L] = sample["attention_mask"]

        # Labels: identical to input_ids except padding positions → ignore_index.
        # Image placeholder tokens (e.g. <|image_pad|>) are real content in the
        # sequence and intentionally included in the loss, consistent with the
        # default UnslothVisionDataCollator behaviour (train_on_responses_only=False).
        labels = input_ids.clone()
        labels[attention_mask == 0] = ignore_index

        result: dict = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

        # Concatenate visual patches from samples that carry images.
        # Samples without images have no <|image_pad|> tokens in their input_ids
        # so the model ignores any pixel_values that aren't addressed by those tokens.
        pv_list  = [s["pixel_values"]   for s in batch if "pixel_values"   in s]
        thw_list = [s["image_grid_thw"] for s in batch if "image_grid_thw" in s]
        if pv_list:
            pv = torch.cat(pv_list, dim=0)
            result["pixel_values"]   = pv.to(pixel_dtype) if pixel_dtype is not None else pv
            result["image_grid_thw"] = torch.cat(thw_list, dim=0)

        # mm_token_type_ids: required by Qwen3.5 for multimodal M-RoPE.
        # Pad with 0 (text token type) at positions beyond each sample's length.
        if any("mm_token_type_ids" in s for s in batch):
            mm_type_ids = torch.zeros(B, max_len, dtype=torch.long)
            for i, (sample, L) in enumerate(zip(batch, seq_lens)):
                if "mm_token_type_ids" in sample:
                    mm_type_ids[i, :L] = sample["mm_token_type_ids"]
            result["mm_token_type_ids"] = mm_type_ids

        return result


def make_parallel_collate_fn(
    pad_token_id: int,
    ignore_index: int = -100,
    pad_to_multiple_of: int | None = None,
    pixel_dtype=None,
):
    """Return a picklable ParallelCollator for use with PreTokenizingDataset.

    Kept as a factory for backward compatibility; returns a top-level
    ``ParallelCollator`` instance (picklable under the ``spawn`` start method,
    unlike a nested closure).  See ``ParallelCollator`` for parameter docs.
    """
    return ParallelCollator(
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        pad_to_multiple_of=pad_to_multiple_of,
        pixel_dtype=pixel_dtype,
    )


class LazyTask1Dataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace items dataset and decodes/resizes images on demand.

    Images are never stored in RAM en masse — each sample is decoded only when
    the DataLoader requests it, keeping peak memory proportional to batch size
    rather than dataset size.
    """

    def __init__(self, hf_dataset, image_size: int = 0):
        self.dataset = hf_dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        content = []
        # Decode and resize images only at access time.
        for img in [sample["image_main"], sample["image_pt01"], sample["image_pt02"]]:
            if img is not None:
                content.append({"type": "image", "image": resize_image(img, self.image_size)})
        content.append({
            "type": "text",
            "text": (
                f"Product title: {sample['title']}\n"
                f"Details: {sample['details']}\n"
                "Predict the semantic ID of this product:"
            ),
        })
        return {"messages": [
            {"role": "user",      "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": sample["semantic_id"]}]},
        ]}


class ListDataset(torch.utils.data.Dataset):
    """Thin torch Dataset wrapper around a plain Python list."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class LazyTask2Dataset(torch.utils.data.Dataset):
    """Task 2 dataset with lazy image loading for image/multimodal history modes.

    Each __getitem__ call decodes only the images needed for that one sample,
    keeping peak RAM proportional to batch size rather than dataset size.
    """

    def __init__(
        self,
        interactions_hf_dataset,
        items_hf_dataset,
        semid_to_index: dict,
        semid_to_text,          # None when mode == "image"
        mode: str,              # "image" | "multimodal"
        image_size: int = 0,
        max_history_items=None, # cap: keep only the most recent N items
        max_images=None,        # hard cap on total images; prevents mid-image truncation
    ):
        self.interactions      = interactions_hf_dataset
        self.items             = items_hf_dataset
        self.semid_to_idx      = semid_to_index
        self.semid_to_text     = semid_to_text  # only used by "multimodal"
        self.mode              = mode
        self.image_size        = image_size
        self.max_history_items = max_history_items
        self.max_images        = max_images      # None means no cap
        # Per-worker in-process cache for popular item rows.
        # Survives across batches when persistent_workers=True.
        self._img_cache: dict[int, dict] = {}
        self._img_cache_maxsize = 512

    def _fetch_item(self, item_idx: int) -> dict:
        """Return item row from cache or HF Arrow, evicting oldest when full."""
        if item_idx not in self._img_cache:
            if len(self._img_cache) >= self._img_cache_maxsize:
                self._img_cache.pop(next(iter(self._img_cache)))
            self._img_cache[item_idx] = self.items[item_idx]
        return self._img_cache[item_idx]

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> dict:
        interaction = self.interactions[idx]
        history_sids = interaction["history_semantic_ids"]

        # Keep only the most recent N items when capped
        if self.max_history_items is not None:
            history_sids = history_sids[-self.max_history_items:]

        content = [{"type": "text", "text": "User interaction history:\n"}]
        count = 0
        image_count = 0
        for sid in history_sids:
            # Stop adding history items once the image budget is exhausted so
            # the sequence never needs to be truncated mid-image-token sequence.
            # Truncating inside an image's visual tokens causes:
            #   ValueError: Mismatch in `image` token count between text and input_ids
            if self.max_images is not None and image_count >= self.max_images:
                break

            item_idx = self.semid_to_idx.get(sid)
            if item_idx is None:
                continue  # skip history entries not found in the items dataset

            item = self._fetch_item(item_idx)

            # ── count images this item would add ──────────────────────────────
            if self.mode == "image":
                incoming = 1 if item.get("image_main") is not None else 0
            else:  # multimodal
                incoming = sum(
                    1 for k in ("image_main", "image_pt01", "image_pt02")
                    if item.get(k) is not None
                )

            # Skip this history item if it would exceed the image budget
            if self.max_images is not None and image_count + incoming > self.max_images:
                continue

            # ── number label ──────────────────────────────────────────────────
            count += 1
            content.append({"type": "text", "text": f"{count}."})

            # ── images ────────────────────────────────────────────────────────
            # image mode: only image_main thumbnail (per plan spec)
            # multimodal mode: all available views (image_main + pt01 + pt02)
            if self.mode == "image":
                img = item.get("image_main")
                if img is not None:
                    content.append({
                        "type": "image",
                        "image": resize_image(img, self.image_size),
                    })
                    image_count += 1
            else:  # multimodal
                for key in ("image_main", "image_pt01", "image_pt02"):
                    img = item.get(key)
                    if img is not None:
                        content.append({
                            "type": "image",
                            "image": resize_image(img, self.image_size),
                        })
                        image_count += 1

            # ── text (multimodal only) ─────────────────────────────────────
            if self.mode == "multimodal" and self.semid_to_text is not None:
                text = self.semid_to_text.get(sid, "")
                if text:
                    content.append({"type": "text", "text": f" {text}"})

            # ── item terminator ───────────────────────────────────────────────
            content.append({"type": "text", "text": "<|im_end|>\n"})

        content.append({
            "type": "text",
            "text": "Predict the next item's semantic ID:",
        })

        return {"messages": [
            {"role": "user",      "content": content},
            {"role": "assistant", "content": [
                {"type": "text", "text": interaction["target_semantic_id"]}
            ]},
        ]}


def build_semid_to_index(items_hf_dataset) -> dict:
    """Return {semantic_id_string: row_index} for items that have image_main.

    Items missing image_main (~0.2% + all text_only items) are excluded so the
    index only points to rows that can actually supply a thumbnail at batch time.

    Iterates a binary-cast view of image_main to check presence without
    triggering PIL decode — 10–50× faster than iterating the Image-typed column.
    """
    # Cast image_main to raw bytes for the scan so HF skips PIL decoding.
    items_raw = items_hf_dataset.select_columns(["semantic_id", "image_main"]) \
                                .cast_column("image_main", HFImage(decode=False))
    mapping = {}
    for idx, sample in enumerate(items_raw):
        sid = (sample.get("semantic_id") or "").strip()
        if sid and sample.get("image_main") is not None:
            mapping[sid] = idx
    return mapping


def build_semid_to_text(items_hf_dataset) -> dict:
    """Return {semantic_id_string: "Title. Details"} from the items dataset.

    Projects down to only the three text columns before iterating so HF never
    decodes any image column during this scan.
    """
    items_text = items_hf_dataset.select_columns(["semantic_id", "title", "details"])
    mapping = {}
    for sample in items_text:
        sid     = (sample.get("semantic_id") or "").strip()
        title   = sample.get("title", "") or ""
        details = sample.get("details", "") or ""
        if sid:
            text = title
            if details and details != "{}":
                text = f"{title}. {details}"
            mapping[sid] = text.strip()
    return mapping


def build_semid_to_index_and_text(items_hf_dataset) -> tuple[dict, dict]:
    """Merged single-pass variant of build_semid_to_index + build_semid_to_text.

    For multimodal mode both maps are needed.  Running two separate scans over
    112K rows doubles startup I/O.  This function builds both in one pass using
    only text columns + image_main presence (decode=False → no PIL overhead).
    Benchmark: 1.57× faster than calling the two functions separately (10s vs 16s at 112K).
    """
    items_raw = (items_hf_dataset
                 .select_columns(["semantic_id", "image_main", "title", "details"])
                 .cast_column("image_main", HFImage(decode=False)))
    idx_map: dict = {}
    txt_map: dict = {}
    for idx, sample in enumerate(items_raw):
        sid = (sample.get("semantic_id") or "").strip()
        if not sid:
            continue
        if sample.get("image_main") is not None:
            idx_map[sid] = idx
        title   = sample.get("title", "") or ""
        details = sample.get("details", "") or ""
        text = title
        if details and details != "{}":
            text = f"{title}. {details}"
        txt_map[sid] = text.strip()
    return idx_map, txt_map


class TimedCollator:
    """Wraps a collate_fn and logs per-batch collation latency to W&B.

    Design note
    -----------
    PyTorch's DataLoader calls ``collate_fn`` **inside worker processes** (via
    ``_MapDatasetFetcher.fetch``), not in the main process.  W&B is only
    initialised in the main process, so ``wandb.log()`` must be guarded with a
    ``wandb.run is not None`` check.  Without this guard, the first log attempt
    in a worker raises ``wandb.errors.Error: You must call wandb.init()`` which
    kills the worker and silently hangs the DataLoader prefetch queue.

    ``wandb`` is imported lazily so workers don't pay the import cost on every
    spawn (workers re-import ``__main__`` = ``train_qwen_full`` via Python's
    spawn bootstrap, but they never reach ``wandb.log()`` because the run check
    short-circuits first).
    """

    LOG_EVERY = 5  # log a W&B point every N collation calls

    def __init__(self, collator):
        self._collator = collator
        self._n = 0
        self._window_s: list[float] = []

    def __call__(self, features):
        t0 = _time.perf_counter()
        batch = self._collator(features)
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        self._window_s.append(elapsed_ms)
        self._n += 1
        if self._n % self.LOG_EVERY == 0:
            import wandb
            # Only log when called from the main process (wandb.run is None in
            # DataLoader workers — they don't call wandb.init()).
            if wandb.run is not None:
                mean_ms = sum(self._window_s) / len(self._window_s)
                wandb.log(
                    {"timings/collate_ms": mean_ms,
                     "timings/collate_ms_peak": max(self._window_s)},
                    commit=False,
                )
            self._window_s.clear()
        return batch
