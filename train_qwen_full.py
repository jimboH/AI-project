import argparse
import os
import time as _time
import torch
import torch._dynamo
import wandb
from datasets import load_dataset

# ── torch.compile recompile-limit bump ────────────────────────────────────────
# Unsloth compiles Qwen3_5RMSNorm_forward with fullgraph=True. That single
# compiled code object is shared across every RMSNorm instance in the model
# (input/post-attention layernorms plus q_norm/k_norm, which have a different
# weight shape). Dynamo guards per instance/shape, so recompiles accumulate
# fast. The torch default recompile_limit is only 8; once exceeded, fullgraph=True
# forbids an eager fallback and hard-fails with:
#   torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to fullgraph=True
# Unsloth normally bumps these to 1024, but set them here too so the limit holds
# regardless of whether Unsloth's import-time patch ran. Re-applied after model
# load (load_model_and_tokenizer) in case Unsloth's patching lowered them.
def _bump_dynamo_recompile_limits():
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 4096


_bump_dynamo_recompile_limits()

# ── Spawn-safe import discipline ──────────────────────────────────────────────
# Python's `spawn` multiprocessing start method re-executes this file in every
# DataLoader worker (via multiprocessing.spawn._fixup_main_from_path).  Any
# module-level code runs again in each worker.
#
# Unsloth's __init__ triggers full CUDA patching on import.  Running that
# inside 4 spawned worker processes simultaneously causes:
#   1. Slow startup (several seconds per worker → prefetch queue starves GPU).
#   2. CUDA deadlock: multiple processes initialise the same GPU concurrently.
#
# Fix: keep module-level imports Unsloth-free.  Import Unsloth inside the
# functions that actually need it (load_model_and_tokenizer, build_trainer).
# Workers never call those functions, so they never touch Unsloth.
#
# TRL (SFTTrainer / SFTConfig) and the custom trainer classes are imported
# lazily inside build_trainer() so that DataLoader workers — which re-execute
# this file as __mp_main__ via Python's spawn bootstrap — never import TRL or
# wandb at startup.  Workers only need torch + data_utils to do their job.

# Dataset & collation machinery lives in an Unsloth-free module so that
# spawned DataLoader workers re-import only torch/PIL/datasets instead of
# re-running the full Unsloth patching on every worker startup.
from data_utils import (
    compute_max_images,
    PreTokenizingDataset,
    make_parallel_collate_fn,
    LazyTask1Dataset,
    LazySemanticIdTask2Dataset,
    ListDataset,
    LazyTask2Dataset,
    LazyMultiModeTask2Dataset,
    build_semid_to_index,
    build_semid_to_text,
    build_semid_to_index_and_text,
    TimedCollator,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps",    type=int,   default=500)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--grad_accum",   type=int,   default=8)
    parser.add_argument("--warmup_steps", type=int,   default=10)
    parser.add_argument("--output_dir",   type=str,   default="outputs/qwen_full")
    parser.add_argument("--max_length",         type=int,   default=2048)
    parser.add_argument("--model_name",         type=str,   default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--max_task1_samples",  type=int,   default=None,
                        help="Limit items dataset size (None = use all ~112K)")
    parser.add_argument("--max_task2_samples",  type=int,   default=None,
                        help="Limit interactions dataset size (None = use all ~5.6K)")
    parser.add_argument("--dataset",            type=str,   default="theblackcat102/amazon-all-beauty-filtered",
                        help="HuggingFace dataset repo to use for both 'items' and 'interactions' configs.")
    parser.add_argument("--wandb_project",      type=str,   default="gen-retrieval-decoder")
    parser.add_argument("--wandb_run_name",     type=str,   default=None,
                        help="W&B run name (default: qwen_full_<max_steps>steps)")
    parser.add_argument("--image_size",         type=int,   default=224,
                        help="Resize images to this square size before the vision processor. "
                             "Default 224 = small fixed size, lower memory and fewer visual tokens. "
                             "Use 0 to skip pre-resize and let the processor handle it (higher quality, more tokens). "
                             "Use 512 to match Unsloth's internal size.")
    parser.add_argument("--dataloader_workers", type=int,   default=4,
                        help="Number of DataLoader worker processes for parallel image loading. "
                             "Each worker forks ~1.7GB RAM; keep <= floor(RAM_GB / 2) to avoid swapping. "
                             "NOTE: bench-optimal workers depends on --image_size. "
                             "At native resolution (image_size=0) more workers bloat the prefetch buffer "
                             "with huge images and stall startup. 4 is safe for 32GB machines.")
    parser.add_argument(
        "--historical_inputs",
        type=str,
        default="semantic_id",
        help=(
            "Comma-separated list of history representations for Task 2 samples.\n"
            "Valid values: semantic_id, text, image, multimodal\n"
            "\n"
            "Single mode (original behaviour):\n"
            "  --historical_inputs image\n"
            "\n"
            "Multi-mode (new): each interaction is repeated once per mode, making\n"
            "the Task 2 dataset N× larger. Every copy predicts the same target\n"
            "semantic ID but sees a different history representation:\n"
            "  --historical_inputs image,text,multimodal\n"
            "\n"
            "  semantic_id  — raw semantic token string for each history item\n"
            "  text         — product title + details looked up from the items dataset\n"
            "  image        — product image thumbnail for each history item\n"
            "  multimodal   — image(s) + text for each history item"
        ),
    )
    parser.add_argument(
        "--max_history_items",
        type=int,
        default=None,
        help=(
            "Cap the number of history items used in the Task 2 prompt. "
            "None = use all. Strongly recommended to set 3–5 for image mode to avoid "
            "exceeding max_length. When capped, the most recent items are kept (tail of list). "
            "Also applied in text mode."
        ),
    )
    parser.add_argument(
        "--max_text_chars_per_item",
        type=int,
        default=512,
        help=(
            "Maximum characters per history item in text mode. "
            "Each item's 'Title. Details' string is hard-clipped to this length before "
            "being inserted into the prompt, preventing runaway sequence lengths when "
            "product details fields are long. Default 512."
        ),
    )
    parser.add_argument(
        "--parallel_collate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move per-sample tokenisation (apply_chat_template + processor encode) into "
            "DataLoader workers so the main thread only pads and stacks tensors. "
            "Requires num_workers >= 1. Falls back to UnslothVisionDataCollator when "
            "num_workers == 0 or --no-parallel_collate is passed."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume training from, or 'latest' to "
            "auto-detect the most recent checkpoint-* folder inside --output_dir. "
            "Omit (default) to start a fresh run."
        ),
    )
    parser.add_argument(
        "--max_images_per_sample",
        type=int,
        default=None,
        help=(
            "Hard cap on total images per Task 2 sample in image/multimodal modes. "
            "If None (default), auto-derived from --max_length and --image_size so "
            "sequences never need mid-image truncation (which causes the Qwen3-VL "
            "processor ValueError). Override only if you know your actual token budget."
        ),
    )
    parser.add_argument(
        "--task2_only",
        action="store_true",
        default=False,
        help=(
            "Train on Task 2 (recsys interactions) only, skipping Task 1 (item indexing). "
            "History items still use the format selected by --historical_inputs. "
            "Uses a standard random sampler instead of HalfHalfBatchSampler."
        ),
    )
    return parser.parse_args()


_VALID_HIST_MODES = {"semantic_id", "text", "image", "multimodal"}


def parse_historical_inputs(value: str) -> list:
    """Parse and validate the ``--historical_inputs`` argument.

    Accepts a single mode name or a comma-separated list of mode names.
    Returns a deduplicated ordered list, preserving the user-specified order.

    Raises ``ValueError`` if any mode is unknown.
    """
    modes = [m.strip() for m in value.split(",") if m.strip()]
    if not modes:
        raise ValueError("--historical_inputs must specify at least one mode.")
    # Deduplicate while preserving order (dict trick, Python 3.7+)
    modes = list(dict.fromkeys(modes))
    invalid = set(modes) - _VALID_HIST_MODES
    if invalid:
        raise ValueError(
            f"--historical_inputs: unknown mode(s) {sorted(invalid)}. "
            f"Valid: {sorted(_VALID_HIST_MODES)}"
        )
    return modes


def build_semantic_tokens():
    # d0–d3, codes 0–255 → 1,024 tokens total
    return [f"<|d{d}_{i}|>" for d in range(4) for i in range(256)]


def load_model_and_tokenizer(args):
    from unsloth import FastVisionModel  # deferred: must not run in DataLoader workers
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=False,
        full_finetuning=True,           # enable full-weight finetuning (no LoRA)
        use_gradient_checkpointing="unsloth",
    )
    # ── No get_peft_model() → all 866M parameters are trained ──

    # FastVisionModel returns a Processor for VL models; the underlying tokenizer
    # is at processor.tokenizer (or the object itself if it's a plain tokenizer).
    inner_tok = getattr(tokenizer, "tokenizer", tokenizer)

    # Add semantic tokens
    new_tokens = build_semantic_tokens()
    num_added = inner_tok.add_special_tokens({"additional_special_tokens": new_tokens})
    new_vocab_size = len(inner_tok)
    print(f"Added {num_added} new tokens. New vocab size: {new_vocab_size}")

    # Resize embedding matrix
    model.resize_token_embeddings(new_vocab_size)

    # ── New-row range ─────────────────────────────────────────────────────────
    # resize_token_embeddings pads the matrix to the nearest multiple (e.g. 64),
    # so weight.shape[0] > new_vocab_size after the call.  Using weight.shape[0]
    # as the split point (the old code) would touch those padding rows too.
    # Correct bounds: the actual new token positions are the last num_added rows
    # of the true vocabulary, i.e. [new_vocab_size - num_added, new_vocab_size).
    new_row_start = len(inner_tok) - num_added   # first real new-token row
    new_row_end   = len(inner_tok)               # one past last real new-token row

    # Mean-init new rows + noise to break symmetry.
    # All 1 024 new rows would receive identical mean_emb without noise, making
    # their gradients collapse to the same direction every step (the backward
    # through an embedding lookup sums identical rows → identical gradient →
    # identical updates).  Noise std 0.02 ≈ per-dimension std of existing rows.
    with torch.no_grad():
        emb_w = model.get_input_embeddings().weight
        mean_emb = emb_w[:new_row_start].mean(dim=0)
        noise = torch.randn(new_row_end - new_row_start, emb_w.shape[1],
                            dtype=emb_w.dtype, device=emb_w.device) * 0.02
        emb_w[new_row_start:new_row_end] = mean_emb.unsqueeze(0) + noise

        lm_w = model.get_output_embeddings().weight
        mean_lm = lm_w[:new_row_start].mean(dim=0)
        noise = torch.randn(new_row_end - new_row_start, lm_w.shape[1],
                            dtype=lm_w.dtype, device=lm_w.device) * 0.02
        lm_w[new_row_start:new_row_end] = mean_lm.unsqueeze(0) + noise

    print(f"New rows [{new_row_start}, {new_row_end}) initialised with mean+noise.")
    return model, tokenizer, new_row_start


def prepare_task1_dataset(max_samples=None, image_size=224, max_text_chars_per_item=512,
                           dataset="theblackcat102/amazon-all-beauty-filtered"):
    items = load_dataset(
        dataset, "items", split="train"
    )
    # Filter rows whose image_main is missing without triggering PIL decode.
    # HF's filter() internally builds a full example dict from every column,
    # which decodes ALL Image-typed columns — including ones with
    # {bytes: None, path: None} that raise ValueError.  The only safe approach
    # is to inspect the raw Arrow struct column directly and call select().
    arrow_img = items.data.table.column("image_main")
    valid_indices = [
        i for i, val in enumerate(arrow_img)
        if (d := val.as_py()) is not None
        and (d.get("bytes") is not None or d.get("path") is not None)
    ]
    n_dropped = len(items) - len(valid_indices)
    if n_dropped:
        print(f"[Task 1] Dropped {n_dropped} rows with missing image_main")
    items = items.select(valid_indices)
    # Slice BEFORE wrapping to avoid materialising the full dataset
    if max_samples is not None:
        items = items.select(range(min(max_samples, len(items))))
        print(f"[Task 1] Using {len(items)} samples (capped at {max_samples})")
    else:
        print(f"[Task 1] Using all {len(items)} samples")

    if image_size > 0:
        print(f"[Task 1] Images will be resized lazily to {image_size}x{image_size} during training")
    else:
        print("[Task 1] Images will be passed at native resolution (processor handles resizing)")
    # Return a lazy dataset — no images are decoded here.
    return LazyTask1Dataset(items, image_size=image_size, max_text_chars=max_text_chars_per_item)


def prepare_task2_dataset(max_samples=None, historical_inputs="semantic_id",
                           image_size=0, max_history_items=None, max_images=None,
                           max_text_chars_per_item=512,
                           dataset="theblackcat102/amazon-all-beauty-filtered"):
    interactions = load_dataset(
        dataset, "interactions", split="train"
    )
    if max_samples is not None:
        interactions = interactions.select(range(min(max_samples, len(interactions))))
        print(f"[Task 2] Using {len(interactions)} samples (capped at {max_samples})")
    else:
        print(f"[Task 2] Using all {len(interactions)} samples")

    # Parse mode list (supports both "image" and "image,text,multimodal")
    modes = parse_historical_inputs(historical_inputs)

    # ── Single-mode fast paths (unchanged legacy behaviour) ───────────────────
    if len(modes) == 1:
        mode = modes[0]

        # ── semantic_id ───────────────────────────────────────────────────────
        if mode == "semantic_id":
            # Use a lazy dataset backed by the HF Arrow interactions table.
            #
            # Why lazy instead of ListDataset (eager)?
            # -----------------------------------------
            # The DataLoader uses multiprocessing_context="fork".  Under fork,
            # Python's reference-counting turns every read of a pre-built Python
            # list element into a copy-on-write page fault.  A ListDataset of 5 600
            # prompt strings (some potentially >10 KB for power users) would cause
            # all workers to COW-copy the entire list into their own address space,
            # bloating RAM and stalling the prefetch queue.
            #
            # HF Arrow datasets use memory-mapped native buffers without Python GC
            # overhead, so forked workers share the same physical pages and no COW
            # faulting occurs.  Per-sample prompt construction (cheap string ops)
            # happens inside the worker on demand.
            #
            # Why max_history_items matters here:
            # ------------------------------------
            # image/multimodal modes are auto-capped to ~1 history item by the
            # compute_max_images() budget (1 native-res image already fills most of
            # the 4 096-token budget).  semantic_id mode has NO equivalent cap unless
            # max_history_items is set — users with 100–500 interactions produce
            # huge prompt strings, making tokenisation in workers 10–100× slower
            # than image mode and turning the DataLoader into the training bottleneck.
            if max_history_items is None:
                print(
                    "[Task 2][semantic_id] WARNING: max_history_items is not set. "
                    "Users with many interactions will produce very long prompts, "
                    "making per-sample tokenisation slow and stalling the DataLoader. "
                    "Pass --max_history_items (e.g. 20) to cap history length."
                )
            return LazySemanticIdTask2Dataset(
                interactions_hf_dataset=interactions,
                max_history_items=max_history_items,
            )

        # ── text ──────────────────────────────────────────────────────────────
        if mode == "text":
            print("[Task 2] Loading items dataset to build semantic-ID → text lookup …")
            items = load_dataset(
                dataset, "items", split="train"
            )
            semid_to_text = build_semid_to_text(items)
            print(f"[Task 2] Lookup built: {len(semid_to_text)} entries")

            converted = []
            for sample in interactions:
                history_sids = sample["history_semantic_ids"]
                if max_history_items is not None:
                    history_sids = history_sids[-max_history_items:]
                history_entries = []
                for sid in history_sids:
                    text = semid_to_text.get(sid)
                    if text:
                        if max_text_chars_per_item is not None:
                            text = text[:max_text_chars_per_item]
                        history_entries.append(text)
                history_lines = "".join(
                    f"{i+1}.{text}<|im_end|>\n" for i, text in enumerate(history_entries)
                )
                prompt = (
                    f"User interaction history:\n{history_lines}"
                    "Predict the next item's semantic ID:"
                )
                converted.append({"messages": [
                    {"role": "user",      "content": [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": sample["target_semantic_id"]}]},
                ]})
            return ListDataset(converted)

        # ── image / multimodal ────────────────────────────────────────────────
        if mode in ("image", "multimodal"):
            print(f"[Task 2] Loading items dataset to build semantic-ID → image index …")
            items = load_dataset(
                dataset, "items", split="train"
            )
            if mode == "multimodal":
                # Single-pass: builds both index and text map in one 112K-row scan.
                # 1.57× faster than two separate calls (saves ~6s startup at 112K rows).
                semid_to_index, semid_to_text = build_semid_to_index_and_text(items)
                print(f"[Task 2] Image index + text map built in one pass: "
                      f"{len(semid_to_index)} image entries, {len(semid_to_text)} text entries")
            else:
                semid_to_index = build_semid_to_index(items)
                print(f"[Task 2] Image index built: {len(semid_to_index)} entries with image_main")
                semid_to_text = None

            # Project items down to only the columns LazyTask2Dataset actually needs.
            # Benchmarks show full-row random access costs 1.95ms vs 0.69ms with
            # projection — a 2.8x speedup per history item lookup.
            # image mode   : only image_main is rendered per history item
            # multimodal   : all three image views + title + details
            if mode == "image":
                items = items.select_columns(["semantic_id", "image_main"])
            else:  # multimodal
                items = items.select_columns(
                    ["semantic_id", "image_main", "image_pt01", "image_pt02", "title", "details"]
                )
            print(f"[Task 2] Items projected to {items.column_names} — {len(items)} rows")

            return LazyTask2Dataset(
                interactions_hf_dataset=interactions,
                items_hf_dataset=items,
                semid_to_index=semid_to_index,
                semid_to_text=semid_to_text,
                mode=mode,
                image_size=image_size,
                max_history_items=max_history_items,
                max_images=max_images,
            )

    # ── Multi-mode path ───────────────────────────────────────────────────────
    # Each base interaction produces len(modes) training samples; all share the
    # same target semantic ID but use a different history representation.
    # Total dataset size = len(interactions) × len(modes).
    print(
        f"[Task 2] Multi-mode active: {modes} — "
        f"each of {len(interactions)} interactions → {len(modes)} samples "
        f"(total {len(interactions) * len(modes)})"
    )

    need_text  = any(m in ("text",  "multimodal") for m in modes)
    need_image = any(m in ("image", "multimodal") for m in modes)

    semid_to_text  = None
    semid_to_index = None
    items          = None

    if need_text or need_image:
        print("[Task 2] Loading items dataset for multi-mode lookup …")
        items = load_dataset(dataset, "items", split="train")

        if need_text and need_image:
            # Single-pass builds both maps — avoids scanning 112K rows twice.
            semid_to_index, semid_to_text = build_semid_to_index_and_text(items)
            print(
                f"[Task 2] Image index + text map built in one pass: "
                f"{len(semid_to_index)} image entries, {len(semid_to_text)} text entries"
            )
        elif need_text:
            semid_to_text = build_semid_to_text(items)
            print(f"[Task 2] Text map built: {len(semid_to_text)} entries")
        else:  # only image
            semid_to_index = build_semid_to_index(items)
            print(f"[Task 2] Image index built: {len(semid_to_index)} entries")

        # Project items to only the columns needed across the active modes.
        # Full-row random access costs ~1.95ms vs ~0.69ms with projection (2.8×).
        cols = ["semantic_id"]
        if need_image:
            cols.append("image_main")
            if any(m == "multimodal" for m in modes):
                cols += ["image_pt01", "image_pt02"]
        if need_text:
            cols += ["title", "details"]
        # Deduplicate while preserving order (dict trick, Python 3.7+).
        seen: set = set()
        cols = [c for c in cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
        items = items.select_columns(cols)
        print(f"[Task 2] Items projected to {items.column_names} — {len(items)} rows")

    return LazyMultiModeTask2Dataset(
        interactions_hf_dataset=interactions,
        items_hf_dataset=items,
        modes=modes,
        semid_to_index=semid_to_index,
        semid_to_text=semid_to_text,
        image_size=image_size,
        max_history_items=max_history_items,
        max_images=max_images,
        max_text_chars=max_text_chars_per_item,
    )


def build_trainer(model, tokenizer, dataset, args, task1_size: int = 0, task2_size: int = 0,
                   processor_path: str | None = None, new_emb_row_start: int | None = None):
    # Deferred imports — only the main process calls build_trainer(), so these
    # heavy packages are never imported by DataLoader workers.
    from unsloth import FastVisionModel  # triggers CUDA patching; main process only
    from trainer_utils import TaskAwareSFTTrainer  # pulls in trl + wandb; main process only
    from trl import SFTConfig

    FastVisionModel.for_training(model)

    # ── 10× gradient-scale hook for new token rows ────────────────────────────
    # Registered AFTER for_training() so any Unsloth parameter surgery is done.
    # Scaling the gradient by 10× gives the new rows an effective lr ≈ 1e-3 when
    # the global lr is 1e-4, matching the recommended separate-param-group LR,
    # without needing a custom optimizer.  Old rows are unaffected (scale = 1).
    if new_emb_row_start is not None:
        _start = new_emb_row_start

        def _scale_new_rows(grad, start=_start, factor=10.0):
            g = grad.clone()
            g[start:] *= factor
            return g

        model.get_input_embeddings().weight.register_hook(_scale_new_rows)
        model.get_output_embeddings().weight.register_hook(_scale_new_rows)
        print(f"[Init] Gradient scale x10 registered for new rows [{_start}:] "
              f"on input_embeddings and output_embeddings.")

    use_parallel = args.parallel_collate and args.dataloader_workers > 0

    if use_parallel:
        # ── Parallel collation path ───────────────────────────────────────────
        # Each DataLoader worker calls apply_chat_template + processor encode for
        # its own samples.  The main-thread collate_fn only pads and stacks the
        # pre-built tensors — O(batch_size) work instead of O(batch_size × seq_len).
        #
        # Buffering with this path:
        #   workers × prefetch_factor × 1 sample (compact tensors)
        # vs the old path:
        #   workers × prefetch_factor × batch_size samples (PIL dicts) → OOM risk
        print(
            f"[Collation] Parallel mode: per-sample tokenisation in {args.dataloader_workers} workers. "
            "Main thread: pad + stack only."
        )
        train_dataset = PreTokenizingDataset(
            dataset,
            processor=tokenizer,
            max_seq_length=args.max_length,
            processor_path=processor_path,
        )
        inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
        pad_token_id = inner_tok.pad_token_id if inner_tok.pad_token_id is not None else inner_tok.eos_token_id
        pixel_dtype = model.get_input_embeddings().weight.dtype
        collate_fn = TimedCollator(
            make_parallel_collate_fn(
                pad_token_id=pad_token_id,
                pad_to_multiple_of=8,
                pixel_dtype=pixel_dtype,
            )
        )
    else:
        # ── Original single-process collation path ────────────────────────────
        # UnslothVisionDataCollator runs apply_chat_template + processor encode
        # for every sample in the batch serially in the main thread.
        from unsloth.trainer import UnslothVisionDataCollator  # deferred: safe here, main process only
        print(
            "[Collation] Serial mode (UnslothVisionDataCollator in main thread). "
            "Use --parallel_collate with num_workers >= 1 to parallelise."
        )
        train_dataset = dataset
        collate_fn = TimedCollator(UnslothVisionDataCollator(model, tokenizer))

    return TaskAwareSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        task1_size=task1_size,
        task2_size=task2_size,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=10,
            save_steps=500,
            save_total_limit=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=10.0,            # default 1.0 clips 6-25× every step; 10.0 lets gradients breathe
            seed=3407,
            output_dir=args.output_dir,
            report_to="wandb",
            bf16=True,                                 # bf16 mixed precision (Blackwell supports it)
            # ── DataLoader throughput ─────────────────────────────────────────
            dataloader_num_workers=args.dataloader_workers,   # parallel tokenise/encode workers
            dataloader_prefetch_factor=2,              # prefetch 2 samples per worker
            dataloader_pin_memory=True,                # faster CPU→GPU transfers
            dataloader_persistent_workers=True,        # keep workers alive between steps (avoids respawn overhead)
            # ── MANDATORY for multimodal SFT ──────────────────────────────────
            remove_unused_columns=False,               # keeps tensor keys alive in dataset
            dataset_text_field="",                     # no single text column; multimodal format
            dataset_kwargs={"skip_prepare_dataset": True},  # bypass SFT's default tokenization
            max_length=args.max_length,                # max token sequence length
            # NOTE: For VLM training the image-per-sample cap (max_images_per_sample /
            # compute_max_images) must ensure sequences fit within this budget *before*
            # reaching the processor.  Truncating mid-image-token sequence causes:
            #   ValueError: Mismatch in `image` token count between text and input_ids
            # The LazyTask2Dataset enforces this via its max_images parameter.
            pad_to_multiple_of=8,                      # align seq-len to Tensor Core boundaries (bf16 → 8)
            # ─────────────────────────────────────────────────────────────────
        ),
    )


def resolve_checkpoint(resume_arg: str | None, output_dir: str) -> str | None:
    """Return the checkpoint directory to resume from, or None for a fresh run.

    Pass ``resume_arg="latest"`` (or ``"true"``) to auto-detect the highest-numbered
    ``checkpoint-*`` folder inside *output_dir*.  Pass an explicit path to use that
    checkpoint directly.
    """
    if resume_arg is None:
        return None
    if resume_arg.lower() in ("latest", "true"):
        import glob as _glob
        candidates = sorted(
            _glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
        )
        if not candidates:
            print(f"[Resume] No checkpoints found in '{output_dir}' — starting fresh.")
            return None
        chosen = candidates[-1]
        print(f"[Resume] Auto-detected latest checkpoint: {chosen}")
        return chosen
    if not os.path.isdir(resume_arg):
        raise ValueError(
            f"--resume_from_checkpoint path does not exist or is not a directory: {resume_arg}"
        )
    print(f"[Resume] Resuming from checkpoint: {resume_arg}")
    return resume_arg


_WANDB_ID_FILE = "wandb_run_id.txt"


def get_or_create_wandb_id(output_dir: str, resuming: bool) -> tuple[str, bool]:
    """Return (run_id, is_resume) and persist the ID to *output_dir*.

    On the first run a fresh ID is generated and saved.  On subsequent runs with
    ``resuming=True`` the saved ID is reused so the W&B run continues seamlessly.
    """
    id_path = os.path.join(output_dir, _WANDB_ID_FILE)
    if resuming and os.path.exists(id_path):
        with open(id_path) as fh:
            run_id = fh.read().strip()
        print(f"[W&B] Resuming existing run id={run_id}")
        return run_id, True
    run_id = wandb.util.generate_id()
    os.makedirs(output_dir, exist_ok=True)
    with open(id_path, "w") as fh:
        fh.write(run_id)
    return run_id, False


def main():
    # Prevent the HuggingFace fast tokenizer's Rust thread-pool from deadlocking
    # inside DataLoader worker processes (spawn or fork).  Must be set before any
    # tokenizer is imported or workers are created.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()

    # Validate --historical_inputs early so the user gets a clear error before
    # any model or dataset loading begins.
    hist_modes = parse_historical_inputs(args.historical_inputs)

    # 0. Resolve checkpoint (None → fresh run)
    checkpoint_path = resolve_checkpoint(args.resume_from_checkpoint, args.output_dir)

    # 0. Initialise W&B — same project used by the rest of this repo
    # Use "+" as separator in run names for multi-mode (commas break some shells)
    _hist_suffix = "+".join(hist_modes)
    if args.max_history_items is not None:
        _hist_suffix += f"x{args.max_history_items}"
    _tasks_suffix = "t2only" if args.task2_only else "t1t2"
    run_name = args.wandb_run_name or (
        f"qwen_full"
        f"_hist={_hist_suffix}"
        f"_{_tasks_suffix}"
        f"_steps={args.max_steps}"
        + (f"_t1={args.max_task1_samples or 'all'}" if not args.task2_only else "")
        + f"_t2={args.max_task2_samples or 'all'}"
    )
    wandb_id, is_wandb_resume = get_or_create_wandb_id(args.output_dir, checkpoint_path is not None)
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        id=wandb_id,
        resume="allow",
        config=vars(args),
    )

    # 1. Load model and tokenizer, add semantic tokens, resize embeddings
    model, tokenizer, new_row_start = load_model_and_tokenizer(args)

    # Re-apply: Unsloth's import-time patching (triggered inside
    # load_model_and_tokenizer) may have reset the dynamo recompile limits.
    _bump_dynamo_recompile_limits()

    # 2. Prepare datasets for both tasks
    # Compute a safe image-per-sample cap for Task 2 image/multimodal modes.
    # This prevents the Qwen3-VL processor ValueError caused by truncation
    # slicing through a visual-token sequence mid-image.
    if args.max_images_per_sample is not None:
        max_images = args.max_images_per_sample
    elif any(m in ("image", "multimodal") for m in hist_modes):
        max_images = compute_max_images(args.max_length, args.image_size)
        print(
            f"[Task 2] Auto-derived max_images_per_sample={max_images} "
            f"(max_length={args.max_length}, image_size={args.image_size or 'native'}). "
            f"Pass --max_images_per_sample to override."
        )
    else:
        max_images = None

    task2_data = prepare_task2_dataset(
        args.max_task2_samples,
        args.historical_inputs,
        image_size=args.image_size,
        max_history_items=args.max_history_items,
        max_images=max_images,
        max_text_chars_per_item=args.max_text_chars_per_item,
        dataset=args.dataset,
    )

    if args.task2_only:
        combined = task2_data
        task1_size = 0
        task2_size = len(task2_data)
        print(f"Task 2 only : {task2_size} samples")
    else:
        # 3. Joint training: ConcatDataset keeps both lazy datasets intact
        task1_data = prepare_task1_dataset(args.max_task1_samples, args.image_size,
                                            max_text_chars_per_item=args.max_text_chars_per_item,
                                            dataset=args.dataset)
        combined = torch.utils.data.ConcatDataset([task1_data, task2_data])
        task1_size = len(task1_data)
        task2_size = len(task2_data)
        print(f"Task 1 samples : {task1_size}")
        print(f"Task 2 samples : {task2_size}")
        print(f"Total combined : {len(combined)}")

    # 4. Build trainer and run
    # ── Save processor so spawned DataLoader workers can reload it from disk ──
    # Unsloth patches Qwen3VLProcessor at the class level, making the live
    # processor object unpicklable under the ``spawn`` multiprocessing context.
    # Saving to disk and passing the path lets each worker call
    # AutoProcessor.from_pretrained() instead of unpickling the patched class.
    import tempfile
    _proc_tmp_dir = tempfile.mkdtemp(prefix="qwen_proc_workers_")
    tokenizer.save_pretrained(_proc_tmp_dir)
    print(f"[DataLoader] Processor saved for workers: {_proc_tmp_dir}")

    trainer = build_trainer(model, tokenizer, combined, args,
                            task1_size=task1_size,
                            task2_size=task2_size,
                            processor_path=_proc_tmp_dir,
                            new_emb_row_start=new_row_start)
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # 5. Save full weights + tokenizer (includes new special tokens)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
