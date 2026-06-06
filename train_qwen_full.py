import argparse
import os
import time as _time
import torch
import wandb
from datasets import load_dataset

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
    ListDataset,
    LazyTask2Dataset,
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
        choices=["semantic_id", "text", "image", "multimodal"],
        help=(
            "What feature to use for history items in the Task 2 decoder prompt.\n"
            "  semantic_id  — raw semantic token string (current default behaviour)\n"
            "  text         — product title + details looked up from the items dataset\n"
            "  image        — product image thumbnail for each history item\n"
            "  multimodal   — (future) image + text for each history item"
        ),
    )
    parser.add_argument(
        "--max_history_items",
        type=int,
        default=None,
        help=(
            "Cap the number of history items used in the Task 2 image/multimodal prompt. "
            "None = use all. Strongly recommended to set 3–5 for image mode to avoid "
            "exceeding max_length. When capped, the most recent items are kept (tail of list)."
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
    return parser.parse_args()


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
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(new_vocab_size)

    # Mean-init new rows for both input embeddings and LM head (output embeddings)
    with torch.no_grad():
        mean_emb = model.get_input_embeddings().weight[:old_vocab_size].mean(dim=0)
        model.get_input_embeddings().weight[old_vocab_size:] = mean_emb

        mean_lm = model.get_output_embeddings().weight[:old_vocab_size].mean(dim=0)
        model.get_output_embeddings().weight[old_vocab_size:] = mean_lm

    return model, tokenizer


def prepare_task1_dataset(max_samples=None, image_size=224, dataset="theblackcat102/amazon-all-beauty-filtered"):
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
    return LazyTask1Dataset(items, image_size=image_size)


def prepare_task2_dataset(max_samples=None, historical_inputs="semantic_id",
                           image_size=0, max_history_items=None, max_images=None,
                           dataset="theblackcat102/amazon-all-beauty-filtered"):
    interactions = load_dataset(
        dataset, "interactions", split="train"
    )
    if max_samples is not None:
        interactions = interactions.select(range(min(max_samples, len(interactions))))
        print(f"[Task 2] Using {len(interactions)} samples (capped at {max_samples})")
    else:
        print(f"[Task 2] Using all {len(interactions)} samples")

    # ── semantic_id mode ──────────────────────────────────────────────────────
    if historical_inputs == "semantic_id":
        converted = []
        for sample in interactions:
            history_lines = "".join(
                f"{i+1}.{sid}<|im_end|>\n"
                for i, sid in enumerate(sample["history_semantic_ids"])
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

    # ── text mode ─────────────────────────────────────────────────────────────
    if historical_inputs == "text":
        print("[Task 2] Loading items dataset to build semantic-ID → text lookup …")
        items = load_dataset(
            dataset, "items", split="train"
        )
        semid_to_text = build_semid_to_text(items)
        print(f"[Task 2] Lookup built: {len(semid_to_text)} entries")

        converted = []
        for sample in interactions:
            history_entries = []
            for sid in sample["history_semantic_ids"]:
                text = semid_to_text.get(sid)
                if text:
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

    # ── image / multimodal modes ───────────────────────────────────────────────
    if historical_inputs in ("image", "multimodal"):
        print(f"[Task 2] Loading items dataset to build semantic-ID → image index …")
        items = load_dataset(
            dataset, "items", split="train"
        )
        if historical_inputs == "multimodal":
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
        if historical_inputs == "image":
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
            mode=historical_inputs,
            image_size=image_size,
            max_history_items=max_history_items,
            max_images=max_images,
        )

    raise ValueError(f"Unknown historical_inputs value: '{historical_inputs}'")


def build_trainer(model, tokenizer, dataset, args, task1_size: int = 0, task2_size: int = 0,
                   processor_path: str | None = None):
    # Deferred imports — only the main process calls build_trainer(), so these
    # heavy packages are never imported by DataLoader workers.
    from unsloth import FastVisionModel  # triggers CUDA patching; main process only
    from trainer_utils import TaskAwareSFTTrainer  # pulls in trl + wandb; main process only
    from trl import SFTConfig

    FastVisionModel.for_training(model)

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

    # 0. Resolve checkpoint (None → fresh run)
    checkpoint_path = resolve_checkpoint(args.resume_from_checkpoint, args.output_dir)

    # 0. Initialise W&B — same project used by the rest of this repo
    _hist_suffix = args.historical_inputs
    if args.max_history_items is not None:
        _hist_suffix += f"x{args.max_history_items}"
    run_name = args.wandb_run_name or (
        f"qwen_full"
        f"_hist={_hist_suffix}"
        f"_steps={args.max_steps}"
        f"_t1={args.max_task1_samples or 'all'}"
        f"_t2={args.max_task2_samples or 'all'}"
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
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. Prepare datasets for both tasks
    # Compute a safe image-per-sample cap for Task 2 image/multimodal modes.
    # This prevents the Qwen3-VL processor ValueError caused by truncation
    # slicing through a visual-token sequence mid-image.
    if args.max_images_per_sample is not None:
        max_images = args.max_images_per_sample
    elif args.historical_inputs in ("image", "multimodal"):
        max_images = compute_max_images(args.max_length, args.image_size)
        print(
            f"[Task 2] Auto-derived max_images_per_sample={max_images} "
            f"(max_length={args.max_length}, image_size={args.image_size or 'native'}). "
            f"Pass --max_images_per_sample to override."
        )
    else:
        max_images = None

    task1_data = prepare_task1_dataset(args.max_task1_samples, args.image_size, dataset=args.dataset)
    task2_data = prepare_task2_dataset(
        args.max_task2_samples,
        args.historical_inputs,
        image_size=args.image_size,
        max_history_items=args.max_history_items,
        max_images=max_images,
        dataset=args.dataset,
    )

    # 3. Joint training: ConcatDataset keeps both lazy datasets intact
    combined = torch.utils.data.ConcatDataset([task1_data, task2_data])
    print(f"Task 1 samples : {len(task1_data)}")
    print(f"Task 2 samples : {len(task2_data)}")
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
                            task1_size=len(task1_data),
                            task2_size=len(task2_data),
                            processor_path=_proc_tmp_dir)
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # 5. Save full weights + tokenizer (includes new special tokens)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
