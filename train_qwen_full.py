import argparse
import os
import torch
import wandb
from datasets import load_dataset
from PIL import Image
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


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
    parser.add_argument("--wandb_project",      type=str,   default="gen-retrieval-decoder")
    parser.add_argument("--wandb_run_name",     type=str,   default=None,
                        help="W&B run name (default: qwen_full_<max_steps>steps)")
    parser.add_argument("--image_size",         type=int,   default=0,
                        help="Resize images to this square size before the vision processor. "
                             "Default 0 = skip pre-resize and let the processor handle it (recommended). "
                             "Use 512 to match Unsloth's internal size, or 224 for lower memory.")
    parser.add_argument("--dataloader_workers", type=int,   default=4,
                        help="Number of DataLoader worker processes for parallel image loading. "
                             "Each worker forks ~1.7GB RAM; keep <= floor(RAM_GB / 2) to avoid swapping. "
                             "Default 4 is safe for 32GB machines.")
    return parser.parse_args()


def build_semantic_tokens():
    # d0–d3, codes 0–255 → 1,024 tokens total
    return [f"<|d{d}_{i}|>" for d in range(4) for i in range(256)]


def load_model_and_tokenizer(args):
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


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """Resize a PIL image to (size x size) using high-quality downsampling.

    Pass size=0 to skip resizing entirely and let the vision processor handle it.
    """
    if size <= 0:
        return img
    return img.resize((size, size), Image.LANCZOS)


class LazyTask1Dataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace items dataset and decodes/resizes images on demand.

    Images are never stored in RAM en masse — each sample is decoded only when
    the DataLoader requests it, keeping peak memory proportional to batch size
    rather than dataset size.
    """

    def __init__(self, hf_dataset, image_size: int = 224):
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


def prepare_task1_dataset(max_samples=None, image_size=224):
    items = load_dataset(
        "theblackcat102/amazon-all-beauty-filtered", "items", split="train"
    )
    # Filter items missing the main image
    items = items.filter(lambda x: x["image_main"] is not None)
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


def prepare_task2_dataset(max_samples=None):
    interactions = load_dataset(
        "theblackcat102/amazon-all-beauty-filtered", "interactions", split="train"
    )
    if max_samples is not None:
        interactions = interactions.select(range(min(max_samples, len(interactions))))
        print(f"[Task 2] Using {len(interactions)} samples (capped at {max_samples})")

    converted = []
    for sample in interactions:
        history_lines = "\n".join(
            f"{i+1}. {sid}"
            for i, sid in enumerate(sample["history_semantic_ids"])
        )
        prompt = (
            f"User interaction history:\n{history_lines}\n"
            "Predict the next item's semantic ID:"
        )

        converted.append({"messages": [
            {"role": "user",      "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["target_semantic_id"]}]},
        ]})

    return ListDataset(converted)


def build_trainer(model, tokenizer, dataset, args):
    FastVisionModel.for_training(model)

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=args.output_dir,
            report_to="wandb",
            bf16=True,                                 # bf16 mixed precision (Blackwell supports it)
            # ── DataLoader throughput ─────────────────────────────────────────
            dataloader_num_workers=args.dataloader_workers,   # parallel image decode/resize workers
            dataloader_prefetch_factor=2,              # prefetch 2 batches per worker
            dataloader_pin_memory=True,                # faster CPU→GPU transfers
            dataloader_persistent_workers=True,        # keep workers alive between steps (avoids respawn overhead)
            # ── MANDATORY for multimodal SFT ──────────────────────────────────
            remove_unused_columns=False,               # keeps 'image' column alive in dataset
            dataset_text_field="",                     # no single text column; multimodal format
            dataset_kwargs={"skip_prepare_dataset": True},  # bypass SFT's default tokenization
            max_length=2048,                           # max token sequence length
            # ─────────────────────────────────────────────────────────────────
        ),
    )


def main():
    args = parse_args()

    # 0. Initialise W&B — same project used by the rest of this repo
    run_name = args.wandb_run_name or f"qwen_full_{args.max_steps}steps"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    # 1. Load model and tokenizer, add semantic tokens, resize embeddings
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. Prepare datasets for both tasks
    task1_data = prepare_task1_dataset(args.max_task1_samples, args.image_size)
    task2_data = prepare_task2_dataset(args.max_task2_samples)

    # 3. Joint training: ConcatDataset keeps both lazy datasets intact
    combined = torch.utils.data.ConcatDataset([task1_data, task2_data])
    print(f"Task 1 samples : {len(task1_data)}")
    print(f"Task 2 samples : {len(task2_data)}")
    print(f"Total combined : {len(combined)}")

    # 4. Build trainer and run
    trainer = build_trainer(model, tokenizer, combined, args)
    trainer.train()

    # 5. Save full weights + tokenizer (includes new special tokens)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
