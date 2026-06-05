# Multimodal Generative Retrieval — Training Plan

Qwen3.5-0.8B full-weight finetuning on `theblackcat102/amazon-all-beauty-filtered` for two jointly-trained tasks:
- **Task 1** — Image + text → product semantic ID
- **Task 2** — User interaction history → next item semantic ID

---

## Background: Reference Notebook

`Qwen3_5_(0_8B)_Vision.ipynb` demonstrates LoRA finetuning using:
- `FastVisionModel.from_pretrained("unsloth/Qwen3.5-0.8B")` + `FastVisionModel.get_peft_model()` (r=16, alpha=16, 1.52% trainable params)
- `UnslothVisionDataCollator` — mandatory for multimodal SFT
- `SFTTrainer` + `SFTConfig` from `trl`
- Four mandatory `SFTConfig` flags for multimodal data (see Section 6 below)

This plan replaces LoRA with **full weight finetuning** and targets a new domain-specific task.

---

## Dataset Analysis

### `items` split (112,590 rows)

| Column | Type | Notes |
|--------|------|-------|
| `parent_asin` | string | Product ID |
| `title` | string | Product title |
| `details` | string | JSON-encoded product details |
| `semantic_id` | string | Target — e.g. `<\|d0_101\|> <\|d1_57\|> <\|d2_29\|>` |
| `semantic_codes` | List[int] | Numeric codes per depth level |
| `image_main` | Image | Main product image (~240 None, 0.2%) |
| `image_pt01` | Image | Secondary image (~26K None, 23%) |
| `image_pt02` | Image | Tertiary image (~35K None, 31%) |

### `interactions` split (5,621 rows, all `split='train'`)

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | string | User identifier |
| `history_semantic_ids` | List[string] | Ordered list of past item semantic IDs |
| `target_semantic_id` | string | Target — next item to predict |
| `target_semantic_codes` | List[int] | Numeric codes for target |
| `split` | string | Always `'train'` in this config |

### Semantic ID Format

Semantic IDs use hierarchical tokens across 4 depth levels:
- Depth 0–2: codes 0–255 (items split uses d0–d2)
- Depth 3: codes 0–39 (interactions split target may include d3)

Example: `<|d0_101|> <|d1_57|> <|d2_29|>`

---

## Critical Issue: Semantic Tokens Not in Vocabulary

The tokens `<|d0_X|>`, `<|d1_X|>` etc. are **not** in the Qwen3.5 tokenizer.
Each token is split into 9+ subword pieces (e.g. `<|d0_50|>` → 9 token IDs).

**Fix:** Add all 4 × 256 = **1,024 new special tokens** and resize the model's embedding matrix with mean initialization.

---

## Script: `train_qwen_full.py`

### Section 1 — Imports & argparse

```python
import argparse
import torch
from datasets import load_dataset
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
    parser.add_argument("--max_length",   type=int,   default=2048)
    parser.add_argument("--model_name",   type=str,   default="unsloth/Qwen3.5-0.8B")
    return parser.parse_args()
```

---

### Section 2 — Semantic Token Generation

```python
def build_semantic_tokens():
    # d0–d3, codes 0–255 → 1,024 tokens total
    return [f"<|d{d}_{i}|>" for d in range(4) for i in range(256)]
```

---

### Section 3 — Model & Tokenizer Loading

Full weight finetuning: `get_peft_model()` is **not called**.
New embedding rows are mean-initialized for both the input embedding table and the LM head.

```python
def load_model_and_tokenizer(args):
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    # ── No get_peft_model() → all 866M parameters are trained ──

    # Add semantic tokens
    new_tokens = build_semantic_tokens()
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {num_added} new tokens. New vocab size: {len(tokenizer)}")

    # Resize embedding matrix
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    # Mean-init new rows for both input embeddings and LM head (output embeddings)
    with torch.no_grad():
        mean_emb = model.get_input_embeddings().weight[:old_vocab_size].mean(dim=0)
        model.get_input_embeddings().weight[old_vocab_size:] = mean_emb

        mean_lm = model.get_output_embeddings().weight[:old_vocab_size].mean(dim=0)
        model.get_output_embeddings().weight[old_vocab_size:] = mean_lm

    return model, tokenizer
```

---

### Section 4 — Task 1 Dataset Preparation

**Task:** Given product images + text metadata, predict the semantic ID string.

- Items with `image_main is None` (~240) are filtered out.
- `image_pt01` and `image_pt02` are included as additional image content blocks only when not None.
- Image content blocks appear **before** the text block.

```python
def prepare_task1_dataset():
    items = load_dataset(
        "theblackcat102/amazon-all-beauty-filtered", "items", split="train"
    )
    # Filter items missing the main image
    items = items.filter(lambda x: x["image_main"] is not None)

    converted = []
    for sample in items:
        content = []
        # Add images first (main always present; pt01, pt02 optional)
        for img in [sample["image_main"], sample["image_pt01"], sample["image_pt02"]]:
            if img is not None:
                content.append({"type": "image", "image": img})

        content.append({
            "type": "text",
            "text": (
                f"Product title: {sample['title']}\n"
                f"Details: {sample['details']}\n"
                "Predict the semantic ID of this product:"
            ),
        })

        converted.append({"messages": [
            {"role": "user",      "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": sample["semantic_id"]}]},
        ]})

    return converted
```

**Example conversation:**

```
User:      [image_main] [image_pt01]
           "Product title: Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)
            Details: {"Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds", ...}
            Predict the semantic ID of this product:"
Assistant: "<|d0_101|> <|d1_57|> <|d2_29|>"
```

---

### Section 5 — Task 2 Dataset Preparation

**Task:** Given a user's ordered interaction history (as semantic IDs), predict the next item's semantic ID.
Text-only — no image content blocks.

```python
def prepare_task2_dataset():
    interactions = load_dataset(
        "theblackcat102/amazon-all-beauty-filtered", "interactions", split="train"
    )

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

    return converted
```

**Example conversation:**

```
User:      "User interaction history:
            1. <|d0_50|> <|d1_102|> <|d2_225|>
            2. <|d0_50|> <|d1_150|> <|d2_11|>
            3. <|d0_153|> <|d1_80|> <|d2_61|>
            4. <|d0_123|> <|d1_38|> <|d2_129|>
            5. <|d0_242|> <|d1_167|> <|d2_12|>
            Predict the next item's semantic ID:"
Assistant: "<|d0_115|> <|d1_131|> <|d2_88|>"
```

---

### Section 6 — Trainer Construction

The four `SFTConfig` flags marked MANDATORY below must always be set for multimodal SFT
with `UnslothVisionDataCollator`. Removing any one of them will cause a crash or silent
data corruption.

```python
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
            report_to="none",
            # ── MANDATORY for multimodal SFT ──────────────────────────────────
            remove_unused_columns=False,               # keeps 'image' column alive in dataset
            dataset_text_field="",                     # no single text column; multimodal format
            dataset_kwargs={"skip_prepare_dataset": True},  # bypass SFT's default tokenization
            max_length=2048,                           # max token sequence length
            # ─────────────────────────────────────────────────────────────────
        ),
    )
```

**Why each flag is mandatory:**

| Flag | Consequence if removed |
|------|------------------------|
| `remove_unused_columns=False` | HuggingFace drops the `image` key → `UnslothVisionDataCollator` crashes with KeyError |
| `dataset_text_field=""` | SFTTrainer tries to find a text column and fails on dict-of-lists format |
| `dataset_kwargs={"skip_prepare_dataset": True}` | SFT's internal tokenizer runs before the collator, corrupting multimodal samples |
| `max_length=2048` | Without an explicit limit, sequences with 3 images can exceed GPU memory silently |

---

### Section 7 — main()

```python
def main():
    args = parse_args()

    # 1. Load model and tokenizer, add semantic tokens, resize embeddings
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. Prepare datasets for both tasks
    task1_data = prepare_task1_dataset()   # ~112,350 vision+text samples
    task2_data = prepare_task2_dataset()   # 5,621 text-only samples

    # 3. Joint training: simple list concatenation (SFTTrainer shuffles internally)
    combined = task1_data + task2_data
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

if __name__ == "__main__":
    main()
```

---

## Configuration Comparison

| Parameter | LoRA (notebook) | Full Weight (this plan) |
|-----------|----------------|------------------------|
| PEFT method | LoRA r=16, α=16 | None — all 866M params trained |
| Trainable params | 13.2M (1.52%) | 866M (100%) |
| Learning rate | 2e-4 | 1e-5 |
| Batch size | 2 | 1 |
| Gradient accumulation | 4 | 8 (effective bs = 8) |
| Optimizer | adamw_8bit | adamw_8bit |
| LR schedule | linear | cosine |
| Warmup steps | 5 | 10 |
| Gradient checkpointing | unsloth | unsloth |
| Max steps (first experiment) | 30 | 500 |
| Special tokens added | 0 | 1,024 |
| Embedding resize | No | Yes (mean-init) |
| Dataset | LaTeX OCR (68K) | Items (112K) + Interactions (5.6K) |
| Tasks | Image → LaTeX | Image+Text → SemanticID (Task 1) |
|  |  | History → SemanticID (Task 2) |
| Multi-image samples | No | Yes (1–3 images, Task 1) |
| Text-only samples | No | Yes (Task 2) |

---

## File to Create

```
train_qwen_full.py   ← single self-contained training script
```

Run with defaults (500 steps, lr=1e-5, effective batch=8):
```bash
python train_qwen_full.py
```

Run with custom settings:
```bash
python train_qwen_full.py \
  --max_steps 2000 \
  --lr 5e-6 \
  --batch_size 1 \
  --grad_accum 16 \
  --output_dir outputs/qwen_full_v1
```
