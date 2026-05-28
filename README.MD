# Generative Recommendation with Multimodal Semantic IDs

A generative sequential recommendation system built on the Amazon Reviews 2023 dataset.
Items are encoded into discrete semantic IDs by a Residual-Quantised VAE (RQ-VAE), and a
Qwen/Qwen3.5-0.8B sequence model predicts the next item a user will interact with.

---

## Architecture Overview

```
Item metadata / images
        │
        ▼
┌─────────────────────────────┐
│  Qwen3-VL-Embedding-2B      │  ← shared backbone for all three modalities
│  text / image / multimodal  │    produces L2-normalised embeddings
└─────────────┬───────────────┘
              │ precomputed embeddings  (outputs/embeddings/)
              ▼
┌─────────────────────────────┐
│  RQ-VAE (3-layer VQ, STE)   │  ← trained per modality per category
│  embedding → 3 × code[0-255]│    EMA codebook updates; distance L2-norm
└─────────────┬───────────────┘
              │ semantic ID corpus  (outputs/rqvae/)
              ▼
┌─────────────────────────────┐
│  QwenRetrievalModel         │  ← trained per modality per category
│  Qwen/Qwen3.5-0.8B          │    encodes user history (semantic IDs) with
│                             │    causal self-attention; last-position hidden
│                             │    state drives 3 classification heads that
│                             │    predict next-item codes hierarchy by hierarchy
│                             │    (teacher forcing during training; top-k beam
│                             │    search with conditioning chain at inference)
└─────────────────────────────┘
```

---

## Models

| Component           | Model                          | Role                                      |
|---------------------|-------------------------------|-------------------------------------------|
| Text encoder        | Qwen/Qwen3-VL-Embedding-2B    | Embed item metadata text                  |
| Image encoder       | Qwen/Qwen3-VL-Embedding-2B    | Embed product images                      |
| Multimodal encoder  | Qwen/Qwen3-VL-Embedding-2B    | Embed text + images jointly (1536-dim)    |
| RQ-VAE              | Custom 3-layer VQ (MLP, STE)  | Quantise embeddings → semantic ID tuples  |
| Sequence model      | Qwen/Qwen3.5-0.8B             | Encode history + predict next-item IDs    |

All three embedding modalities share the same `Qwen/Qwen3-VL-Embedding-2B` backbone.
This enables the 3×3 cross-modal evaluation grid where a decoder trained on one
modality is evaluated with another modality's tokenisation.

---

## Repository Structure

```
project/
├── configs/                   # Gin configuration files
│   ├── rqvae_{mod}_{cat}.gin  # RQ-VAE training configs (2 categories × 3 modalities)
│   └── decoder_{mod}_{cat}.gin# Decoder training configs
├── data/
│   ├── amazon2023.py          # Dataset loading, leave-one-out splits, text-prompt builder
│   ├── schemas.py             # SeqBatch / TokenizedSeqBatch data classes
│   └── utils.py               # DataLoader helpers
├── datasets/                  # Raw metadata, product images, HF cache
├── distributions/
│   └── gumbel.py              # Gumbel-softmax for VQ training
├── embeddings/
│   ├── text_encoder.py        # TextEncoder — Qwen3-VL-Embedding-2B, text-only
│   ├── image_encoder.py       # ImageEncoder — Qwen3-VL-Embedding-2B, image-only
│   └── multimodal_encoder.py  # MultimodalEncoder — Qwen3-VL-Embedding-2B, text+image
├── evaluate/
│   └── metrics.py             # TopKAccumulator, NDCGAccumulator
├── init/
│   └── kmeans.py              # K-means codebook initialisation
├── modules/
│   ├── rqvae.py               # RQ-VAE model
│   ├── model.py               # QwenRetrievalModel (Qwen3.5-0.8B only)
│   ├── quantize.py            # Quantisation layer (STE / Gumbel / Rotation)
│   ├── encoder.py             # MLP encoder/decoder blocks
│   ├── loss.py                # Reconstruction and quantisation losses
│   ├── normalize.py           # Normalisation layers
│   ├── utils.py               # Training utilities
│   ├── scheduler/
│   │   └── inv_sqrt.py        # Inverse square-root LR scheduler
│   └── tokenizer/
│       └── semids.py          # SemanticIdTokenizer (frozen RQ-VAE wrapper)
├── outputs/                   # Training artefacts (gitignored)
│   ├── embeddings/            # Precomputed .pt embedding files
│   ├── rqvae/                 # RQ-VAE checkpoints
│   └── decoder/               # Decoder checkpoints
├── precompute_embeddings.py   # Step 1 — extract and cache embeddings
├── train_rqvae.py             # Step 2 — train RQ-VAE
├── train_decoder.py           # Step 3 — train sequence model
├── evaluate_grid.py           # 3×3 cross-modal evaluation grid
└── requirements.txt
```

---

## Dataset Splits (Leave-One-Out Protocol)

For every user, all interacted items are ordered by exact review timestamp.
Given a user's full sequence of *n* items [i₁, i₂, …, iₙ]:

| Split      | Input history          | Target item | Condition |
|------------|------------------------|-------------|-----------|
| **Test**   | [i₁, …, i_{n-1}]      | i_n         | n ≥ 3     |
| **Valid**  | [i₁, …, i_{n-2}]      | i_{n-1}     | n ≥ 3     |
| **Train**  | [i₁, …, i_{k-1}]      | i_k         | k ∈ {2,…,n-2} |

The training split generates **all valid subsequences** for each user (every prefix of length
1 to n-3 paired with its next item), rather than a single example per user.  This substantially
increases the number of training examples and removes the need for history sub-sampling
(`train_data_subsample = False` in the configs).

The splits are derived from the `5core_last_out_w_his` HuggingFace config:
the test split's `history` field (already timestamp-sorted by HuggingFace)
is concatenated with the `target` to reconstruct each user's full sequence,
then the leave-one-out protocol is applied.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: ≥ 24 GB VRAM for the 2B VL model)
- `conda` or `pip` environment

### Install

```bash
pip install -r requirements.txt
```

The two models are downloaded automatically from HuggingFace on first use:
- `Qwen/Qwen3-VL-Embedding-2B` (~5 GB)
- `Qwen/Qwen3.5-0.8B` (~1.5 GB)

### Data

Download Amazon Reviews 2023 product images and place them under:

```
datasets/images/{category}/{asin}_{variant}.jpg
```

Metadata is streamed directly from HuggingFace (`McAuley-Lab/Amazon-Reviews-2023`) and
cached locally in `datasets/hf_cache/`.

---

## Training Pipeline

### Full pipeline (one command)

```bash
./run_pipeline.sh All_Beauty all
# or
./run_pipeline.sh Musical_Instruments text
```

### Step-by-step

**Step 1 — Precompute embeddings**

Extracts embeddings for all items using `Qwen/Qwen3-VL-Embedding-2B`:

```bash
python3 precompute_embeddings.py --category All_Beauty --modality all
```

Outputs:
```
outputs/embeddings/All_Beauty/
  text_embeddings.pt
  image_embeddings.pt
  multimodal_embeddings.pt   (N, 1536)
  asins.json
```

**Step 2 — Train RQ-VAE**

Compresses embeddings into tuples of 3 discrete codes (0–255):

```bash
python3 train_rqvae.py configs/rqvae_text_all_beauty.gin
python3 train_rqvae.py configs/rqvae_image_all_beauty.gin
python3 train_rqvae.py configs/rqvae_multimodal_all_beauty.gin
```

- 30 000 iterations, batch size 640
- Codebook: 256 entries × 3 layers, STE straight-through estimator
- EMA codebook updates (`vae_use_ema = True`) prevent codebook collapse
- Distance L2 normalisation (`vae_distance_l2_normalize = True`) for directional lookups
- Checkpoints saved every 10 000 iterations; best by eval loss at `checkpoint_best.pt`

**Step 3 — Train sequence model**

Trains one `QwenRetrievalModel` per modality on leave-one-out interaction histories:

```bash
python3 train_decoder.py configs/decoder_text_all_beauty.gin
python3 train_decoder.py configs/decoder_image_all_beauty.gin
python3 train_decoder.py configs/decoder_multimodal_all_beauty.gin
```

- 25 000 iterations, effective batch size 64 (batch=16 × accumulate=4)
- Qwen3.5-0.8B applies causal self-attention over history semantic IDs
- Classification heads use **teacher forcing**: each head is conditioned on the
  ground-truth code from the previous codebook layer during training
- Validation (eval_loss, hit@k, NDCG@k on the **valid** split) runs every 2 500 iterations
- Training loss is logged every 100 iterations
- Checkpoints are saved every 2 500 iterations; the best checkpoint (`checkpoint_best.pt`) is
  updated whenever **NDCG@10** improves
- Progress is recorded in `train.log` (human-readable) and `metrics.jsonl` (JSON lines)
- Optional wandb logging (`wandb_logging = True`)

**Step 4 — 3×3 Cross-modal evaluation grid**

Tests all 9 (train-modality, test-modality) combinations on the **test** split:

```bash
python3 evaluate_grid.py \
    --category All_Beauty \
    --rqvae_dir out/rqvae/ \
    --decoder_dir out/decoder/ \
    --output_dir out/grid_results/
```

Additional flags (must match training configuration):

| Flag                      | Default              | Description                                      |
|---------------------------|---------------------|--------------------------------------------------|
| `--qwen_model_name`       | Qwen/Qwen3.5-0.8B   | Qwen backbone model name                         |
| `--top_k_for_generation`  | 20                  | Beam width for next-item generation              |
| `--freeze_encoder`        | False               | Whether the Qwen encoder was frozen at training  |
| `--should_add_sep_token`  | True                | Insert separator tokens between items            |
| `--num_user_bins`         | None                | Number of user embedding bins (if used)          |
| `--modalities`            | all three           | Restrict evaluation to specific modalities       |
| `--device`                | auto                | Compute device (cuda / cpu)                      |

---

## 3×3 Training and Testing Grid

Nine grid cells cover every combination of training modality and testing modality:

| Cell | Train modality | Test modality |
|------|---------------|---------------|
| 1    | text          | text          |
| 2    | image         | text          |
| 3    | multimodal    | text          |
| 4    | text          | image         |
| 5    | image         | image         |
| 6    | multimodal    | image         |
| 7    | text          | multimodal    |
| 8    | image         | multimodal    |
| 9    | multimodal    | multimodal    |

For cross-modal cells (train ≠ test), test-modality embeddings are projected through the
train-modality RQ-VAE to obtain semantic IDs.  If the embedding dimensions differ, test
embeddings are zero-padded or truncated to match the RQ-VAE input dimension.

---

## Configuration

Training is configured via [gin-config](https://github.com/google/gin-config) files.
Key parameters:

### RQ-VAE (`configs/rqvae_*.gin`)

| Parameter                    | Default      | Description                                           |
|------------------------------|-------------|-------------------------------------------------------|
| `vae_input_dim`              | 1536 / 2048 | Embedding dimension (modality-dependent)              |
| `vae_embed_dim`              | 32           | Codebook embedding dimension                         |
| `vae_hidden_dims`            | [768,512,256]| MLP encoder hidden layers                            |
| `vae_codebook_size`          | 256          | Entries per codebook layer                           |
| `vae_n_layers`               | 3            | Number of quantisation layers                        |
| `vae_codebook_mode`          | STE          | Quantisation estimator (STE / GUMBEL_SOFTMAX)        |
| `vae_use_ema`                | True         | EMA codebook updates (prevents codebook collapse)    |
| `vae_ema_decay`              | 0.99         | EMA decay factor                                     |
| `vae_ema_threshold`          | 1.0          | Dead-entry reseeding threshold                       |
| `vae_distance_l2_normalize`  | True         | L2-normalise encoder outputs and codebook entries    |
| `iterations`                 | 30000        | Training iterations                                  |
| `batch_size`                 | 640          | Batch size                                           |

### Decoder (`configs/decoder_*.gin`)

| Parameter                  | Default             | Description                                |
|----------------------------|--------------------|--------------------------------------------|
| `qwen_model_name`          | Qwen/Qwen3.5-0.8B  | Qwen backbone model                        |
| `freeze_encoder`           | False              | Freeze Qwen weights during training        |
| `top_k_for_generation`     | 20                 | Beam width for next-item generation        |
| `should_add_sep_token`     | True               | Insert separator tokens between items      |
| `train_data_subsample`     | False              | Random history sub-sampling (training)     |
| `iterations`               | 25000              | Training iterations                        |
| `batch_size`               | 16                 | Per-device batch size                      |
| `gradient_accumulate_every`| 4                  | Gradient accumulation steps                |
| `eval_every`               | 2500               | Validation frequency (iterations)          |
| `save_model_every`         | 2500               | Checkpoint save frequency (iterations)     |
| `log_every`                | 100                | Training loss log frequency (iterations)   |
| `run_name`                 | modality name      | Subdirectory name inside save_dir_root     |

---

## Encoder Details

All three encoder classes live in `embeddings/` and share the same interface:

```python
encoder.encode(texts_or_asins, ...)  # → Tensor (N, D)
encoder.dim                          # → embedding dimension
```

### TextEncoder

Builds a single-turn user message with text-only content and runs it through
`Qwen3-VL-Embedding-2B`. Mean-pools the final hidden state over non-padding tokens
and L2-normalises.

### ImageEncoder

Builds a single-turn user message with an image-only payload. Encodes items one at a
time; items with missing or unreadable images receive zero vectors. Mean-pools and
L2-normalises identical to the text encoder.

### MultimodalEncoder

Builds a message with up to 3 product images followed by truncated text metadata.
Items with no images fall back to text-only inputs. Produces 1536-dim embeddings.

---

## Sequence Model Details

`QwenRetrievalModel` in `modules/model.py`:

1. **Semantic ID embedding** — one embedding table maps shifted codebook IDs into
   Qwen's 1024-dim hidden space.

2. **Qwen3.5-0.8B** — takes `inputs_embeds` (bypassing its word-embedding layer)
   and applies causal self-attention over the history sequence.  Optional separator
   tokens are injected between items; an optional user embedding is prepended.

3. **Classification heads (training)** — three linear layers project the hidden state at
   the **last valid sequence position** to `codebook_size` logits and are trained with
   cross-entropy loss.  **Teacher forcing** is applied: each head beyond the first is
   conditioned on the ground-truth code from the previous codebook layer, added to
   the hidden state before the next head's projection.

4. **Beam search generation (inference)** — at inference time each head uses **top-k
   selection** (deterministic, not sampled) to pick `n_cands` candidates.  Validity is
   checked against the precomputed corpus codebook after each level.  Each level's
   hidden state is conditioned on the codes selected at the previous level (mirroring
   teacher forcing), so the hierarchy is decoded autoregressively.

---

## Evaluation

The 3×3 evaluation grid measures cross-modal generalisation.  Each cell corresponds to
a model trained on one embedding modality (rows) and evaluated with semantic IDs
derived from a different modality (columns).  Evaluation uses the **test** split
(absolute last item per user as target):

|              | Test: text | Test: image | Test: multimodal |
|-------------|-----------|-------------|-----------------|
| Train: text  | ✓          | ✓            | ✓                |
| Train: image | ✓          | ✓            | ✓                |
| Train: mm    | ✓          | ✓            | ✓                |

Reported metrics: **hit@1**, **hit@5**, **hit@10**, **NDCG@1**, **NDCG@5**, **NDCG@10**.

Results are saved as `{output_dir}/{category}_grid_results.json` and printed as a
formatted table to stdout.

---

## Supported Categories

| Category              | HF config name                    |
|-----------------------|----------------------------------|
| All Beauty            | `All_Beauty`                      |
| Musical Instruments   | `Musical_Instruments`             |

Additional categories can be added by extending `METADATA_FILES` and
`CATEGORY_TO_HF_CONFIG` in `data/amazon2023.py` and providing the corresponding
image directory.

---

## Citation

This codebase extends the generative retrieval paradigm from:

- **TIGER** — Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023.
- **RQ-VAE** — Lee et al., *Autoregressive Image Generation using Residual Quantization*, CVPR 2022.

The Amazon Reviews 2023 dataset is from:

- Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv 2024.
