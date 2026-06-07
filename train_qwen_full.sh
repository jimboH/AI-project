#!/usr/bin/env bash
# train_qwen_full.sh — full-weight Qwen3.5-0.8B finetuning
# W&B project : gen-retrieval-decoder  (same as the rest of this repo)
# Usage:
#   bash train_qwen_full.sh               # full run (500 steps, all data)
#   bash train_qwen_full.sh --max_steps 2000 --output_dir outputs/qwen_v2
#   MAX_STEPS=50 bash train_qwen_full.sh  # override via env var
set -euo pipefail

# ── Resolve script directory so the script works from any CWD ────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate project virtualenv if present ────────────────────────────────────
if [[ -f "$SCRIPT_DIR/env/bin/activate" ]]; then
    source "$SCRIPT_DIR/env/bin/activate"
fi

# ── Configurable defaults (override with env vars or CLI args below) ──────────
MAX_STEPS="${MAX_STEPS:-500}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_full}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-0.8B}"
HISTORICAL_INPUTS="${HISTORICAL_INPUTS:-semantic_id}"
# Default history cap per mode.
# - semantic_id / text: no images, so the only sequence-length guard is this cap.
#   Without it, users with 100–500 interactions produce prompt strings with
#   thousands of special tokens, making DataLoader tokenisation 10–100× slower
#   than image mode (where compute_max_images() auto-limits to ~1 item).
#   20 items × ~8 tokens/item ≈ 160 extra tokens — well within a 4 096 budget.
# - image / multimodal: compute_max_images() enforces its own hard cap via
#   max_images; MAX_HISTORY_ITEMS here just adds an additional coarser guard.
if [[ -z "${MAX_HISTORY_ITEMS:-}" ]]; then
    if [[ "$HISTORICAL_INPUTS" == "semantic_id" || "$HISTORICAL_INPUTS" == "text" ]]; then
        MAX_HISTORY_ITEMS=20
    fi
fi
MAX_HISTORY_ITEMS="${MAX_HISTORY_ITEMS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-gen-retrieval-decoder}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_full_hist=${HISTORICAL_INPUTS}_${MAX_STEPS}steps}"
# Each worker forks the parent process (~1.7 GB RAM inherited per worker).
# With persistent_workers=True all workers spawn at DataLoader creation and
# immediately fill the prefetch queue.
# RAM budget: 16 workers × 1.7 GB = 27 GB on a 33 GB machine → OOM / hang at step 0.
# Keep DATALOADER_WORKERS × 1.7 GB well below free RAM (free RAM ≈ 14 GB after model load).
# 4 workers = safe ceiling for 32 GB machines at any image_size or batch_size.
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# Task-1 / Task-2 sample caps (empty = use all data)
MAX_TASK1="${MAX_TASK1:-}"
MAX_TASK2="${MAX_TASK2:-}"

# ── Build optional args ───────────────────────────────────────────────────────
EXTRA_ARGS=()
[[ -n "$MAX_TASK1" ]] && EXTRA_ARGS+=(--max_task1_samples "$MAX_TASK1")
[[ -n "$MAX_TASK2" ]] && EXTRA_ARGS+=(--max_task2_samples "$MAX_TASK2")
[[ -n "$MAX_HISTORY_ITEMS" ]] && EXTRA_ARGS+=(--max_history_items "$MAX_HISTORY_ITEMS")

# Pass any CLI arguments straight through to the Python script
EXTRA_ARGS+=("$@")

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " Qwen3.5-0.8B full-weight finetuning"
echo "============================================================"
echo "  model        : $MODEL_NAME"
echo "  max_steps    : $MAX_STEPS"
echo "  lr           : $LR"
echo "  batch_size   : $BATCH_SIZE"
echo "  grad_accum   : $GRAD_ACCUM  (effective bs = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  warmup_steps : $WARMUP_STEPS"
echo "  output_dir   : $OUTPUT_DIR"
echo "  image_size   : $IMAGE_SIZE"
echo "  hist_inputs  : $HISTORICAL_INPUTS"
[[ -n "$MAX_HISTORY_ITEMS" ]] && echo "  max_hist_items: $MAX_HISTORY_ITEMS"
echo "  dl workers   : $DATALOADER_WORKERS"
echo "  wandb project: $WANDB_PROJECT"
echo "  wandb run    : $WANDB_RUN_NAME"
[[ -n "$MAX_TASK1" ]] && echo "  task1 cap    : $MAX_TASK1 samples"
[[ -n "$MAX_TASK2" ]] && echo "  task2 cap    : $MAX_TASK2 samples"
echo "============================================================"
echo ""

# ── RAM snapshot before launch ────────────────────────────────────────────────
echo "[RAM before launch]"
free -h
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
python train_qwen_full.py \
    --max_steps         "$MAX_STEPS" \
    --lr                "$LR" \
    --batch_size        "$BATCH_SIZE" \
    --grad_accum        "$GRAD_ACCUM" \
    --warmup_steps      "$WARMUP_STEPS" \
    --output_dir        "$OUTPUT_DIR" \
    --max_length        "$MAX_LENGTH" \
    --image_size        "$IMAGE_SIZE" \
    --model_name        "$MODEL_NAME" \
    --dataloader_workers "$DATALOADER_WORKERS" \
    --historical_inputs "$HISTORICAL_INPUTS" \
    --wandb_project     "$WANDB_PROJECT" \
    --wandb_run_name    "$WANDB_RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# ── RAM snapshot after finish ─────────────────────────────────────────────────
echo ""
echo "[RAM after finish]"
free -h
