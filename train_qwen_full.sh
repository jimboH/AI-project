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
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_full}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-0.8B}"
WANDB_PROJECT="${WANDB_PROJECT:-gen-retrieval-decoder}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_full_${MAX_STEPS}steps}"
# Each worker forks ~1.7 GB RAM. Keep DATALOADER_WORKERS <= floor(free_RAM_GB / 2).
# On a 32 GB machine with other jobs running, 4 is safe; push to 8 if RAM is free.
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# Task-1 / Task-2 sample caps (empty = use all data)
MAX_TASK1="${MAX_TASK1:-}"
MAX_TASK2="${MAX_TASK2:-}"

# ── Build optional args ───────────────────────────────────────────────────────
EXTRA_ARGS=()
[[ -n "$MAX_TASK1" ]] && EXTRA_ARGS+=(--max_task1_samples "$MAX_TASK1")
[[ -n "$MAX_TASK2" ]] && EXTRA_ARGS+=(--max_task2_samples "$MAX_TASK2")

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
    --wandb_project     "$WANDB_PROJECT" \
    --wandb_run_name    "$WANDB_RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# ── RAM snapshot after finish ─────────────────────────────────────────────────
echo ""
echo "[RAM after finish]"
free -h
