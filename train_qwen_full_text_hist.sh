#!/usr/bin/env bash
# train_qwen_full_text_hist.sh — full-weight Qwen3.5-0.8B finetuning
#                                with text-based interaction history (Task 2)
# Usage:
#   bash train_qwen_full_text_hist.sh
#   MAX_STEPS=1000 bash train_qwen_full_text_hist.sh
#   bash train_qwen_full_text_hist.sh --max_steps 1000 --output_dir outputs/qwen_text_v2
#
# Multi-mode (each interaction → N samples, one per mode):
#   HISTORICAL_INPUTS=text,image,multimodal bash train_qwen_full_text_hist.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/env/bin/activate" ]]; then
    source "$SCRIPT_DIR/env/bin/activate"
fi

MAX_STEPS="${MAX_STEPS:-500}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_full_text_hist}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-0.8B}"
# Default to text-only; override with e.g. HISTORICAL_INPUTS=text,image,multimodal
HISTORICAL_INPUTS="${HISTORICAL_INPUTS:-text}"
WANDB_PROJECT="${WANDB_PROJECT:-gen-retrieval-decoder}"
# Replace commas with "+" so the run name is shell/URL-safe in multi-mode.
HIST_DISPLAY="${HISTORICAL_INPUTS//,/+}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_full_${MAX_STEPS}steps_hist=${HIST_DISPLAY}}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# Apply history cap when any text/semantic_id mode is active (no built-in
# image-budget guard for those modes).
MAX_HISTORY_ITEMS="${MAX_HISTORY_ITEMS:-}"
if [[ -z "$MAX_HISTORY_ITEMS" ]]; then
    if [[ "$HISTORICAL_INPUTS" == *"text"* || "$HISTORICAL_INPUTS" == *"semantic_id"* ]]; then
        MAX_HISTORY_ITEMS=20
    fi
fi

MAX_TASK1="${MAX_TASK1:-}"
MAX_TASK2="${MAX_TASK2:-}"

EXTRA_ARGS=()
[[ -n "$MAX_TASK1"         ]] && EXTRA_ARGS+=(--max_task1_samples  "$MAX_TASK1")
[[ -n "$MAX_TASK2"         ]] && EXTRA_ARGS+=(--max_task2_samples  "$MAX_TASK2")
[[ -n "$MAX_HISTORY_ITEMS" ]] && EXTRA_ARGS+=(--max_history_items  "$MAX_HISTORY_ITEMS")
EXTRA_ARGS+=("$@")

echo "============================================================"
echo " Qwen3.5-0.8B full-weight finetuning  [text history]"
echo "============================================================"
echo "  model            : $MODEL_NAME"
echo "  historical_inputs: $HISTORICAL_INPUTS"
[[ "$HISTORICAL_INPUTS" != "$HIST_DISPLAY" ]] && echo "  hist_display     : $HIST_DISPLAY  (commas → + in run name)" || true
echo "  max_steps        : $MAX_STEPS"
echo "  lr               : $LR"
echo "  batch_size       : $BATCH_SIZE"
echo "  grad_accum       : $GRAD_ACCUM  (effective bs = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  warmup_steps     : $WARMUP_STEPS"
echo "  output_dir       : $OUTPUT_DIR"
echo "  image_size       : $IMAGE_SIZE"
[[ -n "$MAX_HISTORY_ITEMS" ]] && echo "  max_hist_items   : $MAX_HISTORY_ITEMS" || true
echo "  dl workers       : $DATALOADER_WORKERS"
echo "  wandb project    : $WANDB_PROJECT"
echo "  wandb run        : $WANDB_RUN_NAME"
[[ -n "$MAX_TASK1" ]] && echo "  task1 cap        : $MAX_TASK1 samples" || true
[[ -n "$MAX_TASK2" ]] && echo "  task2 cap        : $MAX_TASK2 samples" || true
echo "============================================================"
echo ""

echo "[RAM before launch]"
free -h
echo ""

python train_qwen_full.py \
    --historical_inputs  "$HISTORICAL_INPUTS" \
    --max_steps          "$MAX_STEPS" \
    --lr                 "$LR" \
    --batch_size         "$BATCH_SIZE" \
    --grad_accum         "$GRAD_ACCUM" \
    --warmup_steps       "$WARMUP_STEPS" \
    --output_dir         "$OUTPUT_DIR" \
    --max_length         "$MAX_LENGTH" \
    --image_size         "$IMAGE_SIZE" \
    --model_name         "$MODEL_NAME" \
    --dataloader_workers "$DATALOADER_WORKERS" \
    --wandb_project      "$WANDB_PROJECT" \
    --wandb_run_name     "$WANDB_RUN_NAME" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "[RAM after finish]"
free -h
