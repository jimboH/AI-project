#!/bin/bash
# run_pipeline.sh — Full training and evaluation pipeline.
#
# Usage:
#   ./run_pipeline.sh [category] [modality]
#
#   category  : All_Beauty | Musical_Instruments  (default: All_Beauty)
#   modality  : text | image | multimodal | all   (default: all)
#
# Steps:
#   1. Precompute embeddings
#   2. Train RQ-VAE  (one per modality)
#   3. Train decoder (one per modality)
#   4. Run 3×3 evaluation grid

set -euo pipefail

CATEGORY="${1:-All_Beauty}"
MODALITY="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/env/bin/python3"

cd "$SCRIPT_DIR"

echo "=========================================="
echo " Generative Recommendation Pipeline"
echo " Category : $CATEGORY"
echo " Modality : $MODALITY"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: Precompute embeddings
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Precomputing embeddings..."
#$PYTHON precompute_embeddings.py --category "$CATEGORY" --modality "$MODALITY"

# ---------------------------------------------------------------------------
# Step 2: Train RQ-VAE (one per modality)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Training RQ-VAE..."
CATEGORY_LOWER=$(echo "$CATEGORY" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

run_rqvae() {
    local mod="$1"
    local config="configs/rqvae_${mod}_${CATEGORY_LOWER}.gin"
    if [ -f "$config" ]; then
        echo "  Training RQ-VAE [modality=$mod]..."
        $PYTHON train_rqvae.py "$config"
    else
        echo "  Config not found: $config — skipping"
    fi
}

#if [ "$MODALITY" = "all" ]; then
#    run_rqvae text
#    run_rqvae image
#    run_rqvae multimodal
#else
#    run_rqvae "$MODALITY"
#fi

# ---------------------------------------------------------------------------
# Step 3: Train decoder (one per modality)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Training decoder..."

run_decoder() {
    local mod="$1"
    local config="configs/decoder_${mod}_${CATEGORY_LOWER}.gin"
    if [ -f "$config" ]; then
        echo "  Training decoder [modality=$mod]..."
        $PYTHON train_decoder.py "$config"
    else
        echo "  Config not found: $config — skipping"
    fi
}

if [ "$MODALITY" = "all" ]; then
    # run_decoder text
    run_decoder image
    run_decoder multimodal
else
    run_decoder "$MODALITY"
fi

# ---------------------------------------------------------------------------
# Step 4: 3×3 evaluation grid
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Running 3×3 evaluation grid..."
$PYTHON evaluate_grid.py \
    --category "$CATEGORY" \
    --rqvae_dir out/rqvae/ \
    --decoder_dir out/decoder/ \
    --output_dir out/grid_results/

echo ""
echo "Pipeline complete. Results in out/grid_results/"
