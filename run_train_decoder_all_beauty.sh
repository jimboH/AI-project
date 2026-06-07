#!/bin/bash
# run_train_decoder_all_beauty.sh
#
# Full end-to-end pipeline for All_Beauty — no pseudo queries used at any step.
# Mirrors run_pipeline.sh exactly but is scoped to a single category.
#
# Steps:
#   1. Precompute embeddings  (text, image, multimodal)
#   2. Train RQ-VAE           (text, image, multimodal)
#   3. Train decoder          (text, image, multimodal)
#   4. Run 3x3 evaluation grid
#
# Usage:
#   ./run_train_decoder_all_beauty.sh

set -euo pipefail

CATEGORY="All_Beauty"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/env/bin/python3"

cd "$SCRIPT_DIR"

echo "=========================================="
echo " All_Beauty — Full Training Pipeline"
echo " (no pseudo queries)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: Precompute embeddings
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Precomputing embeddings (text, image, multimodal)..."
$PYTHON precompute_embeddings.py --category "$CATEGORY" --modality all

# ---------------------------------------------------------------------------
# Step 2: Train RQ-VAE (one per modality)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Training RQ-VAE..."

echo "  [2/4a] text..."
$PYTHON train_rqvae.py configs/rqvae_text_all_beauty.gin

echo "  [2/4b] image..."
$PYTHON train_rqvae.py configs/rqvae_image_all_beauty.gin

echo "  [2/4c] multimodal..."
$PYTHON train_rqvae.py configs/rqvae_multimodal_all_beauty.gin

# ---------------------------------------------------------------------------
# Step 3: Train decoder (one per modality)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Training decoder..."

echo "  [3/4a] text..."
$PYTHON train_decoder.py configs/decoder_text_all_beauty_direct.gin

echo "  [3/4b] image..."
$PYTHON train_decoder.py configs/decoder_image_all_beauty_direct.gin

echo "  [3/4c] multimodal..."
$PYTHON train_decoder.py configs/decoder_multimodal_all_beauty_direct.gin

# ---------------------------------------------------------------------------
# Step 4: Evaluation grid
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Running 3x3 evaluation grid..."
$PYTHON evaluate_grid.py \
    --category "$CATEGORY" \
    --rqvae_dir out/rqvae/ \
    --decoder_dir out/decoder/ \
    --output_dir out/grid_results/

echo ""
echo "Pipeline complete. Results in out/grid_results/"
