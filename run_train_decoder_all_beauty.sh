#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== [1/3] Training text modality ==="
python3 train_decoder.py configs/decoder_text_all_beauty.gin

echo "=== [2/3] Training image modality ==="
python3 train_decoder.py configs/decoder_image_all_beauty.gin

echo "=== [3/3] Training multimodal modality ==="
python3 train_decoder.py configs/decoder_multimodal_all_beauty.gin

echo "=== All done ==="
