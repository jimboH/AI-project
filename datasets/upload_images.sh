#!/usr/bin/env bash
# upload_images.sh — resize images to TARGET_SIZE, zip, and upload to HuggingFace.
#
# Resizing happens once here so training workers never resize at runtime.
# At 512px each image is ~50–100 KB (vs ~2–4 MB native) and produces 256 visual
# tokens in Qwen3.5-VL — the same budget Unsloth defaults to for unknown sizes.
#
# Override TARGET_SIZE to use a different resolution:
#   TARGET_SIZE=256 bash upload_images.sh    # faster training, fewer tokens
#   TARGET_SIZE=768 bash upload_images.sh    # higher fidelity
set -euo pipefail

STAGING="/work/u1304848/AI/project/datasets/images_upload_staging"
SRC="/work/u1304848/AI/project/datasets/images"
REPO="jimboHsueh/amazon2023-multimodal"
LOG="/work/u1304848/AI/project/datasets/upload_images.log"
TARGET_SIZE="${TARGET_SIZE:-512}"   # px — resize longest side to this

exec > >(tee -a "$LOG") 2>&1

mkdir -p "$STAGING"

echo "============================================================"
echo " Image upload with pre-resize"
echo "  source      : $SRC"
echo "  staging     : $STAGING"
echo "  target_size : ${TARGET_SIZE}px (longest side)"
echo "  repo        : $REPO"
echo "============================================================"

# ── Helper: resize + zip a category in one Python pass ───────────────────────
# Reads directly from SRC — no raw cp step needed.
# Each image is resized with BILINEAR (2.3× faster than LANCZOS, same quality
# for training) and saved as JPEG quality=90 into the zip.
resize_and_zip() {
    local CATEGORY="$1"
    local SRC_DIR="$SRC/$CATEGORY"
    local OUT_ZIP="$STAGING/${CATEGORY}.zip"

    if [ -f "$OUT_ZIP" ]; then
        echo "[$(date)] $OUT_ZIP already exists — skipping"
        return
    fi

    echo "[$(date)] Resizing + zipping $CATEGORY (target=${TARGET_SIZE}px) …"
    python3 - <<PYEOF
import io, os, zipfile
from PIL import Image

src_dir     = "$SRC_DIR"
out_zip     = "$OUT_ZIP"
target_size = $TARGET_SIZE
resample    = Image.BILINEAR

exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
total = skipped = errors = 0

with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for root, dirs, files in os.walk(src_dir):
        dirs.sort()
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, os.path.dirname(src_dir))
            # Normalise extension to .jpg inside the zip (saves space, avoids
            # format confusion — JPEG is always the right format for product photos)
            arcname = os.path.splitext(arcname)[0] + ".jpg"
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    # Only downscale — never upscale small thumbnails
                    if max(w, h) > target_size:
                        scale = target_size / max(w, h)
                        img = img.resize(
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            resample,
                        )
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90, optimize=True)
                buf.seek(0)
                zf.writestr(arcname, buf.read())
                total += 1
                if total % 5000 == 0:
                    print(f"  {total} images processed …", flush=True)
            except Exception as e:
                print(f"  WARN: skipping {fpath}: {e}", flush=True)
                errors += 1

print(f"Done. {total} images resized to {target_size}px, {errors} errors, {skipped} skipped.")
PYEOF

    SIZE=$(du -sh "$OUT_ZIP" | cut -f1)
    echo "[$(date)] $CATEGORY.zip done ($SIZE)"
}

# ── Process each category ─────────────────────────────────────────────────────
resize_and_zip "All_Beauty"
resize_and_zip "Musical_Instruments"

# ── Upload to HuggingFace ─────────────────────────────────────────────────────
echo "[$(date)] Uploading All_Beauty.zip …"
hf upload "$REPO" "$STAGING/All_Beauty.zip" "images/All_Beauty.zip" --repo-type dataset
echo "[$(date)] All_Beauty.zip uploaded"

echo "[$(date)] Uploading Musical_Instruments.zip …"
hf upload "$REPO" "$STAGING/Musical_Instruments.zip" "images/Musical_Instruments.zip" --repo-type dataset
echo "[$(date)] Musical_Instruments.zip uploaded"

echo "[$(date)] All done!"
