#!/usr/bin/env bash
set -euo pipefail

STAGING="/work/u1304848/AI/project/datasets/images_upload_staging"
SRC="/work/u1304848/AI/project/datasets/images"
REPO="jimboHsueh/amazon2023-multimodal"
LOG="/work/u1304848/AI/project/datasets/upload_images.log"

exec > >(tee -a "$LOG") 2>&1

cd "$STAGING"

# All_Beauty copy already done — skip if already present
if [ ! -d "$STAGING/All_Beauty" ]; then
  echo "[$(date)] Copying All_Beauty..."
  cp -r "$SRC/All_Beauty" "$STAGING/All_Beauty"
fi

# Zip All_Beauty using Python (ZIP_STORED — no compression, fast for JPEGs)
if [ ! -f "$STAGING/All_Beauty.zip" ]; then
  echo "[$(date)] Zipping All_Beauty..."
  python3 - <<'PYEOF'
import zipfile, os, sys
src = "/work/u1304848/AI/project/datasets/images_upload_staging/All_Beauty"
out = "/work/u1304848/AI/project/datasets/images_upload_staging/All_Beauty.zip"
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for root, dirs, files in os.walk(src):
        for f in files:
            fp = os.path.join(root, f)
            arcname = os.path.relpath(fp, os.path.dirname(src))
            zf.write(fp, arcname)
print("Done.")
PYEOF
  echo "[$(date)] All_Beauty.zip done ($(du -sh "$STAGING/All_Beauty.zip" | cut -f1))"
else
  echo "[$(date)] All_Beauty.zip already exists, skipping"
fi

# Copy Musical_Instruments if not already done
if [ ! -d "$STAGING/Musical_Instruments" ]; then
  echo "[$(date)] Copying Musical_Instruments..."
  cp -r "$SRC/Musical_Instruments" "$STAGING/Musical_Instruments"
fi

# Zip Musical_Instruments
if [ ! -f "$STAGING/Musical_Instruments.zip" ]; then
  echo "[$(date)] Zipping Musical_Instruments..."
  python3 - <<'PYEOF'
import zipfile, os, sys
src = "/work/u1304848/AI/project/datasets/images_upload_staging/Musical_Instruments"
out = "/work/u1304848/AI/project/datasets/images_upload_staging/Musical_Instruments.zip"
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for root, dirs, files in os.walk(src):
        for f in files:
            fp = os.path.join(root, f)
            arcname = os.path.relpath(fp, os.path.dirname(src))
            zf.write(fp, arcname)
print("Done.")
PYEOF
  echo "[$(date)] Musical_Instruments.zip done ($(du -sh "$STAGING/Musical_Instruments.zip" | cut -f1))"
else
  echo "[$(date)] Musical_Instruments.zip already exists, skipping"
fi

# Upload to HuggingFace
echo "[$(date)] Uploading All_Beauty.zip to HuggingFace..."
hf upload "$REPO" "$STAGING/All_Beauty.zip" "images/All_Beauty.zip" --repo-type dataset
echo "[$(date)] All_Beauty.zip uploaded"

echo "[$(date)] Uploading Musical_Instruments.zip to HuggingFace..."
hf upload "$REPO" "$STAGING/Musical_Instruments.zip" "images/Musical_Instruments.zip" --repo-type dataset
echo "[$(date)] Musical_Instruments.zip uploaded"

echo "[$(date)] All done!"
