"""Multimodal encoder using google/siglip-so400m-patch14-384.

Combines text (item metadata) and images into joint embeddings by computing
both modality features from SigLIP's shared model and returning their
L2-normalised sum.  Since SigLIP is a contrastive model, text and image
embeddings already occupy the same 1152-dim space, so a simple unit-norm
average is a principled fusion strategy.

When an item has no image, the text embedding is returned unchanged.
Both modalities are computed in a single model load to save GPU memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 8

VARIANT_PRIORITY = ["MAIN", "PT01", "PT02", "PT03"]


class MultimodalEncoder:
    """Encode items using both text metadata and product images via SigLIP.

    For each item the text and image embeddings are each computed via the
    respective SigLIP tower, then fused as:

        multimodal_emb = L2_normalize(text_emb + image_emb)

    Items with no image fall back to the text embedding alone.

    Parameters
    ----------
    image_dir : Path
        Root directory: image_dir/{category}/{asin}_{variant}.jpg
    category : str
        Dataset category name (e.g. 'All_Beauty').
    model_name : str
        HuggingFace model ID for the SigLIP model.
    device : str or None
        Target device.
    embedding_dim : int or None
        Output dimension; projects if different from model native dim (1152).
    """

    def __init__(
        self,
        image_dir: Path,
        category: str,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.image_dir = Path(image_dir) / category
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Single model load — SigLIP text and vision towers share one checkpoint
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.native_dim = getattr(
            cfg, "projection_dim",
            getattr(cfg.text_config, "hidden_size", 1152),
        )
        self.embedding_dim = embedding_dim or self.native_dim

        self.projection = None
        if embedding_dim is not None and embedding_dim != self.native_dim:
            self.projection = torch.nn.Linear(
                self.native_dim, embedding_dim, bias=False
            ).to(self.device)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _find_images(self, asin: str) -> List[Image.Image]:
        """Return up to one usable image for an ASIN (MAIN variant first)."""
        for variant in VARIANT_PRIORITY:
            path = self.image_dir / f"{asin}_{variant}.jpg"
            if path.exists():
                try:
                    return [Image.open(path).convert("RGB")]
                except Exception:
                    pass
        for f in self.image_dir.glob(f"{asin}_*.jpg"):
            try:
                return [Image.open(f).convert("RGB")]
            except Exception:
                pass
        return []

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of text strings. Returns (B, native_dim) L2-normed."""
        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)  # (B, native_dim)
        return F.normalize(emb.float(), p=2, dim=-1)

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL images. Returns (B, native_dim) L2-normed."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)  # (B, native_dim)
        return F.normalize(emb.float(), p=2, dim=-1)

    # ------------------------------------------------------------------
    # Public encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        asins: List[str],
        texts: List[str],
        batch_size: int = BATCH_SIZE,
    ) -> torch.Tensor:
        """Encode items using text + images (falls back to text-only if no image).

        For each item:
        - If image available: multimodal_emb = L2_normalize(text_emb + image_emb)
        - If no image: multimodal_emb = text_emb

        Parameters
        ----------
        asins : List[str]
            ASIN identifiers (used to find image files).
        texts : List[str]
            Pre-formatted text strings for each item.

        Returns
        -------
        Tensor of shape (N, embedding_dim)
        """
        all_embs = []

        for i in tqdm(range(0, len(asins), batch_size), desc="Encoding multimodal"):
            batch_asins = asins[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            B = len(batch_asins)

            # --- Text embeddings (always computed) ---
            try:
                txt_embs = self._encode_texts(batch_texts)  # (B, D)
            except Exception:
                # Fall back to item-by-item text encoding on failure
                txt_embs = torch.zeros(B, self.native_dim, device=self.device)
                for j, t in enumerate(batch_texts):
                    try:
                        txt_embs[j] = self._encode_texts([t])[0]
                    except Exception:
                        pass

            # --- Image embeddings (only for items that have an image) ---
            item_images: List[Optional[Image.Image]] = []
            for asin in batch_asins:
                imgs = self._find_images(asin)
                item_images.append(imgs[0] if imgs else None)

            # Collect valid image indices
            valid_idx = [j for j, img in enumerate(item_images) if img is not None]
            valid_imgs = [item_images[j] for j in valid_idx]

            img_embs = torch.zeros(B, self.native_dim, device=self.device)
            if valid_imgs:
                try:
                    encoded = self._encode_images(valid_imgs)  # (K, D)
                    for k, j in enumerate(valid_idx):
                        img_embs[j] = encoded[k]
                except Exception:
                    # Fall back item-by-item
                    for j, img in zip(valid_idx, valid_imgs):
                        try:
                            img_embs[j] = self._encode_images([img])[0]
                        except Exception:
                            pass

            # --- Fuse: average where image is available, else text-only ---
            batch_embs = torch.zeros(B, self.embedding_dim)
            for j in range(B):
                if item_images[j] is not None and img_embs[j].norm() > 0:
                    # L2-normalised sum (unit-norm average direction)
                    fused = F.normalize(
                        (txt_embs[j] + img_embs[j]).unsqueeze(0), p=2, dim=-1
                    )[0]
                else:
                    fused = txt_embs[j]

                if self.projection is not None:
                    fused = self.projection(fused.unsqueeze(0))[0]
                    fused = F.normalize(fused, p=2, dim=-1)

                batch_embs[j] = fused.cpu()

            all_embs.append(batch_embs)

        return torch.cat(all_embs, dim=0)

    @property
    def dim(self) -> int:
        return self.embedding_dim
