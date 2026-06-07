"""Image encoder using google/siglip-so400m-patch14-384.

Encodes product images into dense embeddings via the SigLIP vision tower,
which applies global average pooling followed by a learned projection,
producing L2-normalised 1152-dim embeddings aligned with the text tower.

Missing images are handled gracefully by returning zero vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 8


class ImageEncoder:
    """Encode product images using SigLIP's vision tower.

    Parameters
    ----------
    image_dir : Path
        Root directory containing category subdirectories with images.
        Expected layout: image_dir/{category}/{asin}_{variant}.jpg
    category : str
        Dataset category name (e.g. 'All_Beauty').
    model_name : str
        HuggingFace model ID.
    device : str or None
        Target device.
    embedding_dim : int or None
        If set, projects to this dimension. Otherwise uses the model's
        native projection dimension (1152).
    """

    VARIANT_PRIORITY = ["MAIN", "PT01", "PT02", "PT03"]

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

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.native_dim = getattr(
            cfg, "projection_dim",
            getattr(cfg.vision_config, "hidden_size", 1152),
        )
        self.embedding_dim = embedding_dim or self.native_dim

        self.projection = None
        if embedding_dim is not None and embedding_dim != self.native_dim:
            self.projection = torch.nn.Linear(
                self.native_dim, embedding_dim, bias=False
            ).to(self.device)

    def _find_image_path(self, asin: str) -> Optional[Path]:
        """Find the best available image for an ASIN."""
        for variant in self.VARIANT_PRIORITY:
            candidate = self.image_dir / f"{asin}_{variant}.jpg"
            if candidate.exists():
                return candidate
        for f in self.image_dir.glob(f"{asin}_*.jpg"):
            return f
        return None

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def _encode_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of PIL images in a single forward pass.

        Returns
        -------
        Tensor of shape (B, native_dim), L2-normalised.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # get_image_features returns the projected, pooled vision embedding.
        emb = self.model.get_image_features(**inputs)  # (B, native_dim)
        return F.normalize(emb.float(), p=2, dim=-1)

    @torch.no_grad()
    def encode(
        self,
        asins: List[str],
        batch_size: int = BATCH_SIZE,
    ) -> torch.Tensor:
        """Encode images for a list of ASINs.

        Items with missing or unreadable images get zero vectors.
        Items within a batch that have valid images are processed together
        in a single forward pass; items without images keep zero vectors.

        Returns
        -------
        Tensor of shape (N, embedding_dim)
        """
        all_embs = []

        for i in tqdm(range(0, len(asins), batch_size), desc="Encoding images"):
            batch_asins = asins[i : i + batch_size]
            batch_embs = torch.zeros(len(batch_asins), self.embedding_dim)

            # Collect valid (index, image) pairs for batched encoding
            valid_indices: List[int] = []
            valid_images: List[Image.Image] = []
            for j, asin in enumerate(batch_asins):
                path = self._find_image_path(asin)
                if path is None:
                    continue
                image = self._load_image(path)
                if image is None:
                    continue
                valid_indices.append(j)
                valid_images.append(image)

            if not valid_images:
                all_embs.append(batch_embs)
                continue

            try:
                embs = self._encode_batch(valid_images)  # (K, native_dim)
                if self.projection is not None:
                    embs = self.projection(embs)
                    embs = F.normalize(embs, p=2, dim=-1)
                for k, j in enumerate(valid_indices):
                    batch_embs[j] = embs[k].cpu()
            except Exception:
                # Fall back to item-by-item on batch failure
                for k, (j, image) in enumerate(zip(valid_indices, valid_images)):
                    try:
                        embs_single = self._encode_batch([image])  # (1, D)
                        emb = embs_single[0]
                        if self.projection is not None:
                            emb = self.projection(emb.unsqueeze(0))[0]
                            emb = F.normalize(emb, p=2, dim=-1)
                        batch_embs[j] = emb.cpu()
                    except Exception:
                        pass

            all_embs.append(batch_embs)

        return torch.cat(all_embs, dim=0)

    @property
    def dim(self) -> int:
        return self.embedding_dim
