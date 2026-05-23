"""Image encoder using Qwen/Qwen3-VL-Embedding-2B.

Encodes product images into dense embeddings by passing image-only inputs
through the VL model. Missing images are handled gracefully by returning
zero vectors. Uses the same backbone as the text and multimodal encoders,
producing 1536-dim embeddings consistent across all modalities.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
BATCH_SIZE = 8


class ImageEncoder:
    """Encode product images using Qwen3-VL-Embedding-2B.

    Parameters
    ----------
    image_dir : Path
        Root directory containing category subdirectories with images.
        Expected layout: image_dir/{category}/{asin}_{variant}.jpg
    category : str
        Dataset category name (e.g. 'All_Beauty').
    model_name : str
        HuggingFace VL model ID.
    device : str or None
        Target device.
    embedding_dim : int or None
        If set, projects to this dimension. Otherwise uses the model's native dim (1536).
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

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.native_dim = (
            cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.text_config.hidden_size
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

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single PIL image through the VL model."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}],
            }
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state  # (1, T, D)
        mask = inputs.get(
            "attention_mask",
            torch.ones(last_hidden.shape[:2], device=self.device),
        ).unsqueeze(-1).float()
        emb = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb.float(), p=2, dim=-1)[0]  # (D,)

    @torch.no_grad()
    def encode(
        self,
        asins: List[str],
        batch_size: int = BATCH_SIZE,
    ) -> torch.Tensor:
        """Encode images for a list of ASINs.

        Items with missing or unreadable images get zero vectors.

        Returns
        -------
        Tensor of shape (N, embedding_dim)
        """
        all_embs = []

        for i in tqdm(range(0, len(asins), batch_size), desc="Encoding images"):
            batch_asins = asins[i : i + batch_size]
            batch_embs = torch.zeros(len(batch_asins), self.embedding_dim)

            for j, asin in enumerate(batch_asins):
                path = self._find_image_path(asin)
                if path is None:
                    continue
                image = self._load_image(path)
                if image is None:
                    continue

                try:
                    emb = self._encode_image(image)
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
