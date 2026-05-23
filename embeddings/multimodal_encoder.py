"""Multimodal encoder using Qwen/Qwen3-VL-Embedding-2B.

Combines text (item metadata) and images into joint embeddings via mean
pooling over the final hidden layer, then L2-normalised. Uses the same
VL backbone as the text and image encoders so all three modalities produce
consistent 1536-dim embeddings. Missing images are handled gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
MAX_TEXT_LEN = 400
MAX_IMAGES_PER_ITEM = 3
BATCH_SIZE = 8

VARIANT_PRIORITY = ["MAIN", "PT01", "PT02", "PT03"]


class MultimodalEncoder:
    """Encode items using both text metadata and product images.

    Parameters
    ----------
    image_dir : Path
        Root directory: image_dir/{category}/{asin}_{variant}.jpg
    category : str
        Dataset category name (e.g. 'All_Beauty').
    model_name : str
        HuggingFace model ID for the multimodal embedding model.
    device : str or None
        Target device.
    embedding_dim : int or None
        Output dimension; projects if different from model native dim.
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

    def _find_images(self, asin: str) -> List[Image.Image]:
        images = []
        for variant in VARIANT_PRIORITY:
            path = self.image_dir / f"{asin}_{variant}.jpg"
            if path.exists():
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    pass
            if len(images) >= MAX_IMAGES_PER_ITEM:
                break
        return images

    def _build_content(self, text: str, images: List[Image.Image]) -> List[dict]:
        """Build multimodal content list for the processor."""
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text[:MAX_TEXT_LEN]})
        return content

    @torch.no_grad()
    def encode(
        self,
        asins: List[str],
        texts: List[str],
        batch_size: int = BATCH_SIZE,
    ) -> torch.Tensor:
        """Encode items using text + images.

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
            batch_embs = torch.zeros(len(batch_asins), self.embedding_dim)

            for j, (asin, text) in enumerate(zip(batch_asins, batch_texts)):
                images = self._find_images(asin)
                content = self._build_content(text, images)

                try:
                    messages = [{"role": "user", "content": content}]
                    text_input = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = self.processor(
                        text=[text_input],
                        images=images if images else None,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs, output_hidden_states=False)
                    last_hidden = outputs.last_hidden_state  # (1, T, D)
                    # Mean pool over all tokens
                    mask = inputs.get("attention_mask", torch.ones_like(
                        last_hidden[:, :, 0]
                    )).unsqueeze(-1).float()
                    emb = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    emb = F.normalize(emb.float(), p=2, dim=-1)

                    if self.projection is not None:
                        emb = self.projection(emb)
                        emb = F.normalize(emb, p=2, dim=-1)

                    batch_embs[j] = emb[0].cpu()

                except Exception as e:
                    # Item with encoding error gets zero vector
                    pass

            all_embs.append(batch_embs)

        return torch.cat(all_embs, dim=0)

    @property
    def dim(self) -> int:
        return self.embedding_dim
