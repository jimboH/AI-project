"""Text encoder using Qwen/Qwen3-VL-Embedding-2B.

Encodes item metadata text prompts into dense embeddings via
last-token mean pooling over the final hidden layer, then L2-normalised.
Uses the same VL backbone as the image and multimodal encoders so all
three modalities share a common 1536-dim embedding space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from typing import List, Optional
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
MAX_TEXT_LEN = 512
BATCH_SIZE = 16


class TextEncoder:
    """Encode text strings into fixed-size embeddings using Qwen3-VL-Embedding-2B.

    Text is formatted as a single-turn user message and processed through
    the VL model's chat template (image slots are omitted).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Defaults to Qwen/Qwen3-VL-Embedding-2B.
    device : str
        Target device ('cuda', 'cpu', etc.).
    embedding_dim : int or None
        If set, adds a linear projection layer to this dimensionality.
        If None, uses the model's native hidden size (1536).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
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

    def _build_inputs(self, texts: List[str]):
        """Build processor inputs for a batch of text-only items."""
        formatted = []
        for text in texts:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text[:MAX_TEXT_LEN]}],
                }
            ]
            formatted.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return self.processor(
            text=formatted,
            images=None,
            return_tensors="pt",
            padding=True,
        )

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE) -> torch.Tensor:
        """Encode a list of texts into embeddings.

        Returns
        -------
        Tensor of shape (N, embedding_dim)
        """
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = texts[i : i + batch_size]
            inputs = self._build_inputs(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, output_hidden_states=False)
            last_hidden = outputs.last_hidden_state  # (B, T, D)

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb.float(), p=2, dim=-1)
            if self.projection is not None:
                emb = self.projection(emb)
                emb = F.normalize(emb, p=2, dim=-1)

            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)

    @property
    def dim(self) -> int:
        return self.embedding_dim
