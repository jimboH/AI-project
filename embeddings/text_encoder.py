"""Text encoder using google/siglip-so400m-patch14-384.

Encodes item metadata text into dense embeddings via the SigLIP text tower,
which applies a learned CLS-style pooling and produces L2-normalised 1152-dim
embeddings aligned with the SigLIP vision tower.

Note: SigLIP's text tokenizer is capped at 64 tokens. Inputs are truncated
automatically by the processor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from typing import List, Optional
from tqdm import tqdm

MODEL_NAME = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 32


class TextEncoder:
    """Encode text strings into fixed-size embeddings using SigLIP's text tower.

    Text is passed directly to the SigLIP processor (no chat template).
    The processor handles tokenisation and padding/truncation to 64 tokens.
    Embeddings are L2-normalised pooled outputs from the text model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Defaults to google/siglip-so400m-patch14-384.
    device : str
        Target device ('cuda', 'cpu', etc.).
    embedding_dim : int or None
        If set, adds a linear projection layer to this dimensionality.
        If None, uses the model's native hidden size (1152).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self.model.eval()

        # SigLIP text projection dimension
        cfg = self.model.config
        # text_config.hidden_size is the projection output dim for SigLIP
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
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # SigLIP exposes get_text_features() which returns the projected,
            # L2-normalised pooled embedding directly.
            emb = self.model.get_text_features(**inputs)  # (B, native_dim)
            emb = F.normalize(emb.float(), p=2, dim=-1)

            if self.projection is not None:
                emb = self.projection(emb)
                emb = F.normalize(emb, p=2, dim=-1)

            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)

    @property
    def dim(self) -> int:
        return self.embedding_dim
