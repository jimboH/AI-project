"""Qwen/Qwen3.5-0.8B-only sequential recommendation model.

User interaction history is represented as semantic-ID sequences and fed
as inputs_embeds to Qwen3.5-0.8B (causal self-attention).  The hidden
state at the last valid sequence position is passed to three linear
classification heads — one per RQ-VAE codebook layer — to predict the
next item's semantic IDs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schemas import TokenizedSeqBatch
from typing import NamedTuple, Optional
from torch import Tensor
from transformers import AutoModel, AutoConfig

torch.set_float32_matmul_precision("high")

QWEN_MODEL_NAME = "Qwen/Qwen3.5-0.8B"


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


def _strip_dedup_col(
    tensor: torch.Tensor, sem_ids_dim: int, n_layers: int
) -> torch.Tensor:
    B, total = tensor.shape
    N = total // sem_ids_dim
    return (
        tensor.view(B, N, sem_ids_dim)[:, :, :n_layers]
        .contiguous()
        .view(B, N * n_layers)
    )


class QwenRetrievalModel(nn.Module):
    """Qwen3.5-0.8B-only model for sequential recommendation.

    History semantic IDs are embedded and fed as inputs_embeds to
    Qwen3.5-0.8B (causal self-attention).  The hidden state at the last
    valid sequence position drives num_hierarchies linear classification
    heads that predict the next item's semantic IDs hierarchy by hierarchy.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        qwen_model_name: str = QWEN_MODEL_NAME,
        top_k_for_generation: int = 10,
        should_add_sep_token: bool = True,
        num_user_bins: Optional[int] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.top_k_for_generation = top_k_for_generation
        self.register_buffer("codebooks", codebooks)

        # Qwen3.5-0.8B backbone
        qwen_cfg = AutoConfig.from_pretrained(qwen_model_name, trust_remote_code=True)
        self.qwen_model = AutoModel.from_pretrained(
            qwen_model_name, trust_remote_code=True
        )
        qwen_hidden = getattr(qwen_cfg, "hidden_size", None) or qwen_cfg.text_config.hidden_size

        if freeze_encoder:
            for p in self.qwen_model.parameters():
                p.requires_grad_(False)

        # Embedding table: shifted semantic IDs → Qwen hidden space
        self.item_embedding = nn.Embedding(
            num_embeddings=num_embeddings_per_hierarchy * num_hierarchies,
            embedding_dim=qwen_hidden,
        )

        # Classification heads: last hidden state → per-hierarchy codebook logits
        self.output_mlp = nn.ModuleList(
            [
                nn.Linear(qwen_hidden, num_embeddings_per_hierarchy, bias=False)
                for _ in range(num_hierarchies)
            ]
        )

        # Optional separator token inserted between items in the input sequence
        self.sep_token = (
            nn.Parameter(torch.randn(1, qwen_hidden), requires_grad=True)
            if should_add_sep_token
            else None
        )

        # Optional per-user contextual embedding
        self.user_embedding = (
            nn.Embedding(num_user_bins, qwen_hidden) if num_user_bins else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")
        _, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )
        num_repeats = (num_cols + num_hierarchies - 1) // num_hierarchies
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]
        result = input_sids + repeated_offsets
        if attention_mask is not None:
            result = result * attention_mask
        return result

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ):
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count = seq_len // num_hierarchies
        reshaped_emb = id_embeddings.view(batch_size, item_count, num_hierarchies, -1)
        reshaped_mask = attention_mask.view(batch_size, item_count, num_hierarchies)
        sep = sep_token.unsqueeze(0).expand(batch_size, item_count, -1).unsqueeze(-2)
        id_embeddings = torch.cat([reshaped_emb, sep], dim=-2)
        attention_mask = torch.cat([reshaped_mask, reshaped_mask[:, :, [-1]]], dim=-1)
        return id_embeddings.reshape(batch_size, -1, emb_dim), attention_mask.reshape(
            batch_size, -1
        )

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)
        trimmed = self.codebooks[:, : prefix.shape[1]]
        results = []
        for i in range(0, prefix.shape[0], batch_size):
            batch = prefix[i : i + batch_size]
            results.append(
                (trimmed.unsqueeze(1) == batch.unsqueeze(0)).all(dim=2).any(dim=0)
            )
        return torch.cat(results)

    def _get_last_valid_hidden(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the hidden state at the last non-padding position for each batch item."""
        seq_lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)
        B, T, D = hidden_states.shape
        idx = seq_lengths.view(B, 1, 1).expand(-1, 1, D)
        return hidden_states.gather(1, idx).squeeze(1)  # (B, D)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
    ):
        """Embed history semantic IDs and run through Qwen3.5-0.8B.

        Returns (hidden_states, attention_mask) where attention_mask may be
        wider than the input due to injected separator tokens and user embedding.
        """
        shifted = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.item_embedding(shifted)  # (B, T, qwen_hidden)

        if self.sep_token is not None:
            inputs_embeds, attention_mask = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        if user_id is not None and self.user_embedding is not None:
            user_embeds = self.user_embedding(
                torch.remainder(user_id[:, 0], self.user_embedding.num_embeddings)
            )
            inputs_embeds = torch.cat([user_embeds.unsqueeze(1), inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.size(0), 1, device=attention_mask.device
                    ),
                    attention_mask,
                ],
                dim=1,
            )

        model_dtype = next(self.qwen_model.parameters()).dtype
        hidden_states = self.qwen_model(
            inputs_embeds=inputs_embeds.to(model_dtype),
            attention_mask=attention_mask,
        ).last_hidden_state.float()  # (B, T, qwen_hidden)

        return hidden_states, attention_mask

    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        sem_ids_dim = self.num_hierarchies + 1
        input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, self.num_hierarchies)
        attention_mask = _strip_dedup_col(
            batch.seq_mask.long(), sem_ids_dim, self.num_hierarchies
        )
        fut_ids = batch.sem_ids_fut[:, : self.num_hierarchies]

        hidden_states, seq_mask = self._encode(input_ids, attention_mask, batch.user_ids)
        last_hidden = self._get_last_valid_hidden(hidden_states, seq_mask)  # (B, D)

        total_loss = torch.tensor(0.0, device=last_hidden.device)
        loss_d = []
        for h in range(self.num_hierarchies):
            logits = self.output_mlp[h](last_hidden)  # (B, num_embeddings_per_hierarchy)
            h_loss = F.cross_entropy(logits, fut_ids[:, h].long())
            total_loss = total_loss + h_loss
            loss_d.append(h_loss.detach())

        return ModelOutput(loss=total_loss, logits=None, loss_d=torch.stack(loss_d))

    @torch.no_grad()
    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
    ):
        B = input_ids.size(0)
        k = self.top_k_for_generation
        n_cands = min(64, self.num_embeddings_per_hierarchy)

        hidden_states, seq_mask = self._encode(input_ids, attention_mask, user_id)
        last_hidden = self._get_last_valid_hidden(hidden_states, seq_mask)  # (B, D)

        generated = None
        log_probas = None

        for h in range(self.num_hierarchies):
            if h == 0:
                probas = F.softmax(self.output_mlp[h](last_hidden), dim=-1)  # (B, E)
                samples = torch.multinomial(probas, num_samples=n_cands)  # (B, n_cands)
                samp_log_p = torch.log(
                    torch.gather(probas, 1, samples) + 1e-12
                )  # (B, n_cands)

                is_valid = self._check_valid_prefix(
                    samples.reshape(-1, 1)
                ).reshape(B, n_cands)
                scores, idx = samp_log_p.masked_fill(
                    ~is_valid, float("-inf")
                ).sort(-1, descending=True)
                top_k_idx = idx[:, :k]
                generated = torch.gather(samples, 1, top_k_idx).unsqueeze(-1)  # (B, k, 1)
                log_probas = scores[:, :k]  # (B, k)

                # Expand last_hidden once for all k beams; same encoding for every beam
                last_hidden = last_hidden.repeat_interleave(k, dim=0)  # (B*k, D)
            else:
                probas = F.softmax(
                    self.output_mlp[h](last_hidden), dim=-1
                )  # (B*k, E)
                samples = torch.multinomial(
                    probas, num_samples=n_cands
                )  # (B*k, n_cands)
                samp_log_p = torch.log(torch.gather(probas, 1, samples) + 1e-12)

                prev = generated.reshape(B * k, h)
                prefix = torch.cat(
                    [prev.repeat_interleave(n_cands, dim=0), samples.reshape(-1, 1)],
                    dim=1,
                )
                is_valid = self._check_valid_prefix(prefix).reshape(B, k * n_cands)

                scores = (
                    samp_log_p.reshape(B, k * n_cands)
                    + log_probas.repeat_interleave(n_cands, dim=1)
                )
                scores = scores.masked_fill(~is_valid, float("-inf"))
                scores, idx = scores.sort(-1, descending=True)

                top_k_idx = idx[:, :k]
                parent_beam_idx = top_k_idx // n_cands

                parent_ids = torch.gather(
                    generated,
                    1,
                    parent_beam_idx.unsqueeze(-1).expand(-1, -1, h),
                )
                new_ids = torch.gather(
                    samples.reshape(B, k * n_cands), 1, top_k_idx
                ).unsqueeze(-1)
                generated = torch.cat([parent_ids, new_ids], dim=-1)
                log_probas = scores[:, :k]

        return generated, log_probas

    @torch.no_grad()
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        top_k: bool = True,
        temperature: int = 1,
    ) -> GenerationOutput:
        sem_ids_dim = self.num_hierarchies + 1
        input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, self.num_hierarchies)
        attention_mask = _strip_dedup_col(
            batch.seq_mask.long(), sem_ids_dim, self.num_hierarchies
        )
        generated_ids, log_probas = self.generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
        )
        return GenerationOutput(sem_ids=generated_ids, log_probas=log_probas)
