"""Multimodal ICLGR model: SigLIP-so400m-patch14-224 + Qwen3.5-0.8B.

Architecture
------------
  SigLIP-so400m-patch14-224
    image (224×224) → 256 patch embeddings (hidden=1152)
         │
  Linear Projector  (1152 → qwen_hidden)
         │
  [img_tok_0 ... img_tok_255]        ← 256 visual tokens in Qwen space
         │
  Qwen3.5-0.8B (CausalLM)
    inputs_embeds = [img_embeds] + [text_embeds]
         │
  Causal LM head → next-token logits
         ▼
  <|d0_X|> <|d1_Y|> <|d2_Z|>   (trie-constrained at inference)

Key design decisions
--------------------
* SigLIP produces exactly 16×16=256 patch tokens when using
  siglip-so400m-patch14-224 (224 / 14 = 16 patches per side).
* Image features are prepended to text token embeddings via ``inputs_embeds``.
* Loss is computed only on the response tokens (doc_id sequence); prompt
  tokens receive label=-100.
* SigLIP is frozen by default; the projector and Qwen are trained.
  Pass ``freeze_siglip=False`` to also fine-tune SigLIP.
* LoRA can be applied to Qwen with ``lora_r > 0``.

Usage
-----
  from modules.multimodal_iclgr_model import MultimodalIclgrModel, build_special_tokens

  special_tokens = build_special_tokens()   # all <|d0_X|> … <|d2_255|>
  model = MultimodalIclgrModel(
      special_tokens=special_tokens,
      freeze_siglip=True,
      lora_r=16,
  )
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipImageProcessor,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-224"
QWEN_MODEL_NAME   = "Qwen/Qwen3.5-0.8B"

# SigLIP so400m-patch14-224: 224/14 = 16 patches per side → 256 total
N_IMAGE_TOKENS  = 256
SIGLIP_HIDDEN   = 1152   # hidden dim of siglip-so400m
# Qwen3.5-0.8B hidden dim — resolved dynamically from model config
# (typically 1024 for 0.8B)

N_CODEBOOK_LEVELS   = 3
N_CODES_PER_LEVEL   = 256   # <|d0_0|>…<|d0_255|>


# ── Special token helpers ─────────────────────────────────────────────────────

def build_special_tokens(
    n_levels: int = N_CODEBOOK_LEVELS,
    n_codes:  int = N_CODES_PER_LEVEL,
) -> List[str]:
    """Return all semantic-ID special token strings e.g. '<|d0_0|>'…'<|d2_255|>'."""
    return [
        f"<|d{level}_{code}|>"
        for level in range(n_levels)
        for code in range(n_codes)
    ]


def semid_str_to_token_ids(semid: str, tokenizer) -> List[int]:
    """Convert '<|d0_X|> <|d1_Y|> <|d2_Z|>' to a list of token ids."""
    tokens = re.findall(r"<\|d\d+_\d+\|>", semid)
    ids = []
    for tok in tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            ids.append(tid)
    return ids


# ── Model ─────────────────────────────────────────────────────────────────────

class MultimodalIclgrModel(nn.Module):
    """SigLIP-so400m + linear projector + Qwen3.5-0.8B CausalLM.

    Parameters
    ----------
    special_tokens : list of str
        Semantic-ID tokens to add to the tokenizer vocabulary.
    freeze_siglip : bool
        If True (default), SigLIP weights are frozen.
    lora_r : int
        LoRA rank to apply to Qwen attention layers.  0 = no LoRA (full fine-tune).
    qwen_model_name : str
        HuggingFace model name for the Qwen CausalLM backbone.
    siglip_model_name : str
        HuggingFace model name for SigLIP vision encoder.
    """

    def __init__(
        self,
        special_tokens:    List[str],
        freeze_siglip:     bool = True,
        load_siglip:       bool = True,
        lora_r:            int  = 16,
        qwen_model_name:   str  = QWEN_MODEL_NAME,
        siglip_model_name: str  = SIGLIP_MODEL_NAME,
    ) -> None:
        """
        Parameters
        ----------
        load_siglip : bool
            Set False for text-only training to skip loading the SigLIP
            encoder entirely (saves ~800 MB VRAM and speeds up startup).
            The image_processor attribute will still be set (needed by the
            collator) but ``self.siglip`` will be None.
        """
        super().__init__()

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add semantic-ID special tokens
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        print(f"[MultimodalIclgrModel] Added {n_added} special tokens to tokenizer.")

        # ── SigLIP vision encoder ─────────────────────────────────────────────
        self.image_processor = SiglipImageProcessor.from_pretrained(siglip_model_name)
        if load_siglip:
            self.siglip = SiglipVisionModel.from_pretrained(
                siglip_model_name,
                torch_dtype=torch.float16,
            )
            if freeze_siglip:
                for p in self.siglip.parameters():
                    p.requires_grad_(False)
                print("[MultimodalIclgrModel] SigLIP frozen.")
            siglip_hidden = self.siglip.config.hidden_size   # 1152
        else:
            self.siglip    = None
            siglip_hidden  = SIGLIP_HIDDEN
            print("[MultimodalIclgrModel] SigLIP NOT loaded (text-only mode).")

        # ── Qwen3.5-0.8B CausalLM ─────────────────────────────────────────────
        # attn_implementation="sdpa" uses PyTorch's built-in Flash Attention
        # kernel (no flash-attn package needed); falls back to eager if unsupported.
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )  # loads in the model's native dtype (bfloat16 for Qwen3.5)
        # Resize embedding table for the new special tokens
        self.qwen.resize_token_embeddings(len(self.tokenizer))

        qwen_hidden = self.qwen.config.hidden_size   # 1024 for 0.8B

        # ── Linear projector: SigLIP → Qwen ─────────────────────────────────
        self.image_projector = nn.Linear(siglip_hidden, qwen_hidden, bias=False)
        # Initialise close to identity-scale (stabilises early training)
        nn.init.normal_(self.image_projector.weight, std=0.02)
        # Cast to Qwen's dtype (bfloat16) so the projector runs efficiently
        # under autocast without unnecessary float32 promotion.
        qwen_dtype = next(self.qwen.parameters()).dtype
        self.image_projector = self.image_projector.to(qwen_dtype)

        # ── Optional LoRA on Qwen ──────────────────────────────────────────────
        if lora_r > 0:
            self._apply_lora(lora_r)

    # ── LoRA ──────────────────────────────────────────────────────────────────

    def _apply_lora(self, r: int) -> None:
        """Wrap Qwen attention q/k/v/o projections with LoRA adapters."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_cfg = LoraConfig(
                task_type    = TaskType.CAUSAL_LM,
                r            = r,
                lora_alpha   = r * 2,
                lora_dropout = 0.05,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                bias         = "none",
            )
            self.qwen = get_peft_model(self.qwen, lora_cfg)
            self.qwen.print_trainable_parameters()
            print(f"[MultimodalIclgrModel] LoRA applied (r={r}).")
        except ImportError:
            print(
                "[MultimodalIclgrModel] WARNING: peft not installed; "
                "running without LoRA (full fine-tune)."
            )

    # ── Image encoding ────────────────────────────────────────────────────────

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run SigLIP and project to Qwen hidden dim.

        Parameters
        ----------
        pixel_values : (B, C, H, W) float16 tensor already on the right device.

        Returns
        -------
        img_embeds : (B, N_IMAGE_TOKENS, qwen_hidden) float16 tensor.
        """
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.siglip.parameters())):
            vision_out = self.siglip(pixel_values=pixel_values)

        # last_hidden_state: (B, num_patches, siglip_hidden)
        patch_embeds = vision_out.last_hidden_state   # (B, 256, 1152)

        # Cast to projector dtype (bfloat16) — no float32 round-trip needed.
        img_embeds = self.image_projector(patch_embeds.to(self.image_projector.weight.dtype))
        return img_embeds   # (B, 256, qwen_hidden)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:       torch.Tensor,                   # (B, T_text)
        attention_mask:  torch.Tensor,                   # (B, T_text)
        labels:          torch.Tensor,                   # (B, T_text)  — -100 for prompt
        pixel_values:    Optional[torch.Tensor] = None,  # (B, C, H, W) or None
    ) -> torch.Tensor:
        """Compute causal-LM loss.

        Image tokens (if present) are prepended to text token embeddings.
        Loss is masked so only the doc_id response tokens contribute.

        Returns
        -------
        loss : scalar tensor.
        """
        B = input_ids.size(0)

        # ── Text embeddings ───────────────────────────────────────────────────
        embed_fn = (
            self.qwen.model.embed_tokens
            if hasattr(self.qwen, "model")
            else self.qwen.get_input_embeddings()
        )
        text_embeds = embed_fn(input_ids)   # (B, T_text, D)

        if pixel_values is not None:
            # ── Image embeddings ─────────────────────────────────────────────
            img_embeds = self.encode_images(pixel_values)   # (B, 256, D)

            # Concatenate [img | text]
            inputs_embeds   = torch.cat([img_embeds, text_embeds], dim=1)
            img_attn        = torch.ones(B, N_IMAGE_TOKENS,
                                         dtype=attention_mask.dtype,
                                         device=attention_mask.device)
            full_attn_mask  = torch.cat([img_attn, attention_mask], dim=1)

            # Extend labels: image positions get -100 (not supervised)
            img_labels      = torch.full(
                (B, N_IMAGE_TOKENS), -100,
                dtype=labels.dtype, device=labels.device
            )
            full_labels     = torch.cat([img_labels, labels], dim=1)
        else:
            inputs_embeds  = text_embeds
            full_attn_mask = attention_mask
            full_labels    = labels

        # ── Qwen forward ──────────────────────────────────────────────────────
        outputs = self.qwen(
            inputs_embeds  = inputs_embeds,
            attention_mask = full_attn_mask,
            labels         = full_labels,
        )
        return outputs.loss

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_docid(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        pixel_values:     Optional[torch.Tensor],
        trie_processor,
        num_beams:        int = 10,
        max_new_tokens:   int = 50,
    ) -> List[List[str]]:
        """Generate candidate doc_ids with trie-constrained beam search.

        Parameters
        ----------
        input_ids, attention_mask : prompt encoding (batch_size=1 recommended).
        pixel_values : (1, C, H, W) or None.
        trie_processor : TrieConstrainedLogitsProcessor instance (ICLGR).
        num_beams : beam width.
        max_new_tokens : generation budget.

        Returns
        -------
        List of decoded candidate strings per batch element.
        """
        from transformers import GenerationConfig, LogitsProcessorList

        B, T_text = input_ids.shape

        embed_fn = (
            self.qwen.model.embed_tokens
            if hasattr(self.qwen, "model")
            else self.qwen.get_input_embeddings()
        )
        text_embeds = embed_fn(input_ids)   # (B, T, D)

        if pixel_values is not None:
            img_embeds      = self.encode_images(pixel_values)    # (B, 256, D)
            inputs_embeds   = torch.cat([img_embeds, text_embeds], dim=1)
            img_attn        = torch.ones(B, N_IMAGE_TOKENS,
                                         dtype=attention_mask.dtype,
                                         device=attention_mask.device)
            full_attn_mask  = torch.cat([img_attn, attention_mask], dim=1)
            prompt_len      = N_IMAGE_TOKENS + T_text
        else:
            inputs_embeds  = text_embeds
            full_attn_mask = attention_mask
            prompt_len     = T_text

        # Re-create the trie processor with the correct prompt length
        from iclgr_src.src.inference_utils import TrieConstrainedLogitsProcessor
        processor = TrieConstrainedLogitsProcessor(
            trie_processor.trie_root,
            prompt_len,
            self.tokenizer.eos_token_id,
        )

        gen_cfg = GenerationConfig(
            max_new_tokens     = max_new_tokens,
            do_sample          = False,
            num_beams          = num_beams,
            num_return_sequences = num_beams,
            pad_token_id       = self.tokenizer.pad_token_id,
            eos_token_id       = self.tokenizer.eos_token_id,
        )

        outputs = self.qwen.generate(
            inputs_embeds  = inputs_embeds,
            attention_mask = full_attn_mask,
            generation_config     = gen_cfg,
            logits_processor      = LogitsProcessorList([processor]),
        )

        # Decode only the newly generated tokens
        generated_ids = outputs[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        return decoded

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save_pretrained(self, save_dir: str) -> None:
        """Save projector, Qwen (or LoRA adapter), and tokenizer."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Tokenizer
        self.tokenizer.save_pretrained(save_dir)

        # Projector
        torch.save(
            self.image_projector.state_dict(),
            f"{save_dir}/image_projector.pt",
        )

        # Qwen (saves LoRA adapter if peft was applied)
        if hasattr(self.qwen, "save_pretrained"):
            self.qwen.save_pretrained(save_dir)
        else:
            torch.save(self.qwen.state_dict(), f"{save_dir}/qwen.pt")

        print(f"[MultimodalIclgrModel] Saved to {save_dir}")

    @classmethod
    def load_pretrained(
        cls,
        save_dir: str,
        special_tokens: Optional[List[str]] = None,
        freeze_siglip: bool = True,
        lora_r: int = 0,
    ) -> "MultimodalIclgrModel":
        """Load a saved model from *save_dir*."""
        if special_tokens is None:
            special_tokens = build_special_tokens()

        model = cls(
            special_tokens = special_tokens,
            freeze_siglip  = freeze_siglip,
            lora_r         = lora_r,
        )

        # Projector
        proj_path = f"{save_dir}/image_projector.pt"
        model.image_projector.load_state_dict(
            torch.load(proj_path, map_location="cpu", weights_only=True)
        )

        # Qwen / LoRA
        try:
            from peft import PeftModel
            model.qwen = PeftModel.from_pretrained(model.qwen, save_dir)
        except Exception:
            qwen_state = torch.load(f"{save_dir}/qwen.pt", map_location="cpu")
            model.qwen.load_state_dict(qwen_state, strict=False)

        print(f"[MultimodalIclgrModel] Loaded from {save_dir}")
        return model
