# T5 Decoder in RQ-VAE-Recommender: Training and Inference

## Overview

The model (`EncoderDecoderRetrievalModel`) is a custom T5 encoder-decoder. The encoder reads the user's interaction history expressed as Semantic IDs; the decoder autoregressively generates the next item's Semantic ID — a tuple of `n_layers=3` integer codes produced by the pre-trained RQ-VAE.

---

## 1. What the Decoder Receives

Every item in the corpus is mapped to a 3-code tuple `(c₀, c₁, c₂)` by the frozen RQ-VAE, where each `cₕ ∈ [0, 255]`. These codes are hierarchical: `c₁` refines the residual left by `c₀`, and `c₂` refines what `c₁` left behind.

A single shared embedding table of size `[n_layers × codebook_size, d_model]` = `[768, 384]` covers all levels. Hierarchy `h`, code `c` maps to embedding index `h × 256 + c`. This is the `item_sid_embedding_table`.

A learned BOS (beginning-of-sequence) token is the decoder's starting signal, playing the same role as `<pad>` in standard T5 generation.

---

## 2. Training — Parallel Teacher Forcing

**All three codes are predicted in a single forward pass.**

The decoder input during training is constructed in `decoder_forward_pass` (lines 229–268):

```
Decoder input sequence:   [BOS, embed(c₀), embed(c₁)]   shape: [B, 3, d_model]
Decoder output sequence:  [h₀,  h₁,        h₂,   h₃]   shape: [B, 4, d_model]
```

The `[:, :-1]` slice in `forward()` (line 288) drops the last output `h₃`, leaving `[B, 3, d_model]`. Then three **independent** linear heads — one per hierarchy — each consume a different decoder position:

```python
for h in range(self.num_hierarchies):        # h = 0, 1, 2
    logits = self.decoder_mlp[h](decoder_output[:, h])   # [B, 256]
    h_loss  = CrossEntropy(logits, fut_ids[:, h])
```

| Decoder position | Hidden state used | Predicts | Target |
|---|---|---|---|
| 0 (BOS output → `h₀`) | `decoder_output[:, 0]` | `c₀` | `fut_ids[:, 0]` |
| 1 (c₀ output → `h₁`) | `decoder_output[:, 1]` | `c₁` | `fut_ids[:, 1]` |
| 2 (c₁ output → `h₂`) | `decoder_output[:, 2]` | `c₂` | `fut_ids[:, 2]` |

This is standard left-to-right teacher forcing: `h₀` has seen only BOS, `h₁` has seen BOS+c₀, `h₂` has seen BOS+c₀+c₁ — so each head has the ground-truth prefix as context, but the computation across all three heads is **parallel** (one T5 forward pass, not a loop). The total loss is the sum of three cross-entropy terms.

---

## 3. Inference — Sequential Autoregressive Decoding with KV-Cache

At inference, the decoder runs **three sequential steps**, one per hierarchy level, with beam search across candidates. One code level is resolved per step.

The loop in `generate()` (lines 301–391):

### Step h=0 — Predict c₀ from BOS

```python
# generated is None → use original encoder outputs (B samples)
dec_out, past_kv = decoder_forward_pass(
    future_ids=None,         # no prior codes yet
    encoder_output=enc_out,  # [B, seq_len, d_model]
    use_cache=True,
    past_key_values=empty_cache,
)
# inputs_embeds = just BOS → decoder runs [BOS] through T5Stack
# dec_out shape: [B, 1, d_model]
probas = softmax(decoder_mlp[0](dec_out[:, -1, :]))   # [B, 256]
samples = multinomial(probas, n_cands=64)              # [B, 64]
```

Each of the B inputs samples 64 candidate `c₀` values. Invalid ones (whose single-code prefix doesn't match any corpus item) are masked to `-inf`. The top-`k` survivors form `generated = [B, k, 1]`.

The cache is then **reset** (`past_kv = EncoderDecoderCache(...)`), because the encoder will be expanded to B×k for subsequent steps and the old B-sized KV entries are incompatible.

### Step h=1 — Predict c₁ given c₀

```python
# generated = [B, k, 1]; squeezed = [B*k, 1] (all selected c₀s)
# encoder expanded: rep_enc = enc_out.repeat_interleave(k, dim=0) → [B*k, seq_len, d_model]

dec_out, past_kv = decoder_forward_pass(
    future_ids=squeezed,        # [B*k, 1] — the selected c₀ for each beam
    encoder_output=rep_enc,
    use_cache=True,
    past_key_values=empty_cache,   # cache was just reset
)
# _is_cache_valid = False → prepend BOS
# inputs_embeds = [BOS, embed(c₀)] → [B*k, 2, d_model]
# Full forward pass through T5Stack; KV cache is now populated
# dec_out shape: [B*k, 2, d_model]
probas = softmax(decoder_mlp[1](dec_out[:, -1, :]))   # [B*k, 256]
```

Each beam now samples 64 candidate `c₁` values. Scores are accumulated with `log_probas` from h=0. The prefix `(c₀, c₁)` is validity-checked against the corpus. New top-k beams are selected; `past_kv.reorder_cache(parent_global)` re-orders the cached K/V tensors to match which parent beams survived.

### Step h=2 — Predict c₂ given c₀, c₁

```python
# generated = [B, k, 2]; squeezed = [B*k, 2]

dec_out, past_kv = decoder_forward_pass(
    future_ids=squeezed,         # [B*k, 2]
    encoder_output=rep_enc,
    use_cache=True,
    past_key_values=past_kv,     # valid cache from h=1 (has 2 positions cached)
)
# _is_cache_valid = True → only process the NEW last token
# inputs_embeds = inputs_embeds[:, -1:, :] = embed(c₁) only → [B*k, 1, d_model]
# Incremental decode with KV-cache
# dec_out shape: [B*k, 1, d_model]
probas = softmax(decoder_mlp[2](dec_out[:, -1, :]))   # [B*k, 256]
```

KV-cache is fully effective here: only `embed(c₁)` is processed as a new query; the attention keys/values from positions 0 (BOS) and 1 (c₀) are reused from cache. The final `generated = [B, k, 3]` contains the top-k complete item Semantic IDs.

---

## 4. Key Design Notes

**Why the cache reset after h=0.** At h=0, the encoder outputs are B-sized. After selecting k beams, the encoder is expanded to B×k for h=1 onward. The h=0 KV cache was built against the B-sized encoder's cross-attention, so it cannot be reused — it is discarded and a fresh full forward pass runs at h=1.

**One head per hierarchy, not one head for all.** There are 3 separate `decoder_mlp` projections (`[d_model → 256]` each). The rationale: the optimal logit space for `c₀` (coarse) differs from that of `c₂` (fine residual). Each head specialises on its own level.

**Training is parallel, inference is sequential.** Training is efficient because teacher forcing lets the decoder see the whole target sequence `[BOS, c₀, c₁]` at once and predict all three codes in one pass. Inference must be sequential because each code depends on the one before it; this is the standard autoregressive trade-off.

**Training vs. inference mismatch (exposure bias).** The model trained here uses independent cross-entropy heads — each head sees only the decoder's self-attention hidden state conditioned on the ground-truth prefix, not the previous head's *predictions*. This is precisely the "independent classification heads" problem described in `train_modify.md` for the current project, which motivates the teacher-forcing fix already applied there.
