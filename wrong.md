# Bug Report: Codebook Collapse and Related Issues

## Root Cause: Total Codebook Collapse

The RQ-VAE checkpoint at `out/rqvae/All_Beauty/text/` has completely collapsed. `out/rqvae/All_Beauty/text/metrics.jsonl` proves it — from iteration 9,999 onward:

```json
{"iter": 9999, "entropy": -0.0, "max_duplicates_frac": 0.9999911,
 "codebook_usage_0": 0.00390625, "codebook_usage_1": 0.00390625, "codebook_usage_2": 0.00390625}
```

`codebook_usage = 1/256` means exactly **1 out of 256 codebook entries is used per layer** across all 112,590 corpus items. Every item maps to `[155, 145, 216]`. This persists all the way to `iter=199999` (the end of training).

**Why `actual` is always `[155, 145, 216]`**: `precompute_corpus_ids()` runs the collapsed RQ-VAE over the entire corpus → `self.cached_ids` has all rows = `[155, 145, 216, dedup_count]`. Every `_tokenize_seq_batch_from_cached(batch.ids_fut)` lookup then returns `[155, 145, 216]` regardless of which item is the target.

**Why h@k = 1.0**: The decoder was trained on these collapsed IDs, so it learned to always output `[155, 145, 216]`. `_check_valid_prefix` validates it (it's a valid corpus prefix). Ground truth always matches → h@1 = h@5 = h@10 = 1.0 trivially.

---

## Bug 1 — No Dead Codebook Recovery (PRIMARY cause of collapse)

**Location**: `modules/quantize.py`, `modules/loss.py`

`QuantizeLoss.forward()` in `modules/loss.py`:
```python
emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])   # only updates codebook[ids]
query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
```

Only the codebook entry AT the assigned index `ids` receives a gradient. Once a codebook entry stops being selected (assignment collapses to a few winners), that entry receives **zero gradient forever**. There is no EMA codebook update, no random restart of dead vectors, and no codebook reset. Once the collapse begins, there is no recovery mechanism.

This is the primary structural defect. Standard VQ-VAE implementations (e.g., the original VQ-VAE-2, Jukebox) use exponential moving average (EMA) updates that also reset dead entries by reinitializing them from random encoder outputs.

---

## Bug 2 — Embedding Dimension Mismatch in Gin Config (contributes to collapse)

**Location**: `configs/rqvae_text_all_beauty.gin`

The config specifies `vae_input_dim = 1536`, but the actual text embeddings are `(112590, 2048)` — a 512-dimensional mismatch. `train_rqvae.py` auto-corrects this silently:
```python
if actual_dim != vae_input_dim:
    vae_input_dim = actual_dim
```

But this means the RQ-VAE was trained with a different architecture than the gin config describes, the encoder's first linear layer `Linear(2048, 768)` was trained rather than `Linear(1536, 768)`. The mismatch suggests either the wrong embedding model was used (2048-dim vs 1536-dim Qwen) or configs were never updated to match. This also silently changes the model's effective capacity and initialization scale, which can destabilize codebook training early on.

---

## Bug 3 — `gumbel_t` Is Dead Code in STE Mode (misleading, wastes a tuning knob)

**Location**: `modules/quantize.py`, `train_rqvae.py`

`train_rqvae.py` passes a temperature schedule (`gumbel_t=0.2`) to `model(batch, t)`. Inside `rqvae.py`, this is forwarded as `layer(res, temperature=gumbel_t)`. But `modules/quantize.py`:

```python
_, ids = (dist.detach()).min(axis=1)   # hard argmin — always used
if self.forward_mode == QuantizeForwardMode.STE:
    emb = self.get_item_embeddings(ids)
    emb_out = x + (emb - x).detach()  # straight-through gradient
```

In `STE` mode (the default, and what the gin config uses since `vae_codebook_mode` is not overridden), `temperature` is completely ignored. The `gumbel_t` parameter only has an effect in `QuantizeForwardMode.GUMBEL_SOFTMAX` mode. So all the temperature annealing logic in `train_rqvae.py` does nothing.

---

## Bug 4 — DataLoader Inconsistency in `precompute_corpus_ids`

**Location**: `modules/tokenizer/semids.py`

In `precompute_corpus_ids()`, the DataLoader is:
```python
DataLoader(
    item_dataset,
    sampler=BatchSampler(SequentialSampler(item_dataset), batch_size=640, drop_last=False),
    batch_size=1,          # ← should be None
    collate_fn=lambda batch: batch[0],
)
```

In `train_rqvae.py`, the correct pattern is `batch_size=None`. Using `batch_size=1` with a `BatchSampler` creates a double-wrapping: DataLoader wraps the already-batched sample in another list of size 1, which `collate_fn=lambda batch: batch[0]` then unwraps. It works, but only coincidentally. This is a latent bug — the `collate_fn` is masking the incorrect `batch_size` setting, and a refactor could break it silently.

---

## Bug 5 — Off-by-One Iteration in `train_rqvae.py`

**Location**: `train_rqvae.py`, training loop

```python
for iter in range(start_iter, start_iter + 1 + iterations):
```

`+ 1 + iterations` instead of `+ iterations` runs one extra iteration. Minor, but the checkpoint saved at the end corresponds to `iterations` steps, not `iterations - 1` as documented.

---

## Summary Table

| # | Bug | Location | Severity |
|---|-----|----------|----------|
| 1 | No dead codebook recovery (no EMA, no random restart) | `modules/quantize.py`, `modules/loss.py` | **Critical** — directly causes collapse |
| 2 | Gin config `vae_input_dim=1536` vs actual 2048-dim embeddings | `configs/rqvae_text_all_beauty.gin` | High — silent mismatch, contributes to instability |
| 3 | `gumbel_t` temperature ignored in STE mode | `modules/quantize.py`, `train_rqvae.py` | Medium — a tuning knob that does nothing |
| 4 | Double-wrapped BatchSampler in `precompute_corpus_ids` | `modules/tokenizer/semids.py` | Low — works by coincidence, fragile |
| 5 | Off-by-one `+ 1 + iterations` in training loop | `train_rqvae.py` | Low — runs one extra iter |

The fix for Bug 1 is the critical one: the RQ-VAE needs either EMA codebook updates with dead-entry reseeding, or a codebook reset triggered when `codebook_usage` drops below a threshold. Without it, collapse is essentially guaranteed on any sufficiently large dataset with random codebook initialization.
