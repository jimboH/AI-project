#!/work/u1304848/.conda/envs/AI1/bin/python3
"""
Train Residual Quantization (RQ) and generate semantic IDs.

Implements the same hierarchical quantization approach used in GRID
(Generative Recommendation with Semantic IDs), but operates on the
multimodal embeddings produced by generate_multimodal_embeddings.py.

Each item embedding is approximated as a sum of codebook vectors:
    emb ≈ c0[q0] + c1[q1] + ... + c_{L-1}[q_{L-1}]
where each layer l quantizes the residual left by layers 0..l-1.
The resulting tuple (q0, q1, ..., q_{L-1}) is the semantic ID.

FAISS KMeans is used for each quantization layer (fast GPU-accelerated
clustering when available, otherwise CPU).

Input:
  /work/u1304848/AI/project/outputs/multimodal_embeddings.npy  -- (N, D) float32
  /work/u1304848/AI/project/outputs/item_ids.json              -- list of N parent_asin

Output (all in /work/u1304848/AI/project/outputs/):
  rq_codebooks.npy    -- (n_levels, n_codes, D) float32  codebook centroids
  rq_codes.npy        -- (N, n_levels) int32              per-item semantic IDs
  semantic_ids.json   -- {parent_asin: [q0, q1, ..., q_{L-1}]}  human-readable map
  rq_stats.json       -- coverage / collision statistics

Usage:
  python3 generate_semantic_ids.py
  # or with the conda env explicitly:
  /work/u1304848/.conda/envs/AI1/bin/python3 generate_semantic_ids.py

Hyperparameters can be overridden via environment variables:
  RQ_N_LEVELS   number of quantization levels  (default: 3)
  RQ_N_CODES    codebook size per level         (default: 256)
  RQ_N_ITER     k-means iterations per level    (default: 50)
  RQ_SEED       random seed                     (default: 42)
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration  (override with env vars)
# ---------------------------------------------------------------------------
N_LEVELS = int(os.environ.get("RQ_N_LEVELS", 3))
N_CODES  = int(os.environ.get("RQ_N_CODES",  256))
N_ITER   = int(os.environ.get("RQ_N_ITER",   50))
SEED     = int(os.environ.get("RQ_SEED",     42))

OUTPUT_DIR = Path("/work/u1304848/AI/project/outputs")
EMB_PATH   = OUTPUT_DIR / "multimodal_embeddings.npy"
IDS_PATH   = OUTPUT_DIR / "item_ids.json"


# ---------------------------------------------------------------------------
# Residual Quantization
# ---------------------------------------------------------------------------

def normalize_l2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (in-place safe copy)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def kmeans_one_level(
    residuals: np.ndarray,
    n_codes: int,
    n_iter: int,
    seed: int,
    level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run FAISS KMeans on residuals.

    Returns
    -------
    codes      : (N,) int32  cluster assignment for each point
    centroids  : (n_codes, D) float32  learned centroids
    """
    d = residuals.shape[1]
    use_gpu = faiss.get_num_gpus() > 0

    km = faiss.Kmeans(
        d,
        n_codes,
        niter=n_iter,
        seed=seed + level,
        verbose=False,
        gpu=use_gpu,
        nredo=3,          # multiple restarts for stability
        max_points_per_centroid=500,
    )
    km.train(residuals.astype(np.float32))

    _, codes = km.index.search(residuals.astype(np.float32), 1)
    codes = codes.reshape(-1).astype(np.int32)
    centroids = km.centroids.astype(np.float32)   # (n_codes, D)
    return codes, centroids


def train_rq(
    embeddings: np.ndarray,
    n_levels: int,
    n_codes: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train Residual Quantization.

    At each level l the current residual is L2-normalised before clustering
    (matches GRID's normalize_residuals=True).

    Returns
    -------
    rq_codes     : (N, n_levels) int32
    rq_codebooks : (n_levels, n_codes, D) float32
    """
    N, D = embeddings.shape
    rq_codes     = np.zeros((N, n_levels), dtype=np.int32)
    rq_codebooks = np.zeros((n_levels, n_codes, D), dtype=np.float32)

    residuals = embeddings.copy().astype(np.float32)

    for level in tqdm(range(n_levels), desc="RQ levels"):
        # Normalise residuals before clustering (as in GRID ResidualQuantization)
        normed = normalize_l2(residuals)

        codes, centroids = kmeans_one_level(normed, n_codes, n_iter, seed, level)

        rq_codes[:, level]     = codes
        rq_codebooks[level]    = centroids

        # Subtract the matched centroid from the *normalised* residuals
        rq_residual_norm = normed - centroids[codes]
        residuals = rq_residual_norm          # carry normalised residuals forward

        coverage = np.unique(codes).shape[0] / n_codes
        mean_res = np.linalg.norm(residuals, axis=1).mean()
        print(
            f"  Level {level}: coverage={coverage:.3f}  "
            f"residual_norm={mean_res:.4f}"
        )

    return rq_codes, rq_codebooks


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(rq_codes: np.ndarray, item_ids: list) -> dict:
    N, L = rq_codes.shape
    unique_ids = set(tuple(row) for row in rq_codes)
    stats = {
        "n_items":       N,
        "n_levels":      L,
        "n_codes":       int(rq_codes.max()) + 1,
        "unique_sem_ids": len(unique_ids),
        "collision_rate": round(1.0 - len(unique_ids) / N, 4),
    }
    for l in range(L):
        u = np.unique(rq_codes[:, l]).shape[0]
        stats[f"level_{l}_coverage"] = round(u / (int(rq_codes[:, l].max()) + 1), 4)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    codes_path     = OUTPUT_DIR / "rq_codes.npy"
    codebooks_path = OUTPUT_DIR / "rq_codebooks.npy"
    semid_path     = OUTPUT_DIR / "semantic_ids.json"
    stats_path     = OUTPUT_DIR / "rq_stats.json"

    # Load embeddings
    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMB_PATH}. "
            "Run generate_multimodal_embeddings.py first."
        )
    print(f"Loading embeddings from {EMB_PATH} ...")
    embeddings = np.load(EMB_PATH).astype(np.float32)
    print(f"  Shape: {embeddings.shape}")

    # Load item IDs
    with open(IDS_PATH) as f:
        item_ids: list[str] = json.load(f)
    assert len(item_ids) == embeddings.shape[0], (
        f"ID/embedding count mismatch: {len(item_ids)} vs {embeddings.shape[0]}"
    )
    print(f"  {len(item_ids):,} items")

    # Skip if outputs already exist
    if codes_path.exists() and codebooks_path.exists():
        print("RQ outputs already exist. Loading and printing stats only.")
        rq_codes     = np.load(codes_path)
        rq_codebooks = np.load(codebooks_path)
    else:
        print(
            f"\nTraining RQ: {N_LEVELS} levels × {N_CODES} codes, "
            f"{N_ITER} k-means iters, seed={SEED}"
        )
        rq_codes, rq_codebooks = train_rq(
            embeddings, N_LEVELS, N_CODES, N_ITER, SEED
        )

        np.save(codes_path, rq_codes)
        np.save(codebooks_path, rq_codebooks)
        print(f"\nSaved rq_codes     → {codes_path}     shape={rq_codes.shape}")
        print(f"Saved rq_codebooks → {codebooks_path}  shape={rq_codebooks.shape}")

    # Build human-readable semantic ID map
    print("Building semantic ID map ...")
    semantic_ids: dict[str, list[int]] = {}
    for i, asin in enumerate(item_ids):
        semantic_ids[asin] = rq_codes[i].tolist()

    with open(semid_path, "w") as f:
        json.dump(semantic_ids, f)
    print(f"Saved semantic_ids → {semid_path}")

    # Compute and save statistics
    stats = compute_stats(rq_codes, item_ids)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n── RQ Statistics ──────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("────────────────────────────────────────────────────")
    print(f"\nDone. All outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
