"""Evaluation metrics for generative recommendation.

TopKAccumulator: accumulates hit@k metrics over batches.
NDCGAccumulator: accumulates NDCG@k metrics over batches.
"""

from collections import defaultdict
from einops import rearrange
from torch import Tensor
import torch
import math


class TopKAccumulator:
    """Accumulates hit@k (recall@k) metrics."""

    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        """
        Parameters
        ----------
        actual : Tensor [B, D]  -- ground-truth semantic ID per sample
        top_k  : Tensor [B, K, D] -- top-k predicted semantic IDs
        """
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        match_found, rank = pos_match.all(axis=-1).max(axis=-1)
        matched_rank = rank[match_found]
        for k in self.ks:
            self.metrics[f"h@{k}"] += len(matched_rank[matched_rank < k])
        self.total += B

    def reduce(self) -> dict:
        if self.total == 0:
            return {f"h@{k}": 0.0 for k in self.ks}
        return {key: val / self.total for key, val in self.metrics.items()}


class NDCGAccumulator:
    """Accumulates NDCG@k metrics."""

    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        match_found, rank = pos_match.all(axis=-1).max(axis=-1)

        for i in range(B):
            if match_found[i]:
                r = rank[i].item()
                for k in self.ks:
                    if r < k:
                        self.metrics[f"ndcg@{k}"] += 1.0 / math.log2(r + 2)
        self.total += B

    def reduce(self) -> dict:
        if self.total == 0:
            return {f"ndcg@{k}": 0.0 for k in self.ks}
        return {key: val / self.total for key, val in self.metrics.items()}


class CombinedMetricsAccumulator:
    """Combines TopK and NDCG accumulators."""

    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.topk = TopKAccumulator(ks)
        self.ndcg = NDCGAccumulator(ks)

    def reset(self):
        self.topk.reset()
        self.ndcg.reset()

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        self.topk.accumulate(actual, top_k)
        self.ndcg.accumulate(actual, top_k)

    def reduce(self) -> dict:
        return {**self.topk.reduce(), **self.ndcg.reduce()}
