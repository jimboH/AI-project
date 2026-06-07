from typing import NamedTuple
from torch import Tensor

FUT_SUFFIX = "_fut"


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor


class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor


class PairBatch(NamedTuple):
    """Batch of (source, target) embedding pairs for cross-modal RQVAE training.

    ``x_src`` is the encoder input and ``x_tgt`` is the reconstruction target.
    For same-modal pairs (text→text, image→image) both tensors are identical.
    For cross-modal pairs (text→image, image→text) they differ.

    ``pair_type`` is an integer code identifying the pair:
        0  text  → text
        1  image → image
        2  text  → image
        3  image → text
    """

    x_src: Tensor       # (B, D) — source modality embedding
    x_tgt: Tensor       # (B, D) — target modality embedding
    pair_type: Tensor   # (B,)   — integer in {0, 1, 2, 3}
