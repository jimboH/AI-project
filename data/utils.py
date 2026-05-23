import torch
from itertools import cycle as _cycle


def batch_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to(v, device) for k, v in batch.items()}
    if hasattr(batch, "_fields"):  # NamedTuple
        return type(batch)(*[batch_to(f, device) for f in batch])
    return batch


def cycle(dataloader):
    return _cycle(dataloader)


def next_batch(dataloader_iter, device):
    batch = next(dataloader_iter)
    return batch_to(batch, device)
