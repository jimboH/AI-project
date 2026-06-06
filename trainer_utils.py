"""Trainer-side utilities: HalfHalfBatchSampler and TaskAwareSFTTrainer.

Kept in a separate module from ``train_qwen_full`` so that DataLoader worker
processes — which re-execute ``train_qwen_full.py`` as ``__mp_main__`` via
Python's spawn bootstrap — do NOT import TRL or wandb at startup.

``train_qwen_full.py`` imports from this module lazily (inside
``build_trainer()``), which only the main process calls.  Workers never call
``build_trainer()``, so they never import this file.
"""

import time as _time

import torch
import wandb
from trl import SFTTrainer, SFTConfig  # noqa: F401  (re-exported for build_trainer)


class HalfHalfBatchSampler(torch.utils.data.Sampler):
    """Yields batches containing exactly half Task 1 (items indexing) and half
    Task 2 (recsys interactions) samples in every batch.

    For batch_size B >= 2:
      - Each batch = ceil(B/2) Task-1 indices  +  floor(B/2) Task-2 indices.
      - The number of batches is driven by one full epoch over Task 1.
      - Task 2 is cycled (with a fresh shuffle each cycle) because it is
        typically ~20× smaller than Task 1.  This ensures the model sees
        recsys interactions at every gradient step throughout the epoch.

    For batch_size = 1:
      - True 50/50 within a single index is impossible, so the sampler
        strictly alternates: [T1], [T2], [T1], [T2], …
        Task 2 is cycled to match the Task 1 epoch length.
        With grad_accum >= 2 the effective ratio per accumulation window
        is still 50/50.

    Padding note: Task 1 samples contain vision tokens and produce much longer
    sequences than Task 2 text samples.  Mixing them in one batch causes Task 2
    samples to be padded to the Task 1 sequence length.  Keep
    per_device_train_batch_size small (1–4) to bound this overhead.
    """

    def __init__(
        self,
        task1_size: int,
        task2_size: int,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 3407,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.task1_size = task1_size
        self.task2_size = task2_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0  # incremented by Trainer when shuffling

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _cycle_t2(self, g: torch.Generator, needed: int) -> list:
        """Return *needed* Task-2 global indices by cycling with re-shuffles."""
        pool: list = []
        while len(pool) < needed:
            pool.extend(
                (torch.randperm(self.task2_size, generator=g) + self.task1_size).tolist()
            )
        return pool[:needed]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Task 1: one full epoch of shuffled global indices (0 … task1_size-1)
        t1_idx = torch.randperm(self.task1_size, generator=g).tolist()

        t1_per = (self.batch_size + 1) // 2   # ceil(B/2); == 1 when B == 1
        t2_per = self.batch_size // 2          # floor(B/2); == 0 when B == 1

        # ── batch_size == 1: strict alternation [T1], [T2], … ────────────────
        if t2_per == 0:
            t2_pool = self._cycle_t2(g, len(t1_idx))
            for t1, t2 in zip(t1_idx, t2_pool):
                yield [t1]
                yield [t2]
            return

        # ── batch_size >= 2: ceil/floor half-and-half within each batch ───────
        if self.drop_last:
            n_batches = len(t1_idx) // t1_per
        else:
            n_batches = (len(t1_idx) + t1_per - 1) // t1_per
            # Pad T1 to fill the final partial batch by cycling from its start
            shortfall = n_batches * t1_per - len(t1_idx)
            if shortfall > 0:
                t1_idx.extend(t1_idx[:shortfall])

        # Cycle Task 2 to cover every batch
        t2_pool = self._cycle_t2(g, n_batches * t2_per)

        for i in range(n_batches):
            t1_slice = t1_idx[i * t1_per : (i + 1) * t1_per]
            t2_slice = t2_pool[i * t2_per : (i + 1) * t2_per]
            yield t1_slice + t2_slice

    def __len__(self) -> int:
        t1_per = (self.batch_size + 1) // 2
        t2_per = self.batch_size // 2
        if t2_per == 0:
            # alternating single-index batches: one T1 + one T2 per T1 sample
            return self.task1_size * 2
        if self.drop_last:
            return self.task1_size // t1_per
        return (self.task1_size + t1_per - 1) // t1_per


class TaskAwareSFTTrainer(SFTTrainer):
    """SFTTrainer that uses HalfHalfBatchSampler to guarantee every batch
    contains equal numbers of Task-1 (items indexing, vision+text) and Task-2
    (recsys interactions) samples.

    Task 2 is ~20× smaller than Task 1, so it is cycled with re-shuffling
    throughout each epoch.  This ensures the model receives recsys gradient
    signal at every training step rather than only during the first ~5 % of
    an epoch.

    Padding trade-off: Task-1 sequences are longer (vision tokens) so Task-2
    samples within the same batch are padded to that length.  Keep
    per_device_train_batch_size small (1–4) to bound padding overhead.

    Also logs per-step GPU time vs collation time to W&B so the pipeline
    bottleneck (data vs compute) is visible.
    """

    def __init__(self, *args, task1_size: int = 0, task2_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.task1_size = task1_size
        self.task2_size = task2_size
        self._gpu_step_times: list[float] = []
        self._data_wait_times: list[float] = []
        self._last_batch_end: float | None = None
        # Pending CUDA event pairs for async GPU timing.
        self._pending_events: list = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Time from the end of the previous step to now = data wait time.
        now = _time.perf_counter()
        if self._last_batch_end is not None:
            self._data_wait_times.append((now - self._last_batch_end) * 1000)

        # Time the GPU step with CUDA Events so we never spin-wait in the
        # CPU while Torch Inductor/Triton is compiling kernels on the first
        # step.  cudaEventRecord() is non-blocking — the event is inserted
        # into the stream and signals when GPU reaches that point.
        # We defer elapsed_time() until the next logging window (LOG_EVERY
        # steps later), when we know the GPU is past those events.
        if torch.cuda.is_available():
            t_start = torch.cuda.Event(enable_timing=True)
            t_end   = torch.cuda.Event(enable_timing=True)
            t_start.record()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if torch.cuda.is_available():
            t_end.record()
            self._pending_events.append((t_start, t_end))

        self._last_batch_end = _time.perf_counter()

        # Log to W&B every 5 steps (same cadence as TimedCollator).
        LOG_EVERY = 5
        # Count pending events as proxy for step count (avoids a separate counter).
        n_pending = len(self._pending_events) if torch.cuda.is_available() else len(self._gpu_step_times)
        if n_pending >= LOG_EVERY:
            if torch.cuda.is_available() and self._pending_events:
                # One synchronize here is safe: all LOG_EVERY steps are done,
                # so the GPU is past both events and elapsed_time() is instant.
                torch.cuda.synchronize()
                gpu_times_ms = [s.elapsed_time(e) for s, e in self._pending_events]
                mean_gpu = sum(gpu_times_ms) / len(gpu_times_ms)
                self._pending_events.clear()
            else:
                mean_gpu = (sum(self._gpu_step_times) / len(self._gpu_step_times)
                            if self._gpu_step_times else 0.0)
                self._gpu_step_times.clear()

            mean_wait = (sum(self._data_wait_times) / len(self._data_wait_times)
                         if self._data_wait_times else 0.0)
            # gpu_util_ratio approaches 1.0 when GPU is the bottleneck;
            # approaches 0.0 when data loading is the bottleneck.
            gpu_util_ratio = mean_gpu / (mean_gpu + mean_wait) if (mean_gpu + mean_wait) > 0 else 1.0
            wandb.log(
                {
                    "timings/gpu_step_ms":    mean_gpu,
                    "timings/data_wait_ms":   mean_wait,
                    "timings/gpu_util_ratio": gpu_util_ratio,
                },
                commit=False,
            )
            self._data_wait_times.clear()
        return loss

    def get_train_dataloader(self):
        dataset = self.train_dataset
        sampler = HalfHalfBatchSampler(
            task1_size=self.task1_size,
            task2_size=self.task2_size,
            batch_size=self._train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )

        num_workers = self.args.dataloader_num_workers
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers and num_workers > 0,
            prefetch_factor=self.args.dataloader_prefetch_factor if num_workers > 0 else None,
            # "fork" is required when Unsloth is active.  Unsloth patches
            # Python/PyTorch's multiprocessing IPC machinery; under "spawn" the
            # patched main process sends a truncated pickle stream to fresh worker
            # processes, which then block forever waiting for bytes that never
            # arrive — GPU stays at 0 %, no error is raised.
            #
            # "fork" sidesteps pickling entirely: workers inherit the live
            # process image (including Unsloth patches and the live processor)
            # via copy-on-write.  Workers are CPU-only (no CUDA calls), so the
            # CUDA-after-fork restriction does not apply.
            # TOKENIZERS_PARALLELISM=false (set in main()) prevents the Rust
            # tokenizer thread-pool from being forked in a bad state.
            multiprocessing_context="fork" if num_workers > 0 else None,
        )
