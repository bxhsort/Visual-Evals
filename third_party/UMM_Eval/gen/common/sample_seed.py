"""Deterministic per-sample seeding for reproducible generation.

The model's `t2i_generate` / `it2i_generate` draw their init latent from the global torch
RNG (`torch.randn(..., device=device)`), so seeding torch right before each generate call
fully determines that sample's noise. To make a run reproducible *regardless of how the
dataset is sharded across nodes/GPUs*, the seed must key off each sample's STABLE dataset
index — the position assigned at load time, before any shuffle/shard — not its runtime order.

Usage in an inference loop::

    from sample_seed import seed_for_sample
    ...
    seed_for_sample(stable_index)   # stable_index: 0..N-1 assigned before shuffle/shard
    images = engine.chat(prompt, ...)

So sample 0 -> seed BASE+0, sample 1 -> BASE+1, ... (all distinct, all reproducible).
"""

import torch

BASE_SEED = 42


def sample_seed(stable_index, base=BASE_SEED):
    """Return the deterministic seed for a sample given its stable dataset index."""
    return int(base) + int(stable_index)


def seed_for_sample(stable_index, base=BASE_SEED):
    """Seed torch (CPU + CUDA) so the next generate call is reproducible for this sample."""
    seed = sample_seed(stable_index, base)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
