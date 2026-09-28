"""Read-only token streams and reproducible next-token minibatches."""

from pathlib import Path

import numpy as np
import torch


def load_token_array(path):
    """Map a raw little-endian uint16 stream without loading the whole file."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0 or size % 2:
        raise ValueError("token file must have a positive, even byte length")
    return np.memmap(path, mode="r", dtype=np.dtype("<u2"))


def get_batch(dataset, batch_size, sequence_length, device, generator):
    """Sample [B, n+1] windows and return their shifted [B, n] inputs/targets."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be a flat token stream")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if len(dataset) <= sequence_length:
        raise ValueError("dataset needs at least sequence_length + 1 tokens")
    starts = torch.randint(0, len(dataset) - sequence_length, (batch_size,),
                           generator=generator, device=generator.device)
    # Only sampled slices are copied and converted; the full stream stays mapped.
    windows = np.stack([dataset[start:start + sequence_length + 1] for start in starts.tolist()])
    tokens = torch.from_numpy(windows.astype(np.int64)).to(device)
    return tokens[:, :-1], tokens[:, 1:]
