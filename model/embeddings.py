"""Positional embeddings used by the iterative tracker."""

import numpy as np
import torch


def sinusoidal_embedding_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    frequencies = np.arange(embed_dim // 2, dtype=np.float64)
    frequencies /= embed_dim / 2.0
    frequencies = 1.0 / 10000**frequencies
    angles = np.einsum("m,d->md", positions.reshape(-1), frequencies)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def embedding_3d(xyz: torch.Tensor, channels: int) -> torch.Tensor:
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [B, N, 3]")
    if channels % 2:
        raise ValueError("channels must be even")

    batch, points, _ = xyz.shape
    divisor = (
        torch.arange(0, channels, 2, device=xyz.device, dtype=torch.float32)
        * (1000.0 / channels)
    ).reshape(1, 1, channels // 2)

    encoded_axes = []
    for axis in range(3):
        coordinate = xyz[:, :, axis : axis + 1]
        encoded = torch.zeros(
            batch, points, channels, device=xyz.device, dtype=torch.float32
        )
        encoded[:, :, 0::2] = torch.sin(coordinate * divisor)
        encoded[:, :, 1::2] = torch.cos(coordinate * divisor)
        encoded_axes.append(encoded)

    return torch.cat([xyz, xyz, *encoded_axes], dim=2)

