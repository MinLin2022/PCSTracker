"""Tensor helpers shared by PCSTracker model and loss code."""

from typing import Optional

import torch


def smart_cat(
    first: Optional[torch.Tensor], second: torch.Tensor, dim: int
) -> torch.Tensor:
    return second if first is None else torch.cat([first, second], dim=dim)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    return torch.sum(values * mask) / (eps + torch.sum(mask))


def interpolate_features(
    point_features: torch.Tensor,
    point_coordinates: torch.Tensor,
    queries: torch.Tensor,
    neighbors: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Interpolate each query from the point cloud associated with its frame."""
    if point_features.ndim != 4 or point_coordinates.ndim != 4 or queries.ndim != 3:
        raise ValueError(
            "point_features, point_coordinates and queries must have ranks 4, 4 and 3"
        )
    batch, query_count, channels, support_count = point_features.shape
    if point_coordinates.shape != (batch, query_count, 3, support_count):
        raise ValueError("point feature and coordinate shapes are not aligned")
    if queries.shape != (batch, query_count, 3):
        raise ValueError("queries must have shape [B, query_count, 3]")
    if support_count < neighbors:
        raise ValueError(f"at least {neighbors} support points are required")

    support_xyz = point_coordinates.permute(0, 1, 3, 2).contiguous()
    distances = torch.norm(support_xyz - queries.unsqueeze(2), dim=-1)
    knn_distances, knn_indices = torch.topk(
        distances, k=neighbors, dim=-1, largest=False, sorted=True
    )
    weights = 1.0 / (knn_distances + eps)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).unsqueeze(2)
    gather_indices = knn_indices.unsqueeze(2).expand(
        batch, query_count, channels, neighbors
    )
    neighbor_features = torch.gather(point_features, dim=3, index=gather_indices)
    return torch.sum(neighbor_features * weights, dim=-1)
