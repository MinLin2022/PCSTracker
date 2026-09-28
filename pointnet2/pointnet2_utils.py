"""Autograd wrappers for the PointNet2 operations used by PCSTracker."""

from typing import Tuple

import torch
from torch.autograd import Function

try:
    import pointnet2_cuda
except ModuleNotFoundError:
    pointnet2_cuda = None


def require_extension() -> None:
    if pointnet2_cuda is None:
        raise RuntimeError(
            "pointnet2_cuda is not installed; run `pip install ./pointnet2` "
            "from the PCSTracker repository root"
        )


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        require_extension()
        if not xyz.is_contiguous():
            raise ValueError("xyz must be contiguous")
        batch, point_count, _ = xyz.size()
        output = torch.empty(
            batch, npoint, dtype=torch.int32, device=xyz.device
        )
        temporary = torch.full(
            (batch, point_count), 1e10, dtype=torch.float32, device=xyz.device
        )
        pointnet2_cuda.furthest_point_sampling_wrapper(
            batch, point_count, npoint, xyz, temporary, output
        )
        return output

    @staticmethod
    def backward(ctx, gradient=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        require_extension()
        if not features.is_contiguous() or not indices.is_contiguous():
            raise ValueError("features and indices must be contiguous")
        batch, channels, point_count = features.size()
        sample_count = indices.size(1)
        output = torch.empty(
            batch,
            channels,
            sample_count,
            dtype=features.dtype,
            device=features.device,
        )
        pointnet2_cuda.gather_points_wrapper(
            batch,
            channels,
            point_count,
            sample_count,
            features,
            indices,
            output,
        )
        ctx.for_backward = (indices, channels, point_count)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        indices, channels, point_count = ctx.for_backward
        batch, sample_count = indices.size()
        feature_gradient = torch.zeros(
            batch,
            channels,
            point_count,
            dtype=gradient.dtype,
            device=gradient.device,
        )
        pointnet2_cuda.gather_points_grad_wrapper(
            batch,
            channels,
            point_count,
            sample_count,
            gradient.contiguous(),
            indices,
            feature_gradient,
        )
        return feature_gradient, None


gather_operation = GatherOperation.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        require_extension()
        if not features.is_contiguous() or not indices.is_contiguous():
            raise ValueError("features and indices must be contiguous")
        batch, grouped_points, sample_count = indices.size()
        _, channels, point_count = features.size()
        output = torch.empty(
            batch,
            channels,
            grouped_points,
            sample_count,
            dtype=features.dtype,
            device=features.device,
        )
        pointnet2_cuda.group_points_wrapper(
            batch,
            channels,
            point_count,
            grouped_points,
            sample_count,
            features,
            indices,
            output,
        )
        ctx.for_backward = (indices, point_count)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        indices, point_count = ctx.for_backward
        batch, channels, grouped_points, sample_count = gradient.size()
        feature_gradient = torch.zeros(
            batch,
            channels,
            point_count,
            dtype=gradient.dtype,
            device=gradient.device,
        )
        pointnet2_cuda.group_points_grad_wrapper(
            batch,
            channels,
            point_count,
            grouped_points,
            sample_count,
            gradient.contiguous(),
            indices,
            feature_gradient,
        )
        return feature_gradient, None


grouping_operation = GroupingOperation.apply
