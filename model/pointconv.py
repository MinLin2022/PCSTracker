"""Minimal PointConv layers used by the PCSTracker feature encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils


LEAKY_RATE = 0.1
USE_BATCH_NORM = False


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        use_leaky: bool = True,
        bn: bool = USE_BATCH_NORM,
        bias: bool = True,
    ):
        super().__init__()
        activation = (
            nn.LeakyReLU(LEAKY_RATE, inplace=True)
            if use_leaky
            else nn.ReLU(inplace=True)
        )
        self.composed_module = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            activation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.composed_module(inputs)


def square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances = -2 * torch.matmul(source, target.permute(0, 2, 1))
    distances += torch.sum(source**2, dim=-1).unsqueeze(-1)
    distances += torch.sum(target**2, dim=-1).unsqueeze(1)
    return distances


def knn_indices(
    sample_count: int, points: torch.Tensor, queries: torch.Tensor
) -> torch.Tensor:
    if points.shape[1] < sample_count:
        raise ValueError(f"at least {sample_count} points are required")
    distances = square_distance(queries, points)
    return torch.topk(
        distances, sample_count, dim=-1, largest=False, sorted=False
    ).indices


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = pointnet2_utils.gather_operation(
        points.permute(0, 2, 1).contiguous(), indices
    )
    return gathered.permute(0, 2, 1).contiguous()


def group_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return pointnet2_utils.grouping_operation(
        points.permute(0, 2, 1).contiguous(), indices.int()
    ).permute(0, 2, 3, 1)


def group_all(
    sample_count: int, xyz: torch.Tensor, features: torch.Tensor
):
    indices = knn_indices(sample_count, xyz, xyz)
    grouped_xyz = group_points(xyz, indices)
    relative_xyz = grouped_xyz - xyz.unsqueeze(2)
    grouped_features = group_points(features, indices)
    return torch.cat([relative_xyz, grouped_features], dim=-1), relative_xyz


def group_queries(
    sample_count: int,
    xyz: torch.Tensor,
    queries: torch.Tensor,
    features: torch.Tensor,
):
    indices = knn_indices(sample_count, xyz, queries)
    grouped_xyz = group_points(xyz, indices)
    relative_xyz = grouped_xyz - queries.unsqueeze(2)
    grouped_features = group_points(features, indices)
    return torch.cat([relative_xyz, grouped_features], dim=-1), relative_xyz


class WeightNet(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        hidden_unit=(8, 8),
        bn: bool = USE_BATCH_NORM,
    ):
        super().__init__()
        self.bn = bn
        channels = [in_channel, *hidden_unit, out_channel]
        self.mlp_convs = nn.ModuleList(
            [
                nn.Conv2d(channels[index], channels[index + 1], 1)
                for index in range(len(channels) - 1)
            ]
        )
        self.mlp_bns = nn.ModuleList(
            [nn.BatchNorm2d(channels[index + 1]) for index in range(len(channels) - 1)]
        )

    def forward(self, localized_xyz: torch.Tensor) -> torch.Tensor:
        weights = localized_xyz
        for index, convolution in enumerate(self.mlp_convs):
            weights = convolution(weights)
            if self.bn:
                weights = self.mlp_bns[index](weights)
            weights = F.relu(weights)
        return weights


class PointConv(nn.Module):
    def __init__(
        self,
        nsample: int,
        in_channel: int,
        out_channel: int,
        weightnet: int = 16,
        bn: bool = USE_BATCH_NORM,
        use_leaky: bool = True,
    ):
        super().__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)
        self.relu = (
            nn.LeakyReLU(LEAKY_RATE, inplace=True)
            if use_leaky
            else nn.ReLU(inplace=True)
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch, _, point_count = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        grouped, relative_xyz = group_all(self.nsample, xyz, features)
        weights = self.weightnet(relative_xyz.permute(0, 3, 2, 1))
        output = torch.matmul(
            grouped.permute(0, 1, 3, 2), weights.permute(0, 3, 2, 1)
        ).reshape(batch, point_count, -1)
        output = self.linear(output)
        output = output.permute(0, 2, 1)
        if self.bn:
            output = self.bn_linear(output)
        return self.relu(output)


class PointConvD(nn.Module):
    def __init__(
        self,
        npoint: int,
        nsample: int,
        in_channel: int,
        out_channel: int,
        weightnet: int = 16,
        bn: bool = USE_BATCH_NORM,
        use_leaky: bool = True,
    ):
        super().__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)
        self.relu = (
            nn.LeakyReLU(LEAKY_RATE, inplace=True)
            if use_leaky
            else nn.ReLU(inplace=True)
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        batch, _, available_points = xyz.shape
        if available_points < self.npoint:
            raise ValueError(f"at least {self.npoint} input points are required")
        xyz = xyz.permute(0, 2, 1).contiguous()
        features = features.permute(0, 2, 1)
        sample_indices = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        sampled_xyz = gather_points(xyz, sample_indices)
        grouped, relative_xyz = group_queries(
            self.nsample, xyz, sampled_xyz, features
        )
        weights = self.weightnet(relative_xyz.permute(0, 3, 2, 1))
        output = torch.matmul(
            grouped.permute(0, 1, 3, 2), weights.permute(0, 3, 2, 1)
        ).reshape(batch, self.npoint, -1)
        output = self.linear(output).permute(0, 2, 1)
        if self.bn:
            output = self.bn_linear(output)
        return sampled_xyz.permute(0, 2, 1), self.relu(output), sample_indices

