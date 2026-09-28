"""Point-cloud feature encoder used by PCSTracker."""

import torch
import torch.nn as nn

from .pointconv import Conv1d, PointConv, PointConvD


class PointConvEncoderLight(nn.Module):
    def __init__(self, weightnet: int = 8):
        super().__init__()
        neighbors = 32
        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(neighbors, 35, 32, weightnet=weightnet)
        self.level0_1 = Conv1d(32, 32)
        self.level1 = PointConvD(4096, neighbors, 35, 32, weightnet=weightnet)
        self.level1_0 = Conv1d(32, 32)
        self.level1_1 = Conv1d(32, 64)
        self.level2 = PointConvD(2048, neighbors, 67, 128, weightnet=weightnet)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        level0_features = self.level0_lift(features)
        level0_features = self.level0(xyz, level0_features)
        level0_refined = self.level0_1(level0_features)

        level1_xyz, level1_features, level1_indices = self.level1(
            xyz, level0_refined
        )
        level1_features = self.level1_0(level1_features)
        level1_refined = self.level1_1(level1_features)

        level2_xyz, level2_features, level2_indices = self.level2(
            level1_xyz, level1_refined
        )
        return (
            [xyz, level1_xyz, level2_xyz],
            [level0_features, level1_features, level2_features],
            [level1_indices, level2_indices],
        )

