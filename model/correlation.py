"""Local voxel and nearest-neighbor correlation features."""

import numpy as np
import torch
import torch.nn as nn


def scatter_add_last(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    output_shape = list(source.shape)
    output_shape[-1] = int(indices.max().item()) + 1
    output = torch.zeros(output_shape, dtype=source.dtype, device=source.device)
    return output.scatter_add_(-1, indices, source)


class CorrelationBlock(nn.Module):
    def __init__(
        self,
        num_levels: int = 3,
        base_scale: float = 0.25,
        resolution: int = 3,
        truncate_k: int = 128,
        knn: int = 32,
    ):
        super().__init__()
        self.truncate_k = truncate_k
        self.num_levels = num_levels
        self.resolution = resolution
        self.base_scale = base_scale
        self.out_conv = nn.Sequential(
            nn.Conv1d((resolution**3) * num_levels, 128, 1),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
            nn.Conv1d(128, 128, 1),
        )
        self.knn = knn
        self.knn_conv = nn.Sequential(
            nn.Conv2d(4, 128, 1), nn.GroupNorm(8, 128), nn.PReLU()
        )
        self.knn_out = nn.Conv1d(128, 128, 1)

    def init_module(
        self, query_features: torch.Tensor, map_features: torch.Tensor, map_xyz: torch.Tensor
    ) -> None:
        batch, frames, map_points, _ = map_xyz.shape
        query_points = query_features.shape[2]
        correlations = self.calculate_corr(query_features, map_features)
        topk = torch.topk(
            correlations.clone(), k=self.truncate_k, dim=3, sorted=True
        )
        self.truncated_corr = topk.values
        expanded_xyz = map_xyz.unsqueeze(2).expand(
            batch, frames, query_points, map_points, 3
        )
        gather_indices = topk.indices.unsqueeze(-1).expand(
            -1, -1, -1, self.truncate_k, 3
        )
        self.truncate_xyz2 = torch.gather(expanded_xyz, dim=3, index=gather_indices)
        self.ones_matrix = torch.ones_like(self.truncated_corr)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.get_voxel_feature(coordinates) + self.get_knn_feature(coordinates)

    def get_voxel_feature(self, coordinates: torch.Tensor) -> torch.Tensor:
        batch, frames, query_points, _ = coordinates.shape
        features = []
        for level in range(self.num_levels):
            with torch.no_grad():
                radius = self.base_scale * (2**level)
                voxel_offsets = torch.round(
                    (self.truncate_xyz2 - coordinates.unsqueeze(-2)) / radius
                )
                valid = (
                    torch.abs(voxel_offsets) <= np.floor(self.resolution / 2)
                ).all(dim=-1)
                voxel_offsets = voxel_offsets + 1
                voxel_indices = (
                    voxel_offsets[..., 0] * (self.resolution**2)
                    + voxel_offsets[..., 1] * self.resolution
                    + voxel_offsets[..., 2]
                ).to(torch.int64)
                scatter_indices = (voxel_indices * valid).detach()
                valid = valid.detach()

            correlation_sum = scatter_add_last(
                self.truncated_corr * valid, scatter_indices
            )
            correlation_count = torch.clamp(
                scatter_add_last(self.ones_matrix * valid, scatter_indices),
                1,
                self.truncate_k,
            )
            correlation = correlation_sum / correlation_count
            required_channels = self.resolution**3
            if correlation.shape[-1] < required_channels:
                padding = torch.zeros(
                    batch,
                    frames,
                    query_points,
                    required_channels - correlation.shape[-1],
                    device=coordinates.device,
                )
                correlation = torch.cat([correlation, padding], dim=-1)
            features.append(correlation.permute(0, 1, 3, 2).contiguous())
        stacked = torch.cat(features, dim=2).reshape(
            batch * frames, -1, query_points
        )
        return self.out_conv(stacked).reshape(batch, frames, -1, query_points)

    def get_knn_feature(self, coordinates: torch.Tensor) -> torch.Tensor:
        batch, frames, query_points, _ = coordinates.shape
        squared_distance = torch.sum(
            (self.truncate_xyz2 - coordinates.unsqueeze(-2)) ** 2, dim=-1
        )
        neighbors = torch.topk(
            -squared_distance, k=self.knn, dim=3
        ).indices
        correlations = torch.gather(
            self.truncated_corr.reshape(
                batch * frames * query_points, self.truncate_k
            ),
            dim=1,
            index=neighbors.reshape(batch * frames * query_points, self.knn),
        ).reshape(batch, frames, 1, query_points, self.knn)
        coordinate_indices = neighbors.unsqueeze(-1).expand(
            batch, frames, query_points, self.knn, 3
        )
        neighbor_xyz = torch.gather(
            self.truncate_xyz2, dim=3, index=coordinate_indices
        ).permute(0, 1, 4, 2, 3)
        neighbor_xyz -= coordinates.permute(0, 1, 3, 2).unsqueeze(-1)
        knn_input = torch.cat([correlations, neighbor_xyz], dim=2).reshape(
            batch * frames, 4, query_points, self.knn
        )
        knn_features = self.knn_conv(knn_input).max(dim=3)[0]
        return self.knn_out(knn_features).reshape(
            batch, frames, -1, query_points
        )

    @staticmethod
    def calculate_corr(
        query_features: torch.Tensor, map_features: torch.Tensor
    ) -> torch.Tensor:
        channels = query_features.shape[-1]
        correlations = torch.matmul(query_features, map_features)
        scale = torch.sqrt(
            torch.tensor(channels, dtype=torch.float32, device=correlations.device)
        )
        return correlations / scale
