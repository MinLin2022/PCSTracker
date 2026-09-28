"""PCSTracker model for long-term point-cloud trajectory estimation."""

from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

from model.blocks import UpdateFormer
from model.correlation import CorrelationBlock
from model.embeddings import embedding_3d, sinusoidal_embedding_1d
from model.feature_encoder import PointConvEncoderLight
from model.losses import sequence_loss
from model.utils import interpolate_features, smart_cat


class TrackerIterationModule(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_size: int = 384,
        additional_dim: int = 2,
        space_depth: int = 6,
        time_depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        add_space_attn: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.additional_dim = additional_dim
        self.corr_levels = 3
        self.base_scales = 0.25
        self.truncate_k = 512
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.add_space_attn = add_space_attn
        self.hidden_size = hidden_size
        self.space_depth = space_depth
        self.time_depth = time_depth

        input_dim = 358 + additional_dim
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, input_dim)
        )
        self.fcorr_fn = CorrelationBlock(
            num_levels=self.corr_levels,
            base_scale=self.base_scales,
            resolution=3,
            truncate_k=self.truncate_k,
        )
        self.updateformer = nn.Sequential(
            UpdateFormer(
                space_depth=space_depth,
                time_depth=time_depth,
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_heads=num_heads,
                output_dim=latent_dim + 3,
                mlp_ratio=mlp_ratio,
                add_space_attn=add_space_attn,
            )
        )
        self.norm = nn.GroupNorm(1, latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU()
        )
        self.vis_predictor = nn.Sequential(nn.Linear(latent_dim, 1))

    def forward(
        self,
        feature_maps: torch.Tensor,
        point_cloud: torch.Tensor,
        initial_coordinates: torch.Tensor,
        initial_features: torch.Tensor,
        initial_visibility: torch.Tensor,
        track_mask: torch.Tensor,
        iterations: int = 4,
    ):
        batch, initial_frames, tracked_points, dimensions = initial_coordinates.shape
        if dimensions != 3 or batch != 1:
            raise ValueError("tracking requires shape [1, frames, points, 3]")
        map_xyz = point_cloud.permute(0, 1, 3, 2)
        _, frames, _, _ = feature_maps.shape

        if initial_frames < frames:
            coordinate_padding = initial_coordinates[:, -1:].repeat(
                1, frames - initial_frames, 1, 1
            )
            coordinates = torch.cat(
                [initial_coordinates, coordinate_padding], dim=1
            )
            visibility_padding = initial_visibility[:, -1:].repeat(
                1, frames - initial_frames, 1, 1
            )
            initial_visibility = torch.cat(
                [initial_visibility, visibility_padding], dim=1
            )
        else:
            coordinates = initial_coordinates.clone()

        tracked_features = initial_features.clone()
        times = torch.linspace(0, frames - 1, frames).reshape(1, frames, 1)
        pose_embedding = self.pos_embed(
            coordinates.reshape(batch * frames, tracked_points, 3)
        )
        pose_embedding = (
            pose_embedding.reshape(batch, frames, tracked_points, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch * tracked_points, frames, -1)
        )
        time_embedding = torch.from_numpy(
            sinusoidal_embedding_1d(358 + self.additional_dim, times[0].numpy())
        )
        time_embedding = (
            time_embedding.unsqueeze(0)
            .repeat(batch, 1, 1)
            .float()
            .to(feature_maps.device)
        )
        coordinate_predictions = []

        for _ in range(iterations):
            coordinates = coordinates.detach()
            self.fcorr_fn.init_module(tracked_features, feature_maps, map_xyz)
            correlations = self.fcorr_fn(coordinates).permute(0, 1, 3, 2)
            correlation_dim = correlations.shape[3]
            correlations = correlations.permute(0, 2, 1, 3).reshape(
                batch * tracked_points, frames, correlation_dim
            )
            flows = (coordinates - coordinates[:, 0:1]).permute(0, 2, 1, 3)
            flows = flows.reshape(batch * tracked_points, frames, 3)
            flow_embedding = embedding_3d(flows, 32)
            flattened_features = tracked_features.permute(0, 2, 1, 3).reshape(
                batch * tracked_points, frames, self.latent_dim
            )

            if track_mask.shape[1] < initial_visibility.shape[1]:
                missing_frames = initial_visibility.shape[1] - track_mask.shape[1]
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0:1]).repeat(
                            1, missing_frames, 1, 1
                        ),
                    ],
                    dim=1,
                )
            status = torch.cat([track_mask, initial_visibility], dim=2)
            status = status.permute(0, 2, 1, 3).reshape(
                batch * tracked_points, frames, 2
            )
            transformer_input = torch.cat(
                [flow_embedding, correlations, flattened_features, status], dim=2
            )
            transformer_input = transformer_input + pose_embedding + time_embedding
            transformer_input = rearrange(
                transformer_input, "(b n) t d -> b n t d", b=batch
            )

            delta = self.updateformer(transformer_input)
            delta = rearrange(delta, "b n t d -> (b n) t d")
            coordinate_delta = delta[:, :, :3]
            feature_delta = delta[:, :, 3:].reshape(
                batch * tracked_points * frames, self.latent_dim
            )
            flat_features = tracked_features.permute(0, 2, 1, 3).reshape(
                batch * tracked_points * frames, self.latent_dim
            )
            flat_features = self.ffeat_updater(self.norm(feature_delta)) + flat_features
            tracked_features = flat_features.reshape(
                batch, tracked_points, frames, self.latent_dim
            ).permute(0, 2, 1, 3)
            coordinates = coordinates + coordinate_delta.reshape(
                batch, tracked_points, frames, 3
            ).permute(0, 2, 1, 3)
            coordinate_predictions.append(coordinates)

        visibility = self.vis_predictor(
            tracked_features.reshape(
                batch * frames * tracked_points, self.latent_dim
            )
        ).reshape(batch, frames, tracked_points)
        return coordinate_predictions, visibility.float(), initial_features


class PCSTracker(nn.Module):
    """Track user-specified 3D query points through a point-cloud sequence."""

    def __init__(self):
        super().__init__()
        self.latent_dim = 128
        self.additional_dim = 2
        self.S = 16
        self.fnet = PointConvEncoderLight(weightnet=8)
        self.iter_module = TrackerIterationModule(
            latent_dim=self.latent_dim,
            hidden_size=256,
            additional_dim=self.additional_dim,
            space_depth=3,
            time_depth=3,
            num_heads=4,
            mlp_ratio=2.0,
            add_space_attn=True,
        )

    def forward(
        self,
        video_pc: torch.Tensor,
        queries: torch.Tensor,
        iters: int = 4,
        is_train: bool = False,
    ):
        batch, total_frames, channels, total_points = video_pc.shape
        query_batch, tracked_points, query_dim = queries.shape
        if batch != 1 or query_batch != 1:
            raise ValueError("PCSTracker supports one sequence per device")
        if channels != 3 or query_dim != 4:
            raise ValueError("video_pc and queries require channel dimensions 3 and 4")
        if total_frames < self.S:
            raise ValueError(f"video_pc must contain at least {self.S} frames")
        if total_points < 4096:
            raise ValueError("video_pc must contain at least 4096 points per frame")

        device = queries.device
        query_frames = queries[:, :, 0].long()
        _, sort_indices = torch.sort(query_frames[0], descending=False)
        inverse_indices = torch.argsort(sort_indices)
        sorted_query_frames = query_frames[0][sort_indices]

        initial_xyz = queries[:, :, 1:4]
        initial_coordinates = initial_xyz.view(batch, 1, tracked_points, 3).repeat(
            1, self.S, 1, 1
        )
        trajectories = torch.zeros(
            (batch, total_frames, tracked_points, 3), device=device
        )
        frame_indices = torch.arange(total_frames, device=device).view(1, -1, 1)
        frame_indices = frame_indices.repeat(batch, 1, tracked_points)
        track_mask = (frame_indices >= query_frames[:, None, :]).unsqueeze(-1)
        initial_visibility = torch.ones(
            (batch, self.S, tracked_points, 1), device=device
        ).float() * 10

        track_mask = track_mask[:, :, sort_indices].clone()
        initial_coordinates = initial_coordinates[:, :, sort_indices].clone()
        initial_visibility = initial_visibility[:, :, sort_indices].clone()

        window_start = 0
        previous_query_count = 0
        cached_features = None
        cached_xyz = None
        track_features = None
        coordinate_windows = []
        window_query_counts = []
        coordinates = None
        visibility = None

        while window_start < total_frames - self.S // 2:
            point_window = video_pc[:, window_start : window_start + self.S]
            local_frames = point_window.shape[1]
            if local_frames < self.S:
                point_window = torch.cat(
                    [
                        point_window,
                        point_window[:, -1:].repeat(
                            1, self.S - local_frames, 1, 1
                        ),
                    ],
                    dim=1,
                )
            flat_window = point_window.reshape(
                batch * self.S, channels, total_points
            )

            if cached_features is None:
                xyz_levels, feature_levels, _ = self.fnet(flat_window, flat_window)
                cached_features = feature_levels[-1]
                cached_xyz = xyz_levels[-1]
            else:
                new_points = flat_window[self.S // 2 :]
                xyz_levels, feature_levels, _ = self.fnet(new_points, new_points)
                cached_features = torch.cat(
                    [cached_features[self.S // 2 :], feature_levels[-1]], dim=0
                )
                cached_xyz = torch.cat(
                    [cached_xyz[self.S // 2 :], xyz_levels[-1]], dim=0
                )

            downsampled_points = cached_xyz.shape[-1]
            feature_maps = cached_features.float().reshape(
                batch, self.S, self.latent_dim, downsampled_points
            )
            downsampled_xyz = cached_xyz.reshape(
                batch, self.S, channels, downsampled_points
            )

            available = torch.nonzero(
                sorted_query_frames < window_start + self.S, as_tuple=False
            )
            if available.shape[0] == 0:
                window_start += self.S // 2
                continue
            current_query_count = int(available[-1].item()) + 1

            if current_query_count > previous_query_count:
                new_query_frames = sorted_query_frames[
                    previous_query_count:current_query_count
                ] - window_start
                sampled_features = feature_maps[
                    :, new_query_frames
                ]
                sampled_xyz = downsampled_xyz[:, new_query_frames]
                new_features = interpolate_features(
                    sampled_features,
                    sampled_xyz,
                    initial_coordinates[
                        :, 0, previous_query_count:current_query_count
                    ],
                )
                new_features = new_features.unsqueeze(1).repeat(
                    1, self.S, 1, 1
                )
                track_features = smart_cat(track_features, new_features, dim=2)

            if previous_query_count > 0:
                carried_coordinates = coordinates[-1][:, self.S // 2 :]
                initial_coordinates[
                    :, : self.S // 2, :previous_query_count
                ] = carried_coordinates
                initial_coordinates[
                    :, self.S // 2 :, :previous_query_count
                ] = carried_coordinates[:, -1:].repeat(1, self.S // 2, 1, 1)

                carried_visibility = visibility[:, self.S // 2 :].unsqueeze(-1)
                initial_visibility[
                    :, : self.S // 2, :previous_query_count
                ] = carried_visibility
                initial_visibility[
                    :, self.S // 2 :, :previous_query_count
                ] = carried_visibility[:, -1:].repeat(1, self.S // 2, 1, 1)

            coordinates, visibility, _ = self.iter_module(
                feature_maps=feature_maps,
                point_cloud=downsampled_xyz,
                initial_coordinates=initial_coordinates[:, :, :current_query_count],
                initial_features=track_features[:, :, :current_query_count],
                initial_visibility=initial_visibility[:, :, :current_query_count],
                track_mask=track_mask[
                    :, window_start : window_start + self.S, :current_query_count
                ],
                iterations=iters,
            )
            if is_train:
                coordinate_windows.append(
                    [prediction[:, :local_frames] for prediction in coordinates]
                )
                window_query_counts.append(current_query_count)

            trajectories[
                :, window_start : window_start + self.S, :current_query_count
            ] = coordinates[-1][:, :local_frames]
            track_mask[:, : window_start + self.S, :current_query_count] = False
            window_start += self.S // 2
            previous_query_count = current_query_count

        trajectories = trajectories[:, :, inverse_indices]
        training_data = (
            (coordinate_windows, window_query_counts, sort_indices)
            if is_train
            else None
        )
        return trajectories, track_features, training_data

    def infer(
        self,
        model: nn.Module,
        input_list: Sequence[torch.Tensor],
        gt_list: Optional[Sequence[torch.Tensor]] = None,
        iters: int = 4,
        is_train: bool = False,
    ):
        video_pc, queries = input_list
        predictions, features, training_data = model(
            video_pc, queries, iters=iters, is_train=is_train
        )
        if gt_list is not None:
            return self.compute_loss(training_data, gt_list)
        return predictions, features, training_data

    def compute_loss(self, training_data, ground_truth):
        coordinate_windows, window_query_counts, sort_indices = training_data
        trajectories, _, valid = ground_truth
        trajectories = trajectories[:, :, sort_indices]
        valid = valid[:, :, sort_indices]
        trajectory_targets = []
        validity_targets = []
        for window_index, query_count in enumerate(window_query_counts):
            window_start = window_index * (self.S // 2)
            trajectory_targets.append(
                trajectories[
                    :, window_start : window_start + self.S, :query_count
                ]
            )
            validity_targets.append(
                valid[:, window_start : window_start + self.S, :query_count]
            )
        loss = sequence_loss(
            coordinate_windows, trajectory_targets, validity_targets, gamma=0.8
        )

        with torch.no_grad():
            prediction = coordinate_windows[-1][-1][:, -1]
            target = trajectories[:, -1]
            endpoint_error = torch.norm(prediction - target, dim=-1).reshape(-1)
            epe_sum = endpoint_error.sum()
            pc1_sum = (endpoint_error < 0.1).float().sum()
            pc2_sum = (endpoint_error < 0.2).float().sum()
            pc4_sum = (endpoint_error < 0.4).float().sum()
            count = torch.tensor(
                endpoint_error.numel(), device=endpoint_error.device, dtype=torch.float32
            )
            if dist.is_available() and dist.is_initialized():
                for value in (epe_sum, pc1_sum, pc2_sum, pc4_sum, count):
                    dist.all_reduce(value)
            metrics = [
                ["epe", (epe_sum / count).item()],
                ["pc1", (pc1_sum / count).item()],
                ["pc2", (pc2_sum / count).item()],
                ["pc4", (pc4_sum / count).item()],
            ]
        return loss, metrics

    def training_infer(self, model: nn.Module, step_data, device: torch.device):
        video_pc = step_data["video_pc"].to(device)
        trajectories = step_data["trajs_3d"].to(device)
        visibility = step_data["visibs"].to(device).float()
        valid = step_data["valids"].to(device).float()
        batch, frames, tracked_points, _ = trajectories.shape
        if batch != 1:
            raise ValueError("training requires one sequence per device")

        first_visible = torch.max(visibility, dim=1).indices
        randomized_count = tracked_points // 4
        visible_frames = [
            torch.nonzero(visibility[0, :, point], as_tuple=False)
            for point in range(tracked_points)
        ]
        if any(frames_for_point.numel() == 0 for frames_for_point in visible_frames):
            raise ValueError("every training trajectory must be visible in at least one frame")
        random_visible = torch.cat(
            [
                frames_for_point[
                    torch.randint(len(frames_for_point), size=(1,), device=device)
                ]
                for frames_for_point in visible_frames
            ],
            dim=1,
        )
        first_visible = torch.cat(
            [random_visible[:, :randomized_count], first_visible[:, randomized_count:]],
            dim=1,
        )
        point_indices = torch.arange(tracked_points, device=device)
        initial_xyz = trajectories[0, first_visible[0], point_indices].unsqueeze(0)
        queries = torch.cat([first_visible.unsqueeze(-1), initial_xyz], dim=2)

        unwrapped_model = model.module if hasattr(model, "module") else model
        return unwrapped_model.infer(
            unwrapped_model,
            input_list=[video_pc, queries],
            gt_list=[trajectories, visibility, valid],
            iters=4,
            is_train=True,
        )


__all__ = ["PCSTracker"]
