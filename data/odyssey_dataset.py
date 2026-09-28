"""Odyssey training dataset used by PCSTracker."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class OdysseyDataset(Dataset):
    REQUIRED_KEYS = ("pc_xyz_seq", "traj_seq", "visibs", "valids")

    def __init__(
        self,
        root,
        seq_len: int = 24,
        track_point_num: int = 256,
        split: str = "train",
    ):
        if seq_len < 16:
            raise ValueError("seq_len must be at least 16")
        if track_point_num < 1:
            raise ValueError("track_point_num must be positive")
        self.seq_len = seq_len
        self.track_point_num = track_point_num
        self.data_directory = Path(root).expanduser().resolve() / split
        if not self.data_directory.is_dir():
            raise FileNotFoundError(
                f"Odyssey split directory does not exist: {self.data_directory}"
            )
        self.sample_paths = sorted(self.data_directory.rglob("*.npz"))
        if not self.sample_paths:
            raise FileNotFoundError(
                f"no .npz samples found under {self.data_directory}"
            )

    def __len__(self) -> int:
        return len(self.sample_paths)

    @staticmethod
    def _visibility_array(values: np.ndarray, name: str) -> np.ndarray:
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 2:
            raise ValueError(f"{name} must have shape [T, M] or [T, M, 1]")
        return values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_path = self.sample_paths[index]
        with np.load(sample_path, allow_pickle=False) as archive:
            missing = [key for key in self.REQUIRED_KEYS if key not in archive]
            if missing:
                raise ValueError(
                    f"{sample_path} is missing required arrays: {', '.join(missing)}"
                )
            point_cloud = np.asarray(archive["pc_xyz_seq"], dtype=np.float32)
            trajectories = np.asarray(archive["traj_seq"], dtype=np.float32)
            visibility = self._visibility_array(archive["visibs"], "visibs")
            validity = self._visibility_array(archive["valids"], "valids")

        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError("pc_xyz_seq must have shape [T, N, 3]")
        if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
            raise ValueError("traj_seq must have shape [T, M, 3]")
        if point_cloud.shape[0] < self.seq_len:
            raise ValueError(
                f"sample has {point_cloud.shape[0]} frames; {self.seq_len} required"
            )
        if point_cloud.shape[1] < 4096:
            raise ValueError("each point-cloud frame must contain at least 4096 points")
        if not (
            trajectories.shape[:2] == visibility.shape == validity.shape
            and trajectories.shape[0] == point_cloud.shape[0]
        ):
            raise ValueError("trajectory, visibility, validity and frame shapes are inconsistent")

        point_cloud = point_cloud[: self.seq_len].copy()
        trajectories = trajectories[: self.seq_len].copy()
        visibility = visibility[: self.seq_len]
        validity = validity[: self.seq_len]
        point_cloud[..., :2] *= -1
        trajectories[..., :2] *= -1

        visible_indices = np.flatnonzero(visibility[0] > 0)
        if visible_indices.size < self.track_point_num:
            raise ValueError(
                f"sample has {visible_indices.size} first-frame visible tracks; "
                f"{self.track_point_num} required"
            )
        selected = visible_indices[
            torch.randperm(visible_indices.size)[: self.track_point_num].numpy()
        ]

        return {
            "video_pc": torch.from_numpy(point_cloud).permute(0, 2, 1).contiguous(),
            "trajs_3d": torch.from_numpy(trajectories[:, selected]).float(),
            "visibs": torch.from_numpy(visibility[:, selected]).float(),
            "valids": torch.from_numpy(validity[:, selected]).float(),
        }

