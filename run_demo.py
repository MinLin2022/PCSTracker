"""Run PCSTracker on a point-cloud sequence stored in an NPZ archive."""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from PCSTracker import PCSTracker


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "PCSTracker.pth"


def load_demo_input(path) -> Tuple[torch.Tensor, torch.Tensor]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input archive does not exist: {input_path}")
    with np.load(input_path, allow_pickle=False) as archive:
        missing = [key for key in ("video_pc", "query_points") if key not in archive]
        if missing:
            raise ValueError(f"input archive is missing: {', '.join(missing)}")
        video_pc = np.asarray(archive["video_pc"], dtype=np.float32)
        query_points = np.asarray(archive["query_points"], dtype=np.float32)

    if video_pc.ndim != 3 or video_pc.shape[-1] != 3:
        raise ValueError("video_pc must have shape [T, N, 3]")
    if query_points.ndim != 2 or query_points.shape[-1] != 4:
        raise ValueError("query_points must have shape [M, 4]")
    if video_pc.shape[0] < 16:
        raise ValueError("video_pc must contain at least 16 frames")
    if video_pc.shape[1] < 1 or query_points.shape[0] < 1:
        raise ValueError("video_pc and query_points must be non-empty")
    if not np.isfinite(video_pc).all() or not np.isfinite(query_points).all():
        raise ValueError("video_pc and query_points must contain only finite values")

    query_frames = query_points[:, 0]
    if not np.equal(query_frames, np.floor(query_frames)).all():
        raise ValueError("query frame indices must be integers")
    if (query_frames < 0).any() or (query_frames >= video_pc.shape[0]).any():
        raise ValueError("query frame indices must fall within the video frame range")

    video_tensor = torch.from_numpy(video_pc).permute(0, 2, 1).unsqueeze(0)
    query_tensor = torch.from_numpy(query_points).unsqueeze(0)
    return video_tensor.contiguous(), query_tensor.contiguous()


def load_checkpoint(model: PCSTracker, path, device: torch.device) -> None:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, (dict, OrderedDict)):
        raise ValueError("checkpoint must contain a model state dictionary")
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = OrderedDict(
            (key[len("module.") :], value) for key, value in checkpoint.items()
        )
    model.load_state_dict(checkpoint, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track 3D query points through an NPZ point-cloud sequence."
    )
    parser.add_argument("--input", type=Path, required=True, help="input NPZ archive")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pcstracker_predictions.npz"),
        help="output NPZ archive",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device")
    parser.add_argument("--iters", type=int, default=4, help="refinement iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iters < 1:
        raise ValueError("--iters must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    video_pc, query_points = load_demo_input(args.input)
    if video_pc.shape[-1] < 4096:
        raise ValueError("video_pc must contain at least 4096 points per frame")
    model = PCSTracker().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        trajectories, _, _ = model(
            video_pc.to(device), query_points.to(device), iters=args.iters
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trajs_xyz=trajectories[0].cpu().numpy(),
        query_points=query_points[0].numpy(),
    )
    print(f"Saved trajectories to {output_path}")


if __name__ == "__main__":
    main()
