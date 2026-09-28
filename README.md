# PCSTracker

PCSTracker estimates long-term 3D trajectories for query points in a temporal
point-cloud sequence. This release contains the model code, the Odyssey
training entry point, and a file-driven inference demo.

## Requirements

- Linux
- Python 3.8 or newer
- NVIDIA GPU with a CUDA toolkit compatible with the installed PyTorch build

Create an environment and install the Python dependencies:

```bash
conda create -n pcstracker python=3.8
conda activate pcstracker
pip install -r requirements.txt
```

Compile and install the PointNet2 CUDA extension from the repository root:

```bash
pip install ./pointnet2
```

The extension must be rebuilt after changing the PyTorch or CUDA version.

## Pretrained weights

Pretrained weights are not included in this repository. Place the checkpoint at:

```text
checkpoints/PCSTracker.pth
```

`run_demo.py` uses this file by default. A different raw state dictionary can be
selected with `--checkpoint`.

## Demo input

The demo reads one `.npz` file with exactly these arrays:

- `video_pc`: `float32` array shaped `[T, N, 3]`, containing `T` point-cloud
  frames with `N` points per frame.
- `query_points`: `float32` array shaped `[M, 4]`. Each row is
  `[start_frame, x, y, z]`; `start_frame` must be an integer in `[0, T)`.

Coordinates must use the same metric coordinate system in both arrays. The demo
does not apply dataset-specific coordinate conversion.

Run inference from the repository root:

```bash
python run_demo.py \
  --input /path/to/input.npz \
  --output outputs/predictions.npz
```

The output archive contains:

- `trajs_xyz`: predicted trajectories shaped `[T, M, 3]`;
- `query_points`: a copy of the input queries.

Use `--device cuda:1` to select another GPU and `--iters` to change the number
of refinement iterations.

## Odyssey training data

The training root must contain a `train` split. Samples may be grouped into any
number of sequence subdirectories:

```text
DATA_ROOT/
└── train/
    ├── sequence_001/
    │   ├── 000001.npz
    │   └── 000002.npz
    └── sequence_002/
        └── 000001.npz
```

Each training archive must contain:

- `pc_xyz_seq`: `[T, N, 3]` point clouds;
- `traj_seq`: `[T, M, 3]` target trajectories;
- `visibs`: `[T, M]` or `[T, M, 1]` visibility values;
- `valids`: `[T, M]` or `[T, M, 1]` validity values.

The loader changes the sign of the first two coordinate axes to match the
training convention used by PCSTracker.

## Training

Training uses DistributedDataParallel and requires one sequence per GPU. The
global `--batch-size` must therefore equal the number of processes. For four
GPUs:

```bash
torchrun --standalone --nproc_per_node=4 run_train.py \
  --data-root /path/to/PointOdyssey3D \
  --batch-size 4 \
  --output-dir exp
```

Experiment tracking is disabled by default. Enable it with
`--wandb-mode online --wandb-project PCSTracker` and optionally
`--wandb-entity`.

## License

See [LICENSE](LICENSE).
