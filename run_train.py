"""Distributed Odyssey training entry point for PCSTracker."""

import argparse
import datetime
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from PCSTracker import PCSTracker
from data.odyssey_dataset import OdysseyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCSTracker on Odyssey data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("exp"))
    parser.add_argument("--exp-name", default="train_odyssey")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--track-point-num", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1, help="global batch size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="PCSTracker")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    minutes, _ = divmod(max(seconds, 0), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{int(days)}/{int(hours)}/{int(minutes)}"


def initialize_distributed(args):
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError(
            "distributed training is required; launch with torchrun --nproc_per_node=<gpus>"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PCSTracker training")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, dist.get_rank(), dist.get_world_size()


def create_tracker(args, is_master: bool, run_name: str):
    if args.wandb_mode == "disabled" or not is_master:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        mode=args.wandb_mode,
    )


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = initialize_distributed(args)
    is_master = rank == 0
    if args.batch_size != world_size:
        raise ValueError(
            "the model requires one sequence per GPU, so --batch-size must equal WORLD_SIZE"
        )
    if min(args.max_steps, args.log_every, args.save_every) < 1:
        raise ValueError("step and interval arguments must be positive")

    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", local_rank)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.exp_name}_{timestamp}_pcstracker"
    checkpoint_dir = args.output_dir.expanduser().resolve() / run_name / "checkpoints"
    if is_master:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    tracker = create_tracker(args, is_master, run_name)

    dataset = OdysseyDataset(
        root=args.data_root,
        seq_len=args.seq_len,
        track_point_num=args.track_point_num,
        split="train",
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if not loader:
        raise RuntimeError("training loader is empty")

    model = PCSTracker().to(device)
    if is_master:
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(f"Trainable parameters: {parameter_count:,}")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).train()
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.max_steps,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    optimizer.zero_grad(set_to_none=True)
    step = 0
    epoch = 0
    start_time = interval_start = time.time()
    try:
        while step < args.max_steps:
            sampler.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}",
                disable=not is_master,
                ncols=100,
            )
            for batch in progress:
                step += 1
                loss, metrics = model.module.training_infer(model, batch, device)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                metric_values = dict(metrics)
                if tracker is not None:
                    tracker.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            **{
                                f"train/{name}": value
                                for name, value in metric_values.items()
                            },
                        },
                        step=step,
                    )
                if is_master and step % args.log_every == 0:
                    now = time.time()
                    elapsed = now - start_time
                    interval = now - interval_start
                    interval_start = now
                    remaining = interval * (args.max_steps - step) / args.log_every
                    summary = "  ".join(
                        f"{name}: {value:.3f}" for name, value in metric_values.items()
                    )
                    progress.write(
                        f"[{step}/{args.max_steps}] loss: {loss.item():.3f}  "
                        f"{summary}  lr: {scheduler.get_last_lr()[0]:.6f}  "
                        f"time: [{format_duration(elapsed)},{format_duration(remaining)}]"
                    )
                if is_master and step % args.save_every == 0:
                    torch.save(
                        model.module.state_dict(),
                        checkpoint_dir / f"model_step_{step}.pth",
                    )
                if step >= args.max_steps:
                    break
            epoch += 1

        if is_master:
            torch.save(
                model.module.state_dict(), checkpoint_dir / "model_step_finished.pth"
            )
    finally:
        if tracker is not None:
            tracker.finish()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

