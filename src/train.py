"""Train the MotionDenoiser and save the best checkpoint (by val loss).

Usage:
    python -m src.train
    python -m src.train --epochs 200 --lr 2e-3 --hidden-channels 96
    python -m src.train --velocity-weight 0      # position MSE only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import create_dataloaders
from src.model import MotionDenoiser, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def denoising_loss(pred: torch.Tensor, clean: torch.Tensor, velocity_weight: float) -> torch.Tensor:
    """Position MSE + velocity_weight * velocity MSE.

    The velocity term compares frame-to-frame differences. Plain position MSE
    tolerates small high-frequency wobble (it is only a small position error),
    but wobble shows up strongly in velocity error, so this term pushes the net
    toward smooth output, which is what matters for animation.
    """
    loss = nn.functional.mse_loss(pred, clean)
    if velocity_weight > 0:
        loss = loss + velocity_weight * nn.functional.mse_loss(
            pred.diff(dim=-1), clean.diff(dim=-1))
    return loss


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float,
        velocity_weight: float) -> float:
    """Returns the mean training loss (position + weighted velocity)."""
    model.train()
    total, count = 0.0, 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        loss = denoising_loss(model(noisy), clean, velocity_weight)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += loss.item() * noisy.shape[0]
        count += noisy.shape[0]
    return total / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Position MSE and velocity MSE over every element of the split
    (not an average of batch means)."""
    model.eval()
    pos_sum, vel_sum, pos_count, vel_count = 0.0, 0.0, 0, 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        pos_sum += nn.functional.mse_loss(pred, clean, reduction="sum").item()
        vel_sum += nn.functional.mse_loss(
            pred.diff(dim=-1), clean.diff(dim=-1), reduction="sum").item()
        pos_count += clean.numel()
        vel_count += clean[..., 1:].numel()
    return pos_sum / pos_count, vel_sum / vel_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-frames", type=int, default=60)
    p.add_argument("--num-joints", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    # model
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--velocity-weight", type=float, default=1.0,
                   help="weight of the velocity (frame-difference) loss term; 0 = position MSE only")
    p.add_argument("--patience", type=int, default=30,
                   help="stop after this many epochs without val improvement")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_config = dict(
        num_samples=args.num_samples, num_frames=args.num_frames,
        num_joints=args.num_joints, batch_size=args.batch_size, seed=args.seed)
    train_loader, val_loader, _, normalizer = create_dataloaders(**data_config)

    model_config = dict(
        in_channels=args.num_joints * 3, hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size, dilations=tuple(args.dilations), dropout=args.dropout)
    model = MotionDenoiser(**model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    noisy_mse, noisy_vel = evaluate(model, val_loader, device)  # untrained model == identity
    print(f"Device: {device} | Parameters: {count_parameters(model):,} | "
          f"velocity weight: {args.velocity_weight}")
    print(f"Noisy input (do-nothing baseline): val MSE {noisy_mse:.5f} | "
          f"val velocity MSE {noisy_vel:.5f}")

    checkpoint_path = args.output_dir / "best_model.pt"
    history: list[dict] = []
    best_loss, best_epoch, best_mse, best_vel = float("inf"), 0, float("nan"), float("nan")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, args.grad_clip, args.velocity_weight)
        val_mse, val_vel = evaluate(model, val_loader, device)
        # Select checkpoints on the same objective we train on.
        val_loss = val_mse + args.velocity_weight * val_vel
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_mse=val_mse, val_velocity_mse=val_vel, lr=lr))

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, best_mse, best_vel = val_loss, epoch, val_mse, val_vel
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model_config,
                "data_config": data_config,
                "train_config": {"velocity_weight": args.velocity_weight, "lr": args.lr,
                                 "epochs": args.epochs, "weight_decay": args.weight_decay},
                "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                "epoch": epoch,
                "val_mse": val_mse,
                "val_velocity_mse": val_vel,
            }, checkpoint_path)

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:4d} | train loss {train_loss:.5f} | val loss {val_loss:.5f} "
                  f"(mse {val_mse:.5f}, vel {val_vel:.5f}) | lr {lr:.2e}{'  *' if improved else ''}")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest epoch {best_epoch}: val MSE {best_mse:.5f} "
          f"({100 * (1 - best_mse / noisy_mse):.1f}% below noisy), "
          f"val velocity MSE {best_vel:.5f} ({100 * (1 - best_vel / noisy_vel):.1f}% below noisy) "
          f"| {time.time() - start:.0f}s")
    print(f"Saved checkpoint to {checkpoint_path} and history to {args.output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
