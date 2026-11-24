"""Final evaluation on the held-out TEST split: neural net vs. classical baselines.

Produces:
  * a metrics table (printed + outputs/results.json)
  * a robustness table: how each method holds up at noise levels never seen in training
  * plots: outputs/example_clip.png, outputs/loss_curve.png, outputs/robustness.png

Usage:
    python -m src.evaluate
    python -m src.evaluate --checkpoint outputs/best_model.pt --clip 3
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baselines import _stack, tune_baselines
from src.dataset import Normalizer, add_noise, create_dataloaders
from src.model import MotionDenoiser

# A method maps normalized noisy clips (N, C, T) -> normalized denoised clips.
Method = Callable[[np.ndarray], np.ndarray]

ROBUSTNESS_LEVELS = (1.0, 1.5, 2.0, 3.0)  # multipliers on jitter_std and spike_prob


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path: Path, device: torch.device) -> tuple[MotionDenoiser, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = MotionDenoiser(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def make_net_method(model: MotionDenoiser, device: torch.device, batch_size: int = 256) -> Method:
    @torch.no_grad()
    def run(x: np.ndarray) -> np.ndarray:
        out = [model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu().numpy()
               for i in range(0, len(x), batch_size)]
        return np.concatenate(out)
    return run


def build_methods(val_x, val_y, model, device) -> tuple[dict[str, Method], dict[str, dict]]:
    """All methods to compare, with baselines tuned on val."""
    methods: dict[str, Method] = {"noisy input": lambda x: x}
    params: dict[str, dict] = {"noisy input": {}}
    for name, (fn, best) in tune_baselines(val_x, val_y).items():
        methods[name] = lambda x, fn=fn, best=best: fn(x, **best)
        params[name] = best
    methods["neural net"] = make_net_method(model, device)
    params["neural net"] = {}
    return methods, params


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
        pred_norm: np.ndarray,
        clean_norm: np.ndarray,
        normalizer: Normalizer,
        spike_mask: np.ndarray) -> dict[str, float]:
    """
    mse_norm      MSE in normalized units, the same quantity the net is trained on.
    rmse          root-mean-square error in original coordinate units.
    rmse_spike    RMSE only on values hit by a spike (how well outliers are removed).
    rmse_jitter   RMSE on everything else (how well small jitter is smoothed).
    velocity_rmse RMSE of frame-to-frame differences: penalizes wobbly output
                  even when positions are close. Matters a lot for animation.
    """
    mean, std = normalizer.mean.numpy(), normalizer.std.numpy()
    pred = pred_norm * std + mean
    clean = clean_norm * std + mean
    err = pred - clean
    vel_err = np.diff(pred, axis=-1) - np.diff(clean, axis=-1)
    return {
        "mse_norm": float(np.mean((pred_norm - clean_norm) ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmse_spike": float(np.sqrt(np.mean(err[spike_mask] ** 2))) if spike_mask.any() else float("nan"),
        "rmse_jitter": float(np.sqrt(np.mean(err[~spike_mask] ** 2))),
        "velocity_rmse": float(np.sqrt(np.mean(vel_err ** 2))),
    }


def spike_mask_from_noise(noisy_raw: np.ndarray, clean_raw: np.ndarray, jitter_std: float) -> np.ndarray:
    """Values whose noise exceeds 4 jitter standard deviations.

    Jitter alone exceeds 4 std with probability ~6e-5, so this is almost exactly
    the set of spikes. (Tiny spikes fall below the threshold, but those are
    indistinguishable from jitter anyway.)
    """
    return np.abs(noisy_raw - clean_raw) > 4.0 * jitter_std


def robustness_sweep(methods, clean_raw: torch.Tensor, normalizer: Normalizer,
                     jitter_std: float, spike_prob: float, spike_scale: float,
                     seed: int) -> dict[float, dict[str, float]]:
    """Re-noise the test clips at higher noise levels. Every method keeps the
    settings it got at the training noise level, so nothing is re-tuned."""
    clean_norm = normalizer.normalize(clean_raw).numpy()
    results = {}
    for i, level in enumerate(ROBUSTNESS_LEVELS):
        generator = torch.Generator().manual_seed(seed + 100 + i)
        noisy_raw = add_noise(clean_raw, jitter_std * level, min(1.0, spike_prob * level),
                              spike_scale, generator)
        noisy_norm = normalizer.normalize(noisy_raw).numpy()
        results[level] = {name: float(np.mean((fn(noisy_norm) - clean_norm) ** 2))
                          for name, fn in methods.items()}
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_example_clip(path, noisy, clean, baseline, net, baseline_name, spike_mask, clip_idx):
    """Three channels of one clip: the one with the largest spike, plus two others."""
    num_channels = clean.shape[1]
    spike_channel = int(np.argmax(np.abs(noisy[clip_idx] - clean[clip_idx]).max(axis=-1)))
    others = [c for c in (num_channels - 3, 1) if c != spike_channel]
    channels = [spike_channel] + others[:2]
    axis_names = "xyz"

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.8 * len(channels)), sharex=True)
    frames = np.arange(clean.shape[-1])
    for ax, ch in zip(axes, channels):
        ax.plot(frames, noisy[clip_idx, ch], color="0.6", marker=".", lw=0.8, label="noisy")
        ax.plot(frames, clean[clip_idx, ch], color="black", lw=2, label="clean (target)")
        ax.plot(frames, baseline[clip_idx, ch], color="tab:orange", lw=1.5, label=baseline_name)
        ax.plot(frames, net[clip_idx, ch], color="tab:blue", lw=1.5, ls="--", label="neural net")
        spikes = np.where(spike_mask[clip_idx, ch])[0]
        ax.scatter(spikes, noisy[clip_idx, ch, spikes], color="red", s=25, zorder=5,
                   label="spike" if len(spikes) else None)
        ax.set_ylabel(f"joint {ch // 3} {axis_names[ch % 3]}")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=5)
    axes[0].set_title(f"Test clip {clip_idx} (original units)")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss_curve(path, history_path: Path, baseline_val_mse: float, baseline_name: str):
    history = json.loads(history_path.read_text())
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # train_loss includes the velocity term (if any), so it is not directly
    # comparable to the MSE lines; it is plotted to show convergence.
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss", alpha=0.7)
    ax.plot(epochs, [h["val_mse"] for h in history], label="val position MSE")
    ax.axhline(baseline_val_mse, color="tab:orange", ls="--", label=f"{baseline_name} val MSE")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / MSE (normalized)")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_robustness(path, sweep: dict[float, dict[str, float]]):
    levels = list(sweep)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in next(iter(sweep.values())):
        ax.plot(levels, [sweep[lv][name] for lv in levels], marker="o", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("noise level (x training jitter_std and spike_prob)")
    ax.set_ylabel("test MSE (normalized)")
    ax.set_title("Robustness to unseen noise levels")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(rows: dict[str, dict[str, float]], columns: list[str], title: str) -> None:
    print(f"\n{title}")
    print(f"{'method':<16}" + "".join(f"{c:>15}" for c in columns))
    for name, row in rows.items():
        print(f"{name:<16}" + "".join(f"{row[c]:>15.5f}" for c in columns))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--clip", type=int, default=0, help="test clip index to plot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(args.checkpoint, device)
    # Rebuild exactly the data the model was trained with (same seed -> same splits/noise).
    _, val_loader, test_loader, normalizer = create_dataloaders(**checkpoint["data_config"])
    test_ds = test_loader.dataset
    jitter_std, spike_prob, spike_scale = test_ds.jitter_std, test_ds.spike_prob, test_ds.spike_scale

    val_x, val_y = _stack(val_loader)
    test_x, test_y = _stack(test_loader)
    methods, params = build_methods(val_x, val_y, model, device)

    spike_mask = spike_mask_from_noise(
        test_ds.noisy_clips.numpy(), test_ds.clean_clips.numpy(), jitter_std)
    predictions = {name: fn(test_x) for name, fn in methods.items()}
    results = {name: compute_metrics(pred, test_y, normalizer, spike_mask)
               for name, pred in predictions.items()}

    velocity_weight = checkpoint.get("train_config", {}).get("velocity_weight", 0.0)
    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, "
          f"val MSE {checkpoint['val_mse']:.5f}, velocity weight {velocity_weight})")
    print(f"Test set: {len(test_x)} clips | {100 * spike_mask.mean():.1f}% of values hit by spikes")
    print_table(results, ["mse_norm", "rmse", "rmse_spike", "rmse_jitter", "velocity_rmse"],
                "TEST metrics (rmse columns in original units; lower is better)")

    baselines = [n for n in results if n not in ("noisy input", "neural net")]
    best_baseline = min(baselines, key=lambda n: results[n]["mse_norm"])
    net_mse, base_mse = results["neural net"]["mse_norm"], results[best_baseline]["mse_norm"]
    verdict = "beats" if net_mse < base_mse else "does NOT beat"
    print(f"\nNeural net {verdict} best baseline ({best_baseline}, {params[best_baseline]}): "
          f"{net_mse:.5f} vs {base_mse:.5f} ({100 * (1 - net_mse / base_mse):+.1f}%)")

    sweep = robustness_sweep(methods, test_ds.clean_clips, normalizer,
                             jitter_std, spike_prob, spike_scale, checkpoint["data_config"]["seed"])
    print_table({name: {f"{lv}x": sweep[lv][name] for lv in sweep} for name in methods},
                [f"{lv}x" for lv in sweep],
                "ROBUSTNESS: test MSE (normalized) at noise levels x training level")

    # Plots (in original units)
    mean, std = normalizer.mean.numpy(), normalizer.std.numpy()
    raw = {name: pred * std + mean for name, pred in predictions.items()}
    plot_example_clip(args.output_dir / "example_clip.png", raw["noisy input"],
                      test_y * std + mean, raw[best_baseline], raw["neural net"],
                      best_baseline, spike_mask, args.clip)
    history_path = args.checkpoint.parent / "history.json"
    if history_path.exists():
        val_baseline = float(np.mean((methods[best_baseline](val_x) - val_y) ** 2))
        plot_loss_curve(args.output_dir / "loss_curve.png", history_path, val_baseline, best_baseline)
    plot_robustness(args.output_dir / "robustness.png", sweep)

    (args.output_dir / "results.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "test_metrics": results,
        "baseline_params": params,
        "best_baseline": best_baseline,
        "robustness_mse_norm": {str(lv): row for lv, row in sweep.items()},
    }, indent=2))
    print(f"\nSaved results.json and plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
