"""Classical (non-learned) denoising baselines.

All filters operate along the LAST axis (time), so they accept a single clip
of shape (features, frames) or a batch of shape (batch, features, frames).
Edges are handled by repeating the first/last frame ("edge" padding), so the
output always has the same shape as the input.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.dataset import create_dataloaders


def _check_window(window: int) -> int:
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    return window // 2


def _pad_time(x: np.ndarray, half: int) -> np.ndarray:
    pad = [(0, 0)] * (x.ndim - 1) + [(half, half)]
    return np.pad(x, pad, mode="edge")


def _filter_time(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a symmetric-length FIR kernel along the time axis."""
    half = len(kernel) // 2
    windows = sliding_window_view(_pad_time(x, half), len(kernel), axis=-1)
    return windows @ kernel


def median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Sliding median along time. Removes isolated spikes without blurring.

    A spike that lasts fewer than window // 2 + 1 frames is removed completely.
    """
    half = _check_window(window)
    windows = sliding_window_view(_pad_time(x, half), window, axis=-1)
    return np.median(windows, axis=-1).astype(x.dtype)


def savgol_coefficients(window: int, polyorder: int) -> np.ndarray:
    """Weights that fit a degree-`polyorder` polynomial to each window by least
    squares and return its value at the center frame."""
    half = _check_window(window)
    if polyorder >= window:
        raise ValueError("polyorder must be less than window")
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    vander = np.vander(offsets, polyorder + 1, increasing=True)  # (window, p+1)
    # Row 0 of the pseudo-inverse gives the constant term of the fit = value at offset 0.
    return np.linalg.pinv(vander)[0]


def savgol_filter(x: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing. Removes jitter but keeps peaks sharper than a
    moving average does."""
    kernel = savgol_coefficients(window, polyorder)
    return _filter_time(x, kernel).astype(x.dtype)


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(truncate * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_filter(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smoothing along time."""
    return _filter_time(x, gaussian_kernel(sigma)).astype(x.dtype)


def median_then_savgol(
    x: np.ndarray, median_window: int = 5, window: int = 7, polyorder: int = 2
) -> np.ndarray:
    """Remove spikes with a median filter, then smooth the remaining jitter."""
    return savgol_filter(median_filter(x, median_window), window, polyorder)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _stack(loader) -> tuple[np.ndarray, np.ndarray]:
    noisy, clean = zip(*[(x.numpy(), y.numpy()) for x, y in loader])
    return np.concatenate(noisy), np.concatenate(clean)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


# Each baseline gets a small parameter grid. We pick the best parameters on the
# VAL split and report on TEST, the same treatment the neural net gets
# (model selection on val, final number on test).
#
# The loaders yield normalized clips. All these filters commute with the
# per-channel affine normalization (filter(a*x + b) == a*filter(x) + b), so
# filtering normalized data is equivalent to filtering raw data, and the MSE
# is in the same normalized units the neural net will be trained on.
BASELINE_GRIDS = {
    "median": (median_filter, [{"window": w} for w in (3, 5, 7, 9)]),
    "savgol": (
        savgol_filter,
        [{"window": w, "polyorder": p} for w in (5, 7, 9, 11, 15) for p in (2, 3) if p < w],
    ),
    "gaussian": (gaussian_filter, [{"sigma": s} for s in (0.5, 1.0, 1.5, 2.0, 3.0)]),
    "median+savgol": (
        median_then_savgol,
        [
            {"median_window": m, "window": w, "polyorder": 2}
            for m in (3, 5, 7)
            for w in (5, 7, 9, 11)
        ],
    ),
}


def tune_baselines(val_x: np.ndarray, val_y: np.ndarray) -> dict[str, tuple]:
    """Grid-search each baseline on val. Returns {name: (fn, best_params)}."""
    tuned = {}
    for name, (fn, grid) in BASELINE_GRIDS.items():
        best = min(grid, key=lambda params: mse(fn(val_x, **params), val_y))
        tuned[name] = (fn, best)
    return tuned


def tune_and_evaluate(val_loader, test_loader) -> dict[str, tuple[dict, float]]:
    """Returns {name: (best_params, test_mse)}."""
    test_x, test_y = _stack(test_loader)
    results = {"identity (noisy)": ({}, mse(test_x, test_y))}
    for name, (fn, best) in tune_baselines(*_stack(val_loader)).items():
        results[name] = (best, mse(fn(test_x, **best), test_y))
    return results


if __name__ == "__main__":
    _, val_loader, test_loader, _ = create_dataloaders(num_samples=1000, batch_size=32, seed=42)
    results = tune_and_evaluate(val_loader, test_loader)

    noisy_mse = results["identity (noisy)"][1]
    print(f"{'method':<18} {'test MSE':>10} {'vs noisy':>9}  best params (tuned on val)")
    for name, (params, score) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        reduction = 100.0 * (1.0 - score / noisy_mse)
        print(f"{name:<18} {score:>10.5f} {reduction:>8.1f}%  {params}")
