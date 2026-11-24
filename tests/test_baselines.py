import numpy as np
import pytest
import torch

from src.baselines import (
    gaussian_filter,
    gaussian_kernel,
    median_filter,
    median_then_savgol,
    savgol_coefficients,
    savgol_filter,
)
from src.dataset import add_noise, generate_clean_clip


@pytest.mark.parametrize(
    "fn", [median_filter, savgol_filter, gaussian_filter, median_then_savgol]
)
def test_shape_and_dtype_preserved(fn):
    x = np.random.default_rng(0).normal(size=(4, 15, 60)).astype(np.float32)
    out = fn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_median_removes_single_spike():
    x = np.zeros((1, 20), dtype=np.float32)
    x[0, 10] = 5.0
    np.testing.assert_array_equal(median_filter(x, window=3), np.zeros_like(x))


def test_savgol_preserves_polynomials():
    # A degree-2 S-G filter reproduces a quadratic exactly away from the edges.
    t = np.arange(30, dtype=np.float64)
    x = (0.3 * t**2 - 2 * t + 1)[None, :]
    out = savgol_filter(x, window=7, polyorder=2)
    np.testing.assert_allclose(out[:, 3:-3], x[:, 3:-3], rtol=1e-9, atol=1e-9)


def test_savgol_coefficients_known_values():
    # Classic 5-point quadratic smoothing weights: (-3, 12, 17, 12, -3) / 35
    expected = np.array([-3, 12, 17, 12, -3]) / 35
    np.testing.assert_allclose(savgol_coefficients(5, 2), expected, atol=1e-12)


def test_gaussian_kernel_normalized_and_constant_preserved():
    assert gaussian_kernel(1.5).sum() == pytest.approx(1.0)
    x = np.full((3, 40), 2.5, dtype=np.float32)
    np.testing.assert_allclose(gaussian_filter(x, sigma=2.0), x, rtol=1e-6)


@pytest.mark.parametrize("window", [0, 4, -1])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError):
        median_filter(np.zeros((1, 10)), window=window)


def test_filters_reduce_error_on_synthetic_clip():
    clean = generate_clean_clip(np.random.default_rng(1))
    noisy = add_noise(torch.from_numpy(clean), generator=torch.Generator().manual_seed(1)).numpy()
    noisy_err = np.mean((noisy - clean) ** 2)
    for fn in (median_filter, savgol_filter, gaussian_filter, median_then_savgol):
        assert np.mean((fn(noisy) - clean) ** 2) < noisy_err
