"""Tests for the data -> model -> train -> evaluate pipeline.

Everything uses small datasets and few epochs so the whole file runs in seconds.
"""

import json
import sys

import numpy as np
import pytest
import torch

from src import evaluate as evaluate_module
from src import train as train_module
from src.dataset import Normalizer, add_noise, create_dataloaders, generate_clean_clip
from src.evaluate import compute_metrics, load_model, spike_mask_from_noise
from src.model import MotionDenoiser
from src.train import denoising_loss, evaluate, set_seed, train_one_epoch

SMALL = dict(num_samples=100, batch_size=16, seed=0)


@pytest.fixture(scope="module")
def loaders():
    return create_dataloaders(**SMALL)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def test_generate_clean_clip_shape():
    clip = generate_clean_clip(np.random.default_rng(0), num_frames=40, num_joints=3)
    assert clip.shape == (9, 40)
    assert clip.dtype == np.float32


def test_add_noise_is_reproducible_with_generator():
    clean = torch.zeros(15, 60)
    a = add_noise(clean, generator=torch.Generator().manual_seed(1))
    b = add_noise(clean, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    assert not torch.equal(a, clean)


def test_split_sizes_and_batch_shapes(loaders):
    train_loader, val_loader, test_loader, _ = loaders
    sizes = [len(dl.dataset) for dl in (train_loader, val_loader, test_loader)]
    assert sizes == [70, 15, 15]
    noisy, clean = next(iter(train_loader))
    assert noisy.shape == clean.shape == (16, 15, 60)


def test_splits_do_not_share_clips(loaders):
    train_loader, val_loader, test_loader, _ = loaders
    clips = [dl.dataset.clean_clips.flatten(1) for dl in (train_loader, val_loader, test_loader)]
    for i in range(3):
        for j in range(i + 1, 3):
            # Pairwise distance between every clip in split i and split j.
            assert torch.cdist(clips[i], clips[j]).min() > 0


def test_same_seed_gives_same_data():
    a = create_dataloaders(**SMALL)
    b = create_dataloaders(**SMALL)
    for dl_a, dl_b in zip(a[1:3], b[1:3]):  # val and test (fixed noise)
        assert torch.equal(dl_a.dataset.noisy_clips, dl_b.dataset.noisy_clips)
    assert torch.equal(a[0].dataset.clean_clips, b[0].dataset.clean_clips)


def test_normalizer_fit_on_train_only(loaders):
    train_loader, _, _, normalizer = loaders
    normed = normalizer.normalize(train_loader.dataset.clean_clips)
    assert normed.mean(dim=(0, 2)).abs().max() < 1e-4
    assert torch.allclose(normed.std(dim=(0, 2)), torch.ones(15), atol=1e-3)


def test_normalizer_round_trip():
    norm = Normalizer(mean=torch.randn(15, 1), std=torch.rand(15, 1) + 0.5)
    x = torch.randn(4, 15, 60)
    assert torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-5)


def test_train_noise_resampled_val_noise_fixed(loaders):
    train_loader, val_loader, _, _ = loaders
    assert not torch.equal(train_loader.dataset[0][0], train_loader.dataset[0][0])
    assert torch.equal(val_loader.dataset[0][0], val_loader.dataset[0][0])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_model_preserves_shape_for_any_length():
    model = MotionDenoiser()
    for frames in (1, 37, 60, 200):
        x = torch.randn(2, 15, frames)
        assert model(x).shape == x.shape


def test_untrained_model_is_identity():
    model = MotionDenoiser()
    x = torch.randn(3, 15, 60)
    assert torch.equal(model(x), x)


def test_receptive_field_covers_clip():
    assert MotionDenoiser().receptive_field >= 60


@pytest.mark.parametrize("kwargs", [{"kernel_size": 4}, {"hidden_channels": 30}])
def test_model_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MotionDenoiser(**kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_velocity_loss_term():
    clean = torch.randn(2, 15, 60)
    pred = clean + torch.randn_like(clean)
    plain = torch.nn.functional.mse_loss(pred, clean)
    assert torch.allclose(denoising_loss(pred, clean, velocity_weight=0.0), plain)
    assert denoising_loss(pred, clean, velocity_weight=1.0) > plain
    # A constant offset is a position error but has zero velocity error.
    offset = clean + 0.5
    assert torch.allclose(denoising_loss(offset, clean, 1.0), denoising_loss(offset, clean, 0.0))


def test_short_training_beats_noisy_input(loaders):
    set_seed(0)
    train_loader, val_loader, _, _ = loaders
    model = MotionDenoiser(hidden_channels=32)
    device = torch.device("cpu")
    noisy_mse, noisy_vel = evaluate(model, val_loader, device)  # identity at init

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(15):
        train_one_epoch(model, train_loader, optimizer, device, grad_clip=1.0, velocity_weight=1.0)

    # After 15 epochs this reaches ~0.37x (position) and ~0.13x (velocity) of the
    # noisy input across seeds; thresholds leave a wide margin to avoid flakiness.
    val_mse, val_vel = evaluate(model, val_loader, device)
    assert val_mse < 0.6 * noisy_mse
    assert val_vel < 0.3 * noisy_vel


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_prediction_is_zero():
    norm = Normalizer(mean=torch.zeros(15, 1), std=torch.full((15, 1), 2.0))
    clean = np.random.default_rng(0).normal(size=(4, 15, 60)).astype(np.float32)
    mask = np.zeros_like(clean, dtype=bool)
    mask[0, 0, 0] = True
    metrics = compute_metrics(clean, clean, norm, mask)
    assert all(v == 0.0 for v in metrics.values())


def test_compute_metrics_reports_original_units():
    std = 2.0
    norm = Normalizer(mean=torch.zeros(15, 1), std=torch.full((15, 1), std))
    clean = np.zeros((1, 15, 60), dtype=np.float32)
    pred = clean + 0.1  # 0.1 in normalized units = 0.2 in original units
    metrics = compute_metrics(pred, clean, norm, np.zeros_like(clean, dtype=bool))
    assert metrics["mse_norm"] == pytest.approx(0.01, rel=1e-5)
    assert metrics["rmse"] == pytest.approx(0.1 * std, rel=1e-5)
    assert metrics["velocity_rmse"] == pytest.approx(0.0, abs=1e-6)


def test_spike_mask_threshold():
    clean = np.zeros((1, 2, 4))
    noisy = np.array([[[0.0, 0.1, 0.2, -0.5], [0.15, 0.17, 0.0, 0.0]]])
    mask = spike_mask_from_noise(noisy, clean, jitter_std=0.04)  # threshold 0.16
    expected = np.array([[[False, False, True, True], [False, True, False, False]]])
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# End to end: train.py main -> checkpoint -> evaluate.py main
# ---------------------------------------------------------------------------

def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train", "--num-samples", "100", "--batch-size", "16", "--epochs", "3",
        "--hidden-channels", "16", "--output-dir", str(tmp_path), "--device", "cpu"])
    train_module.main()

    checkpoint_path = tmp_path / "best_model.pt"
    history = json.loads((tmp_path / "history.json").read_text())
    assert checkpoint_path.exists()
    assert [h["epoch"] for h in history] == [1, 2, 3]

    # Checkpoint restores the exact same model.
    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert checkpoint["train_config"]["velocity_weight"] == 1.0
    x = torch.randn(2, 15, 60)
    reloaded, _ = load_model(checkpoint_path, torch.device("cpu"))
    assert torch.equal(model(x), reloaded(x))

    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--checkpoint", str(checkpoint_path),
        "--output-dir", str(tmp_path), "--device", "cpu"])
    evaluate_module.main()

    results = json.loads((tmp_path / "results.json").read_text())
    assert set(results["test_metrics"]) >= {"noisy input", "median+savgol", "neural net"}
    for name in ("example_clip.png", "loss_curve.png", "robustness.png"):
        assert (tmp_path / name).stat().st_size > 0
