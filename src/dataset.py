from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def generate_clean_clip(
        rng: np.random.Generator,
        num_frames: int = 60,
        num_joints: int = 5) -> np.ndarray:
    """Generate one clean motion clip of shape (num_joints * 3, num_frames)."""
    t = np.linspace(0.0, 2.0 * np.pi, num_frames, dtype=np.float32)
    num_features = num_joints * 3
    clean = np.zeros((num_features, num_frames), dtype=np.float32)

    base_freq = rng.uniform(0.8, 2.2)
    base_phase = rng.uniform(0.0, 2.0 * np.pi)

    for joint in range(num_joints):
        phase_lag = base_phase + joint * 0.35
        amp_x = rng.uniform(0.4, 1.0) * (1.0 + 0.88 * joint)
        amp_y = rng.uniform(0.3, 0.8)
        amp_z = rng.uniform(0.4, 1.0)

        idx = joint * 3
        clean[idx] = amp_x * np.sin(base_freq * t + phase_lag)
        clean[idx + 1] = amp_y * np.cos(base_freq * 1.5 * t + phase_lag)
        clean[idx + 2] = amp_z * np.sin(base_freq * 0.5 * t + 0.5 + phase_lag)

    return clean


def add_noise(
        clean: torch.Tensor,
        jitter_std: float = 0.04,
        spike_prob: float = 0.03,
        spike_scale: float = 0.35,
        generator: torch.Generator | None = None) -> torch.Tensor:
    """Add Gaussian jitter and sparse spikes. Works on (C, T) or (N, C, T)."""
    jitter = torch.randn(clean.shape, generator=generator) * jitter_std
    spike_mask = torch.rand(clean.shape, generator=generator) < spike_prob
    spikes = torch.randn(clean.shape, generator=generator) * spike_scale * spike_mask
    return clean + jitter + spikes


@dataclass
class Normalizer:
    """Per-channel standardization, fit on training clean clips only."""
    mean: torch.Tensor  # (C, 1)
    std: torch.Tensor   # (C, 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class MotionDenoisingDataset(Dataset):
    """Pairs of (noisy, clean) clips, both normalized.

    Noise is added in the original coordinate space (so the noise parameters keep
    their meaning) and both clips are then normalized with the same statistics.

    resample_noise=True  -> fresh noise on every access (use for training).
    resample_noise=False -> noise generated once from `seed` (use for val/test,
                            so evaluation is deterministic).
    """

    def __init__(
            self,
            clean_clips: torch.Tensor,
            normalizer: Normalizer,
            jitter_std: float = 0.04,
            spike_prob: float = 0.03,
            spike_scale: float = 0.35,
            resample_noise: bool = False,
            seed: int = 0) -> None:
        self.clean_clips = clean_clips
        self.normalizer = normalizer
        self.jitter_std = jitter_std
        self.spike_prob = spike_prob
        self.spike_scale = spike_scale
        self.resample_noise = resample_noise

        self.noisy_clips: torch.Tensor | None = None
        if not resample_noise:
            generator = torch.Generator().manual_seed(seed)
            self.noisy_clips = add_noise(
                clean_clips, jitter_std, spike_prob, spike_scale, generator)

    def __len__(self) -> int:
        return self.clean_clips.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_clips[idx]
        if self.noisy_clips is not None:
            noisy = self.noisy_clips[idx]
        else:
            noisy = add_noise(clean, self.jitter_std, self.spike_prob, self.spike_scale)
        return self.normalizer.normalize(noisy), self.normalizer.normalize(clean)


def create_dataloaders(
        num_samples: int = 1000,
        num_frames: int = 60,
        num_joints: int = 5,
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.15,
        jitter_std: float = 0.04,
        spike_prob: float = 0.03,
        spike_scale: float = 0.35,
        seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer]:
    """Create reproducible train/val/test DataLoaders plus the fitted Normalizer.

    Clean clips are split first, so the same clip never appears in two splits.
    """
    rng = np.random.default_rng(seed)
    clean_all = torch.from_numpy(np.stack([
        generate_clean_clip(rng, num_frames=num_frames, num_joints=num_joints)
        for _ in range(num_samples)
    ]))  # (N, C, T)

    val_size = int(num_samples * val_split)
    test_size = int(num_samples * test_split)
    train_size = num_samples - val_size - test_size
    if train_size <= 0:
        raise ValueError("val_split + test_split leaves no training data")

    perm = np.random.default_rng(seed + 1).permutation(num_samples)
    train_idx = torch.from_numpy(perm[:train_size])
    val_idx = torch.from_numpy(perm[train_size:train_size + val_size])
    test_idx = torch.from_numpy(perm[train_size + val_size:])

    train_clean = clean_all[train_idx]
    val_clean = clean_all[val_idx]
    test_clean = clean_all[test_idx]

    # Fit normalization on training clean clips only (no leakage from val/test).
    mean = train_clean.mean(dim=(0, 2)).unsqueeze(1)
    std = train_clean.std(dim=(0, 2)).unsqueeze(1).clamp_min(1e-6)
    normalizer = Normalizer(mean=mean, std=std)

    noise_kwargs = dict(
        jitter_std=jitter_std, spike_prob=spike_prob, spike_scale=spike_scale)
    train_ds = MotionDenoisingDataset(
        train_clean, normalizer, resample_noise=True, **noise_kwargs)
    val_ds = MotionDenoisingDataset(
        val_clean, normalizer, resample_noise=False, seed=seed + 2, **noise_kwargs)
    test_ds = MotionDenoisingDataset(
        test_clean, normalizer, resample_noise=False, seed=seed + 3, **noise_kwargs)

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, normalizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        num_samples=200, batch_size=16, seed=42)
    x_batch, y_batch = next(iter(train_loader))
    print(f"Train/val/test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print(f"Batch X shape: {x_batch.shape} | Batch Y shape: {y_batch.shape}")

    # Normalization: clean training targets should be ~zero mean, ~unit std per channel.
    train_ds = train_loader.dataset
    y_all = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    print(f"Normalized clean mean (abs max over channels): {y_all.mean(dim=(0, 2)).abs().max():.4f}")
    print(f"Normalized clean std range: {y_all.std(dim=(0, 2)).min():.3f} - {y_all.std(dim=(0, 2)).max():.3f}")

    # Train noise is resampled each access; val noise is fixed.
    print("Train noise resampled:", not torch.equal(train_ds[0][0], train_ds[0][0]))
    val_ds = val_loader.dataset
    print("Val noise fixed:", torch.equal(val_ds[0][0], val_ds[0][0]))








