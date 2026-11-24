"""1D convolutional denoiser for motion clips of shape (batch, channels, frames)."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Dilated conv -> GELU -> 1x1 conv, with a skip connection.

    Dilation lets later blocks see far-apart frames without extra parameters.
    'replicate' padding repeats the edge frames, same as the classical baselines,
    so the clip length is preserved.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation, padding_mode="replicate")
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.GELU()
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mix(self.act(self.norm(self.conv(x)))))


class MotionDenoiser(nn.Module):
    """Predicts the noise in a clip and subtracts it: output = x - noise_hat.

    Predicting the residual is easier than predicting clean motion directly,
    because the noise (jitter + spikes) is small and simple while the motion
    itself is large and varied.

    The output layer starts at zero, so an untrained model returns its input
    unchanged. Training therefore starts from the "do nothing" baseline instead
    of from random output.

    Fully convolutional, so it works on any number of frames.
    """

    def __init__(
            self,
            in_channels: int = 15,
            hidden_channels: int = 64,
            kernel_size: int = 5,
            dilations: tuple[int, ...] = (1, 2, 4, 8),
            dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the output stays aligned in time")
        if hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8 (GroupNorm groups)")

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, kernel_size, d, dropout) for d in dilations
        ])
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Number of frames each output frame can "see".
        self.receptive_field = 1 + sum((kernel_size - 1) * d for d in dilations)

    def predict_noise(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.blocks(self.input_proj(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.predict_noise(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MotionDenoiser()
    x = torch.randn(16, 15, 60)
    y = model(x)
    print(f"Input {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Receptive field: {model.receptive_field} frames")
    print("Untrained model is identity:", torch.equal(y, x))
    print("Works on other lengths:", tuple(model(torch.randn(2, 15, 120)).shape))
