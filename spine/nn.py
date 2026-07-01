"""Shared convolutional and activation primitives for the Spine codec."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def kaiser_sinc_filter(cutoff: float, kernel_size: int, beta: float) -> torch.Tensor:
    """Linear-phase Kaiser-windowed sinc lowpass FIR; ``cutoff`` in cycles/sample (0.5 = Nyquist)."""
    half = kernel_size // 2
    if kernel_size % 2 == 0:
        n = torch.arange(kernel_size).float() - half + 0.5
    else:
        n = torch.arange(kernel_size).float() - half
    fir = torch.sinc(2 * cutoff * n) * torch.kaiser_window(kernel_size, periodic=False, beta=beta)
    return fir / fir.sum()


def WNConv1d(*args: Any, **kwargs: Any) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args: Any, **kwargs: Any) -> nn.ConvTranspose1d:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Snake(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


class FilteredSnake(nn.Module):
    """Anti-aliased Snake: nearest-upsample ×2 → Snake → sinc lowpass → decimate."""

    fir_filter: torch.Tensor

    def __init__(
        self,
        channels: int,
        upsample_factor: int = 2,
        cutoff: float = 0.25,
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.snake = Snake(channels)
        self.channels = channels
        self.upsample_factor = upsample_factor
        self.padding = (kernel_size - 1) // 2
        fir = kaiser_sinc_filter(cutoff, kernel_size, beta=14.0)
        self.register_buffer("fir_filter", fir.view(1, 1, -1).expand(channels, -1, -1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=float(self.upsample_factor), mode="nearest")
        x = self.snake(x)
        x = F.conv1d(x, self.fir_filter, padding=self.padding, groups=self.channels)
        return x[:, :, :: self.upsample_factor]


class ResidualUnit(nn.Module):
    """Dilated residual unit: ``(act → 7-conv → act → 1×1-conv) + skip``."""

    def __init__(
        self, dim: int, dilation: int, activation_cls: Callable[[int], nn.Module] = Snake
    ) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            activation_cls(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            activation_cls(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class NoiseBlock(nn.Module):
    """Signal-dependent noise injection ``x + scale(x)·ε``; the 1×1 conv is zero-init (identity at step 0)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Conv1d(dim, dim, kernel_size=1)
        assert self.scale.bias is not None
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale(x) * torch.randn_like(x)
