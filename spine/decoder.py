"""Convolutional decoder with an always-on DDSP high-frequency split."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from .hf import Crossover, FilteredNoiseBranch, HFHead
from .nn import NoiseBlock, ResidualUnit, Snake, WNConv1d, WNConvTranspose1d
from .transformer import TransformerStack


class DecoderBlock(nn.Module):
    """Upsample block: ``act → transposed-conv (×stride) → NoiseBlock → 3 dilated residual units``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        activation_cls: Callable[[int], nn.Module] = Snake,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            activation_cls(in_dim),
            WNConvTranspose1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            NoiseBlock(out_dim),
            *(ResidualUnit(out_dim, dilation=d, activation_cls=activation_cls) for d in (1, 3, 9)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """Decodes a ``(B, input_dim, T)`` latent to ``(B, 1, T·prod(strides))`` audio with an always-on DDSP HF split."""

    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, ...],
        strides: tuple[int, ...],
        sample_rate: int,
        crossover_hz: float,
        crossover_taps: int,
        noise_branch_bands: int,
        noise_branch_n_fft: int,
        noise_branch_hop: int,
        noise_branch_foothold_bias: float,
        hf_head_n_fft: int,
        transformer: TransformerStack | None = None,
        activation_cls: Callable[[int], nn.Module] = Snake,
        output_bound: str | None = None,
    ) -> None:
        super().__init__()
        assert len(channels) == len(strides) + 1, (
            f"len(channels)={len(channels)} must equal len(strides)+1={len(strides) + 1}"
        )

        self.transformer = transformer
        self.output_bound = output_bound

        self.crossover = Crossover(crossover_hz, crossover_taps, sample_rate)
        self.noise_branch = FilteredNoiseBranch(
            input_dim,
            noise_branch_bands,
            noise_branch_n_fft,
            noise_branch_hop,
            noise_branch_foothold_bias,
            sample_rate,
            hp_cutoff_hz=crossover_hz,
        )
        self.hf_head = HFHead(
            input_dim,
            hf_head_n_fft,
            noise_branch_hop,
            sample_rate,
            hp_cutoff_hz=crossover_hz,
        )

        layers: list[nn.Module] = [
            WNConv1d(input_dim, input_dim, kernel_size=7, padding=3, groups=input_dim),
            WNConv1d(input_dim, channels[0], kernel_size=1),
        ]
        layers += [
            DecoderBlock(channels[i], channels[i + 1], stride, activation_cls)
            for i, stride in enumerate(strides)
        ]
        layers += [
            activation_cls(channels[-1]),
            WNConv1d(channels[-1], 1, kernel_size=7, padding=3),
        ]
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a latent ``(B, input_dim, T)`` to audio ``(B, 1, T·prod(strides))``."""
        if self.transformer is not None:
            x = self.transformer(x)
        wav = self.conv_stack(x)
        length = wav.shape[-1]
        wav = self.crossover.lowpass(wav)
        hf = self.noise_branch(x, length) + self.hf_head(x, length)
        wav = wav + self.crossover.highpass(hf)
        if self.output_bound == "tanh":
            wav = torch.tanh(wav)
        return wav
