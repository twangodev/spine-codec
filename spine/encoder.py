"""Convolutional encoder: a strided Snake conv pyramid then a bidirectional transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .nn import ResidualUnit, Snake, WNConv1d
from .transformer import TransformerStack


class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            *(ResidualUnit(in_dim, dilation=d) for d in (1, 3, 9)),
            Snake(in_dim),
            WNConv1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        strides: tuple[int, ...],
        transformer: TransformerStack | None = None,
    ) -> None:
        super().__init__()
        assert len(channels) == len(strides) + 1, (
            f"len(channels)={len(channels)} must equal len(strides)+1={len(strides) + 1}"
        )
        self.hop_length = math.prod(strides)
        self.output_dim = channels[-1]

        layers: list[nn.Module] = [WNConv1d(1, channels[0], kernel_size=7, padding=3)]
        layers += [
            EncoderBlock(channels[i], channels[i + 1], stride) for i, stride in enumerate(strides)
        ]
        layers += [
            Snake(channels[-1]),
            WNConv1d(channels[-1], channels[-1], kernel_size=7, padding=3, groups=channels[-1]),
        ]
        self.conv_stack = nn.Sequential(*layers)
        self.transformer = transformer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        if self.transformer is not None:
            x = self.transformer(x)
        return x
