"""Multi-scale Finite Scalar Quantization: pool -> FSQ -> repeat per temporal scale."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ

from spine.nn import WNConv1d


class FSQQuantizer(nn.Module):
    def __init__(self, d_latent: int, levels: list[int], stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.codebook_size = math.prod(levels)
        self.in_proj = WNConv1d(d_latent, len(levels), kernel_size=1)
        self.out_proj = WNConv1d(len(levels), d_latent, kernel_size=1)
        self.fsq = FSQ(levels)

    def _upsample(self, codes_latent: torch.Tensor) -> torch.Tensor:
        z_q = self.out_proj(codes_latent.transpose(1, 2))
        if self.stride > 1:
            z_q = z_q.repeat_interleave(self.stride, dim=-1)
        return z_q

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pooled = F.avg_pool1d(z, self.stride, self.stride) if self.stride > 1 else z
        codes_latent, codes = self.fsq(self.in_proj(z_pooled).transpose(1, 2))
        return self._upsample(codes_latent)[..., : z.shape[-1]], codes

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        codes_latent = self.fsq.indices_to_codes(codes).to(self.out_proj.weight.dtype)
        return self._upsample(codes_latent)


class MultiScaleFSQ(nn.Module):
    """Residual multi-scale FSQ: each scale quantizes the residual at its own temporal rate."""

    def __init__(
        self,
        d_latent: int,
        levels_per_scale: list[list[int]],
        vq_strides: list[int],
    ) -> None:
        super().__init__()
        assert len(levels_per_scale) == len(vq_strides)
        self._scales = [
            FSQQuantizer(d_latent, levels, stride)
            for levels, stride in zip(levels_per_scale, vq_strides)
        ]
        self.quantizers = nn.ModuleList(self._scales)

    @property
    def codebook_size(self) -> int:
        """Per-scale codebook size (identical across scales)."""
        return self._scales[0].codebook_size

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z_q = torch.zeros_like(z)
        residual = z
        codes: list[torch.Tensor] = []
        for quantizer in self._scales:
            z_q_i, codes_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(codes_i)
        return z_q, codes

    def from_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Reconstruct the summed latent from per-scale codes (time set by the finest scale)."""
        z_q: torch.Tensor | None = None
        for quantizer, codes_i in zip(self._scales, codes):
            z_q_i = quantizer.from_codes(codes_i)
            if z_q is None:
                z_q = z_q_i
            else:
                n = min(z_q.shape[-1], z_q_i.shape[-1])
                z_q = z_q[..., :n] + z_q_i[..., :n]
        assert z_q is not None
        return z_q
