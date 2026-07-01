"""Bidirectional transformer stack: RMSNorm, RoPE, SDPA, GELU MLP, LayerScale."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


class LayerScale(nn.Module):
    """Per-channel learnable residual scale, near-zero init (Mimi requirement for GAN stability)."""

    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 8192) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cache_len = seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cache_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_dim: int,
        layerscale_init: float,
        causal: bool,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal

        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ls1 = LayerScale(dim, layerscale_init)

        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.ls2 = LayerScale(dim, layerscale_init)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        q, k = apply_rotary_emb(q, k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.ls1(self.attn_out(attn))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class TransformerStack(nn.Module):
    """Channel-first ``(B, dim, T)`` transformer stack with shared RoPE and a final RMSNorm."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        layerscale_init: float = 0.01,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.causal = causal
        self.rope = RotaryEmbedding(dim // n_heads)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, mlp_dim, layerscale_init, causal)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        cos, sin = self.rope(x.shape[1])
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x).transpose(1, 2)
