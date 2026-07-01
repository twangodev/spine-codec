"""DDSP high-frequency split for the Spine decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spine.nn import Snake, WNConv1d, kaiser_sinc_filter


def _highpass_bin_mask(n_bins: int, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    """``(1, n_bins, 1)`` 0/1 mask keeping STFT bins strictly above ``cutoff_hz``."""
    freqs = torch.linspace(0.0, sample_rate / 2, n_bins)
    return (freqs > cutoff_hz).float()[None, :, None]


class Crossover(nn.Module):
    """Complementary linear-phase crossover; ``hp = δ − lp`` so ``lp + hp`` is a bit-exact allpass."""

    lp_kernel: torch.Tensor
    hp_kernel: torch.Tensor

    def __init__(self, cutoff_hz: float, taps: int, sample_rate: int) -> None:
        super().__init__()
        self.pad = (taps - 1) // 2
        lp = kaiser_sinc_filter(cutoff_hz / sample_rate, taps, beta=14.0)
        hp = -lp.clone()
        hp[self.pad] += 1.0
        self.register_buffer("lp_kernel", lp.view(1, 1, -1), persistent=False)
        self.register_buffer("hp_kernel", hp.view(1, 1, -1), persistent=False)

    def _filter(self, wav: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            out = F.conv1d(wav.float(), kernel, padding=self.pad)
        return out.to(wav.dtype)

    def lowpass(self, wav: torch.Tensor) -> torch.Tensor:
        return self._filter(wav, self.lp_kernel)

    def highpass(self, wav: torch.Tensor) -> torch.Tensor:
        return self._filter(wav, self.hp_kernel)


class FilteredNoiseBranch(nn.Module):
    """DDSP aperiodic HF: a regressed per-band/frame envelope shapes fresh, never-fit white noise."""

    window: torch.Tensor
    hp_bin_mask: torch.Tensor

    def __init__(
        self,
        feat_dim: int,
        n_bands: int,
        n_fft: int,
        hop_length: int,
        foothold_bias: float,
        sample_rate: int,
        hp_cutoff_hz: float,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1
        self.proj = WNConv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.act = Snake(feat_dim)
        self.head = nn.Conv1d(feat_dim, n_bands, kernel_size=1)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, foothold_bias)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "hp_bin_mask",
            _highpass_bin_mask(self.n_bins, sample_rate, hp_cutoff_hz),
            persistent=False,
        )

    @staticmethod
    def _exp_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(x).pow(2.302585093) + 1e-7

    def forward(self, feat: torch.Tensor, length: int) -> torch.Tensor:
        """Synthesize ``(B, 1, length)`` of envelope-shaped noise from ``(B, feat_dim, T)`` features."""
        amp = self._exp_sigmoid(self.head(self.act(self.proj(feat))))
        out_dtype = amp.dtype
        b = amp.shape[0]
        device = amp.device
        with torch.autocast(device_type=device.type, enabled=False):
            amp = amp.float()
            window = self.window.float()
            noise = torch.randn(b, length, device=device, dtype=torch.float32)
            spec = torch.stft(
                noise,
                self.n_fft,
                self.hop_length,
                self.n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            t_frames = spec.shape[-1]
            env = F.interpolate(amp, size=t_frames, mode="linear", align_corners=False)
            env = F.interpolate(
                env.transpose(1, 2),
                size=self.n_bins,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            env = env * self.hp_bin_mask
            y = torch.istft(
                spec * env,
                self.n_fft,
                self.hop_length,
                self.n_fft,
                window=window,
                center=True,
                length=length,
            )
        return y.unsqueeze(1).to(out_dtype)


class HFHead(nn.Module):
    """DDSP tonal HF: predicts a complex STFT per bin, masks to >cutoff, iSTFTs; zero-init starts silent."""

    window: torch.Tensor
    hp_bin_mask: torch.Tensor

    def __init__(
        self,
        feat_dim: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        hp_cutoff_hz: float,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1
        self.proj = WNConv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.act = Snake(feat_dim)
        self.head = nn.Conv1d(feat_dim, 2 * self.n_bins, kernel_size=1)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "hp_bin_mask",
            _highpass_bin_mask(self.n_bins, sample_rate, hp_cutoff_hz),
            persistent=False,
        )

    def forward(self, feat: torch.Tensor, length: int) -> torch.Tensor:
        """Synthesize ``(B, 1, length)`` of deterministic HF from ``(B, feat_dim, T)`` features."""
        h = self.head(self.act(self.proj(feat)))
        out_dtype = h.dtype
        device = h.device
        with torch.autocast(device_type=device.type, enabled=False):
            h = h.float()
            t_frames = length // self.hop_length + 1
            ri = F.interpolate(h, size=t_frames, mode="linear", align_corners=False)
            spec = torch.complex(ri[:, : self.n_bins], ri[:, self.n_bins :])
            spec = spec * self.hp_bin_mask
            y = torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.n_fft,
                window=self.window.float(),
                center=True,
                length=length,
            )
        return y.unsqueeze(1).to(out_dtype)
