"""GAN discriminators: DAC-style multi-period + band-split multi-resolution STFT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def WNConv2d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv2d(*args, **kwargs))


class PeriodDiscriminator(nn.Module):
    """Folds the waveform by ``period`` into 2-D and applies strided 2-D convs."""

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        channels = (1, 32, 128, 512, 1024, 1024)
        self.convs = nn.ModuleList(
            WNConv2d(
                channels[i],
                channels[i + 1],
                kernel_size=(5, 1),
                stride=(3, 1),
                padding=(2, 0),
            )
            for i in range(len(channels) - 1)
        )
        self.final = WNConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps with the discriminator score as the last element."""
        t = x.shape[-1]
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = rearrange(x, "b c (t p) -> b c t p", p=self.period)

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmaps.append(x)
        fmaps.append(self.final(x))
        return fmaps


class MPD(nn.Module):
    """Multi-Period Discriminator: an ensemble of ``PeriodDiscriminator``s."""

    def __init__(self, periods: list[int]) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        return [d(x) for d in self.discriminators]


class BandSplitSubDiscriminator(nn.Module):
    """2-D conv stack processing one frequency band of the complex STFT."""

    def __init__(self, in_channels: int = 2, channels: int = 32) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [WNConv2d(in_channels, channels, kernel_size=(3, 9), padding=(1, 4))]
            + [
                WNConv2d(channels, channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                for _ in range(3)
            ]
            + [WNConv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
        return x


class ResolutionDiscriminator(nn.Module):
    """STFT at one resolution, split into frequency bands processed independently."""

    BAND_SPLITS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    window: torch.Tensor

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.n_bins = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.band_processors = nn.ModuleList(
            BandSplitSubDiscriminator(in_channels=2, channels=channels) for _ in self.BAND_SPLITS
        )
        self.final = WNConv2d(channels, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-band feature maps followed by the discriminator score."""
        stft = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        stft_2ch = rearrange(torch.stack([stft.real, stft.imag], dim=1), "b c f t -> b c t f")

        fmaps = []
        processed_bands = []
        for (low, high), processor in zip(self.BAND_SPLITS, self.band_processors):
            low_bin = int(low * self.n_bins)
            high_bin = max(int(high * self.n_bins), low_bin + 1)
            processed = processor(stft_2ch[..., low_bin:high_bin])
            processed_bands.append(processed)
            fmaps.append(processed)

        fmaps.append(self.final(torch.cat(processed_bands, dim=-1)))
        return fmaps


class MRD(nn.Module):
    """Multi-Resolution STFT Discriminator: one ``ResolutionDiscriminator`` per resolution."""

    def __init__(self, resolutions: list[tuple[int, int, int]], channels: int) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            ResolutionDiscriminator(n_fft=n_fft, hop_length=hop, win_length=win, channels=channels)
            for n_fft, hop, win in resolutions
        )

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        return [d(x) for d in self.discriminators]


class Discriminator(nn.Module):
    """Combined MPD + MRD discriminator with DC-removal preprocessing."""

    def __init__(
        self,
        mpd_periods: list[int],
        mrd_resolutions: list[tuple[int, int, int]],
        mrd_channels: int,
    ) -> None:
        super().__init__()
        self.mpd = MPD(periods=mpd_periods)
        self.mrd = MRD(resolutions=mrd_resolutions, channels=mrd_channels)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Remove DC offset only; no peak-normalization (would divide out dynamics error and couple adv gradients to the loudest sample)."""
        return x - x.mean(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        """Return feature-map lists from every MPD and MRD sub-discriminator."""
        x = self.preprocess(x)
        return self.mpd(x) + self.mrd(x)
