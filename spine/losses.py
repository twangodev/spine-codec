"""Composite GAN + reconstruction losses for Spine codec training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from spine.nn import kaiser_sinc_filter


class _MultiResSTFT(nn.Module):
    """Base for multi-resolution STFT losses: caches one Hann window per resolution."""

    def __init__(self, resolutions: list[tuple[int, int, int]]) -> None:
        super().__init__()
        self.resolutions = resolutions
        for i, (_, _, win_length) in enumerate(resolutions):
            self.register_buffer(f"window_{i}", torch.hann_window(win_length), persistent=False)

    def _stft(self, wav: torch.Tensor, i: int) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolutions[i]
        return torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=getattr(self, f"window_{i}"),
            return_complex=True,
        )


class MultiScaleSTFTLoss(_MultiResSTFT):
    """Multi-scale STFT L1 (linear + log magnitude); the log term keeps pressure on quiet HF detail."""

    def __init__(
        self,
        resolutions: list[tuple[int, int, int]],
        clamp_eps: float = 1e-5,
        hf_weight: float = 1.0,
        hf_cutoff_frac: float = 0.5,
    ) -> None:
        super().__init__(resolutions)
        self.clamp_eps = clamp_eps
        for i, (n_fft, _, _) in enumerate(resolutions):
            freqs = torch.linspace(0.0, 1.0, n_fft // 2 + 1)
            ramp = ((freqs - hf_cutoff_frac) / (1.0 - hf_cutoff_frac)).clamp(0.0, 1.0)
            self.register_buffer(f"hfw_{i}", 1.0 + (hf_weight - 1.0) * ramp, persistent=False)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Scalar linear+log magnitude L1 averaged over resolutions; ``x`` is the target."""
        x_flat, x_hat_flat = x.squeeze(1), x_hat.squeeze(1)
        loss = torch.zeros((), device=x.device)
        for i in range(len(self.resolutions)):
            with torch.no_grad():
                mag_x = self._stft(x_flat, i).abs()
            mag_x_hat = self._stft(x_hat_flat, i).abs()
            hfw = getattr(self, f"hfw_{i}")[None, :, None]
            wmean = hfw.mean()
            linear = (hfw * (mag_x_hat - mag_x).abs()).mean() / wmean
            log = (
                hfw
                * (
                    mag_x_hat.clamp(min=self.clamp_eps).log()
                    - mag_x.clamp(min=self.clamp_eps).log()
                ).abs()
            ).mean() / wmean
            loss = loss + linear + log
        return loss / len(self.resolutions)


class MelSpectrogramLoss(nn.Module):
    """Multi-scale log-mel L1 across several STFT window sizes (fine to broad spectral)."""

    def __init__(
        self,
        sample_rate: int,
        n_ffts: list[int],
        n_mels_list: list[int],
        clamp_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert len(n_ffts) == len(n_mels_list)
        self.clamp_eps = clamp_eps
        self.mel_transforms = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                power=1.0,
            )
            for n_fft, n_mels in zip(n_ffts, n_mels_list)
        )

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Scalar log-mel L1 averaged over scales; ``x`` is the (detached) target."""
        x_sq, x_hat_sq = x.squeeze(1), x_hat.squeeze(1)
        loss = torch.zeros((), device=x.device)
        for mel_transform in self.mel_transforms:
            mel_x = mel_transform(x_sq)
            mel_x_hat = mel_transform(x_hat_sq)
            loss = loss + F.l1_loss(
                mel_x.clamp(min=self.clamp_eps).log().detach(),
                mel_x_hat.clamp(min=self.clamp_eps).log(),
            )
        return loss / len(self.mel_transforms)


class HFEnergyLoss(_MultiResSTFT):
    """Realization-invariant HF energy-envelope loss: matches per-frame energy above cutoff, so honest noise satisfies it."""

    def __init__(
        self,
        resolutions: list[tuple[int, int, int]],
        sample_rate: int,
        cutoff_hz: float,
        clamp_eps: float = 1e-7,
    ) -> None:
        super().__init__(resolutions)
        self.clamp_eps = clamp_eps
        for i, (n_fft, _, _) in enumerate(resolutions):
            freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
            self.register_buffer(f"mask_{i}", (freqs > cutoff_hz).float(), persistent=False)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Scalar log-energy L1 of the >cutoff band averaged over resolutions."""
        x_flat, x_hat_flat = x.squeeze(1), x_hat.squeeze(1)
        loss = torch.zeros((), device=x.device)
        for i in range(len(self.resolutions)):
            mask = getattr(self, f"mask_{i}")[None, :, None]
            with torch.no_grad():
                e_x = (self._stft(x_flat, i).abs().pow(2) * mask).sum(dim=1)
            e_x_hat = (self._stft(x_hat_flat, i).abs().pow(2) * mask).sum(dim=1)
            loss = loss + F.l1_loss(
                (e_x_hat + self.clamp_eps).log(),
                (e_x + self.clamp_eps).log(),
            )
        return loss / len(self.resolutions)


def generator_loss(fmaps_fake: list[list[torch.Tensor]]) -> torch.Tensor:
    """LSGAN generator loss: push discriminator scores toward 1."""
    loss = torch.zeros((), device=fmaps_fake[0][-1].device)
    for fmaps in fmaps_fake:
        loss = loss + torch.mean((1.0 - fmaps[-1]) ** 2)
    return loss / len(fmaps_fake)


def discriminator_loss(
    fmaps_real: list[list[torch.Tensor]],
    fmaps_fake: list[list[torch.Tensor]],
) -> torch.Tensor:
    """LSGAN discriminator loss: real -> 1, fake -> 0."""
    loss = torch.zeros((), device=fmaps_real[0][-1].device)
    for fmaps_r, fmaps_f in zip(fmaps_real, fmaps_fake):
        loss = loss + torch.mean((1.0 - fmaps_r[-1]) ** 2) + torch.mean(fmaps_f[-1] ** 2)
    return loss / len(fmaps_real)


def feature_matching_loss(
    fmaps_real: list[list[torch.Tensor]],
    fmaps_fake: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 between real (detached) and fake discriminator features, excluding the score."""
    loss = torch.zeros((), device=fmaps_real[0][-1].device)
    n_features = 0
    for fmaps_r, fmaps_f in zip(fmaps_real, fmaps_fake):
        for feat_r, feat_f in zip(fmaps_r[:-1], fmaps_f[:-1]):
            loss = loss + F.l1_loss(feat_f, feat_r.detach())
            n_features += 1
    return loss / max(n_features, 1)


class SpineLoss(nn.Module):
    """Composite Spine objective: recon (mel+STFT) lowpassed below recon_cutoff_hz, plus HF energy, LSGAN, and feature matching."""

    def __init__(
        self,
        sample_rate: int,
        n_ffts: list[int],
        n_mels_list: list[int],
        lambda_mel: float,
        lambda_stft: float,
        lambda_feat: float,
        lambda_adv: float,
        stft_resolutions: list[tuple[int, int, int]] | None = None,
        hf_weight: float = 1.0,
        hf_cutoff_frac: float = 0.5,
        lambda_hf_energy: float = 0.0,
        hf_energy_cutoff_hz: float = 6000.0,
        hf_energy_resolutions: list[tuple[int, int, int]] | None = None,
        recon_cutoff_hz: float = 6000.0,
    ) -> None:
        super().__init__()
        self.mel_loss = MelSpectrogramLoss(sample_rate, n_ffts, n_mels_list)
        self.stft_loss = (
            MultiScaleSTFTLoss(stft_resolutions, hf_weight=hf_weight, hf_cutoff_frac=hf_cutoff_frac)
            if stft_resolutions
            else None
        )
        self.hf_energy_loss = (
            HFEnergyLoss(
                hf_energy_resolutions,
                sample_rate=sample_rate,
                cutoff_hz=hf_energy_cutoff_hz,
            )
            if lambda_hf_energy > 0.0 and hf_energy_resolutions
            else None
        )
        self.lambda_mel = lambda_mel
        self.lambda_stft = lambda_stft
        self.lambda_feat = lambda_feat
        self.lambda_adv = lambda_adv
        self.lambda_hf_energy = lambda_hf_energy
        self.recon_cutoff_hz = recon_cutoff_hz

        taps = 255
        lp = kaiser_sinc_filter(recon_cutoff_hz / sample_rate, taps, beta=14.0)
        self.recon_lp_pad = (taps - 1) // 2
        self.register_buffer("recon_lp", lp.view(1, 1, -1), persistent=False)

    recon_lp: torch.Tensor

    def _recon_lp(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            out = F.conv1d(wav.float(), self.recon_lp, padding=self.recon_lp_pad)
        return out.to(wav.dtype)

    def generator_total(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        fmaps_real: list[list[torch.Tensor]],
        fmaps_fake: list[list[torch.Tensor]],
        use_adversarial: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Total generator loss and its components; ``use_adversarial=False`` skips adv+feat (warmup)."""
        zero = torch.zeros((), device=x.device)
        x_lo, xh_lo = self._recon_lp(x), self._recon_lp(x_hat)

        mel = self.mel_loss(x_lo, xh_lo)
        stft = self.stft_loss(x_lo, xh_lo) if self.stft_loss is not None else zero
        hf_energy = self.hf_energy_loss(x, x_hat) if self.hf_energy_loss is not None else zero

        if use_adversarial:
            adv = generator_loss(fmaps_fake)
            feat = feature_matching_loss(fmaps_real, fmaps_fake)
        else:
            adv = feat = zero

        total = (
            self.lambda_mel * mel
            + self.lambda_stft * stft
            + self.lambda_hf_energy * hf_energy
            + self.lambda_adv * adv
            + self.lambda_feat * feat
        )
        return {
            "total": total,
            "mel": mel,
            "stft": stft,
            "hf_energy": hf_energy,
            "adv": adv,
            "feat": feat,
        }

    def discriminator_total(
        self,
        fmaps_real: list[list[torch.Tensor]],
        fmaps_fake: list[list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Total discriminator loss (single ``total`` key)."""
        return {"total": discriminator_loss(fmaps_real, fmaps_fake)}
