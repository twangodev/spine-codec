import torch

from spine.hf import Crossover, FilteredNoiseBranch, HFHead

SR = 24000
CUTOFF = 6000.0
N_FFT = 512
HOP = 128


def _below_cutoff_fraction(y: torch.Tensor) -> float:
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        y.squeeze(1), N_FFT, HOP, N_FFT, window=window, center=True, return_complex=True
    )
    mag2 = spec.abs().pow(2)
    freqs = torch.linspace(0.0, SR / 2, N_FFT // 2 + 1)
    below = (freqs <= CUTOFF)[None, :, None]
    return (mag2 * below).sum().item() / mag2.sum().item()


def test_crossover_is_allpass() -> None:
    torch.manual_seed(0)
    xo = Crossover(CUTOFF, 255, SR)
    x = torch.randn(2, 1, 8192)
    recon = xo.lowpass(x) + xo.highpass(x)
    assert (recon - x).abs().max().item() < 1e-4


def test_noise_branch_confined_above_cutoff() -> None:
    torch.manual_seed(0)
    branch = FilteredNoiseBranch(16, 8, N_FFT, HOP, -3.7, SR, hp_cutoff_hz=CUTOFF)
    feat = torch.randn(2, 16, 40)
    y = branch(feat, 4096)
    assert torch.isfinite(y).all()
    assert _below_cutoff_fraction(y) < 0.02


def test_hf_head_confined_above_cutoff() -> None:
    torch.manual_seed(0)
    head = HFHead(16, N_FFT, HOP, SR, hp_cutoff_hz=CUTOFF)
    assert head.head.bias is not None
    torch.nn.init.normal_(head.head.weight, std=1.0)
    torch.nn.init.normal_(head.head.bias, std=1.0)
    feat = torch.randn(2, 16, 40)
    y = head(feat, 4096)
    assert torch.isfinite(y).all()
    assert y.abs().max().item() > 0.0
    assert _below_cutoff_fraction(y) < 0.02
