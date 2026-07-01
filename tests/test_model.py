import torch

from spine.config import tiny_config
from spine.model import Spine


def test_forward_shape_and_finite() -> None:
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Spine(cfg.model).eval()

    audio = torch.randn(1, 1, 4000)
    with torch.no_grad():
        out = model(audio)

    assert out["audio_hat"].shape == audio.shape
    assert torch.isfinite(out["audio_hat"]).all()
    assert len(out["codes"]) == len(cfg.model.vq_strides)


def test_encode_decode_roundtrip() -> None:
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Spine(cfg.model).eval()

    audio = torch.randn(1, 1, 4000)
    with torch.no_grad():
        codes = model.encode(audio)
        audio_hat = model.decode(codes)

    assert isinstance(codes, list)
    assert len(codes) == len(cfg.model.vq_strides)
    assert all(torch.is_tensor(c) for c in codes)
    assert audio_hat.ndim == 3 and audio_hat.shape[1] == 1
    assert torch.isfinite(audio_hat).all()
