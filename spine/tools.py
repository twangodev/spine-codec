"""Inference helpers: load a trained Spine from a checkpoint and roundtrip audio through the codec."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from .config import SpineConfig, from_dict
from .model import Spine


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> Spine:
    """Build a Spine from the checkpoint's config and load its EMA (eval) weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = from_dict(ckpt["config"]) if "config" in ckpt else SpineConfig()

    model = Spine(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])

    ema_state = ckpt.get("ema")
    if ema_state:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.copy_(ema_state[name].to(param))

    model.eval()
    return model


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load an audio file as mono ``(1, T)`` resampled to ``sample_rate``."""
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


@torch.no_grad()
def encode_file(model: Spine, audio_path: str | Path, device: str = "cpu") -> list[torch.Tensor]:
    """Encode an audio file to one discrete-token tensor per quantization scale."""
    audio = load_audio(audio_path, model.sample_rate).to(device)
    return model.encode(audio.unsqueeze(0))


@torch.no_grad()
def reconstruct_file(
    model: Spine, input_path: str | Path, output_path: str | Path, device: str = "cpu"
) -> None:
    """Encode then decode an audio file, writing the reconstruction to ``output_path``."""
    audio = load_audio(input_path, model.sample_rate).to(device)
    x = audio.unsqueeze(0)
    audio_hat = model.decode(model.encode(x))[..., : x.shape[-1]]
    torchaudio.save(str(output_path), audio_hat.squeeze(0).cpu(), model.sample_rate)
