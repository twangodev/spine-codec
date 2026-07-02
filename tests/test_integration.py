"""End-to-end: encode a seeded 5% sample of LibriSpeech test-clean with the pretrained model."""

import os
import random
from pathlib import Path

import pytest
import torch
import torchaudio

from spine import Spine

pytestmark = pytest.mark.integration

REPO_ID = "twangodev/spine-codec"
DATA_DIR = Path(os.environ.get("SPINE_IT_DATA_DIR", "~/.cache/librispeech")).expanduser()
SAMPLE_FRACTION = 0.05
MAX_TOTAL_SECONDS = float(os.environ.get("SPINE_IT_MAX_SECONDS", "960"))
DECODE_COUNT = 3
SEED = 0


@pytest.fixture(scope="module")
def model() -> Spine:
    return Spine.from_pretrained(REPO_ID)


@pytest.fixture(scope="module")
def dataset() -> torchaudio.datasets.LIBRISPEECH:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return torchaudio.datasets.LIBRISPEECH(str(DATA_DIR), url="test-clean", download=True)


def _sampled_indices(dataset: torchaudio.datasets.LIBRISPEECH) -> list[int]:
    count = max(1, int(len(dataset) * SAMPLE_FRACTION))
    return random.Random(SEED).sample(range(len(dataset)), count)


def _load_resampled(
    dataset: torchaudio.datasets.LIBRISPEECH, idx: int, target_sr: int
) -> tuple[torch.Tensor, float]:
    wav, sr, *_ = dataset[idx]
    seconds = wav.shape[-1] / sr
    return torchaudio.functional.resample(wav, sr, target_sr), seconds


def test_encode_dataset_sample(model: Spine, dataset: torchaudio.datasets.LIBRISPEECH) -> None:
    n_scales = len(model.vq_strides)
    total_seconds = 0.0
    encoded = 0
    for idx in _sampled_indices(dataset):
        if total_seconds >= MAX_TOTAL_SECONDS:
            break
        wav, seconds = _load_resampled(dataset, idx, model.sample_rate)
        total_seconds += seconds
        with torch.no_grad():
            codes = model.encode(wav.unsqueeze(0))
        assert len(codes) == n_scales
        for c in codes:
            assert torch.is_tensor(c)
            assert not torch.is_floating_point(c)
            assert c.min() >= 0
        encoded += 1
    assert encoded > 0


def test_decode_roundtrip_subset(model: Spine, dataset: torchaudio.datasets.LIBRISPEECH) -> None:
    for idx in _sampled_indices(dataset)[:DECODE_COUNT]:
        wav, _ = _load_resampled(dataset, idx, model.sample_rate)
        with torch.no_grad():
            codes = model.encode(wav.unsqueeze(0))
            audio_hat = model.decode(codes)
        assert audio_hat.ndim == 3 and audio_hat.shape[1] == 1
        assert torch.isfinite(audio_hat).all()
