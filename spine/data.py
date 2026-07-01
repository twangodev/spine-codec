"""Basic training data: fixed-length mono crops from a directory of audio files at the codec sample rate."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


def find_audio_files(root: str | Path) -> list[Path]:
    """Recursively collect audio files under ``root``, sorted by path."""
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


class AudioDataset(Dataset[torch.Tensor]):
    """Random fixed-length mono crops from a directory of audio files, resampled to ``sample_rate``."""

    def __init__(
        self,
        audio_dir: str | Path,
        sample_rate: int,
        clip_duration: float = 3.0,
        pad_to: int = 1,
        deterministic: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.clip_samples = (int(clip_duration * sample_rate) // pad_to) * pad_to
        self.deterministic = deterministic
        self.files = find_audio_files(audio_dir)
        if not self.files:
            raise ValueError(f"No audio files found under {audio_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        for _ in range(5):
            try:
                return self._load_clip(index)
            except Exception:
                index = random.randint(0, len(self.files) - 1)
        return torch.zeros(1, self.clip_samples)

    def _load_clip(self, idx: int) -> torch.Tensor:
        audio, sr = torchaudio.load(str(self.files[idx]))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        n = audio.shape[-1]
        if n > self.clip_samples:
            start = 0 if self.deterministic else random.randint(0, n - self.clip_samples)
            audio = audio[:, start : start + self.clip_samples]
        elif n < self.clip_samples:
            audio = F.pad(audio, (0, self.clip_samples - n))

        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak
        return audio


def build_dataloader(
    dataset: AudioDataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap an AudioDataset in a DataLoader; fixed-length clips stack into ``(B, 1, T)``."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
