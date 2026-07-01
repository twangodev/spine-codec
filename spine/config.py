"""Centralized configuration for the Spine codec (single source of truth; YAML gives sparse overrides via ``from_yaml``)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Spine generator hyperparameters (Mimi-aligned v2 with an always-on DDSP HF split)."""

    sample_rate: int = 24000

    encoder_channels: tuple[int, ...] = (96, 192, 384, 512, 512)
    encoder_rates: tuple[int, ...] = (2, 4, 8, 8)
    decoder_channels: tuple[int, ...] = (1280, 640, 320, 160, 80)
    decoder_rates: tuple[int, ...] = (8, 8, 4, 2)

    fsq_levels_per_scale: tuple[tuple[int, ...], ...] = (
        (5, 5, 5, 5),
        (8, 5, 5, 5, 5),
        (8, 6, 5, 5, 5, 5, 5, 5),
        (8, 6, 5, 5, 5, 5, 5, 5),
    )
    vq_strides: tuple[int, ...] = (8, 4, 2, 1)

    transformer_dim: int = 512
    transformer_mlp_dim: int = 2048
    transformer_n_heads: int = 8
    encoder_transformer_n_layers: int = 8
    decoder_transformer_n_layers: int = 12
    layerscale_init: float = 0.01

    use_filtered_snake: bool = True
    filtered_snake_cutoff: float = 0.25
    filtered_snake_kernel_size: int = 12

    crossover_hz: float = 6000.0
    crossover_taps: int = 255
    noise_branch_bands: int = 64
    noise_branch_n_fft: int = 2048
    noise_branch_foothold_bias: float = -3.7
    hf_head_n_fft: int = 1024
    output_bound: str | None = None

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)


@dataclass(frozen=True)
class DiscriminatorConfig:
    mpd_periods: tuple[int, ...] = (2, 3, 5, 7, 11, 17, 23, 37)
    mrd_resolutions: tuple[tuple[int, int, int], ...] = (
        (2048, 512, 2048),
        (1024, 120, 600),
        (2048, 240, 1200),
        (4096, 480, 2400),
        (512, 50, 240),
    )
    mrd_channels: int = 32


@dataclass(frozen=True)
class LossConfig:
    """Loss weights and spectrogram parameters; recon is always bandlimited below recon_cutoff_hz."""

    lambda_mel: float = 15.0
    lambda_stft: float = 3.0
    lambda_feat: float = 2.0
    lambda_adv: float = 1.0
    mel_n_ffts: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)
    mel_n_mels: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)
    stft_resolutions: tuple[tuple[int, int, int], ...] = (
        (2048, 512, 2048),
        (1024, 120, 600),
        (2048, 240, 1200),
        (4096, 480, 2400),
        (512, 50, 240),
    )
    hf_weight: float = 1.0
    hf_cutoff_frac: float = 0.5
    lambda_hf_energy: float = 2.0
    hf_energy_cutoff_hz: float = 6000.0
    hf_energy_resolutions: tuple[tuple[int, int, int], ...] = (
        (1024, 256, 1024),
        (2048, 512, 2048),
    )
    recon_cutoff_hz: float = 6000.0


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer and LR schedule; weight decay applies to transformer blocks only (Mimi rule)."""

    lr: float = 1e-4
    lr_d: float = 1e-4
    betas: tuple[float, float] = (0.8, 0.9)
    grad_clip_g: float = 1.0
    grad_clip_d: float = 1.0
    lr_gamma: float = 1.0
    weight_decay_transformer: float = 5e-2
    ema_decay: float = 0.9999


@dataclass(frozen=True)
class DataConfig:
    train_dir: str = "data/train"
    clip_duration: float = 3.0
    deterministic: bool = False


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 1_000_000
    max_epochs: int = 10_000
    num_workers: int = 8
    device: str = "cuda"
    precision: str = "bf16"
    compile: bool = False
    disc_start_step: int = 0
    log_every: int = 100
    use_wandb: bool = True
    wandb_project: str = "spine"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 25_000


@dataclass(frozen=True)
class SpineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "discriminator": DiscriminatorConfig,
    "loss": LossConfig,
    "optimizer": OptimizerConfig,
    "data": DataConfig,
    "training": TrainConfig,
}


def _coerce_sequences(dc_class: type, d: dict[str, Any]) -> dict[str, Any]:
    """Keep only known fields, converting YAML lists (including nested) to tuples."""
    valid = {f.name for f in fields(dc_class)}
    coerced: dict[str, Any] = {}
    for key, value in d.items():
        if key not in valid:
            continue
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                value = tuple(tuple(item) for item in value)
            else:
                value = tuple(value)
        coerced[key] = value
    return coerced


def from_dict(d: dict[str, Any]) -> SpineConfig:
    """Build a SpineConfig from a plain dict; missing sections/keys use dataclass defaults."""
    return SpineConfig(
        **{name: cls(**_coerce_sequences(cls, d.get(name, {}))) for name, cls in _SECTIONS.items()}
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on leaf conflicts)."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_chain(path: Path) -> dict[str, Any]:
    """Load a YAML file, resolving any ``extends`` parent chain relative to it."""
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if "extends" in d:
        parent = (path.parent / d.pop("extends")).resolve()
        d = _deep_merge(_load_yaml_chain(parent), d)
    return d


def from_yaml(path: str | Path) -> SpineConfig:
    """Load a SpineConfig from a YAML file of sparse overrides (supports ``extends``)."""
    return from_dict(_load_yaml_chain(Path(path)))


def to_dict(cfg: SpineConfig) -> dict[str, Any]:
    """Serialize a SpineConfig to a plain dict (tuples flattened to lists for YAML)."""

    def to_lists(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: to_lists(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_lists(item) for item in obj]
        return obj

    return to_lists(asdict(cfg))


def to_yaml(cfg: SpineConfig, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.dump(to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def tiny_config() -> SpineConfig:
    """Minimal config for fast unit tests (keeps ``encoder_channels[-1] == transformer_dim``)."""
    return SpineConfig(
        model=ModelConfig(
            encoder_channels=(8, 16, 32),
            encoder_rates=(2, 4),
            decoder_channels=(32, 16, 8),
            decoder_rates=(4, 2),
            fsq_levels_per_scale=(
                (5, 5, 5, 5),
                (5, 5, 5, 5),
                (5, 5, 5, 5),
                (5, 5, 5, 5),
            ),
            vq_strides=(8, 4, 2, 1),
            transformer_dim=32,
            transformer_mlp_dim=64,
            transformer_n_heads=4,
            encoder_transformer_n_layers=1,
            decoder_transformer_n_layers=1,
        ),
    )
