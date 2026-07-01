"""The Spine codec: encoder + multi-scale FSQ quantizer + DDSP decoder."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .decoder import Decoder
from .encoder import Encoder
from .nn import FilteredSnake, Snake
from .quantizer import MultiScaleFSQ
from .transformer import TransformerStack


class Spine(nn.Module):
    """Encodes 24 kHz mono audio to multi-scale FSQ tokens and decodes them back to waveforms."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        tx_dim = cfg.transformer_dim
        assert cfg.encoder_channels[-1] == tx_dim, (
            f"encoder_channels[-1]={cfg.encoder_channels[-1]} must equal "
            f"transformer_dim={tx_dim} (no projection around the encoder transformer)"
        )

        self.sample_rate = cfg.sample_rate
        self.hop_length = cfg.hop_length
        self.vq_strides = list(cfg.vq_strides)
        self.transformer_dim = tx_dim

        make_transformer = partial(
            TransformerStack,
            dim=tx_dim,
            n_heads=cfg.transformer_n_heads,
            mlp_dim=cfg.transformer_mlp_dim,
            layerscale_init=cfg.layerscale_init,
            causal=False,
        )

        self.encoder = Encoder(
            channels=tuple(cfg.encoder_channels),
            strides=tuple(cfg.encoder_rates),
            transformer=make_transformer(n_layers=cfg.encoder_transformer_n_layers),
        )
        self.quantizer = MultiScaleFSQ(
            d_latent=tx_dim,
            levels_per_scale=[list(levels) for levels in cfg.fsq_levels_per_scale],
            vq_strides=self.vq_strides,
        )

        decoder_activation: Callable[[int], nn.Module] = (
            partial(
                FilteredSnake,
                cutoff=cfg.filtered_snake_cutoff,
                kernel_size=cfg.filtered_snake_kernel_size,
            )
            if cfg.use_filtered_snake
            else Snake
        )
        self.decoder = Decoder(
            input_dim=tx_dim,
            channels=tuple(cfg.decoder_channels),
            strides=tuple(cfg.decoder_rates),
            transformer=make_transformer(n_layers=cfg.decoder_transformer_n_layers),
            activation_cls=decoder_activation,
            sample_rate=cfg.sample_rate,
            crossover_hz=cfg.crossover_hz,
            crossover_taps=cfg.crossover_taps,
            noise_branch_bands=cfg.noise_branch_bands,
            noise_branch_n_fft=cfg.noise_branch_n_fft,
            noise_branch_hop=self.hop_length,
            noise_branch_foothold_bias=cfg.noise_branch_foothold_bias,
            hf_head_n_fft=cfg.hf_head_n_fft,
            output_bound=cfg.output_bound,
        )

    def preprocess(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Right-pad audio to a multiple of ``hop_length * max(vq_strides)`` (the coarsest frame)."""
        original_length = audio.shape[-1]
        pad_to = self.hop_length * max(self.vq_strides)
        remainder = original_length % pad_to
        if remainder != 0:
            audio = F.pad(audio, (0, pad_to - remainder))
        return audio, original_length

    def encode_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """Voice-facing tap: continuous pre-quantization latent ``(B, transformer_dim, T // hop_length)``."""
        audio, _ = self.preprocess(audio)
        return self.encoder(audio)

    def encode(self, audio: torch.Tensor) -> list[torch.Tensor]:
        """Encode ``(B, 1, T)`` audio to one discrete-token tensor per quantization scale."""
        audio, _ = self.preprocess(audio)
        _, codes = self.quantizer(self.encoder(audio))
        return codes

    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Decode per-scale tokens back to ``(B, 1, T)`` audio."""
        return self.decoder(self.quantizer.from_codes(codes))

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Full pass over ``(B, 1, T)`` audio; returns audio_hat, codes, and the pre/post-quant latents."""
        audio, original_length = self.preprocess(audio)
        z = self.encoder(audio)
        z_q, codes = self.quantizer(z)
        audio_hat = self.decoder(z_q)[..., :original_length]
        return {"audio_hat": audio_hat, "codes": codes, "z": z, "z_q": z_q}
