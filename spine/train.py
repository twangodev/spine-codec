"""Single-GPU GAN training loop for the Spine codec: AdamW, exponential LR decay, EMA, and checkpointing."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from spine.config import SpineConfig, to_dict
from spine.data import AudioDataset, build_dataloader
from spine.discriminators import Discriminator
from spine.ema import EMA
from spine.losses import SpineLoss
from spine.model import Spine
from spine.transformer import TransformerStack

logger = logging.getLogger(__name__)

_PRECISION_DTYPES: dict[str, torch.dtype] = {"fp32": torch.float32, "bf16": torch.bfloat16}


def build_from_config(
    cfg: SpineConfig, device: torch.device
) -> tuple[Spine, Discriminator, SpineLoss]:
    """Build the generator, discriminator, and composite loss on ``device`` from config."""
    model = Spine(cfg.model).to(device)
    disc = Discriminator(
        mpd_periods=list(cfg.discriminator.mpd_periods),
        mrd_resolutions=[tuple(r) for r in cfg.discriminator.mrd_resolutions],
        mrd_channels=cfg.discriminator.mrd_channels,
    ).to(device)
    criterion = SpineLoss(
        sample_rate=cfg.model.sample_rate,
        n_ffts=list(cfg.loss.mel_n_ffts),
        n_mels_list=list(cfg.loss.mel_n_mels),
        lambda_mel=cfg.loss.lambda_mel,
        lambda_stft=cfg.loss.lambda_stft,
        lambda_feat=cfg.loss.lambda_feat,
        lambda_adv=cfg.loss.lambda_adv,
        stft_resolutions=[tuple(r) for r in cfg.loss.stft_resolutions],
        hf_weight=cfg.loss.hf_weight,
        hf_cutoff_frac=cfg.loss.hf_cutoff_frac,
        lambda_hf_energy=cfg.loss.lambda_hf_energy,
        hf_energy_cutoff_hz=cfg.loss.hf_energy_cutoff_hz,
        hf_energy_resolutions=[tuple(r) for r in cfg.loss.hf_energy_resolutions],
        recon_cutoff_hz=cfg.loss.recon_cutoff_hz,
    ).to(device)
    return model, disc, criterion


def _generator_param_groups(model: nn.Module, weight_decay_transformer: float) -> list[dict]:
    """Split generator params so only transformer blocks receive weight decay (Mimi rule)."""
    tx_ids = {
        id(p) for m in model.modules() if isinstance(m, TransformerStack) for p in m.parameters()
    }
    tx_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (tx_params if id(p) in tx_ids else other_params).append(p)
    return [
        {"params": tx_params, "weight_decay": weight_decay_transformer},
        {"params": other_params, "weight_decay": 0.0},
    ]


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=_PRECISION_DTYPES[precision])


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    disc: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    """Save raw generator + EMA shadow + discriminator + optimizer states and the config."""
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "ema": ema.state_dict(),
            "discriminator": disc.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )
    logger.info("saved checkpoint to %s", path)


def train(cfg: SpineConfig, resume_from: str | None = None) -> None:
    """Run single-GPU GAN training from ``cfg``, optionally resuming from a checkpoint."""
    device = torch.device(cfg.training.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    model, disc, criterion = build_from_config(cfg, device)
    logger.info("generator params: %d", sum(p.numel() for p in model.parameters()))
    logger.info("discriminator params: %d", sum(p.numel() for p in disc.parameters()))

    ema = EMA(model, decay=cfg.optimizer.ema_decay)

    optimizer_g = torch.optim.AdamW(
        _generator_param_groups(model, cfg.optimizer.weight_decay_transformer),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
    )
    optimizer_d = torch.optim.AdamW(
        disc.parameters(),
        lr=cfg.optimizer.lr_d,
        betas=cfg.optimizer.betas,
        weight_decay=0.0,
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=cfg.optimizer.lr_gamma)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=cfg.optimizer.lr_gamma)

    pad_to = cfg.model.hop_length * max(cfg.model.vq_strides)
    dataset = AudioDataset(
        audio_dir=cfg.data.train_dir,
        sample_rate=cfg.model.sample_rate,
        clip_duration=cfg.data.clip_duration,
        pad_to=pad_to,
        deterministic=cfg.data.deterministic,
    )
    dataloader = build_dataloader(
        dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers
    )

    config_dict = to_dict(cfg)
    if cfg.training.use_wandb:
        import wandb

        wandb.init(project=cfg.training.wandb_project, config=config_dict)

    if cfg.training.compile:
        model = cast(Spine, torch.compile(model))

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(ckpt["model"])
        disc.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        ema.load_state_dict(ckpt["ema"])
        step = ckpt["step"]
        scheduler_g.last_epoch = step
        scheduler_d.last_epoch = step
        logger.info("resumed from %s at step %d", resume_from, step)

    for epoch in range(cfg.training.max_epochs):
        if step >= cfg.training.max_steps:
            break
        model.train()
        disc.train()

        for batch in dataloader:
            if step >= cfg.training.max_steps:
                break

            audio = batch.to(device)
            with _autocast(device, cfg.training.precision):
                audio_hat = model(audio)["audio_hat"]
            min_len = min(audio.shape[-1], audio_hat.shape[-1])
            audio, audio_hat = audio[..., :min_len], audio_hat[..., :min_len]

            use_disc = step >= cfg.training.disc_start_step

            fmaps_real_cached: list[list[torch.Tensor]] = []
            if use_disc:
                optimizer_d.zero_grad()
                with _autocast(device, cfg.training.precision):
                    fmaps_real = disc(audio)
                    fmaps_fake = disc(audio_hat.detach())
                    d_loss = criterion.discriminator_total(fmaps_real, fmaps_fake)["total"]
                fmaps_real_cached = [[f.detach() for f in sub] for sub in fmaps_real]
                d_loss.backward()
                d_grad = nn.utils.clip_grad_norm_(disc.parameters(), cfg.optimizer.grad_clip_d)
                optimizer_d.step()
                scheduler_d.step()
            else:
                d_loss = torch.zeros((), device=device)
                d_grad = torch.zeros((), device=device)

            optimizer_g.zero_grad()
            with _autocast(device, cfg.training.precision):
                fmaps_fake = disc(audio_hat) if use_disc else []
                g_losses = criterion.generator_total(
                    audio, audio_hat, fmaps_real_cached, fmaps_fake, use_adversarial=use_disc
                )
            g_losses["total"].backward()
            g_grad = nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_g)
            optimizer_g.step()
            scheduler_g.step()

            ema.update(_unwrap(model))

            step += 1
            if step % cfg.training.log_every == 0:
                metrics = {
                    "train/g_total": g_losses["total"].item(),
                    "train/g_mel": g_losses["mel"].item(),
                    "train/g_stft": g_losses["stft"].item(),
                    "train/g_hf_energy": g_losses["hf_energy"].item(),
                    "train/g_adv": g_losses["adv"].item(),
                    "train/g_feat": g_losses["feat"].item(),
                    "train/d_total": float(d_loss),
                    "train/g_grad_norm": float(g_grad),
                    "train/d_grad_norm": float(d_grad),
                    "train/lr": scheduler_g.get_last_lr()[0],
                }
                logger.info(
                    "step %d | G %.4f (mel %.4f stft %.4f hf %.4f adv %.4f feat %.4f) | D %.4f",
                    step,
                    metrics["train/g_total"],
                    metrics["train/g_mel"],
                    metrics["train/g_stft"],
                    metrics["train/g_hf_energy"],
                    metrics["train/g_adv"],
                    metrics["train/g_feat"],
                    metrics["train/d_total"],
                )
                if cfg.training.use_wandb:
                    import wandb

                    wandb.log(metrics, step=step)

            if step % cfg.training.save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{step}.pt",
                    model,
                    ema,
                    disc,
                    optimizer_g,
                    optimizer_d,
                    step,
                    config_dict,
                )

    save_checkpoint(
        checkpoint_dir / "final.pt",
        model,
        ema,
        disc,
        optimizer_g,
        optimizer_d,
        step,
        config_dict,
    )
    logger.info("training complete at step %d", step)

    if cfg.training.use_wandb:
        import wandb

        wandb.finish()
