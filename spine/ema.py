"""EMA of the generator weights; the shadow is the canonical Spine output for eval/inference (generator-only)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Shadow parameter dict updated each step as ``decay·shadow + (1-decay)·param``."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        assert 0.0 < decay < 1.0, f"decay={decay} must be in (0, 1)"
        self.decay = decay
        self.step = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._params: list[torch.Tensor] = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        self._shadows = [self.shadow[n] for n, p in model.named_parameters() if p.requires_grad]

    def _effective_decay(self) -> float:
        """Warm decay up ``(1+step)/(10+step)`` so a high target can't pin random init in the shadow."""
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow from current raw parameters (uses the module EMA was built from)."""
        del model
        self.step += 1
        decay = self._effective_decay()
        torch._foreach_mul_(self._shadows, decay)
        torch._foreach_add_(self._shadows, self._params, alpha=1.0 - decay)

    @torch.no_grad()
    def reset_to(self, model: nn.Module) -> None:
        """Re-seed the shadow from current weights and restart warmup (after a strict=False resume)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].copy_(param.detach())
        self.step = 0

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily swap raw weights with EMA weights; restore on exit."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        try:
            yield model
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Snapshot for checkpointing; carries the warmup step so the decay ramp stays continuous across resumes."""
        sd = {k: v.detach().clone() for k, v in self.shadow.items()}
        sd["__ema_step__"] = torch.tensor(self.step)
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load a shadow snapshot; shapes must match the current model."""
        step = sd.get("__ema_step__")
        if step is not None:
            self.step = int(step.item())
        missing = set(self.shadow) - set(sd)
        if missing:
            raise KeyError(f"EMA state missing {len(missing)} keys, first: {next(iter(missing))}")
        for name in self.shadow:
            self.shadow[name].copy_(sd[name])
