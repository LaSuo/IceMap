"""Six staged physical training losses. No convergence-triggered schedule."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrainingConfig:
    stage_epochs: tuple = (20, 20, 40)
    warmup_epochs: int = 5
    lambda_b: float = 1.0
    lambda_count: float = 0.5
    lambda_pre_rho: float = 0.5
    lambda_rho: float = 1.0
    lambda_h: float = 2.0
    lambda_profile: float = 1.0
    # Focal alpha/gamma have no numerical values in the reference specification.
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0


def stage_weights(epoch, cfg):
    if not 0 <= epoch < sum(cfg.stage_epochs):
        raise ValueError("Epoch outside the 20/20/40 training schedule.")
    start2, start3 = cfg.stage_epochs[0], sum(cfg.stage_epochs[:2])
    stage = 1 if epoch < start2 else (2 if epoch < start3 else 3)

    def ramp(start):
        if epoch < start:
            return 0.0
        # The first and fifth epochs of activation use 0 and target weight.
        return min((epoch - start) / max(cfg.warmup_epochs - 1, 1), 1.0)

    return stage, {
        "boundary": cfg.lambda_b * ramp(0),
        "count": cfg.lambda_count * ramp(0),
        "pre_rho": cfg.lambda_pre_rho * ramp(start2),
        "rho": cfg.lambda_rho * ramp(start3),
        "h": cfg.lambda_h * ramp(start3),
        "profile": cfg.lambda_profile * ramp(start3),
    }


def _layer_l1(pred, truth, counts):
    mask = torch.arange(pred.shape[-1], device=pred.device) < counts.unsqueeze(-1)
    return (((pred - truth).abs() * mask).sum(-1) / counts.clamp_min(1)).mean()


class MineLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TrainingConfig()

    def forward(self, out, batch, epoch):
        cfg = self.config
        stage, weights = stage_weights(epoch, cfg)
        logits = out["boundary_logits"]
        target = batch["boundary_target"]
        p = logits.sigmoid()
        pt = torch.where(target > 0.5, p, 1 - p)
        alpha = torch.where(target > 0.5, cfg.focal_alpha, 1 - cfg.focal_alpha)
        # specified formulation divides by N, not by the number of positives.
        focal = (
            alpha
            * (1 - pt).pow(cfg.focal_gamma)
            * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        )
        losses = {
            "boundary": focal.mean(),
            "count": (out["boundary_count"] - (batch["layer_count"] + 1)).abs().mean(),
        }
        if stage >= 2:
            losses["pre_rho"] = _layer_l1(
                out["rho_tilde"], batch["relative_density_target"], batch["layer_count"]
            )
        if stage >= 3:
            losses["rho"] = _layer_l1(
                out["layer_density"], batch["layer_density_gt"], batch["layer_count"]
            )
            losses["h"] = _layer_l1(
                out["layer_thickness_m"],
                batch["layer_thickness_m_gt"],
                batch["layer_count"],
            )
            losses["profile"] = (
                (out["physical_profile"] - batch["density_profile_gt"]).square().mean()
            )
        total = sum(weights[name] * value for name, value in losses.items())
        return total, losses, weights
