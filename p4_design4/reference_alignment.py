"""Uniform reference-aligned posterior regularization, with no labels/IDs."""

from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass
class ReferenceAlignmentConfig:
    count_prior_weight: float = 0.0
    count_prior_sigma: float = 0.5
    density_shrinkage: float = 1.0
    range_shrinkage: float = 1.0
    regression_temporal_consensus: str = "none"
    count_prior_disagreement_limit: float | None = None
    density_offset_g_cm3: float = 0.0


def align_prediction(prediction, base, config, physical_config):
    c = config
    if not 0 <= c.density_shrinkage <= 1 or not 0 <= c.range_shrinkage <= 1:
        raise ValueError("Shrinkage weights must be in [0,1].")
    if c.count_prior_weight < 0 or c.count_prior_sigma <= 0:
        raise ValueError("Invalid count-prior regularization.")
    if (
        c.count_prior_disagreement_limit is not None
        and c.count_prior_disagreement_limit <= 0
    ):
        raise ValueError("Count disagreement limit must be positive.")
    if not math.isfinite(c.density_offset_g_cm3):
        raise ValueError("Density offset must be finite.")
    rho = (
        base["layer_density"]
        + c.density_shrinkage * prediction["density_correction"]
        + c.density_offset_g_cm3
    ).clamp_min(0.01)
    r0 = (base["layer_delays_s"] * physical_config.c0_m_s / 2).clamp_min(0.0001)
    scale = physical_config.range_softplus_scale_m
    inverse = r0 + scale * torch.log(-torch.expm1(-r0 / scale))
    distance = scale * F.softplus(
        (inverse + c.range_shrinkage * prediction["range_correction_m"]) / scale
    )
    if c.regression_temporal_consensus == "mean":
        rho = rho.mean(dim=1, keepdim=True).expand_as(rho)
        distance = distance.mean(dim=1, keepdim=True).expand_as(distance)
    elif c.regression_temporal_consensus == "median":
        # Linear interpolation between the middle two observations for even T.
        rho = torch.quantile(rho, 0.5, dim=1, keepdim=True).expand_as(rho)
        distance = torch.quantile(distance, 0.5, dim=1, keepdim=True).expand_as(
            distance
        )
    elif c.regression_temporal_consensus != "none":
        raise ValueError("Unknown temporal consensus method.")
    classes = torch.arange(
        1, physical_config.max_layers + 1, device=rho.device, dtype=rho.dtype
    )
    expected = (base["boundary_count"] - 1).unsqueeze(-1)
    prior = -0.5 * ((expected - classes) / c.count_prior_sigma).square()
    if c.count_prior_disagreement_limit is not None:
        learned_count = (prediction["count_logits"].softmax(-1) * classes).sum(
            -1, keepdim=True
        )
        prior = prior * (
            (expected - learned_count).abs() <= c.count_prior_disagreement_limit
        )
    logits = prediction["count_logits"] + c.count_prior_weight * prior
    tau = 2 * distance / physical_config.c0_m_s
    velocity = physical_config.c0_m_s / (
        1 + physical_config.dielectric_density_coefficient * rho
    )
    out = dict(prediction)
    out.update(
        count_logits=logits,
        layer_count=logits.argmax(-1) + 1,
        layer_density=rho,
        layer_delays_s=tau,
        velocity_m_s=velocity,
        layer_thickness_m=velocity * tau / 2,
    )
    return out
