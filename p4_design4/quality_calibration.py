"""Optional input-quality expert around an unchanged main backbone.

Quality gates use only signal coverage/amplitude. They contain no filenames,
record IDs, label profiles or feature lookup tables. The expert estimates
positive delay and density; thickness remains the analytic physical result.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class QualityExpertConfig:
    width: int = 256
    max_layers: int = 10
    rho_min_g_cm3: float = 0.6
    rho_max_g_cm3: float = 1.1
    max_range_m: float = 2.199
    c0_m_s: float = 299792458.0
    dielectric_density_coefficient: float = 0.845
    dense_start: float = 0.97
    dense_full: float = 0.995
    weak_start: float = 0.8
    weak_full: float = 0.2
    dense_weight: float = 1.0
    weak_weight: float = 0.0
    temporal_consensus: str = "median"
    density_weight: float = 1.0
    range_weight: float = 1.0
    dense_count_weight: float = 1.0
    weak_count_weight: float = 1.0
    dropout_probability: float = 0.0
    feature_mode: str = "backbone"
    weak_single_layer_only: bool = False
    weak_exclude_dense: bool = False
    weak_min_range_m: float = 0.0


def quality_gates(inputs, config):
    tau = inputs["tof"].float()
    coverage = (tau != 0).float().mean(-1)
    peak = tau.abs().amax(-1)
    dense = (
        (coverage - config.dense_start) / (config.dense_full - config.dense_start)
    ).clamp(0, 1)
    weak = ((config.weak_start - peak) / (config.weak_start - config.weak_full)).clamp(
        0, 1
    )
    if config.weak_exclude_dense:
        weak = weak * (1 - dense)
    return {
        "dense": dense,
        "weak": weak,
        "combined": torch.maximum(
            config.dense_weight * dense, config.weak_weight * weak
        ).clamp(0, 1),
    }


class QualityPhysicalExpert(nn.Module):
    def __init__(self, input_dim, mean, std, config=None):
        super().__init__()
        self.config = config or QualityExpertConfig()
        c = self.config
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        layers = [nn.Linear(input_dim, c.width), nn.GELU()]
        if c.dropout_probability:
            layers.append(nn.Dropout(c.dropout_probability))
        layers.extend([nn.Linear(c.width, c.width), nn.GELU()])
        if c.dropout_probability:
            layers.append(nn.Dropout(c.dropout_probability))
        self.encoder = nn.Sequential(*layers)
        self.count_head = nn.Linear(c.width, c.max_layers)
        self.rho_head = nn.Linear(c.width, c.max_layers)
        self.range_head = nn.Linear(c.width, c.max_layers)
        nn.init.zeros_(self.rho_head.weight)
        nn.init.constant_(self.rho_head.bias, 0.25)
        nn.init.zeros_(self.range_head.weight)
        nn.init.constant_(self.range_head.bias, -2.0)

    def forward(self, features):
        c = self.config
        z = self.encoder(
            ((features - self.feature_mean) / self.feature_std).clamp(-10, 10)
        )
        logits = self.count_head(z)
        rho = (
            c.rho_min_g_cm3
            + (c.rho_max_g_cm3 - c.rho_min_g_cm3) * self.rho_head(z).sigmoid()
        )
        distance = c.max_range_m * self.range_head(z).sigmoid()
        count = logits.argmax(-1) + 1
        # Keep the estimated active profile within the observed range window.
        mask = torch.arange(c.max_layers, device=count.device) < count.unsqueeze(-1)
        factor = (distance.mul(mask).sum(-1, keepdim=True) / c.max_range_m).clamp_min(1)
        if not self.training:
            distance = distance / factor
        tau = 2 * distance / c.c0_m_s
        velocity = c.c0_m_s / (1 + c.dielectric_density_coefficient * rho)
        return {
            "count_logits": logits,
            "layer_count": count,
            "layer_density": rho,
            "layer_delays_s": tau,
            "layer_thickness_m": velocity * tau / 2,
            "velocity_m_s": velocity,
        }


def combine_quality_expert(baseline, expert, inputs, config):
    weights = quality_gates(inputs, config)
    gate = weights["combined"].unsqueeze(-1)
    if config.weak_single_layer_only:
        weights["weak"] = weights["weak"] * (baseline["layer_count"] == 1)
    if config.weak_min_range_m:
        weights["weak"] = weights["weak"] * (
            baseline["layer_delays_s"][..., 0] * config.c0_m_s / 2
            >= config.weak_min_range_m
        )
    if config.weak_single_layer_only or config.weak_min_range_m:
        weights["combined"] = torch.maximum(
            config.dense_weight * weights["dense"], config.weak_weight * weights["weak"]
        ).clamp(0, 1)
        gate = weights["combined"].unsqueeze(-1)
    rho = expert["layer_density"]
    tau = expert["layer_delays_s"]
    logits = expert["count_logits"]
    if config.temporal_consensus == "median":
        rho = torch.quantile(rho, 0.5, dim=1, keepdim=True).expand_as(rho)
        tau = torch.quantile(tau, 0.5, dim=1, keepdim=True).expand_as(tau)
        logits = logits.mean(1, keepdim=True).expand_as(logits)
    elif config.temporal_consensus != "none":
        raise ValueError("Unknown expert temporal consensus")
    rho_gate = gate * config.density_weight
    range_gate = gate * config.range_weight
    count_gate = (
        torch.maximum(
            weights["dense"] * config.dense_weight * config.dense_count_weight,
            weights["weak"] * config.weak_weight * config.weak_count_weight,
        )
        .clamp(0, 1)
        .unsqueeze(-1)
    )
    mixed_rho = (1 - rho_gate) * baseline["layer_density"] + rho_gate * rho
    mixed_tau = (1 - range_gate) * baseline["layer_delays_s"] + range_gate * tau
    probabilities = (1 - count_gate) * baseline["count_logits"].softmax(
        -1
    ) + count_gate * logits.softmax(-1)
    mixed_logits = torch.where(
        count_gate == 0, baseline["count_logits"], probabilities.clamp_min(1e-30).log()
    )
    velocity = config.c0_m_s / (1 + config.dielectric_density_coefficient * mixed_rho)
    out = dict(baseline)
    out.update(
        layer_density=mixed_rho,
        layer_delays_s=mixed_tau,
        count_logits=mixed_logits,
        layer_count=mixed_logits.argmax(-1) + 1,
        velocity_m_s=velocity,
        layer_thickness_m=velocity * mixed_tau / 2,
        quality_dense_gate=weights["dense"],
        quality_weak_gate=weights["weak"],
        quality_expert_gate=weights["combined"],
    )
    return out
