"""Optional label-preserving receiver perturbations and inference uncertainty."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class AugmentationConfig:
    gain_log_std: float = 0.0
    additive_noise_fraction: float = 0.0
    phase_noise_rad: float = 0.0
    modality_dropout_probability: float = 0.0

    def validate(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid " + key)
        if self.modality_dropout_probability > 1:
            raise ValueError("Dropout probability exceeds one.")
        return self


def perturb_inputs(batch, config):
    """Noise/gain are empirical receiver perturbations, not climate simulation."""
    config.validate()
    out = dict(batch)
    for name in ["phase", "scatter", "tof"]:
        x = batch[name]
        B = x.shape[0]
        if config.gain_log_std:
            x = x * torch.exp(
                config.gain_log_std * torch.randn(B, 1, 1, device=x.device)
            )
        if config.additive_noise_fraction:
            scale = x.abs().amax(-1, keepdim=True)
            x = x + config.additive_noise_fraction * scale * torch.randn_like(x)
        if name == "phase" and config.phase_noise_rad:
            x = x + config.phase_noise_rad * torch.randn_like(x)
            x = torch.atan2(x.sin(), x.cos())
        if config.modality_dropout_probability:
            keep = (
                torch.rand(B, 1, 1, device=x.device)
                >= config.modality_dropout_probability
            )
            x = x * keep
            if name == "scatter":
                out["scatter_valid"] = batch["scatter_valid"] & keep
        out[name] = x
    return out


@torch.no_grad()
def monte_carlo_dropout(model, inputs, samples=20):
    """Marginal diagnostic moments, not a replacement physical prediction.

    BatchNorm remains in evaluation mode. Each returned draw obeys the model's
    physical relation; a mean of marginal thickness/density/delay need not do so.
    This is epistemic-dropout variability, not calibrated confidence intervals.
    """
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("At least two draws are required.")
    states = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(
                module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
            ):
                module.train()
        draws = [model(inputs) for _ in range(samples)]
    finally:
        for module, flag in states.items():
            module.training = flag
    result = {
        "samples": samples,
        "layer_count_draws": torch.stack([x["layer_count"] for x in draws]),
    }
    K = draws[0]["layer_density"].shape[-1]
    active = torch.arange(K, device=result["layer_count_draws"].device) < result[
        "layer_count_draws"
    ].unsqueeze(-1)
    result["active_layer_draws"] = active
    result["layer_presence_probability"] = active.float().mean(0)
    n = active.sum(0)
    result["conditional_statistics_available"] = n >= 2
    for key in ["layer_density", "layer_delays_s", "layer_thickness_m"]:
        x = torch.stack([d[key] for d in draws])
        result[key + "_draws"] = x
        result[key + "_marginal_mean"] = x.mean(0)
        result[key + "_marginal_std"] = x.std(0)
        conditional = (x * active).sum(0) / n.clamp_min(1)
        variance = ((x - conditional).square() * active).sum(0) / (n - 1).clamp_min(1)
        result[key + "_conditional_mean"] = torch.where(
            n > 0, conditional, torch.full_like(conditional, float("nan"))
        )
        result[key + "_conditional_std"] = torch.where(
            n >= 2, variance.sqrt(), torch.full_like(conditional, float("nan"))
        )
    return result


class PhysicalEnsemble(nn.Module):
    """Probability-mixture count and weighted rho/tau, with analytic h."""

    def __init__(
        self,
        models,
        weights=None,
        c0_m_s=299792458.0,
        dielectric_density_coefficient=0.845,
    ):
        super().__init__()
        if not models:
            raise ValueError("Empty ensemble.")
        self.models = nn.ModuleList(models)
        w = (
            torch.ones(len(models))
            if weights is None
            else torch.as_tensor(weights, dtype=torch.float32)
        )
        if (
            w.shape != (len(models),)
            or not torch.isfinite(w).all()
            or (w < 0).any()
            or w.sum() <= 0
        ):
            raise ValueError("Invalid global ensemble weights.")
        if (
            not math.isfinite(c0_m_s)
            or c0_m_s <= 0
            or not math.isfinite(dielectric_density_coefficient)
            or dielectric_density_coefficient < 0
        ):
            raise ValueError("Invalid physical constants.")
        self.register_buffer("weights", w / w.sum())
        self.c0 = c0_m_s
        self.coefficient = dielectric_density_coefficient

    def forward(self, inputs):
        out = [m(inputs) for m in self.models]
        for value in out:
            if not all(
                k in value for k in ["count_logits", "layer_density", "layer_delays_s"]
            ):
                raise ValueError(
                    "Ensemble requires calibrated members with count logits."
                )
            if (
                not torch.isfinite(value["layer_density"]).all()
                or (value["layer_density"] <= 0).any()
                or not torch.isfinite(value["layer_delays_s"]).all()
                or (value["layer_delays_s"] < 0).any()
            ):
                raise ValueError("Invalid physical member output.")
        if len({tuple(x["layer_density"].shape) for x in out}) != 1:
            raise ValueError("Ensemble members must share layer slot shape/order.")

        def average(key, probability=False):
            values = torch.stack(
                [x[key].softmax(-1) if probability else x[key] for x in out]
            )
            w = self.weights.to(values).reshape(-1, *([1] * (values.ndim - 1)))
            return (values * w).sum(0)

        rho, tau = average("layer_density"), average("layer_delays_s")
        probabilities = average("count_logits", True)
        velocity = self.c0 / (1 + self.coefficient * rho)
        return {
            "layer_count": probabilities.argmax(-1) + 1,
            "count_logits": probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ).log(),
            "layer_density": rho,
            "layer_delays_s": tau,
            "velocity_m_s": velocity,
            "layer_thickness_m": velocity * tau / 2,
        }
