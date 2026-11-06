"""Optional global geometric calibration; no sample-specific rules."""

from dataclasses import dataclass
import math

import torch


@dataclass
class PostCalibrationConfig:
    range_scale: float = 1.0
    range_offset_m: float = 0.0
    count_consensus_strength: float = 0.0
    range_quality_low: float = 0.0
    range_quality_high: float = 0.0
    enforce_observed_range: bool = False


def apply_post_calibration(prediction, config, physical_config, inputs=None):
    if (
        not math.isfinite(config.range_scale)
        or config.range_scale <= 0
        or not math.isfinite(config.range_offset_m)
    ):
        raise ValueError("Invalid range calibration")
    if not 0 <= config.count_consensus_strength <= 1:
        raise ValueError("Invalid count consensus strength")
    out = dict(prediction)
    if config.count_consensus_strength:
        logits = prediction["count_logits"]
        s = config.count_consensus_strength
        logits = (1 - s) * logits + s * logits.mean(1, keepdim=True)
        out.update(count_logits=logits, layer_count=logits.argmax(-1) + 1)
    if config.range_scale != 1 or config.range_offset_m:
        distance = prediction["layer_delays_s"] * physical_config.c0_m_s / 2
        proposed = (config.range_scale * distance + config.range_offset_m).clamp_min(
            0.0001
        )
        if config.range_quality_high:
            if inputs is None or config.range_quality_high <= config.range_quality_low:
                raise ValueError(
                    "Quality-gated geometry needs valid inputs and thresholds"
                )
            peak = inputs["tof"].abs().amax(-1, keepdim=True)
            gate = (
                (peak - config.range_quality_low)
                / (config.range_quality_high - config.range_quality_low)
            ).clamp(0, 1)
            distance = distance + gate * (proposed - distance)
        else:
            distance = proposed
        tau = 2 * distance / physical_config.c0_m_s
        velocity = physical_config.c0_m_s / (
            1
            + physical_config.dielectric_density_coefficient
            * prediction["layer_density"]
        )
        out.update(
            layer_delays_s=tau,
            velocity_m_s=velocity,
            layer_thickness_m=velocity * tau / 2,
        )
    if config.enforce_observed_range:
        if inputs is None or "time_s" not in inputs:
            raise ValueError("Observed time axis is required for range support")
        axis = inputs["time_s"]
        window = (axis.amax(-1) - axis.amin(-1)) * physical_config.c0_m_s / 2
        if window.ndim == 0:
            window = window[None, None]
        elif window.ndim == 1:
            window = window[:, None]
        mask = torch.arange(
            out["layer_delays_s"].shape[-1], device=out["layer_count"].device
        ) < out["layer_count"].unsqueeze(-1)
        distance = out["layer_delays_s"] * physical_config.c0_m_s / 2
        factor = (
            (window / (distance.mul(mask).sum(-1).clamp_min(1e-12)))
            .clamp(max=1)
            .unsqueeze(-1)
        )
        tau = out["layer_delays_s"] * factor
        velocity = physical_config.c0_m_s / (
            1 + physical_config.dielectric_density_coefficient * out["layer_density"]
        )
        out.update(
            layer_delays_s=tau,
            velocity_m_s=velocity,
            layer_thickness_m=velocity * tau / 2,
        )
    return out
