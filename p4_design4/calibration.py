"""Optional learned residual calibration on frozen SPAI fingerprints.
The complete main backbone is still evaluated. Counts/density/delay get an
explicit extra head. Thickness is never regressed independently.
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CalibrationConfig:
    width: int = 128
    max_layers: int = 10
    density_correction_limit: float = 0.15
    range_correction_limit_m: float = 0.8
    range_softplus_scale_m: float = 0.005
    c0_m_s: float = 299792458.0
    dielectric_density_coefficient: float = 0.845
    dropout_probability: float = 0.0


class ResidualPhysicalCalibration(nn.Module):
    def __init__(self, input_dim, mean, std, config=None):
        super().__init__()
        self.config = config or CalibrationConfig()
        c = self.config
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        if not 0 <= c.dropout_probability < 1:
            raise ValueError("Invalid additional-head dropout probability.")
        layers = [nn.Linear(input_dim, c.width), nn.GELU()]
        if c.dropout_probability:
            layers.append(nn.Dropout(c.dropout_probability))
        layers.extend([nn.Linear(c.width, c.width), nn.GELU()])
        if c.dropout_probability:
            layers.append(nn.Dropout(c.dropout_probability))
        self.encoder = nn.Sequential(*layers)
        self.count_head = nn.Linear(c.width, c.max_layers)
        self.density_head = nn.Linear(c.width, c.max_layers)
        self.delay_head = nn.Linear(c.width, c.max_layers)
        nn.init.zeros_(self.density_head.weight)
        nn.init.zeros_(self.density_head.bias)
        nn.init.zeros_(self.delay_head.weight)
        nn.init.zeros_(self.delay_head.bias)

    def forward(self, features, base_density, base_range_m):
        c = self.config
        z = self.encoder(
            ((features - self.feature_mean) / self.feature_std).clamp(-10, 10)
        )
        logits = self.count_head(z)
        correction_rho = c.density_correction_limit * torch.tanh(self.density_head(z))
        rho = (base_density + correction_rho).clamp_min(0.01)
        r0 = base_range_m.clamp_min(0.0001)
        s = c.range_softplus_scale_m
        inverse = r0 + s * torch.log(-torch.expm1(-r0 / s))
        correction_range = c.range_correction_limit_m * torch.tanh(self.delay_head(z))
        distance = s * F.softplus((inverse + correction_range) / s)
        tau = 2 * distance / c.c0_m_s
        velocity = c.c0_m_s / (1 + c.dielectric_density_coefficient * rho)
        h = velocity * tau / 2
        return {
            "count_logits": logits,
            "layer_count": logits.argmax(-1) + 1,
            "layer_density": rho,
            "layer_thickness_m": h,
            "layer_delays_s": tau,
            "velocity_m_s": velocity,
            "density_correction": correction_rho,
            "range_correction_m": correction_range,
        }


def calibration_features(out, pool_bins=16):
    B, T, N, C = out["phase_features"].shape
    vectors = []
    for name in ["phase_features", "density_evidence", "delay_sample_features"]:
        z = out[name].reshape(B * T, N, -1).transpose(1, 2)
        vectors.append(F.adaptive_avg_pool1d(z, pool_bins).flatten(1))
    vectors.extend(
        [
            out["layer_density"].reshape(B * T, -1),
            out["layer_delays_s"].reshape(B * T, -1) * 299792458 / 2,
            out["boundary_count"].reshape(B * T, 1),
            out["boundary_times_s"].reshape(B * T, -1) * 299792458 / 2,
        ]
    )
    return torch.cat(vectors, -1).reshape(B, T, -1)
