"""SPAI and ERF: ordered physical inversion. Units: s, m, g/cm^3.

Full SPAI has no independent thickness regression or detached layer assignment
in training. CNN ablations are explicitly selected through mine_ablations.py.
Architectural widths are explicit implementation choices.
The implementation uses L+1 interfaces for L physical ice-layer slots.
"""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    max_layers: int = 10
    assignment_temperature: float = 0.23
    c0_m_s: float = 299792458.0
    dielectric_density_coefficient: float = 0.845
    use_erf: bool = True
    erf_windows: tuple = ((30, 2), (60, 3), (90, 4))
    erf_beta: tuple = (1.0, 1.0, 1.0)
    # Not numerically specified by the reference specification:
    cnn_channels: int = 64
    gru_hidden: int = 64
    unet_base: int = 32
    delay_channels: int = 32
    timing_support_temperature_bins: float = 1.0
    profile_temperature_m: float = 0.001
    count_temperature: float = 0.23

    def validate(self):
        for name in [
            "max_layers",
            "cnn_channels",
            "gru_hidden",
            "unet_base",
            "delay_channels",
        ]:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer.")
        for name in [
            "assignment_temperature",
            "c0_m_s",
            "timing_support_temperature_bins",
            "profile_temperature_m",
            "count_temperature",
        ]:
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(name + " must be positive and finite.")
        if (
            not math.isfinite(self.dielectric_density_coefficient)
            or self.dielectric_density_coefficient < 0
        ):
            raise ValueError("Invalid dielectric coefficient.")
        if type(self.use_erf) is not bool:
            raise ValueError("use_erf must be Boolean.")
        if len(self.erf_windows) != 3 or any(
            len(w) != 2 or any(type(v) is not int or v < 1 for v in w)
            for w in self.erf_windows
        ):
            raise ValueError("Three positive depth/frame window pairs required.")
        if len(self.erf_beta) != 3 or any(
            not math.isfinite(v) or v < 0 for v in self.erf_beta
        ):
            raise ValueError("Three finite nonnegative ERF strengths required.")
        return self


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, padding=kernel // 2),
            nn.BatchNorm1d(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class EntropyReliabilityRefinement(nn.Module):
    """Reliability refinement. Input [B,T,N,D]; LayerNorm is along latent D.

    Windows include only observed neighbors at edges, with the even-window
    extra sample on the positive side. No cross-modality softmax or threshold.
    """

    def __init__(self, channels, depth_window, frame_window, beta=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.depth_window, self.frame_window = depth_window, frame_window
        self.beta = beta

    def _mean(self, x):
        kd, kt = self.depth_window, self.frame_window
        pad = ((kd - 1) // 2, kd // 2, (kt - 1) // 2, kt // 2)
        sums = F.avg_pool2d(F.pad(x, pad), (kt, kd), stride=1, divisor_override=1)
        counts = F.avg_pool2d(
            F.pad(torch.ones_like(x[:, :1]), pad),
            (kt, kd),
            stride=1,
            divisor_override=1,
        )
        return sums / counts.clamp_min(1)

    def forward(self, features):
        z = self.norm(features).permute(0, 3, 1, 2)
        consensus = self._mean(z)
        variance = (self._mean(z.square()) - consensus.square()).clamp_min(0)
        H = 1 - torch.exp(-0.5 * torch.log1p(variance).mean(dim=1, keepdim=True))
        omega = torch.exp(-self.beta * H)
        refined = omega * z + (1 - omega) * consensus
        return refined.permute(0, 2, 3, 1), {
            "entropy": H.squeeze(1),
            "reliability": omega.squeeze(1),
        }


class SkeletonAnchoringNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c, h = cfg.cnn_channels, cfg.gru_hidden
        self.cnn = nn.Sequential(ConvBlock(1, c, 7), ConvBlock(c, c, 5))
        self.bigru = nn.GRU(c, h, batch_first=True, bidirectional=True)
        self.erf = (
            EntropyReliabilityRefinement(2 * h, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )
        self.head = nn.Linear(2 * h, 1)

    def forward(self, phase):
        B, T, N = phase.shape
        x = self.cnn(phase.reshape(B * T, 1, N)).transpose(1, 2)
        z, _ = self.bigru(x)
        z = z.reshape(B, T, N, -1)
        reliability = {}
        if self.erf is not None:
            z, reliability = self.erf(z)
        logits = self.head(z).squeeze(-1).float()
        return logits, z, reliability


class UNet1D(nn.Module):
    """Three-level 1D U-Net, output latent sample-wise density evidence r."""

    def __init__(self, base):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(1, base, 7), ConvBlock(base, base))
        self.enc2 = nn.Sequential(
            ConvBlock(base, base * 2), ConvBlock(base * 2, base * 2)
        )
        self.enc3 = nn.Sequential(
            ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 4)
        )
        self.bottom = nn.Sequential(
            ConvBlock(base * 4, base * 8), ConvBlock(base * 8, base * 8)
        )
        self.dec3 = ConvBlock(base * 12, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)

    def forward(self, scatter):
        B, T, N = scatter.shape
        e1 = self.enc1(scatter.reshape(B * T, 1, N))
        e2 = self.enc2(F.avg_pool1d(e1, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool1d(e2, 2, ceil_mode=True))
        z = self.bottom(F.avg_pool1d(e3, 2, ceil_mode=True))
        z = self.dec3(
            torch.cat(
                (
                    F.interpolate(
                        z, size=e3.shape[-1], mode="linear", align_corners=False
                    ),
                    e3,
                ),
                dim=1,
            )
        )
        z = self.dec2(
            torch.cat(
                (
                    F.interpolate(
                        z, size=e2.shape[-1], mode="linear", align_corners=False
                    ),
                    e2,
                ),
                dim=1,
            )
        )
        z = self.dec1(
            torch.cat(
                (F.interpolate(z, size=N, mode="linear", align_corners=False), e1),
                dim=1,
            )
        )
        return z.transpose(1, 2).reshape(B, T, N, -1)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        z = F.gelu(self.bn1(self.conv1(x)))
        return F.gelu(x + self.bn2(self.conv2(z)))


class ResNet8ToFEncoder(nn.Module):
    """8 convolutions: stem + 3 two-convolution blocks + embedding head."""

    def __init__(self, c):
        super().__init__()
        self.stem = ConvBlock(1, c, 7)
        self.blocks = nn.Sequential(*(ResidualBlock(c) for _ in range(3)))
        self.output = nn.Conv1d(c, c + 1, 1)

    def forward(self, tof):
        B, T, N = tof.shape
        z = self.output(self.blocks(self.stem(tof.reshape(B * T, 1, N))))
        z = z.transpose(1, 2).reshape(B, T, N, -1)
        return z


def soft_layer_assignment(q, max_layers, temperature):
    layers = torch.arange(1, max_layers + 1, device=q.device, dtype=q.dtype)
    return torch.softmax(-(q.unsqueeze(-1) - layers).abs() / temperature, dim=-1)


def pool_layers(evidence, assignment):
    return torch.einsum("btnk,btnc->btkc", assignment, evidence) / assignment.sum(
        dim=2
    ).unsqueeze(-1).clamp_min(1e-12)


def _positive_head(cin, hidden=32):
    head = nn.Sequential(
        nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Softplus()
    )
    nn.init.constant_(head[-2].bias, math.log(math.expm1(0.85)))
    return head


class SPAIIceInversionNet(nn.Module):
    def __init__(self, config=None, *, max_layers=None, use_entropy_gate=None):
        super().__init__()
        self.config = config or ModelConfig()
        if max_layers is not None:
            self.config.max_layers = max_layers
        if use_entropy_gate is not None:
            self.config.use_erf = use_entropy_gate
        cfg = self.config
        cfg.validate()
        self.skeleton = SkeletonAnchoringNet(cfg)
        self.density_unet = UNet1D(cfg.unet_base)
        self.density_erf = (
            EntropyReliabilityRefinement(
                cfg.unet_base, *cfg.erf_windows[1], cfg.erf_beta[1]
            )
            if cfg.use_erf
            else None
        )
        self.tof_encoder = ResNet8ToFEncoder(cfg.delay_channels)
        self.delay_erf = (
            EntropyReliabilityRefinement(
                cfg.delay_channels + 1, *cfg.erf_windows[2], cfg.erf_beta[2]
            )
            if cfg.use_erf
            else None
        )
        self.g_rho = _positive_head(cfg.unet_base)
        self.g_cal = _positive_head(cfg.delay_channels + 1)
        self.g_ref = _positive_head(cfg.delay_channels + 2)

    @staticmethod
    def _axis(axis, B, T, N, device, dtype):
        axis = torch.as_tensor(axis, device=device, dtype=dtype)
        if axis.ndim == 1:
            axis = axis[None, None, :].expand(B, T, -1)
        elif axis.ndim == 2:
            axis = axis[:, None, :].expand(-1, T, -1)
        if (
            axis.shape != (B, T, N)
            or not torch.isfinite(axis).all()
            or not (axis.diff(dim=-1) > 0).all()
        ):
            raise ValueError(
                "Axis must be finite, strictly increasing and broadcast to [B,T,N]."
            )
        return axis

    def forward(
        self,
        tof,
        phase,
        scatter,
        time_s=None,
        depth_m=None,
        scatter_valid=None,
        need_layer=True,
        need_physical=True,
        observed_delay_events_s=None,
        observed_delay_event_valid=None,
        event_tolerance_s=None,
    ):
        for name, value in [("tof", tof), ("phase", phase), ("scatter", scatter)]:
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise TypeError(name + " must be a floating point tensor.")
            if value.ndim != 3 or min(value.shape) < 1 or value.shape[-1] < 8:
                raise ValueError(name + " requires nonempty [B,T,N] with N>=8.")
            if not torch.isfinite(value).all():
                raise ValueError(name + " contains NaN/Inf.")
        if (
            phase.device != scatter.device
            or phase.device != tof.device
            or phase.dtype != scatter.dtype
            or phase.dtype != tof.dtype
        ):
            raise ValueError("Fingerprint devices and dtypes must match.")
        if scatter_valid is not None:
            if (
                not torch.is_tensor(scatter_valid)
                or scatter_valid.shape != scatter.shape
                or scatter_valid.device != scatter.device
            ):
                raise ValueError("scatter_valid must match shape and device.")
            if scatter_valid.dtype != torch.bool:
                if (
                    not torch.isfinite(scatter_valid).all()
                    or not ((scatter_valid == 0) | (scatter_valid == 1)).all()
                ):
                    raise ValueError("scatter_valid must be Boolean or binary.")
                scatter_valid = scatter_valid.bool()
        if phase.shape != scatter.shape or phase.shape != tof.shape or phase.ndim != 3:
            raise ValueError("Fingerprints must share [B,T,N].")
        cfg = self.config
        B, T, N = phase.shape
        if time_s is None:
            raise ValueError(
                "Calibrated two-way time_s in seconds is required; N cannot determine dt."
            )
        time = self._axis(time_s, B, T, N, phase.device, phase.dtype)
        logits, phase_features, phase_erf = self.skeleton(phase)
        b = logits.sigmoid()
        count = b.sum(dim=-1)
        q = 1 + b.cumsum(dim=-1)  # specified formulation
        mine_M = soft_layer_assignment(q, cfg.max_layers, cfg.assignment_temperature)
        out = {
            "boundary_logits": logits,
            "boundary_prob": b,
            "q": q,
            "M_mine": mine_M,
            "boundary_count": count,
            "layer_count": (count.round().long() - 1).clamp(0, cfg.max_layers),
            "phase_features": phase_features,
            "erf_phase": phase_erf,
        }
        if not need_layer and not need_physical:
            return out
        # Soft ordered boundary event times (K+1) preserve gradients to b.
        # Ordinal likelihood uses b as event mass, with half-bin centering.
        ranks = b.cumsum(dim=-1) - 0.5 * b
        event_ranks = (
            torch.arange(cfg.max_layers + 1, device=b.device, dtype=b.dtype) + 0.5
        )
        event_logits = (
            b.clamp_min(1e-12).log().unsqueeze(-1)
            - (ranks.unsqueeze(-1) - event_ranks).abs() / cfg.assignment_temperature
        )
        delay_erf = {}
        delay_z = None
        timing_evidence = torch.zeros_like(tof)
        if need_physical:
            delay_z = self.tof_encoder(tof)
            if self.delay_erf is not None:
                delay_z, delay_erf = self.delay_erf(delay_z)
            timing_evidence = delay_z[..., -1].sigmoid()
            event_logits = event_logits + (0.5 + timing_evidence).log().unsqueeze(-1)
        event_weights = event_logits.softmax(dim=2)
        boundary_times = (event_weights * time.unsqueeze(-1)).sum(dim=2)
        if observed_delay_events_s is not None:
            if self.training:
                raise ValueError(
                    "Observed D3 peak association is an explicit inference-only alternative; train the learned-timing model first."
                )
            from event_association import associate_events

            boundary_times, associated = associate_events(
                boundary_times,
                out["layer_count"],
                observed_delay_events_s,
                observed_delay_event_valid,
                event_tolerance_s,
            )
            out["observed_delay_events_used"] = associated
        tau_layer = (boundary_times[..., 1:] - boundary_times[..., :-1]).clamp_min(0)
        # q counts the air/ice surface. Physical ice slot l corresponds to
        # full-coordinate center l+1. M_mine above retains specified formulation.
        ice_M = soft_layer_assignment(q - 1, cfg.max_layers, cfg.assignment_temperature)
        index = torch.arange(cfg.max_layers, device=b.device, dtype=b.dtype)
        # Continuous clipped ramp: integer L+1 boundary count gives exactly
        # L present layers, without a 10% profile shrink at small L.
        presence = (
            (count.unsqueeze(-1) - (index + 1.5)) / cfg.count_temperature + 0.5
        ).clamp(0, 1)
        if not self.training:
            presence = (index < out["layer_count"].unsqueeze(-1)).to(b.dtype)
        # Differentiable surface/bottom support; bottom timing interpolated
        # at the estimated continuous layer count, never a detached index.
        last_rank = presence.sum(dim=-1).clamp(0, cfg.max_layers)
        end_weights = torch.softmax(
            -(event_ranks - 0.5 - last_rank.unsqueeze(-1)).abs()
            / cfg.assignment_temperature,
            dim=-1,
        )
        t_top = boundary_times[..., 0]
        t_bottom = (boundary_times * end_weights).sum(dim=-1)
        dt = (
            time.diff(dim=-1).mean(dim=-1).clamp_min(1e-15)
            * cfg.timing_support_temperature_bins
        )
        roi = torch.sigmoid(
            (time - t_top.unsqueeze(-1)) / dt.unsqueeze(-1)
        ) * torch.sigmoid((t_bottom.unsqueeze(-1) - time) / dt.unsqueeze(-1))
        M = ice_M * roi.unsqueeze(-1) * presence.unsqueeze(2)
        density_features = self.density_unet(scatter)
        density_erf = {}
        if self.density_erf is not None:
            density_features, density_erf = self.density_erf(density_features)
        if scatter_valid is not None:
            if scatter_valid.shape != scatter.shape:
                raise ValueError("scatter_valid must match scatter.")
            density_M = M * scatter_valid.unsqueeze(-1).to(M.dtype)
        else:
            density_M = M
        r_bar = pool_layers(density_features, density_M)
        rho_tilde = self.g_rho(r_bar).squeeze(-1)
        u = pool_layers(delay_z[..., :-1], M) if delay_z is not None else None
        out.update(
            {
                "M_ice": M,
                "density_pool_weights": density_M,
                "density_evidence": density_features,
                "r_bar": r_bar,
                "rho_tilde": rho_tilde,
                "boundary_times_s": boundary_times,
                "layer_delays_s": tau_layer,
                "delay_features": u,
                "tof_node_prob": timing_evidence,
                "layer_presence": presence,
                "delay_sample_features": (
                    delay_z[..., :-1] if delay_z is not None else None
                ),
                "erf_density": density_erf,
                "erf_delay": delay_erf,
            }
        )
        if not need_physical:
            return out
        if getattr(self, "direct_physical_head", None) is not None:
            # Explicit CNN ablation only. The full SPAI never takes this path.
            rho, thickness = self.direct_physical_head(rho_tilde, u)
            rho0 = rho
            h0 = thickness
            velocity = cfg.c0_m_s / (1 + cfg.dielectric_density_coefficient * rho)
            v0 = velocity
        else:
            rho0 = self.g_cal(torch.cat((rho_tilde.unsqueeze(-1), u), dim=-1)).squeeze(
                -1
            )
            v0 = cfg.c0_m_s / (1 + cfg.dielectric_density_coefficient * rho0)
            h0 = v0 * tau_layer / 2
            # One density refinement only; full-SPAI thickness is analytic.
            rho = self.g_ref(
                torch.cat((rho_tilde.unsqueeze(-1), u, h0.unsqueeze(-1)), dim=-1)
            ).squeeze(-1)
            velocity = cfg.c0_m_s / (1 + cfg.dielectric_density_coefficient * rho)
            thickness = velocity * tau_layer / 2
        edges = F.pad((thickness * presence).cumsum(dim=-1), (1, 0))
        out.update(
            {
                "rho0": rho0,
                "velocity0_m_s": v0,
                "thickness0_m": h0,
                "layer_density": rho,
                "velocity_m_s": velocity,
                "layer_thickness_m": thickness,
                "depth_edges_m": edges,
            }
        )
        if depth_m is not None:
            Nd = depth_m.shape[-1]
            depth = self._axis(depth_m, B, T, Nd, phase.device, phase.dtype)
            s = cfg.profile_temperature_m
            if s <= 0:
                raise ValueError("profile_temperature_m must be positive.")
            profile_M = torch.sigmoid(
                (depth.unsqueeze(-1) - edges[..., :-1].unsqueeze(2)) / s
            ) - torch.sigmoid((depth.unsqueeze(-1) - edges[..., 1:].unsqueeze(2)) / s)
            out["physical_profile"] = (profile_M * rho.unsqueeze(2)).sum(dim=-1)
        return out

    def configuration(self):
        return asdict(self.config)
