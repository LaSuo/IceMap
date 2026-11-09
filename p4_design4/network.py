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
    input_stride: int = 1
    timing_fingerprint_weight: float = 0.0
    timing_peak_radius_bins: int = 0
    timing_min_separation_bins: int = 0
    minimum_layers: int = 0
    timing_learned_weight: float = 0.0
    boundary_context_frames: int = 1
    timing_condition_on_count: bool = False
    timing_observed_peaks_only: bool = False
    phase_residual_blocks: int = 0
    delay_residual_blocks: int = 0
    compact_calibration: bool = False
    phase_contrast: bool = False
    calibration_phase_prior: float = 0.25
    calibration_strength: float = 1.0
    identity_extra_blocks: bool = False
    compact_interface_roles: bool = False
    interface_role_penalty: float = 1.0
    interface_role_count_prior: float = 0.5
    interface_role_tof_weight: float = 0.5
    interface_role_interval_pooling: bool = True
    interface_role_local_kernel: int = 0
    interface_role_phase_weight: float = 0.0
    interface_role_evidence_radius_bins: int = 0
    echo_path: dict | None = None

    def validate(self):
        for name in [
            "max_layers",
            "cnn_channels",
            "gru_hidden",
            "unet_base",
            "delay_channels",
            "input_stride",
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
        if (
            not math.isfinite(self.timing_fingerprint_weight)
            or self.timing_fingerprint_weight < 0
        ):
            raise ValueError(
                "timing_fingerprint_weight must be finite and nonnegative."
            )
        if (
            type(self.timing_peak_radius_bins) is not int
            or self.timing_peak_radius_bins < 0
        ):
            raise ValueError("timing_peak_radius_bins must be a nonnegative integer.")
        if (
            type(self.timing_min_separation_bins) is not int
            or self.timing_min_separation_bins < 0
        ):
            raise ValueError(
                "timing_min_separation_bins must be a nonnegative integer."
            )
        if (
            type(self.minimum_layers) is not int
            or not 0 <= self.minimum_layers <= self.max_layers
        ):
            raise ValueError("minimum_layers must be between zero and max_layers.")
        if (
            not math.isfinite(self.timing_learned_weight)
            or self.timing_learned_weight < 0
        ):
            raise ValueError("timing_learned_weight must be finite and nonnegative.")
        if (
            type(self.boundary_context_frames) is not int
            or self.boundary_context_frames < 1
            or self.boundary_context_frames % 2 == 0
        ):
            raise ValueError("boundary_context_frames must be a positive odd integer.")
        if type(self.timing_condition_on_count) is not bool:
            raise ValueError("timing_condition_on_count must be Boolean.")
        if type(self.timing_observed_peaks_only) is not bool:
            raise ValueError("timing_observed_peaks_only must be Boolean.")
        for name in ("phase_residual_blocks", "delay_residual_blocks"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 4:
                raise ValueError(f"{name} must be an integer between zero and four.")
        if type(self.compact_calibration) is not bool:
            raise ValueError("compact_calibration must be Boolean.")
        if type(self.phase_contrast) is not bool:
            raise ValueError("phase_contrast must be Boolean.")
        if type(self.identity_extra_blocks) is not bool:
            raise ValueError("identity_extra_blocks must be Boolean.")
        for name in ("compact_interface_roles", "interface_role_interval_pooling"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean.")
        if self.compact_interface_roles and not self.compact_calibration:
            raise ValueError("Interface roles require compact calibration.")
        if not math.isfinite(self.interface_role_penalty):
            raise ValueError("interface_role_penalty must be finite.")
        for name in (
            "interface_role_count_prior",
            "interface_role_tof_weight",
            "interface_role_phase_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        radius = self.interface_role_evidence_radius_bins
        if type(radius) is not int or radius < 0:
            raise ValueError("Interface evidence radius must be a nonnegative integer.")
        kernel = self.interface_role_local_kernel
        if type(kernel) is not int or (
            kernel != 0
            and (not self.compact_interface_roles or kernel < 3 or kernel % 2 == 0)
        ):
            raise ValueError(
                "A local role kernel requires interface roles and odd width >=3."
            )
        if not 0 <= self.calibration_strength <= 1:
            raise ValueError("calibration_strength must be in [0,1].")
        if (
            not math.isfinite(self.calibration_phase_prior)
            or self.calibration_phase_prior < 0
        ):
            raise ValueError("calibration_phase_prior must be finite and nonnegative.")
        if self.echo_path is not None:
            from echo_path import EchoPathConfig

            if not isinstance(self.echo_path, dict):
                raise ValueError("echo_path must be a calibration mapping.")
            EchoPathConfig(**self.echo_path).validate()
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
        # A rectangular mean is separable, including at truncated edges.
        # Prefix differences avoid work proportional to the window area.
        x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        for axis, width in ((-1, self.depth_window), (-2, self.frame_window)):
            x = x.transpose(axis, -1)
            n = x.shape[-1]
            indices = torch.arange(n, device=x.device)
            lo = (indices - (width - 1) // 2).clamp_min(0)
            hi = (indices + width // 2 + 1).clamp_max(n)
            prefix = F.pad(x.cumsum(dim=-1), (1, 0))
            x = ((prefix[..., hi] - prefix[..., lo]) / (hi - lo)).transpose(axis, -1)
        return x

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
        self.cnn = nn.Sequential(
            ConvBlock(2 if cfg.phase_contrast else 1, c, 7),
            ConvBlock(c, c, 5),
            *(
                ResidualBlock(c, cfg.identity_extra_blocks)
                for _ in range(cfg.phase_residual_blocks)
            ),
        )
        self.bigru = nn.GRU(c, h, batch_first=True, bidirectional=True)
        self.erf = (
            EntropyReliabilityRefinement(2 * h, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )
        self.head = nn.Linear(2 * h, 1)

    def forward(self, phase, contrast=None):
        B, T, N = phase.shape
        x = phase.reshape(B * T, 1, N)
        if contrast is not None:
            x = torch.cat((x, contrast.reshape(B * T, 1, N)), dim=1)
        x = self.cnn(x).transpose(1, 2)
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
    def __init__(self, c, start_identity=False):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c)
        self.skip_gain = nn.Parameter(torch.zeros(())) if start_identity else None

    def forward(self, x):
        z = F.gelu(self.bn1(self.conv1(x)))
        y = F.gelu(x + self.bn2(self.conv2(z)))
        return y if self.skip_gain is None else x + self.skip_gain.tanh() * (y - x)


class ResNet8ToFEncoder(nn.Module):
    """8 convolutions: stem + 3 two-convolution blocks + embedding head."""

    def __init__(self, c, extra_blocks=0, start_identity=False):
        super().__init__()
        self.stem = ConvBlock(1, c, 7)
        self.blocks = nn.Sequential(
            *(ResidualBlock(c) for _ in range(3)),
            *(ResidualBlock(c, start_identity) for _ in range(extra_blocks)),
        )
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


def temporal_mean(values, window):
    """Average neighboring frames, with truncated support at the ends."""
    frames = values.shape[1]
    centers = torch.arange(frames, device=values.device)
    left = (centers - window // 2).clamp_min(0)
    right = (centers + window // 2 + 1).clamp_max(frames)
    prefix = F.pad(values.flatten(2).transpose(1, 2).cumsum(-1), (1, 0))
    means = (prefix[..., right] - prefix[..., left]) / (right - left)
    return means.transpose(1, 2).reshape_as(values)


def _positive_head(cin, hidden=32):
    head = nn.Sequential(
        nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Softplus()
    )
    nn.init.constant_(head[-2].bias, math.log(math.expm1(0.85)))
    return head


class CompactInterfaceCalibration(nn.Module):
    """Small shared count and local timing readout from existing SPAI latents."""

    def __init__(
        self, phase_channels, delay_channels, max_layers, interface_roles, local_kernel
    ):
        super().__init__()
        width = 12
        self.project = nn.Linear(phase_channels + delay_channels + 2, width)
        self.context = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.timing = nn.Linear(width, 1)
        self.count = nn.Sequential(
            nn.Linear(12 * width, 24), nn.GELU(), nn.Linear(24, max_layers + 1)
        )
        self.roles = (
            nn.Sequential(nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, 3))
            if interface_roles
            else None
        )
        self.local_roles = (
            nn.Conv1d(3, width, local_kernel, padding=local_kernel // 2)
            if local_kernel
            else None
        )
        nn.init.zeros_(self.timing.weight)
        nn.init.zeros_(self.timing.bias)

    def forward(self, phase_features, delay_features, tof, scatter, phase, contrast):
        B, T, N = tof.shape
        peaks = tof.clamp_min(0)
        peaks = peaks / peaks.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = torch.cat(
            (
                phase_features,
                delay_features,
                peaks.unsqueeze(-1),
                scatter.unsqueeze(-1),
            ),
            dim=-1,
        )
        z = F.gelu(self.project(evidence)).flatten(0, 1).transpose(1, 2)
        z = z + F.gelu(self.context(z))
        pooled = torch.cat(
            (F.adaptive_avg_pool1d(z, 6), F.adaptive_max_pool1d(z, 6)), dim=1
        )
        counts = self.count(pooled.flatten(1)).reshape(B, T, -1).float()
        timing = self.timing(z.transpose(1, 2)).reshape(B, T, N).float()
        role_logits = None
        if self.roles is not None:
            local = contrast.reshape(B * T, 1, N)
            contrast_pyramid = torch.cat(
                (local, F.max_pool1d(local, 5, 1, 2), F.max_pool1d(local, 13, 1, 6)),
                dim=1,
            )
            role_features = z
            if self.local_roles is not None:
                signals = torch.stack((phase, contrast, peaks), dim=-2).reshape(
                    B * T, 3, N
                )
                role_features = role_features + F.gelu(self.local_roles(signals))
            role_input = torch.cat((role_features, contrast_pyramid), dim=1).transpose(
                1, 2
            )
            role_logits = self.roles(role_input).reshape(B, T, N, 3).float()
        return temporal_mean(counts, 3), 2 * timing.tanh(), role_logits


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
        self.tof_encoder = ResNet8ToFEncoder(
            cfg.delay_channels, cfg.delay_residual_blocks, cfg.identity_extra_blocks
        )
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
        self.interface_calibration = (
            CompactInterfaceCalibration(
                2 * cfg.gru_hidden,
                cfg.delay_channels + 1,
                cfg.max_layers,
                cfg.compact_interface_roles,
                cfg.interface_role_local_kernel,
            )
            if cfg.compact_calibration
            else None
        )

    def invert_layer_delays(self, rho_tilde, delay_features, delays_s):
        """Apply the full physical inversion to pooled layer evidence."""
        cfg = self.config
        inputs = torch.cat((rho_tilde.unsqueeze(-1), delay_features), dim=-1)
        rho0 = self.g_cal(inputs).squeeze(-1)
        v0 = cfg.c0_m_s / (1 + cfg.dielectric_density_coefficient * rho0)
        h0 = v0 * delays_s / 2
        rho = self.g_ref(torch.cat((inputs, h0.unsqueeze(-1)), dim=-1)).squeeze(-1)
        velocity = cfg.c0_m_s / (1 + cfg.dielectric_density_coefficient * rho)
        return rho0, v0, h0, rho, velocity, velocity * delays_s / 2

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
        calibrated_path = None
        if (
            cfg.echo_path is not None
            and not self.training
            and need_physical
            and observed_delay_events_s is None
        ):
            from echo_path import read_echo_paths

            calibrated_path = read_echo_paths(
                phase, tof, time, cfg.echo_path, cfg.max_layers, cfg.minimum_layers
            )
        fine_time = time
        contrast = None
        if cfg.phase_contrast or cfg.compact_interface_roles:
            x = phase.reshape(B * T, 1, N)
            stride = cfg.input_stride
            if stride > 1:
                upper = F.max_pool1d(x, stride, ceil_mode=True)
                lower = -F.max_pool1d(-x, stride, ceil_mode=True)
            else:
                upper = F.max_pool1d(x, 3, 1, 1)
                lower = -F.max_pool1d(-x, 3, 1, 1)
            contrast = (upper - lower).reshape(B, T, -1)
        if cfg.input_stride > 1:
            stride = cfg.input_stride

            def pool(x):
                x = x.reshape(B * T, 1, N)
                return F.avg_pool1d(x, stride, ceil_mode=True).reshape(B, T, -1)

            phase = pool(phase)
            tof_pool, tof_indices = F.max_pool1d(
                tof.reshape(B * T, 1, N), stride, ceil_mode=True, return_indices=True
            )
            tof = tof_pool.reshape(B, T, -1)
            fine_time = time.gather(-1, tof_indices.reshape(B, T, -1))
            if scatter_valid is None:
                scatter = pool(scatter)
            else:
                fraction = pool(scatter_valid.float())
                scatter = pool(
                    scatter.masked_fill(~scatter_valid, 0)
                ) / fraction.clamp_min(1e-8)
                scatter_valid = fraction > 0
            time = time[..., ::stride]
            N = phase.shape[-1]
            if N < 8:
                raise ValueError("input_stride must retain at least eight time bins.")
        if not cfg.phase_contrast:
            logits, phase_features, phase_erf = self.skeleton(phase)
        else:
            logits, phase_features, phase_erf = self.skeleton(phase, contrast)
        b = logits.sigmoid()
        use_context = not self.training and cfg.boundary_context_frames > 1
        if use_context:
            b = temporal_mean(b, cfg.boundary_context_frames)
        count = b.sum(dim=-1)
        q = 1 + b.cumsum(dim=-1)  # specified formulation
        mine_M = soft_layer_assignment(q, cfg.max_layers, cfg.assignment_temperature)
        minimum_layers = 0 if self.training else cfg.minimum_layers
        out = {
            "boundary_logits": logits,
            "boundary_prob": b,
            "q": q,
            "M_mine": mine_M,
            "boundary_count": count,
            "layer_count": (count.round().long() - 1).clamp(
                minimum_layers, cfg.max_layers
            ),
            "phase_features": phase_features,
            "erf_phase": phase_erf,
        }
        if not need_layer and not need_physical:
            return out
        delay_erf = {}
        delay_z = None
        timing_logits = None
        timing_evidence = torch.zeros_like(tof)
        physical_mass = b
        physical_count = count
        role_logits = None
        if need_physical:
            delay_z = self.tof_encoder(tof)
            if self.delay_erf is not None:
                delay_z, delay_erf = self.delay_erf(delay_z)
            timing_logits = delay_z[..., -1].float()
            if self.interface_calibration is not None:
                count_logits, correction, role_logits = self.interface_calibration(
                    phase_features, delay_z, tof, scatter, phase, contrast
                )
                if role_logits is not None:
                    out["interface_role_logits"] = role_logits
                slots = torch.arange(cfg.max_layers + 1, device=b.device, dtype=b.dtype)
                count_logits = (
                    count_logits
                    - cfg.calibration_phase_prior
                    * (slots + 1 - count.unsqueeze(-1)).abs()
                )
                out["calibrated_count_logits"] = count_logits
                strength = cfg.calibration_strength
                if strength:
                    decoded = count_logits
                    if not self.training and cfg.minimum_layers:
                        decoded = decoded.masked_fill(
                            slots < cfg.minimum_layers, -torch.inf
                        )
                    expected = (decoded.softmax(-1) * (slots + 1)).sum(-1)
                    physical_count = torch.lerp(count, expected, strength)
                    if not self.training:
                        if strength == 1:
                            out["layer_count"] = decoded.argmax(-1)
                        else:
                            out["layer_count"] = (
                                physical_count.round().long() - 1
                            ).clamp(minimum_layers, cfg.max_layers)
                        physical_count = out["layer_count"].to(b.dtype) + 1
                    physical_mass = b * (
                        physical_count / count.clamp_min(1e-8)
                    ).unsqueeze(-1)
                    timing_logits = timing_logits + strength * correction
            timing_evidence = timing_logits.sigmoid()
        # Soft ordered boundary event times (K+1) preserve gradients to b.
        # Ordinal likelihood uses b as event mass, with half-bin centering.
        ranks = physical_mass.cumsum(dim=-1) - 0.5 * physical_mass
        if (
            not self.training
            and need_physical
            and cfg.timing_condition_on_count
            and physical_mass is b
        ):
            ranks = ranks * (
                (out["layer_count"] + 1) / count.clamp_min(1e-8)
            ).unsqueeze(-1)
        event_ranks = (
            torch.arange(cfg.max_layers + 1, device=b.device, dtype=b.dtype) + 0.5
        )
        event_logits = (
            b.clamp_min(1e-12).log().unsqueeze(-1)
            - (ranks.unsqueeze(-1) - event_ranks).abs() / cfg.assignment_temperature
        )
        if need_physical:
            if cfg.timing_learned_weight:
                event_logits = event_logits + cfg.timing_learned_weight * F.logsigmoid(
                    timing_logits
                ).unsqueeze(-1)
            else:
                event_logits = event_logits + (0.5 + timing_evidence).log().unsqueeze(
                    -1
                )
            if cfg.timing_fingerprint_weight:
                peaks = tof.clamp_min(0)
                peaks = peaks / peaks.amax(dim=-1, keepdim=True).clamp_min(1e-8)
                event_logits = event_logits + cfg.timing_fingerprint_weight * (
                    0.05 + peaks
                ).log().unsqueeze(-1)
        if use_context:
            event_logits = temporal_mean(event_logits, cfg.boundary_context_frames)
        local_peak = cfg.timing_peak_radius_bins and need_physical and not self.training
        ordered_peak = (
            cfg.timing_min_separation_bins and need_physical and not self.training
        )
        joint_roles = role_logits is not None and not self.training
        if joint_roles:
            from event_association import decode_interface_roles

            # Echo amplitude is supporting evidence, since strong clutter peaks
            # alone cannot determine the identity of an interface.
            peaks = tof.clamp_min(0)
            peaks = peaks / peaks.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            radius = cfg.interface_role_evidence_radius_bins
            if radius:
                peaks = F.max_pool1d(
                    peaks.flatten(0, 1).unsqueeze(1), 2 * radius + 1, 1, radius
                ).reshape(B, T, N)
            evidence = role_logits + cfg.interface_role_tof_weight * (
                0.1 + peaks
            ).log().unsqueeze(-1)
            if cfg.interface_role_phase_weight:
                p = phase.flatten(0, 1).unsqueeze(1).abs()
                center = F.avg_pool1d(p, 3, 1, 1, count_include_pad=False)
                background = F.avg_pool1d(p, 13, 1, 6, count_include_pad=False)
                prominence = ((center - background) / (0.1 + background)).reshape(
                    B, T, N
                )
                evidence = (
                    evidence
                    + cfg.interface_role_phase_weight * prominence.unsqueeze(-1)
                )
            prior = count_logits.log_softmax(-1) * cfg.interface_role_count_prior
            if use_context:
                evidence = temporal_mean(evidence, cfg.boundary_context_frames)
            decoded_count, chosen = decode_interface_roles(
                evidence,
                prior,
                cfg.timing_min_separation_bins or 1,
                minimum_layers,
                cfg.interface_role_penalty,
            )
            out["layer_count"] = decoded_count
            physical_count = decoded_count.to(b.dtype) + 1
            physical_mass = b * (physical_count / count.clamp_min(1e-8)).unsqueeze(-1)
            boundary_times = fine_time.gather(-1, chosen)
        elif ordered_peak:
            from event_association import ordered_interface_bins

            # Keep the learned ordinal score and favor actual local delay peaks.
            observed = F.max_pool1d(tof.reshape(B * T, 1, N), 3, 1, 1).reshape(B, T, N)
            supported = (tof == observed) & (tof > 0)
            score = event_logits + supported.unsqueeze(-1).to(event_logits.dtype)
            peaks_only = None
            if cfg.timing_observed_peaks_only:
                left = F.pad(tof[..., :-1], (1, 0), value=-torch.inf)
                right = F.pad(tof[..., 1:], (0, 1), value=-torch.inf)
                peaks_only = (tof > left) & (tof >= right) & (tof > 0)
            chosen = ordered_interface_bins(
                score, out["layer_count"], cfg.timing_min_separation_bins, peaks_only
            )
            boundary_times = fine_time.gather(-1, chosen)
        elif local_peak:
            # Use a local posterior centroid at inference while retaining the
            # differentiable soft association throughout training.
            center = event_logits.argmax(dim=2, keepdim=True)
            bins = torch.arange(N, device=event_logits.device)[None, None, :, None]
            near = (bins - center).abs() <= cfg.timing_peak_radius_bins
            event_logits = event_logits.masked_fill(~near, -torch.inf)
        if not (ordered_peak or joint_roles):
            event_weights = event_logits.softmax(dim=2)
            boundary_times = (event_weights * time.unsqueeze(-1)).sum(dim=2)
            if local_peak:
                boundary_times = boundary_times.cummax(dim=-1).values
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
        if calibrated_path is not None:
            path_times, path_count = calibrated_path
            used = (path_count > 0)[:, None].expand(-1, T)
            boundary_times = torch.where(used.unsqueeze(-1), path_times, boundary_times)
            out["layer_count"] = torch.where(
                used, path_count[:, None], out["layer_count"]
            )
            out["echo_path_used"] = used
        tau_layer = (boundary_times[..., 1:] - boundary_times[..., :-1]).clamp_min(0)
        # q counts the air/ice surface. Physical ice slot l corresponds to
        # full-coordinate center l+1. M_mine above retains specified formulation.
        ice_coordinate = q - 1 if physical_mass is b else physical_mass.cumsum(-1)
        ice_M = soft_layer_assignment(
            ice_coordinate, cfg.max_layers, cfg.assignment_temperature
        )
        index = torch.arange(cfg.max_layers, device=b.device, dtype=b.dtype)
        # Continuous clipped ramp: integer L+1 boundary count gives exactly
        # L present layers, without a 10% profile shrink at small L.
        presence = (
            (physical_count.unsqueeze(-1) - (index + 1.5)) / cfg.count_temperature + 0.5
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
        if joint_roles and cfg.interface_role_interval_pooling:
            edges = boundary_times.unsqueeze(2)
            cumulative = torch.sigmoid(
                (time.unsqueeze(-1) - edges) / dt[..., None, None]
            )
            M = (cumulative[..., :-1] - cumulative[..., 1:]) * presence.unsqueeze(2)
        else:
            M = ice_M * roi.unsqueeze(-1) * presence.unsqueeze(2)
        if calibrated_path is not None:
            axis = time.unsqueeze(-1)
            inside = (axis >= boundary_times[..., :-1].unsqueeze(2)) & (
                axis < boundary_times[..., 1:].unsqueeze(2)
            )
            M = torch.where(used[..., None, None], inside.to(M.dtype), M)
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
                "tof_node_logits": timing_logits,
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
        if self.training:
            out["event_association_logits"] = event_logits
        if getattr(self, "direct_physical_head", None) is not None:
            # Explicit CNN ablation only. The full SPAI never takes this path.
            rho, thickness = self.direct_physical_head(rho_tilde, u)
            rho0 = rho
            h0 = thickness
            velocity = cfg.c0_m_s / (1 + cfg.dielectric_density_coefficient * rho)
            v0 = velocity
        else:
            rho0, v0, h0, rho, velocity, thickness = self.invert_layer_delays(
                rho_tilde, u, tau_layer
            )
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
