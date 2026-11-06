import torch
from torch import nn
from torch.nn import functional as F
from network import SPAIIceInversionNet
from .parameter_matching import matched_plain_cnn


class DirectFusionCNN(nn.Module):
    """Parameter-matched direct fingerprint fusion with independent h/rho heads.

    This is an explicitly specified control, not the original undisclosed CNN
    architecture. Delays are derived diagnostics; thickness is directly learned.
    """

    def __init__(self, cfg, target_parameters):
        super().__init__()
        self.config = cfg
        C = cfg.cnn_channels
        K = cfg.max_layers
        self.encoder, self.parameter_match = matched_plain_cnn(
            3, C, target_parameters, fixed_parameters=(C + 1) * (4 * K + 2)
        )
        self.boundary = nn.Linear(C, 1)
        self.count = nn.Linear(C, K + 1)
        self.relative = nn.Linear(C, K)
        self.density = nn.Linear(C, K)
        self.thickness = nn.Linear(C, K)

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
        **options,
    ):
        if options:
            raise ValueError("Direct fusion does not use observed-event reassociation.")
        if (
            phase.ndim != 3
            or phase.shape != tof.shape
            or scatter.shape != phase.shape
            or phase.shape[-1] < 8
        ):
            raise ValueError("Fingerprints must share [B,T,N], N>=8.")
        if any(
            not x.is_floating_point()
            or not torch.isfinite(x).all()
            or x.device != phase.device
            or x.dtype != phase.dtype
            for x in (phase, tof, scatter)
        ):
            raise ValueError(
                "Fingerprint dtypes/devices/values must agree and be finite floats."
            )
        B, T, N = phase.shape
        cfg = self.config
        K = cfg.max_layers
        if time_s is None:
            raise ValueError("Calibrated time_s is required.")
        SPAIIceInversionNet._axis(time_s, B, T, N, phase.device, phase.dtype)
        if (
            scatter_valid is None
            or scatter_valid.shape != phase.shape
            or scatter_valid.dtype != torch.bool
        ):
            raise ValueError("A Boolean scatter_valid mask is required.")
        x = torch.stack((phase, scatter * scatter_valid, tof), dim=2).reshape(
            B * T, 3, N
        )
        z = self.encoder(x).transpose(1, 2).reshape(B, T, N, -1)
        pooled = z.mean(2)
        logits = self.boundary(z).squeeze(-1)
        prob = self.count(pooled).softmax(-1)
        ranks = torch.arange(K + 1, device=phase.device, dtype=phase.dtype)
        presence = prob[..., 1:].flip(-1).cumsum(-1).flip(-1)
        count = prob.argmax(-1)
        if not self.training:
            presence = (torch.arange(K, device=phase.device) < count[..., None]).to(
                phase
            )
        rho = F.softplus(self.density(pooled))
        h = F.softplus(self.thickness(pooled))
        edges = F.pad((h * presence).cumsum(-1), (1, 0))
        output = {
            "boundary_logits": logits,
            "boundary_prob": logits.sigmoid(),
            "boundary_count": 1 + (prob * ranks).sum(-1),
            "layer_count": count,
            "rho_tilde": F.softplus(self.relative(pooled)),
            "layer_density": rho,
            "layer_thickness_m": h,
            "layer_presence": presence,
            "depth_edges_m": edges,
            "layer_delays_s": 2
            * h
            * (1 + cfg.dielectric_density_coefficient * rho)
            / cfg.c0_m_s,
        }
        if depth_m is not None:
            depth = SPAIIceInversionNet._axis(
                depth_m, B, T, depth_m.shape[-1], phase.device, phase.dtype
            )
            profile = torch.sigmoid(
                (depth[..., None] - edges[..., None, :-1]) / cfg.profile_temperature_m
            ) - torch.sigmoid(
                (depth[..., None] - edges[..., None, 1:]) / cfg.profile_temperature_m
            )
            output["physical_profile"] = (profile * rho[..., None, :]).sum(-1)
        return output
