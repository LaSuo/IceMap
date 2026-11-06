"""Calibrated SPAI and the disclosed supplementary training losses.

The six main losses remain in losses.py; additions are configurable in
training_config.py. The runnable training entry is train.py.
"""

import torch
from torch import nn
from torch.nn import functional as F
from network import SPAIIceInversionNet, ModelConfig
from model_catalog import build_network, ArchitectureConfig, ALIASES
from dataclasses import asdict

TARGET = {
    "LER_percent": 2.18,
    "LE_h_percent": 4.39,
    "LE_rho_percent": 3.69,
    "AE_h_percent": 4.26,
    "AE_rho_percent": 3.29,
}


class CalibratedSPAI(nn.Module):
    def __init__(self, config=None, variant="mine", architecture_config=None):
        super().__init__()
        self.variant_name = ALIASES.get(variant, variant)
        research = (
            architecture_config
            if isinstance(architecture_config, ArchitectureConfig)
            else ArchitectureConfig(**(architecture_config or {}))
        ).validate()
        self.architecture_config = asdict(research)
        self.core = build_network(config or ModelConfig(), self.variant_name, research)
        self.input_log_gain = nn.Parameter(torch.zeros(3))
        self.input_bias = nn.Parameter(torch.zeros(3))

    def forward(self, batch, stage=3):
        inputs = [batch[k] for k in ["tof", "phase", "scatter"]]
        inputs = [
            (x - self.input_bias[i]) * self.input_log_gain[i].exp()
            for i, x in enumerate(inputs)
        ]
        extra = {"context": batch["context"]} if "context" in batch else {}
        return self.core(
            tof=inputs[0],
            phase=inputs[1],
            scatter=inputs[2],
            time_s=batch["time_s"],
            depth_m=batch["depth_m"],
            scatter_valid=batch["scatter_valid"],
            need_layer=stage >= 2,
            need_physical=stage >= 3,
            **extra,
        )


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def auxiliary_losses(out, batch, stage, args):
    target = batch["boundary_target"]
    counts = batch["layer_count"]
    # Per-frame class balance; the reference specification focal/N and count losses stay intact.
    posweight = (
        (target.shape[-1] - target.sum(-1))
        .div(target.sum(-1).clamp_min(1))
        .unsqueeze(-1)
    )
    balanced = -(
        posweight * target * F.logsigmoid(out["boundary_logits"])
        + (1 - target) * F.logsigmoid(-out["boundary_logits"])
    ).mean()
    gaussian = F.conv1d(
        target.reshape(-1, 1, target.shape[-1]), args.kernel, padding=args.sigma * 3
    ).reshape_as(target)
    mass = gaussian / gaussian.sum(-1, keepdim=True).clamp_min(1e-8)
    location = -(mass * F.log_softmax(out["boundary_logits"], dim=-1)).sum(-1).mean()
    terms = {"balanced_boundary": balanced, "boundary_location": location}
    total = args.aux_bce * balanced + args.aux_location * location
    mask = torch.arange(10, device=target.device) < counts.unsqueeze(-1)

    def meanlayers(x):
        return ((x * mask).sum(-1) / counts).mean()

    if stage >= 2 and args.aux_time > 0:
        N = target.shape[-1]
        bins = torch.arange(N, device=target.device).view(1, 1, -1).expand_as(target)
        marked = torch.where(target > 0.5, bins, torch.full_like(bins, N))
        positions = torch.topk(
            marked, 11, dim=-1, largest=False, sorted=True
        ).values.clamp_max(N - 1)
        times = torch.gather(
            batch["time_s"].unsqueeze(1).expand(-1, target.shape[1], -1), 2, positions
        )
        valid = torch.arange(11, device=target.device) < (counts + 1).unsqueeze(-1)
        error = (out["boundary_times_s"] - times).abs() * 299792458 / 0.002
        timing = (error * valid).sum() / valid.sum() / 10
        terms["boundary_time_bins_div10"] = timing
        total = total + args.aux_time * timing
    if stage >= 2:
        pre = meanlayers(
            (out["rho_tilde"] - batch["relative_density_target"]).abs()
            / batch["relative_density_target"].clamp_min(0.1)
        )
        terms["pre_density_relative"] = pre
        total = total + args.aux_density * pre
    if stage >= 3:
        h = meanlayers(
            (out["layer_thickness_m"] - batch["layer_thickness_m_gt"]).abs()
            / batch["layer_thickness_m_gt"].clamp_min(0.01)
        )
        rho = meanlayers(
            (out["layer_density"] - batch["layer_density_gt"]).abs()
            / batch["layer_density_gt"].clamp_min(0.1)
        )
        terms.update(thickness_relative=h, density_relative=rho)
        total = total + args.aux_thickness * h + args.aux_density * rho
    return total, terms


if __name__ == "__main__":
    from train import main

    main()
