from dataclasses import replace
import torch
from torch import nn
from network import EntropyReliabilityRefinement
from .tcn import TCNSkeleton
from .patch_transformer import TransformerSkeleton


class GatedSkeletonFusion(nn.Module):
    """Parallel local/global encoders with sample-wise learned softmax fusion."""

    def __init__(self, cfg, options):
        super().__init__()
        local = replace(cfg, use_erf=False)
        width = 2 * cfg.gru_hidden
        self.local = TCNSkeleton(local, options)
        self.global_path = TransformerSkeleton(local, options)
        self.gate = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 2)
        )
        self.head = nn.Linear(width, 1)
        self.erf = (
            EntropyReliabilityRefinement(width, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )

    def forward(self, phase):
        _, a, _ = self.local(phase)
        _, b, _ = self.global_path(phase)
        weights = self.gate(torch.cat([a, b], -1)).softmax(-1)
        z = weights[..., 0, None] * a + weights[..., 1, None] * b
        info = {"local_global_weights": weights}
        if self.erf is not None:
            z, entropy = self.erf(z)
            info.update(entropy)
        return self.head(z).squeeze(-1).float(), z, info
