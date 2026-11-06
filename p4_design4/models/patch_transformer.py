import math
import torch
from torch import nn
from torch.nn import functional as F
from network import ConvBlock, EntropyReliabilityRefinement


class TransformerSkeleton(nn.Module):
    def __init__(self, cfg, options):
        super().__init__()
        self.options = options
        width = 2 * cfg.gru_hidden
        if width % options.transformer_heads:
            raise ValueError("Skeleton width must divide attention heads.")
        self.stem = ConvBlock(1, width, 7)
        layer = nn.TransformerEncoderLayer(
            width,
            options.transformer_heads,
            width * options.transformer_ff_multiplier,
            options.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, options.transformer_depth, enable_nested_tensor=False
        )
        self.erf = (
            EntropyReliabilityRefinement(width, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )
        self.head = nn.Linear(width, 1)

    def forward(self, phase):
        B, T, N = phase.shape
        x = self.stem(phase.reshape(B * T, 1, N))
        tokens = min(
            self.options.transformer_max_tokens,
            max(1, math.ceil(N / self.options.transformer_patch_bins)),
        )
        x = F.adaptive_avg_pool1d(x, tokens).transpose(1, 2)
        rate = torch.exp(
            torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / x.shape[-1])
        )
        angle = (
            torch.arange(tokens, device=x.device, dtype=x.dtype)[:, None] * rate[None]
        )
        position = torch.zeros_like(x[0])
        position[:, 0::2] = angle.sin()
        position[:, 1::2] = angle[:, : position[:, 1::2].shape[-1]].cos()
        x = self.encoder(x + position).transpose(1, 2)
        z = (
            F.interpolate(x, size=N, mode="linear", align_corners=False)
            .transpose(1, 2)
            .reshape(B, T, N, -1)
        )
        diagnostic = {}
        if self.erf is not None:
            z, diagnostic = self.erf(z)
        return self.head(z).squeeze(-1).float(), z, diagnostic
