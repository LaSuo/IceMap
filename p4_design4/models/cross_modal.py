import math
import torch
from torch import nn
from torch.nn import functional as F


class CrossModalAdapter(nn.Module):
    """Patch-wise attention across three modality tokens, bounded residuals.

    Attention is across modalities, not all raw depth-bin pairs. This bounds
    memory while retaining a separate residual for each source stream.
    """

    def __init__(self, options):
        super().__init__()
        self.options = options
        w = options.crossmodal_width
        self.stems = nn.ModuleList([nn.Conv1d(1, w, 5, padding=2) for _ in range(3)])
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, w) * 0.02)
        layer = nn.TransformerEncoderLayer(
            w,
            options.crossmodal_heads,
            w * options.expansion,
            options.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            layer, options.crossmodal_depth, enable_nested_tensor=False
        )
        self.output = nn.Linear(w, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, phase, scatter, tof):
        B, T, N = phase.shape
        n = max(1, math.ceil(N / self.options.crossmodal_patch_bins))
        arrays = [phase, scatter, tof]
        encoded = []
        for stem, x in zip(self.stems, arrays):
            feature = stem(x.reshape(B * T, 1, N))
            encoded.append(F.adaptive_avg_pool1d(feature, n).transpose(1, 2))
        tokens = (
            torch.stack(encoded, 2).reshape(B * T * n, 3, -1) + self.modality_embedding
        )
        residual = (
            self.output(self.attention(tokens))
            .squeeze(-1)
            .reshape(B * T, n, 3)
            .transpose(1, 2)
        )
        residual = (
            F.interpolate(residual, size=N, mode="linear", align_corners=False)
            .reshape(B, T, 3, N)
            .tanh()
        )
        result = []
        for i, x in enumerate(arrays):
            scale = x.abs().mean(-1, keepdim=True).clamp_min(1e-6)
            result.append(
                x + self.options.fusion_residual_limit * scale * residual[:, :, i]
            )
        return result
