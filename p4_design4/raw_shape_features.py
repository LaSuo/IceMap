"""Optional additional head evidence; no change to main backbone dimensions.

Per-frame gain-normalized shape descriptors retain boundary timing while the
unscaled summary statistics retain amplitude information. All operations use
only the three model inputs, never physical labels, filenames or test IDs.
"""

import torch
from torch.nn import functional as F


def raw_shape_features(inputs, bins=64):
    vectors = []
    for name in ["phase", "scatter", "tof"]:
        x = inputs[name].float()
        B, T, N = x.shape
        rms = x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        y = x / rms
        v = y.reshape(B * T, 1, N)
        difference = F.pad(y.diff(dim=-1).abs(), (0, 1)).reshape(B * T, 1, N)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool1d(v, bins).flatten(1),
                F.adaptive_max_pool1d(v, bins).flatten(1),
                F.adaptive_avg_pool1d(difference, bins).flatten(1),
            ],
            -1,
        ).reshape(B, T, -1)
        stats = torch.stack(
            [
                rms.squeeze(-1),
                x.mean(-1),
                x.std(-1, unbiased=False),
                x.amax(-1),
                x.amin(-1),
                x.abs().mean(-1),
                (x.abs() > 0.01 * rms).float().mean(-1),
                torch.log1p(x.abs().amax(-1)),
            ],
            -1,
        )
        vectors.extend([pooled, stats])
    return torch.cat(vectors, -1)
