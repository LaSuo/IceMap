"""Input-only descriptors for the optional quality calibration expert.

These spectra describe the supplied fingerprint sequences, not additional
radar-frequency channels or the reference method's unavailable spectral ablation input.
"""

import torch
from torch.nn import functional as F


def build_quality_features(inputs, backbone_features, config):
    if config.feature_mode == "backbone":
        return backbone_features
    if config.feature_mode != "peak_spectral":
        raise ValueError("Unknown quality feature mode")
    x = {k: inputs[k].float() for k in ["phase", "scatter", "tof"]}
    B, T, N = x["tof"].shape
    M = B * T
    vectors = []
    for name in ["phase", "scatter", "tof"]:
        v = x[name]
        centered = v - v.mean(-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1).abs()[..., 1:129]
        spectrum = spectrum / spectrum.square().mean(-1, keepdim=True).sqrt().clamp_min(
            1e-8
        )
        spectrum = F.pad(spectrum, (0, max(0, 128 - spectrum.shape[-1])))
        stats = torch.stack(
            [
                v.mean(-1),
                v.std(-1, unbiased=False),
                v.square().mean(-1).sqrt(),
                v.abs().amax(-1),
                torch.quantile(v, 0.2, dim=-1),
                torch.quantile(v, 0.5, dim=-1),
                torch.quantile(v, 0.8, dim=-1),
                (v != 0).float().mean(-1),
            ],
            -1,
        )
        vectors.extend([spectrum, stats])
    tau = x["tof"].reshape(M, 1, N)
    axis = torch.arange(-9, 10, device=tau.device).float()
    kernel = torch.exp(-axis.square() / 18)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    smooth = F.conv1d(tau, kernel, padding=9).squeeze(1)
    smooth = (smooth - torch.quantile(smooth, 0.2, dim=-1, keepdim=True)).clamp_min(0)
    normalized = smooth / smooth.amax(-1, keepdim=True).clamp_min(1e-8)
    local = F.max_pool1d(normalized[:, None], 31, stride=1, padding=15).squeeze(1)
    bins = torch.arange(N, device=tau.device)[None]
    valid = (normalized >= local) & (normalized > 0.05) & (bins >= 5) & (bins < N - 5)
    heights = torch.where(valid, normalized, torch.full_like(normalized, -1))
    top = torch.topk(heights, min(12, N), dim=-1)
    selected = torch.where(top.values > 0, top.indices, torch.full_like(top.indices, N))
    positions = selected.sort(-1).values
    present = positions < N
    positions = positions.clamp_max(N - 1)
    amplitude = normalized.gather(-1, positions) * present
    relative = (positions - positions[:, :1]).float() * 0.001
    scatter = x["scatter"].reshape(M, N)
    phase = x["phase"].reshape(M, N)
    scatter = scatter / scatter.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    phase = (phase - phase.mean(-1, keepdim=True)) / phase.std(
        -1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    offset = torch.arange(-40, 41, device=tau.device)
    window = (positions[..., None] + offset).clamp(0, N - 1)
    nearby = normalized.gather(1, window.flatten(1)).reshape(M, positions.shape[-1], -1)
    width = (nearby > amplitude[..., None] * 0.5).float().sum(-1) * 0.001
    ev = scatter.gather(-1, positions)
    pv = phase.gather(-1, positions)
    sc = (
        F.avg_pool1d(scatter[:, None], 11, stride=1, padding=5)
        .squeeze(1)
        .gather(-1, positions)
    )
    peaks = (
        torch.stack(
            [
                present.float(),
                positions.float() * 0.001,
                relative,
                amplitude,
                width,
                ev,
                pv,
                sc,
            ],
            -1,
        )
        * present[..., None]
    )
    peaks = F.pad(peaks.flatten(1), (0, max(0, 96 - peaks.shape[1] * 8))).reshape(
        B, T, 96
    )
    vectors.append(peaks)
    return torch.cat(vectors, -1)
