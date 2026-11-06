"""Inference-only boundary peak decoding and density-continuity merging.
No label, filename, sample index or ground-truth layer count is accepted.
This is an added decoder, not a change to the reference method's neural block widths.
"""

import numpy as np
import torch


def candidate_peaks(prob, min_height=0.03, min_distance=8):
    candidates = np.flatnonzero((prob[1:-1] >= prob[:-2]) & (prob[1:-1] > prob[2:])) + 1
    candidates = sorted(candidates, key=lambda i: prob[i], reverse=True)
    selected = []
    for index in candidates:
        if prob[index] >= min_height and all(
            abs(index - j) >= min_distance for j in selected
        ):
            selected.append(int(index))
    return sorted(selected)


@torch.no_grad()
def interval_parameters(core, density_features, delay_features, boundaries, time_s):
    if len(boundaries) < 2:
        return np.array([]), np.array([])
    r = torch.stack(
        [density_features[a:b].mean(0) for a, b in zip(boundaries[:-1], boundaries[1:])]
    )
    u = torch.stack(
        [delay_features[a:b].mean(0) for a, b in zip(boundaries[:-1], boundaries[1:])]
    )
    pre = core.g_rho(r).squeeze(-1)
    rho0 = core.g_cal(torch.cat((pre[:, None], u), -1)).squeeze(-1)
    tau = torch.stack(
        [time_s[b] - time_s[a] for a, b in zip(boundaries[:-1], boundaries[1:])]
    )
    h0 = (
        core.config.c0_m_s
        * tau
        / (2 * (1 + core.config.dielectric_density_coefficient * rho0))
    )
    rho = core.g_ref(torch.cat((pre[:, None], u, h0[:, None]), -1)).squeeze(-1)
    h = (
        core.config.c0_m_s
        * tau
        / (2 * (1 + core.config.dielectric_density_coefficient * rho))
    )
    return h.cpu().numpy(), rho.cpu().numpy()


def decode(core, out, time_s, merge_delta=0.02, min_height=0.03):
    B, T, N = out["boundary_prob"].shape
    decoded = []
    for b in range(B):
        sample = []
        for t in range(T):
            boundaries = candidate_peaks(
                out["boundary_prob"][b, t].detach().cpu().numpy(), min_height
            )
            if len(boundaries) > core.config.max_layers + 1:
                prob = out["boundary_prob"][b, t].detach().cpu().numpy()
                boundaries = sorted(
                    sorted(boundaries, key=lambda i: prob[i], reverse=True)[
                        : core.config.max_layers + 1
                    ]
                )
            density = out["density_evidence"][b, t]
            delay = out["delay_sample_features"][b, t]
            times = time_s[b] if time_s.ndim == 2 else time_s[b, t]
            h, rho = interval_parameters(core, density, delay, boundaries, times)
            while len(rho) > 1:
                delta = np.abs(np.diff(rho))
                i = int(np.argmin(delta))
                if delta[i] >= merge_delta:
                    break
                boundaries.pop(i + 1)
                h, rho = interval_parameters(core, density, delay, boundaries, times)
            sample.append((h, rho, boundaries))
        decoded.append(sample)
    return decoded
