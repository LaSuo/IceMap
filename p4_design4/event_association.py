"""Monotone one-to-one association of learned interfaces and observed D3 peaks."""

import math
import torch
from torch.nn import functional as F


def ordered_interface_bins(logits, layer_count, min_gap, allowed=None):
    """Maximize the ordinal timing score over a distinct ordered peak path.

    The lattice uses observed time bins. Padding slots repeat the last selected
    interface so they cannot contribute a nonzero physical delay. With a peak
    mask, use an unrestricted path only when the peaks cannot support the count.
    """
    batch, frames, samples, events = logits.shape
    if min_gap < 1 or samples < (events - 1) * min_gap + 1:
        raise ValueError("Ordered interface spacing exceeds the timing grid.")
    scores = logits.reshape(-1, samples, events)
    counts = layer_count.reshape(-1).long()
    if allowed is not None:
        constrained = scores.masked_fill(~allowed.reshape(-1, samples, 1), -torch.inf)
        scores = torch.cat((scores, constrained), dim=0)
        counts = counts.repeat(2)
    best = scores[..., 0]
    previous = []
    value, end = best.max(-1)
    endpoints = [end]
    best_values = [value]
    for rank in range(1, events):
        prefix, pointers = best.cummax(-1)
        best = F.pad(prefix[..., :-min_gap], (min_gap, 0), value=-torch.inf)
        best = best + scores[..., rank]
        previous.append(F.pad(pointers[..., :-min_gap], (min_gap, 0)))
        value, end = best.max(-1)
        endpoints.append(end)
        best_values.append(value)
    point = torch.stack(endpoints, -1).gather(1, counts[:, None]).squeeze(1)
    selected = point[:, None].expand(-1, events).clone()
    for rank in range(events - 1, -1, -1):
        active = counts >= rank
        selected[:, rank] = torch.where(active, point, selected[:, rank])
        if rank:
            parent = previous[rank - 1].gather(1, point[:, None]).squeeze(1)
            point = torch.where(active, parent, point)
    if allowed is not None:
        valid = torch.stack(best_values, -1).gather(1, counts[:, None]).isfinite()
        unrestricted, constrained = selected.chunk(2)
        selected = torch.where(valid.chunk(2)[1], constrained, unrestricted)
    return selected.reshape(batch, frames, events)


def decode_interface_roles(role_logits, count_prior, min_gap, minimum_layers, penalty):
    """Jointly select a surface, zero or more inner interfaces, and a bottom.

    Each score is a local log odds against clutter. Each added inner interface
    must improve the best complete path enough to overcome the count penalty.
    """
    batch, frames, samples, _ = role_logits.shape
    n_counts = count_prior.shape[-1]
    if min_gap < 1 or samples < min_gap * minimum_layers + 1:
        raise ValueError("Minimum layer count and spacing exceed the timing grid.")
    feasible = min(n_counts - 1, (samples - 1) // min_gap)
    surface, inner, bottom = role_logits.reshape(-1, samples, 3).unbind(-1)
    prior = count_prior.reshape(-1, n_counts)

    best = surface
    value, end = best.max(-1)
    endpoints, path_scores = [end], [value]
    parents = []
    for count in range(1, feasible + 1):
        prefix, parent = best.cummax(-1)
        prefix = F.pad(prefix[..., :-min_gap], (min_gap, 0), value=-torch.inf)
        parents.append(F.pad(parent[..., :-min_gap], (min_gap, 0)))
        value, end = (prefix + bottom).max(-1)
        endpoints.append(end)
        path_scores.append(value - penalty * (count - 1))
        best = prefix + inner

    scores = torch.stack(path_scores, -1) + prior[:, : feasible + 1]
    count = scores[..., minimum_layers:].argmax(-1) + minimum_layers
    point = torch.stack(endpoints, -1).gather(1, count[:, None]).squeeze(1)
    chosen = point[:, None].expand(-1, n_counts).clone()
    for rank in range(feasible, -1, -1):
        active = count >= rank
        chosen[:, rank] = torch.where(active, point, chosen[:, rank])
        if rank:
            parent = parents[rank - 1].gather(1, point[:, None]).squeeze(1)
            point = torch.where(active, parent, point)
    return count.reshape(batch, frames), chosen.reshape(batch, frames, n_counts)


def associate_events(predicted_s, layer_count, observed_s, valid, tolerance_s):
    """Return associated interface times; missing/ambiguous evidence is an error.

    Inference-only alternative: no labels are accessed, no missing event is
    invented. Dynamic programming minimizes squared timing error in depth order.
    """
    if (
        not isinstance(tolerance_s, (int, float))
        or not math.isfinite(tolerance_s)
        or tolerance_s <= 0
    ):
        raise ValueError(
            "Supply positive finite event association tolerance, in seconds."
        )
    if (
        predicted_s.ndim != 3
        or observed_s.ndim != 3
        or predicted_s.shape[:2] != observed_s.shape[:2]
        or valid.shape != observed_s.shape
        or layer_count.shape != predicted_s.shape[:2]
    ):
        raise ValueError("Event association shape mismatch.")
    if (
        valid.dtype != torch.bool
        or not torch.isfinite(observed_s).all()
        or not torch.isfinite(predicted_s).all()
    ):
        raise ValueError("Finite events and a Boolean support mask are required.")
    if (
        (layer_count < 0).any()
        or (layer_count >= predicted_s.shape[-1]).any()
        or not torch.equal(layer_count, layer_count.round())
    ):
        raise ValueError("Invalid predicted layer counts for event association.")
    result = predicted_s.clone()
    associated = torch.zeros_like(layer_count, dtype=torch.bool)
    for b in range(predicted_s.shape[0]):
        for t in range(predicted_s.shape[1]):
            layers = int(layer_count[b, t])
            needed = layers + 1
            if layers == 0:
                continue
            obs = observed_s[b, t][valid[b, t]]
            if obs.numel() < needed or (obs.diff() <= 0).any():
                raise ValueError(
                    f"Frame {b},{t}: insufficient or unordered observed D3 peaks for {layers} predicted layers."
                )
            target = predicted_s[b, t, :needed].detach()
            costs = (
                ((target[:, None] - obs[None, :]) / tolerance_s).square().cpu().tolist()
            )
            M = obs.numel()
            dp = [[float("inf")] * (M + 1) for _ in range(needed + 1)]
            take = {}
            dp[0] = [0.0] * (M + 1)
            for i in range(1, needed + 1):
                for j in range(1, M + 1):
                    dp[i][j] = dp[i][j - 1]
                    candidate = dp[i - 1][j - 1] + costs[i - 1][j - 1]
                    if costs[i - 1][j - 1] <= 1.0 and candidate < dp[i][j]:
                        dp[i][j] = candidate
                        take[i, j] = True
            if not math.isfinite(dp[needed][M]):
                raise ValueError(
                    f"Frame {b},{t}: no monotone peak association inside tolerance."
                )
            indices = []
            i = needed
            j = M
            while i:
                if take.get((i, j), False):
                    indices.append(j - 1)
                    i -= 1
                j -= 1
            matched = obs[indices[::-1]].to(result)
            result[b, t, :needed] = matched
            result[b, t, needed:] = matched[-1]
            associated[b, t] = True
    return result, associated
