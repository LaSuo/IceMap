"""Monotone one-to-one association of learned interfaces and observed D3 peaks."""

import math
import torch


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
