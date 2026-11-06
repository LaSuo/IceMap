"""Depth-monotone matching, LER, LE_h/rho, AE_h/rho.

The reference specification omits matching tolerance and unmatched-layer AE convention. Here
matching maximizes match count then total overlap, preserving depth order.
Unmatched layers count as misses/spurious layers and incur full AE mass.
These explicit choices must be held fixed when comparing experiments.
"""

import numpy as np
from protocol import METRICS


def monotone_match(h_pred, h_gt, min_iou=0.0):
    if not 0 <= min_iou < 1:
        raise ValueError("min_iou must be in [0,1).")
    ep = np.r_[0.0, np.cumsum(h_pred)]
    eg = np.r_[0.0, np.cumsum(h_gt)]
    P, G = len(h_pred), len(h_gt)
    scores = [[(0, 0.0) for _ in range(G + 1)] for _ in range(P + 1)]
    actions = {}
    for p in range(P + 1):
        for g in range(G + 1):
            if p == g == 0:
                continue
            options = []
            if p:
                options.append((scores[p - 1][g], "spurious"))
            if g:
                options.append((scores[p][g - 1], "missed"))
            if p and g:
                overlap = max(0.0, min(ep[p], eg[g]) - max(ep[p - 1], eg[g - 1]))
                union = max(ep[p], eg[g]) - min(ep[p - 1], eg[g - 1])
                iou = overlap / max(union, 1e-15)
                if overlap > 0 and iou >= min_iou:
                    n, value = scores[p - 1][g - 1]
                    options.append(((n + 1, value + iou), "match"))
            scores[p][g], actions[p, g] = max(options, key=lambda x: x[0])
    matches, spurious, missed = [], [], []
    p, g = P, G
    while p or g:
        action = actions[p, g]
        if action == "match":
            p -= 1
            g -= 1
            matches.append((p, g))
        elif action == "spurious":
            p -= 1
            spurious.append(p)
        else:
            g -= 1
            missed.append(g)
    return matches[::-1], spurious[::-1], missed[::-1]


class MineMetrics:
    def __init__(self, min_iou=0.0):
        if not isinstance(min_iou, (int, float)) or not 0 <= min_iou < 1:
            raise ValueError("min_iou must be in [0,1).")
        self.min_iou = min_iou
        self.gt_count = self.spurious_count = self.missed_count = 0
        self.le_h, self.le_rho, self.ae_h, self.ae_rho = [], [], [], []

    def update(self, h_pred, rho_pred, h_gt, rho_gt):
        hp, rp, hg, rg = (
            np.asarray(x, dtype=float) for x in (h_pred, rho_pred, h_gt, rho_gt)
        )
        if any(x.ndim != 1 for x in (hp, rp, hg, rg)):
            raise ValueError("Physical layers must be one-dimensional vectors.")
        if len(hp) != len(rp) or len(hg) != len(rg) or not len(hg):
            raise ValueError(
                "Layer arrays must agree and ground truth must be nonempty."
            )
        if (
            any(x.ndim != 1 or not np.isfinite(x).all() for x in (hp, rp, hg, rg))
            or np.any(hp < 0)
            or np.any(rp <= 0)
            or np.any(hg <= 0)
            or np.any(rg <= 0)
        ):
            raise ValueError("Invalid physical layer values.")
        matches, spurious, missed = monotone_match(hp, hg, self.min_iou)
        self.gt_count += len(hg)
        self.spurious_count += len(spurious)
        self.missed_count += len(missed)
        err_h = err_rho = 0.0
        for p, g in matches:
            self.le_h.append(abs(hp[p] - hg[g]) / hg[g])
            self.le_rho.append(abs(rp[p] - rg[g]) / rg[g])
            err_h += abs(hp[p] - hg[g])
            err_rho += hg[g] * abs(rp[p] - rg[g])
        err_h += hg[missed].sum() + hp[spurious].sum()
        err_rho += (hg[missed] * rg[missed]).sum() + (hp[spurious] * rp[spurious]).sum()
        self.ae_h.append(err_h / hg.sum())
        self.ae_rho.append(err_rho / (hg * rg).sum())

    def summary(self):
        mean = lambda values: float(np.mean(values) * 100) if values else None
        return {
            "LER_percent": (
                100 * (self.spurious_count + self.missed_count) / self.gt_count
                if self.gt_count
                else None
            ),
            "LE_h_percent": mean(self.le_h),
            "LE_rho_percent": mean(self.le_rho),
            "AE_h_percent": mean(self.ae_h),
            "AE_rho_percent": mean(self.ae_rho),
            "gt_layers": self.gt_count,
            "matched_layers": len(self.le_h),
            "spurious_layers": self.spurious_count,
            "missed_layers": self.missed_count,
            "matching_min_iou": self.min_iou,
            "metrics_version": METRICS,
            "aggregation": {
                "LER": "micro over true layers",
                "LE": "mean over matched layers",
                "AE": "macro over frames; full unmatched-layer mass",
            },
        }
