"""Ordered one- and two-layer echo paths for calibrated observation windows."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import torch


@dataclass(frozen=True)
class EchoPathConfig:
    native_step_s: float
    surface_bin: float
    surface_scale_bins: float
    minimum_gap_bins: int
    maximum_single_span_bins: float
    maximum_double_span_bins: float
    inner_fraction: float
    minimum_tof_support: float
    minimum_relative_phase: float
    tof_weight: float = 0.25
    future_weight: float = 0.85
    joint_future: bool = False
    fraction_spread: float = 0.02
    inner_threshold: float = 0.55
    inner_bonus: float = 2.8

    def validate(self):
        positive = (
            self.native_step_s,
            self.surface_scale_bins,
            self.maximum_single_span_bins,
            self.maximum_double_span_bins,
            self.fraction_spread,
            self.inner_threshold,
            self.inner_bonus,
        )
        nonnegative = (
            self.surface_bin,
            self.minimum_relative_phase,
            self.tof_weight,
            self.future_weight,
        )
        if (
            any(not math.isfinite(x) or x <= 0 for x in positive)
            or any(not math.isfinite(x) or x < 0 for x in nonnegative)
            or type(self.minimum_gap_bins) is not int
            or self.minimum_gap_bins < 1
            or self.maximum_single_span_bins < self.minimum_gap_bins
            or self.maximum_double_span_bins < 2 * self.minimum_gap_bins
            or not 0 < self.inner_fraction < 1
            or not 0 < self.minimum_tof_support <= 1
            or type(self.joint_future) is not bool
        ):
            raise ValueError("Invalid echo path calibration.")
        return self


@dataclass(frozen=True)
class EchoCandidates:
    bins: np.ndarray
    prominence: np.ndarray
    support: np.ndarray
    future: np.ndarray


def extract_echo_candidates(phase, tof):
    """Find distinct phase lobes and ToF peaks in the median observation."""
    if phase.ndim != 2 or phase.shape != tof.shape or min(phase.shape) < 1:
        raise ValueError("Expected phase and ToF observations of shape [frames, bins].")
    phase = np.median(phase, axis=0)
    delay = np.median(tof, axis=0).clip(0)
    if delay.max() <= 0 or len(phase) < 9:
        return None
    phase_prominence = (gaussian_filter1d(phase, 2) - uniform_filter1d(phase, 37)).clip(
        0
    )
    broad_prominence = gaussian_filter1d(phase, 4) - gaussian_filter1d(phase, 14)
    delay_max = max(delay.max(), 1e-8)
    support = maximum_filter1d(delay, 29) / delay_max
    strengths = []
    for signal, weight in (
        (phase_prominence, 1.0),
        (broad_prominence, 0.7),
        (delay / delay_max, 0.2),
    ):
        locations, _ = find_peaks(signal, distance=18)
        strongest = locations[np.argsort(signal[locations])[-24:]]
        scale = max(signal.max(), 1e-7)
        strengths.extend((float(weight * signal[x] / scale), int(x)) for x in strongest)
    chosen = []
    local_score = phase_prominence + 0.03 * support
    for _, x in sorted(strengths, reverse=True):
        if any(abs(x - previous) < 15 for previous in chosen):
            continue
        if support[x] < 0.0005 and phase_prominence[x] < 0.06:
            continue
        lo, hi = max(4, x - 8), min(len(phase) - 4, x + 9)
        if hi <= lo:
            continue
        x = lo + int(local_score[lo:hi].argmax())
        if all(abs(x - previous) >= 14 for previous in chosen):
            chosen.append(x)
        if len(chosen) == 28:
            break
    if len(chosen) < 2:
        return None
    bins = np.array(sorted(chosen), dtype=np.int64)
    future = np.asarray(
        [delay[x + 40 : x + 1200].max(initial=0) / delay_max for x in bins]
    )
    return EchoCandidates(bins, phase_prominence[bins], support[bins], future)


def echo_path(candidates, config):
    """Score complete single and double paths, requiring both fingerprints."""
    if candidates is None:
        return None
    bins = candidates.bins
    p, t = candidates.prominence, candidates.support
    top_score = -0.5 * ((bins - config.surface_bin) / config.surface_scale_bins) ** 2
    top_score = top_score + np.log(0.045 + p) + 0.2 * np.log(0.015 + t)
    top = int(top_score.argmax())
    if (
        abs(bins[top] - config.surface_bin) > 3 * config.surface_scale_bins
        or t[top] < config.minimum_tof_support
    ):
        return None
    delta = bins - bins[top]
    gap = config.minimum_gap_bins
    supported = (t >= config.minimum_tof_support) & (
        p >= config.minimum_relative_phase * max(0.06, p[top])
    )
    ends = np.flatnonzero(
        supported
        & (delta >= gap)
        & (
            delta
            <= max(config.maximum_single_span_bins, config.maximum_double_span_bins)
        )
    )
    if not len(ends):
        return None
    outer = np.log(0.025 + p[ends]) + config.tof_weight * np.log(0.003 + t[ends])
    if config.joint_future:
        support = p * np.sqrt(t)
        after = bins[None, :] > bins[ends, None] + 0.75 * gap
        future = np.where(after, support, 0).max(axis=-1)
        ratio = future / (0.012 + support[ends])
    else:
        ratio = candidates.future[ends] / (0.018 + t[ends])
    outer -= config.future_weight * np.log1p(ratio)

    fractions = delta[None, :] / delta[ends, None]
    closeness = np.exp(
        -0.5 * ((fractions - config.inner_fraction) / config.fraction_spread) ** 2
    )
    inner = p[None, :] / max(0.065, p[top]) * closeness * (t[None, :] + 0.05) ** 0.07
    valid = supported & (delta >= gap) & (bins[ends, None] - bins[None, :] >= gap)
    inner = np.where(valid, inner, 0)
    middles = inner.argmax(axis=-1)
    strength = inner[np.arange(len(ends)), middles]
    single = np.where(delta[ends] <= config.maximum_single_span_bins, outer, -np.inf)
    double = np.where(
        (strength > config.inner_threshold)
        & (delta[ends] <= config.maximum_double_span_bins),
        outer + config.inner_bonus * np.log((strength + 1e-6) / config.inner_threshold),
        -np.inf,
    )
    best = np.maximum(single, double)
    if not np.isfinite(best).any():
        return None
    pick = int(best.argmax())
    selected = (
        (top, middles[pick], ends[pick])
        if double[pick] > single[pick]
        else (top, ends[pick])
    )
    return tuple(int(bins[index]) for index in selected)


def read_echo_paths(phase, tof, time, settings, max_layers, minimum_layers):
    """Read a record-consensus path when the deployment grid matches calibration."""
    config = EchoPathConfig(**settings)
    observed = torch.stack((phase, tof), dim=0).detach().cpu().numpy()
    dt = time.diff(dim=-1).detach().cpu().numpy()
    counts = np.zeros(len(dt), dtype=np.int64)
    padded = np.zeros((len(dt), max_layers + 1), dtype=np.int64)
    for i, steps in enumerate(dt):
        if not np.allclose(steps, config.native_step_s, rtol=1e-3, atol=0):
            continue
        chosen = echo_path(
            extract_echo_candidates(observed[0, i], observed[1, i]), config
        )
        if chosen is None or not minimum_layers <= len(chosen) - 1 <= max_layers:
            continue
        counts[i] = len(chosen) - 1
        padded[i, : len(chosen)] = chosen
        padded[i, len(chosen) :] = chosen[-1]
    counts = torch.as_tensor(counts, device=phase.device)
    indices = torch.as_tensor(padded, device=phase.device)[:, None, :].expand(
        -1, phase.shape[1], -1
    )
    return time.gather(-1, indices), counts
