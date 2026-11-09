"""Fit an OPF-specific ordered echo readout from Design3 v6 observations."""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class BandPathOptions:
    surface_threshold: float
    high_radius: int
    minimum_gap: int
    inner_spread: float
    low_double_threshold: float
    high_double_threshold: float
    high_penalty: float
    overlap_penalty: float
    surface_penalty: float
    cross_band_bonus: float
    fraction_shrink: float


@dataclass(frozen=True)
class EchoPathFit:
    options: BandPathOptions
    maximum_single_span: float
    maximum_double_span: float
    inner_fraction: float
    scale_by_band_and_count: tuple
    validation_loss: float
    recurring_clutter: tuple


def median_bands(low, high, step_m):
    """Convert the measured, registered envelope to normalized millimeter axes."""
    if (
        low.ndim != 2
        or low.shape != high.shape
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or min(low.shape) < 2
        or np.any(low < 0)
        or np.any(high < 0)
        or not np.isfinite(step_m)
        or step_m <= 0
    ):
        raise ValueError("Expected finite nonnegative paired [frames, bins] envelopes.")
    native_mm = np.arange(low.shape[-1]) * (1000 * step_m)
    if native_mm[-1] < 1960:
        raise ValueError(
            "Ordered OPF echo extraction needs at least 1.96 m of support."
        )
    grid_mm = np.arange(1961)
    result = []
    for band, sigma, gap, prominence in ((low, 4, 27, 0.016), (high, 3, 21, 0.028)):
        profile = np.interp(grid_mm, native_mm, np.median(band, axis=0))
        profile = gaussian_filter1d(profile, sigma)
        peak_level = profile[550:1600].max()
        if peak_level <= 1e-12:
            result.append((profile, np.empty(0, dtype=int)))
            continue
        profile /= peak_level
        peaks, _ = find_peaks(profile, distance=gap, prominence=prominence)
        peaks = peaks[(peaks >= 540) & (peaks <= 1950)]
        result.append((profile, peaks))
    return tuple(result)


def fit_clutter(bands, expected, fit):
    """Estimate recurring high-band peaks outside ice in the fitting records."""
    support = np.zeros_like(bands[0][1][0])
    for i in np.flatnonzero(fit):
        (low, lp), (high, hp) = bands[i]
        tops = lp[(lp < 1120) & (low[lp] >= 0.19)]
        if not len(tops):
            continue
        # A guard band keeps uncertain bottom picks out of the clutter profile.
        start = tops[0] + expected[i].sum() + 150
        artifact = hp[(hp > start) & (high[hp] >= 0.055)]
        active = np.zeros_like(support)
        active[artifact] = 1
        support += maximum_filter1d(active, 23)
    prevalence = support / fit.sum()
    locations = np.flatnonzero(prevalence > 0.20)
    return tuple((int(k), float(prevalence[k])) for k in locations)


def clutter_weights(clutter):
    weights = np.ones(1961)
    for index, prevalence in clutter:
        weights[index] = 1.0 / (1.0 + 5.0 * (prevalence - 0.20))
    return weights


def candidate_paths(bands, surface, radius, gap, spread, limits, fraction, clutter):
    """Enumerate physically ordered complete paths within a single band."""
    (low, lp), (high, hp) = bands
    first = lp[(lp < 1120) & (low[lp] >= surface)]
    if not len(first) and low[540:1120].max() < 0.10:
        first = hp[(hp < 1120) & (high[hp] >= surface)]
        if not len(first):
            return ([], [], [], [])
    top = int(first[0]) if len(first) else 540 + int(low[540:1120].argmax())
    high_peaks = hp[(np.abs(hp - top) <= radius) & (high[hp] > 0.12)]
    high_top = int(high_peaks[np.argmax(high[high_peaks])]) if len(high_peaks) else None
    high_noise = max(0.035, np.quantile(high[1250:1951], 0.75))
    candidates = []
    for band, (start, (env, peaks)) in enumerate(zip((top, high_top), bands)):
        for count in (1, 2):
            paths = []
            if start is not None and env[start] >= 0.10:
                bound = limits[count - 1]
                if band == 1 and count == 2:
                    bound = min(bound, 500)
                ends = peaks[(peaks - start >= gap) & (peaks - start <= bound)]
                for end in ends:
                    middles = ends[ends <= end - gap] if count == 2 else (end,)
                    for middle in middles:
                        span = end - start
                        split = middle - start
                        first_gap, last_gap = (
                            (split, end - middle) if count == 2 else (span, 0)
                        )
                        width = max(spread, 0.18) if band == 1 else spread
                        balance = np.exp(
                            -0.5 * ((split / span - fraction) / width) ** 2
                        )
                        end_amp = env[end]
                        mid_amp = env[middle] if count == 2 else 0.0
                        # Weak distant returns need additional evidence to justify a second layer.
                        contrast = (
                            env[start] / max(0.08, mid_amp) if count == 2 else 0.0
                        )
                        witness = hp[(hp >= end - 48) & (hp <= end + 76)]
                        high_support = (high[witness] * clutter[witness]).max(
                            initial=0.0
                        )
                        cross_band = np.log1p(high_support / high_noise)
                        paths.append(
                            (
                                (first_gap, last_gap),
                                (start, middle, end),
                                end_amp,
                                mid_amp * balance,
                                contrast,
                                cross_band,
                            )
                        )
            candidates.append(paths)
    return candidates


def candidate_table(bands, options, limits, fraction, clutter):
    paths = [
        candidate_paths(
            x,
            options.surface_threshold,
            options.high_radius,
            options.minimum_gap,
            options.inner_spread,
            limits,
            fraction,
            clutter,
        )
        for x in bands
    ]
    width = max(1, max(len(p) for row in paths for p in row))
    shape = (len(bands), 4, width)
    gaps = np.zeros((*shape, 2))
    bins = np.zeros((*shape, 3), dtype=int)
    evidence = np.zeros((*shape, 4))
    for i, row in enumerate(paths):
        for j, choices in enumerate(row):
            for k, (d, e, *f) in enumerate(choices):
                gaps[i, j, k], bins[i, j, k], evidence[i, j, k] = d, e, f
    return gaps, bins, evidence


def path_scores(table, cfg):
    gaps, _, evidence = table
    amp, inner, contrast, cross_band = np.moveaxis(evidence, -1, 0)
    minimum_amp = np.array((0.09, 0.065, 0.075, 0.38))[None, :, None]
    threshold = np.array((0, cfg.low_double_threshold, 0, cfg.high_double_threshold))[
        None, :, None
    ]
    score = np.log(amp + 0.03)
    for slot in (1, 3):
        score[:, slot] += 1.6 * np.log((inner[:, slot] + 1e-9) / threshold[:, slot])
    score[:, 2:] -= cfg.high_penalty
    score[:, 0] -= cfg.overlap_penalty * (gaps[:, 0, :, 0] < 135)
    score[:, 2] -= 0.18 * (gaps[:, 2, :, 0] > 320)
    score[:, 1] -= cfg.surface_penalty * np.clip(contrast[:, 1], 0, 2)
    score[:, 1] += cfg.cross_band_bonus * (cross_band[:, 1] - 1.35)
    score[(amp < minimum_amp) | (inner < threshold)] = -np.inf
    return score


def choose_paths(table, cfg):
    gaps, events, _ = table
    score = path_scores(table, cfg)
    flat = score.reshape(len(gaps), -1)
    band_count, candidate = np.divmod(flat.argmax(1), score.shape[-1])
    row = np.arange(len(gaps))
    return (
        gaps[row, band_count, candidate],
        events[row, band_count, candidate],
        band_count,
        np.isfinite(score[row, band_count, candidate]),
    )


def calibrated_gaps(gaps, kind, scale, fraction, shrink):
    result = gaps * np.asarray(scale)[kind, None]
    double = (kind % 2 == 1) & (gaps[:, 1] > 0)
    total = result[double].sum(1)
    first = (1 - shrink) * result[double, 0] + shrink * fraction * total
    result[double] = np.column_stack((first, total - first))
    return result


def fit_echo_paths(bands, delays_mm, fit, validation):
    """Select shared rules on fit/validation groups and leave scoring groups untouched."""
    expected = np.asarray(delays_mm, dtype=float)
    if (
        expected.shape != (len(bands), 2)
        or not fit.any()
        or not validation.any()
        or np.any(fit & validation)
    ):
        raise ValueError(
            "Need disjoint fitting and validation groups with padded equivalent delays."
        )
    known = fit | validation
    if (
        not np.isfinite(expected[known]).all()
        or np.any(expected[known, 0] <= 0)
        or np.any(expected[known, 1] < 0)
    ):
        raise ValueError(
            "Need a valid one- or two-layer reference in each development window."
        )
    count = (expected > 0).sum(1)
    if not np.any(fit & (count == 1)) or not np.any(fit & (count == 2)):
        raise ValueError("Fitting requires both single- and double-layer examples.")
    total = expected.sum(1)
    limits = tuple(1.45 * total[fit & (count == k)].max() for k in (1, 2))
    double_fit = fit & (count == 2)
    fraction = float(np.median(expected[double_fit, 0] / total[double_fit]))
    recurring = fit_clutter(bands, expected, fit)
    clutter = clutter_weights(recurring)
    best = None
    # Candidate enumeration is shared across the score-only parameter grid.
    for surface, radius, gap, spread in product(
        (0.19, 0.28, 0.39), (76, 115), (48, 59), (0.10, 0.18)
    ):
        base = BandPathOptions(
            surface, radius, gap, spread, 0.22, 0.32, 0.55, 0.8, 0.0, 0.0, 0.0
        )
        table = candidate_table(bands, base, limits, fraction, clutter)
        for low, high, high_bias, overlap, surface_bias, cross_bonus in product(
            (0.16, 0.22, 0.30, 0.38),
            (0.24, 0.34),
            (0.30, 0.55, 0.80),
            (0.55, 1.35),
            (0.0, 0.45, 0.9),
            (0.0, 0.6, 1.2, 1.8),
        ):
            cfg = replace(
                base,
                low_double_threshold=low,
                high_double_threshold=high,
                high_penalty=high_bias,
                overlap_penalty=overlap,
                surface_penalty=surface_bias,
                cross_band_bonus=cross_bonus,
            )
            gap_mm, _, kind, valid = choose_paths(table, cfg)
            wrong = (gap_mm > 0).sum(1) != count
            scale = np.ones(4)
            for k in range(4):
                used = fit & valid & ~wrong & (kind == k)
                if used.any():
                    ratio = np.median(total[used] / gap_mm[used].sum(1))
                    scale[k] = np.clip(
                        (2 + used.sum() * ratio) / (2 + used.sum()), 0.80, 1.18
                    )
            for shrink in (0.0, 0.35, 0.65):
                prediction = calibrated_gaps(gap_mm, kind, scale, fraction, shrink)
                aggregate_error = (
                    abs(prediction[known].sum(1) - total[known]) / total[known]
                )
                layer_error = (
                    np.sum(
                        abs(prediction[known] - expected[known])
                        / np.maximum(expected[known], 1)
                        * (expected[known] > 0),
                        axis=1,
                    )
                    / count[known]
                )
                loss = 0.95 * wrong[known] + 0.65 * aggregate_error
                loss += 0.65 * layer_error * ~wrong[known] + 0.35 * ~valid[known]
                score = (
                    0.5 * loss[fit[known]].mean() + 0.5 * loss[validation[known]].mean()
                )
                score += 0.005 * (shrink + surface_bias + cross_bonus)
                if best is None or score < best.validation_loss:
                    best = EchoPathFit(
                        replace(cfg, fraction_shrink=shrink),
                        *limits,
                        fraction,
                        tuple(map(float, scale)),
                        float(score),
                        recurring,
                    )
    return best


def decode_paths(bands, calibration):
    limits = calibration.maximum_single_span, calibration.maximum_double_span
    table = candidate_table(
        bands,
        calibration.options,
        limits,
        calibration.inner_fraction,
        clutter_weights(calibration.recurring_clutter),
    )
    gaps, events, kind, valid = choose_paths(table, calibration.options)
    corrected = calibrated_gaps(
        gaps,
        kind,
        calibration.scale_by_band_and_count,
        calibration.inner_fraction,
        calibration.options.fraction_shrink,
    )
    return events, corrected, kind, valid
