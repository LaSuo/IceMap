"""Cross-frequency support for resolving weak and overlapping OPF echoes."""

from dataclasses import dataclass
import math

import numpy as np

from opf_band_paths import (
    calibrated_gaps,
    candidate_table,
    clutter_weights,
    path_scores,
)
from opf_thickness_prior import apply_thickness_prior


@dataclass(frozen=True)
class EchoWitnessOptions:
    early_window_mm: int
    short_gap_mm: int
    internal_support_ratio: float
    weak_terminal_support: float
    deep_gap_mm: int
    deep_terminal_support: float
    deep_bonus: float
    high_span_weight: float
    paired_size_weight: float

    def validate(self):
        distances = self.early_window_mm, self.short_gap_mm, self.deep_gap_mm
        thresholds = (
            self.weak_terminal_support,
            self.deep_terminal_support,
            self.deep_bonus,
        )
        fractions = (
            self.internal_support_ratio,
            self.high_span_weight,
            self.paired_size_weight,
        )
        if (
            any(type(x) is not int or x <= 0 for x in distances)
            or self.deep_gap_mm <= self.short_gap_mm
            or any(not math.isfinite(x) or x < 0 for x in thresholds)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in fractions)
        ):
            raise ValueError("Invalid cross-frequency echo witness settings.")
        return self


def high_band_witness(bands, low_double_events, clutter, early_window_mm):
    """Measure companion peaks for the low-band interior and terminal echoes."""
    shape = low_double_events.shape[:-1]
    inner_ratio = np.zeros(shape)
    terminal_support = np.zeros(shape)
    companion_span = np.zeros(shape)
    for i, (_, (high, peaks)) in enumerate(bands):
        strength = high[peaks] * clutter[peaks]
        noise = max(0.035, np.quantile(high[1250:1951], 0.75))
        for j, (start, middle, end) in enumerate(low_double_events[i]):
            if not start:
                continue
            terminal = np.flatnonzero(
                (peaks >= end - early_window_mm) & (peaks <= end + 76)
            )
            if not len(terminal):
                continue
            best = terminal[strength[terminal].argmax()]
            amplitude = strength[best]
            terminal_support[i, j] = np.log1p(amplitude / noise)
            interior = (peaks >= middle - 48) & (peaks <= middle + 76)
            inner_ratio[i, j] = strength[interior].max(initial=0.0) / max(
                amplitude, 0.025
            )
            start_candidates = peaks[(peaks >= start - 60) & (peaks <= start + 150)]
            if len(start_candidates):
                surface = start_candidates[high[start_candidates].argmax()]
                companion_span[i, j] = peaks[best] - surface
    return inner_ratio, terminal_support, companion_span


def decode_with_echo_witness(bands, calibration, prior, options, dielectric):
    """Resolve path ambiguity and calibrate the time grid for controlled specimens."""
    options.validate()
    clutter = clutter_weights(calibration.recurring_clutter)
    limits = calibration.maximum_single_span, calibration.maximum_double_span
    table = candidate_table(
        bands, calibration.options, limits, calibration.inner_fraction, clutter
    )
    gaps, events, evidence = table
    score = path_scores(table, calibration.options)
    short_gap = gaps[:, 1].min(axis=-1)
    deep = short_gap >= options.deep_gap_mm
    inner_ratio, wide_support, high_span = high_band_witness(
        bands, events[:, 1], clutter, options.early_window_mm
    )
    amplitude = evidence[:, 1, :, 0]
    narrow_support = evidence[:, 1, :, 3]

    # A close low-band interior lobe needs independent high-band evidence.
    unsupported_short = (short_gap < options.short_gap_mm) & (
        inner_ratio < options.internal_support_ratio
    )
    unsupported_weak = (
        ~deep & (amplitude < 0.25) & (narrow_support < options.weak_terminal_support)
    )
    score[:, 1][unsupported_short | unsupported_weak] = -np.inf

    # Permit a wider timing disagreement for weak returns from two long layers.
    gain = calibration.options.cross_band_bonus * (
        np.maximum(wide_support, narrow_support) - narrow_support
    )
    strong = wide_support >= options.deep_terminal_support
    score[:, 1] += np.where(deep, gain + options.deep_bonus * strong, 0.0)
    kind, index = np.divmod(score.reshape(len(gaps), -1).argmax(1), score.shape[-1])
    row = np.arange(len(gaps))
    valid = np.isfinite(score[row, kind, index])
    measured = calibrated_gaps(
        gaps[row, kind, index],
        kind,
        calibration.scale_by_band_and_count,
        calibration.inner_fraction,
        calibration.options.fraction_shrink,
    )
    measured[~valid] = 0
    delays = apply_thickness_prior(measured, prior, dielectric)

    # On long pairs, combine corroborating total spans before applying the
    # shared nominal-size prior. Preserve the measured split as a contribution.
    use = valid & (kind == 1) & deep[row, index] & strong[row, index]
    totals = measured[use].sum(axis=1)
    high_total = high_span[row[use], index[use]]
    consistent = (high_total > 0.7 * totals) & (high_total < 1.3 * totals)
    fused = np.where(
        consistent,
        (1 - options.high_span_weight) * totals + options.high_span_weight * high_total,
        totals,
    )
    nominal_pair = 2 * prior.step_mm * (1 + dielectric * prior.density_g_cm3)
    nominal = nominal_pair * np.maximum(1, np.rint(fused / nominal_pair))
    fraction = calibration.inner_fraction
    target = nominal[:, None] * np.array((fraction, 1 - fraction))
    continuous = measured[use] * (fused / totals)[:, None]
    delays[use] = (
        1 - options.paired_size_weight
    ) * continuous + options.paired_size_weight * target
    return events[row, kind, index], measured, delays, valid
