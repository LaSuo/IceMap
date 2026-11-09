"""Learn a soft thickness lattice for controlled OPF specimens."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThicknessPrior:
    step_mm: int
    density_g_cm3: float
    weight: float
    fitting_structures: int


def fit_thickness_prior(thickness_mm, density, structure_ids, fit):
    """Use fitting structures to estimate specimen spacing and confidence."""
    thickness = np.asarray(thickness_mm, dtype=float)
    density = np.asarray(density, dtype=float)
    group_ids = np.asarray(structure_ids)
    fit = np.asarray(fit)
    if (
        thickness.ndim != 2
        or thickness.shape != density.shape
        or group_ids.shape != (len(thickness),)
        or fit.shape != group_ids.shape
        or fit.dtype != np.bool_
    ):
        raise ValueError("Expected paired layer arrays and a Boolean structure split.")
    thickness, density, group_ids = thickness[fit], density[fit], group_ids[fit]
    if (
        not np.isfinite(thickness).all()
        or not np.isfinite(density).all()
        or np.any(thickness < 0)
        or np.any(density < 0)
        or np.any((thickness > 0) != (density > 0))
    ):
        raise ValueError("Invalid paired fitting thickness and density labels.")
    structures = np.unique(group_ids)
    if len(structures) < 5:
        return None
    profiles = [thickness[group_ids == g] for g in structures]
    points = [x[x > 0] for x in profiles]
    if any(len(x) == 0 for x in points):
        raise ValueError("Every fitting structure must have a physical ice layer.")
    all_points = np.concatenate(points)
    for step in range(100, 19, -1):
        snapped = np.rint(all_points / step)
        if len(np.unique(snapped[snapped > 0])) < 3:
            continue
        agreement = np.array(
            [
                np.mean((x > step / 2) & (np.abs(x - step * np.rint(x / step)) <= 1))
                for x in points
            ]
        )
        if agreement.mean() < 0.90 or (agreement < 0.70).any():
            continue
        # Smooth over structures to retain a continuous component for unseen specimens.
        weight = (float(agreement.sum()) + 2) / (len(structures) + 4)
        group_density = []
        for g in structures:
            values = density[group_ids == g]
            group_density.append(np.median(values[values > 0]))
        return ThicknessPrior(
            step, float(np.median(group_density)), weight, len(structures)
        )
    return None


def apply_thickness_prior(delay_mm, prior, dielectric_coefficient):
    """Blend the measured delay with the nearest supported specimen size."""
    delay = np.asarray(delay_mm, dtype=float)
    if not np.isfinite(delay).all() or (delay < 0).any():
        raise ValueError("Equivalent delays must be finite and nonnegative.")
    if prior is None:
        return delay.copy()
    if not np.isfinite(dielectric_coefficient) or dielectric_coefficient <= 0:
        raise ValueError("The dielectric coefficient must be positive.")
    if (
        prior.step_mm <= 0
        or not np.isfinite(prior.density_g_cm3)
        or prior.density_g_cm3 <= 0
        or not 0 < prior.weight < 1
    ):
        raise ValueError("Invalid fitted thickness prior.")
    conversion = 1 + dielectric_coefficient * prior.density_g_cm3
    thickness = delay / conversion
    nearest = prior.step_mm * np.rint(thickness / prior.step_mm)
    change = nearest - thickness
    usable = (
        (thickness > 0)
        & (nearest > 0)
        & (np.abs(change) < 0.5 * prior.step_mm - 1e-9)
        & (np.abs(change) <= 0.4 * thickness)
    )
    return delay + np.where(usable, prior.weight * change * conversion, 0)
