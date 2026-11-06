"""Layered capacity model (m, g/cm^3, kg) and explicit spatial interpolation.

Kriging covariance parameters are engineering inputs, not measured constants.
Reference capacities must come from independent, safety-adjusted load tests.
"""

import numpy as np
import torch


def _distances(a, b):
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))


def bearing_capacity(h_m, rho_g_cm3, coefficient_kg_cm2=3.5, density_threshold=0.90):
    h = np.asarray(h_m, dtype=float)
    rho = np.asarray(rho_g_cm3, dtype=float)
    if (
        h.ndim != 1
        or h.shape != rho.shape
        or not h.size
        or not np.isfinite(h).all()
        or not np.isfinite(rho).all()
        or (h <= 0).any()
        or (rho <= 0).any()
    ):
        raise ValueError(
            "Supply nonempty matching positive finite physical layer vectors."
        )
    if (
        not np.isfinite(coefficient_kg_cm2)
        or coefficient_kg_cm2 <= 0
        or not np.isfinite(density_threshold)
        or density_threshold <= 0
    ):
        raise ValueError("Invalid capacity coefficient or density threshold.")
    H_cm = 100 * h.sum()
    low_fraction = float(h[rho < density_threshold].sum() / h.sum())
    return {
        "capacity_kg": float(coefficient_kg_cm2 * H_cm**2 * (2 - low_fraction) / 2),
        "total_thickness_m": float(h.sum()),
        "low_density_fraction": low_fraction,
    }


def thickness_capacity_relative_bias(thickness_relative_bias):
    e = np.asarray(thickness_relative_bias, dtype=float)
    if not np.isfinite(e).all() or (e <= -1).any():
        raise ValueError("Thickness bias must be finite and > -1.")
    return 2 * e + e**2


def composition_capacity_relative_bias(true_low_fraction, predicted_low_fraction):
    true, pred = np.broadcast_arrays(
        np.asarray(true_low_fraction, dtype=float),
        np.asarray(predicted_low_fraction, dtype=float),
    )
    if (
        not np.isfinite(true).all()
        or not np.isfinite(pred).all()
        or ((true < 0) | (true > 1) | (pred < 0) | (pred > 1)).any()
    ):
        raise ValueError("Composition fractions must lie in [0,1].")
    return (true - pred) / (2 - true)


def capacity_error(predicted_kg, reference_kg):
    p = np.asarray(predicted_kg, dtype=float)
    r = np.asarray(reference_kg, dtype=float)
    if (
        p.ndim != 1
        or p.shape != r.shape
        or not p.size
        or not np.isfinite(p).all()
        or not np.isfinite(r).all()
        or (p < 0).any()
        or (r <= 0).any()
    ):
        raise ValueError("Use matching finite capacity vectors with reference >0.")
    error = np.abs(p - r) / r
    return {
        "BCE_percent": float(100 * error.mean()),
        "sites": len(r),
        "absolute_relative_error": error.tolist(),
        "reference_required": "independent safety-adjusted load test; not capacity computed from core labels",
    }


def validation_grid(rows=5, columns=5, spacing_m=2.0):
    if (
        type(rows) is not int
        or type(columns) is not int
        or min(rows, columns) < 1
        or not np.isfinite(spacing_m)
        or spacing_m <= 0
    ):
        raise ValueError("Invalid grid dimensions/spacing.")
    x, y = np.meshgrid(np.arange(columns) * spacing_m, np.arange(rows) * spacing_m)
    return np.column_stack((x.ravel(), y.ravel()))


def ordinary_kriging(
    points_m,
    capacity_kg,
    query_m,
    *,
    correlation_length_m,
    variance_kg2,
    nugget_kg2=0.0,
):
    """Ordinary kriging with exponential covariance; variance returned in kg^2.

    Fix/fit covariance using training sites only. Kriging maps do not create
    additional independent validation sites or load-test ground truth.
    """
    p = np.asarray(points_m, dtype=float)
    v = np.asarray(capacity_kg, dtype=float)
    q = np.asarray(query_m, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or q.ndim != 2
        or q.shape[1] != 2
        or v.shape != (len(p),)
        or len(p) < 2
        or not all(np.isfinite(a).all() for a in (p, v, q))
        or (v < 0).any()
    ):
        raise ValueError(
            "Expected finite [sites,2] coordinates, capacities and [queries,2]."
        )
    if len(np.unique(p, axis=0)) != len(p):
        raise ValueError(
            "Duplicate site coordinates require explicit measurement aggregation."
        )
    if (
        not np.isfinite([correlation_length_m, variance_kg2, nugget_kg2]).all()
        or min(correlation_length_m, variance_kg2) <= 0
        or nugget_kg2 < 0
    ):
        raise ValueError(
            "Supply positive covariance length/variance and nonnegative nugget."
        )
    C = (
        variance_kg2 * np.exp(-_distances(p, p) / correlation_length_m)
        + np.eye(len(p)) * nugget_kg2
    )
    c = variance_kg2 * np.exp(-_distances(p, q) / correlation_length_m)
    system = np.block(
        [[C, np.ones((len(p), 1))], [np.ones((1, len(p))), np.zeros((1, 1))]]
    )
    # Share the existing PyTorch linear algebra runtime, avoiding a second
    # Intel OpenMP runtime when conda NumPy/MKL and PyTorch coexist on Windows.
    try:
        solution = torch.linalg.solve(
            torch.from_numpy(system),
            torch.from_numpy(np.vstack((c, np.ones((1, len(q)))))),
        ).numpy()
    except RuntimeError as error:
        raise ValueError(
            "Singular kriging covariance; inspect sites/covariance inputs."
        ) from error
    weights = solution[:-1]
    return {
        "capacity_kg": np.sum(weights * v[:, None], axis=0),
        "variance_kg2": np.maximum(
            0, variance_kg2 - np.sum(weights * c, axis=0) - solution[-1]
        ),
        "weights": weights.T,
    }
