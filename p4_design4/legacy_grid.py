"""Explicit axes for legacy N,T,S records; never infer meters from array size."""

import numpy as np


def legacy_axes(file, step_m=None):
    N = file["phase"].shape[0]
    if "time_s" in file and "depth_m" in file:
        time = np.asarray(file["time_s"], dtype=np.float64).reshape(-1)
        depth = np.asarray(file["depth_m"], dtype=np.float64).reshape(-1)
        provenance = "explicit_hdf5_axes"
    else:
        step = (
            step_m
            if step_m is not None
            else file.attrs.get("confirmed_legacy_grid_step_m")
        )
        if step is None or not np.isscalar(step) or not np.isfinite(step) or step <= 0:
            raise ValueError(
                "Legacy records lack calibrated axes: supply --legacy-grid-mm explicitly (confirmed dataset: 1 mm) or export time_s/depth_m."
            )
        time = np.arange(N, dtype=np.float64) * (2 * step / 299792458.0)
        depth = np.arange(N, dtype=np.float64) * step
        provenance = "explicit_confirmed_legacy_grid_step_m"
    if (
        time.shape != (N,)
        or depth.shape != (N,)
        or not np.isfinite(time).all()
        or not np.isfinite(depth).all()
        or depth[0] < 0
    ):
        raise ValueError("Legacy axes must be finite N-vectors, depth>=0.")
    for axis in (time, depth):
        delta = np.diff(axis)
        if not (delta > 0).all() or not np.allclose(
            delta, delta.mean(), rtol=1e-5, atol=0
        ):
            raise ValueError("Legacy conversion requires uniform increasing axes.")
    return time, depth, provenance
