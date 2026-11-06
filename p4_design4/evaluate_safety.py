"""Evaluate independent load-test BCE and optionally export a kriging map.

JSON input is a list of site_id, xy_m, pred_h_m, pred_rho_g_cm3,
reference_capacity_kg. Reference loads must already include the experimental
safety factor; core-derived capacity is not an independent load-test label.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from safety import bearing_capacity, capacity_error, ordinary_kriging
from protocol import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--map-step-m", type=float)
    p.add_argument("--correlation-length-m", type=float)
    p.add_argument("--variance-kg2", type=float)
    p.add_argument("--nugget-kg2", type=float, default=0.0)
    a = p.parse_args()
    rows = json.loads(Path(a.input).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        p.error("Expected a nonempty list of independent validation sites.")
    names = [r["site_id"] for r in rows]
    if any(not isinstance(x, str) or not x for x in names) or len(names) != len(
        set(names)
    ):
        p.error("Site identifiers must be nonempty and unique.")
    predicted = []
    details = []
    for r in rows:
        if not r["pred_h_m"] and not r["pred_rho_g_cm3"]:
            value = {"capacity_kg": 0.0, "missed_ice": True}
        else:
            value = bearing_capacity(r["pred_h_m"], r["pred_rho_g_cm3"])
        predicted.append(value["capacity_kg"])
        details.append(
            {
                "site_id": r["site_id"],
                **value,
                "reference_capacity_kg": r["reference_capacity_kg"],
            }
        )
    report = capacity_error(predicted, [r["reference_capacity_kg"] for r in rows])
    report.update(
        input_sha256=sha256_file(a.input),
        site_results=details,
        coefficient_kg_cm2=3.5,
        density_threshold_g_cm3=0.9,
        mine_site_count_matched=len(rows) == 25,
        reference_origin_required="independent_safety_adjusted_loading_tests",
    )
    output = Path(a.output_dir)
    if output.exists():
        p.error("Refusing to overwrite an existing safety evaluation.")
    output.mkdir(parents=True)
    if a.map_step_m is not None:
        if (
            not np.isfinite(a.map_step_m)
            or a.map_step_m <= 0
            or a.correlation_length_m is None
            or a.variance_kg2 is None
        ):
            p.error(
                "Mapping needs positive --map-step-m and explicit covariance length/variance."
            )
        xy = np.asarray([r["xy_m"] for r in rows], dtype=float)
        if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
            p.error("Mapping needs finite xy_m for each site.")
        lower = xy.min(0)
        upper = xy.max(0)
        estimated_shape = np.floor((upper - lower) / a.map_step_m) + 1
        if (
            not np.isfinite(estimated_shape).all()
            or (estimated_shape > 1000000).any()
            or np.prod(estimated_shape) > 1000000
        ):
            p.error("Map exceeds one million cells; increase --map-step-m.")
        shape = estimated_shape.astype(int)
        x, y = np.meshgrid(
            lower[0] + np.arange(shape[0]) * a.map_step_m,
            lower[1] + np.arange(shape[1]) * a.map_step_m,
        )
        query = np.column_stack((x.ravel(), y.ravel()))
        result = ordinary_kriging(
            xy,
            predicted,
            query,
            correlation_length_m=a.correlation_length_m,
            variance_kg2=a.variance_kg2,
            nugget_kg2=a.nugget_kg2,
        )
        np.savez_compressed(
            output / "capacity_map.npz",
            x_m=x,
            y_m=y,
            capacity_kg=result["capacity_kg"].reshape(x.shape),
            variance_kg2=result["variance_kg2"].reshape(x.shape),
        )
        report["map"] = {
            "shape": list(x.shape),
            "step_m": a.map_step_m,
            "correlation_length_m": a.correlation_length_m,
            "variance_kg2": a.variance_kg2,
            "nugget_kg2": a.nugget_kg2,
            "negative_estimate_cells": int((result["capacity_kg"] < 0).sum()),
            "covariance_choice": "exponential; explicit engineering choice, not specified by the article",
        }
    (output / "safety_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {
                "BCE_percent": report["BCE_percent"],
                "sites": report["sites"],
                "map": report.get("map"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
