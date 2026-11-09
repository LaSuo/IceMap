"""Register Design3 v6 fingerprints with ordered, measured OPF echo paths."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
import torch

from align_opf_features import (
    FINGERPRINTS,
    GRID,
    SURFACE,
    correlation,
    finite_native,
    projection,
    protocol_key,
    relative_rmse,
    replay,
    write_mirror,
)
from echo_path import EchoPathConfig
from inference import load_model
from legacy_data import physical_layers
from opf_band_paths import decode_paths, fit_echo_paths, median_bands
from opf_thickness_prior import apply_thickness_prior, fit_thickness_prior
from protocol import sha256_file
from retraining_data import read_legacy, supervised_arrays
from train_opf_cv import TARGET, write_json


def read_inputs(args, rows, fit, validation):
    bands, expected, shift = [], np.zeros((len(rows), 2)), []
    thickness = np.zeros_like(expected)
    density = np.zeros_like(expected)
    fields = ("time_s", "shift_bins", "raw_low_envelope", "raw_high_envelope")
    with h5py.File(args.match_dir / "labels_matched.h5") as labels:
        for i, row in enumerate(rows):
            key = row["key"]
            data = loadmat(
                args.native_dir / f"{key}.mat",
                squeeze_me=True,
                variable_names=(*fields, "design3_version"),
            )
            if (
                str(data.get("design3_version")) != "v6"
                or not set(fields) <= data.keys()
            ):
                raise ValueError(f"{key} is missing the v6 raw envelope export.")
            dt = np.diff(data["time_s"])
            if (
                dt.size < 8
                or dt[0] <= 0
                or not np.allclose(dt, dt[0], rtol=1e-6, atol=0)
            ):
                raise ValueError(f"{key} needs a uniformly sampled time axis.")
            step = float(dt[0] * 299792458.0 / 2)
            bands.append(
                median_bands(data["raw_low_envelope"], data["raw_high_envelope"], step)
            )
            shift.append((float(np.median(data["shift_bins"])), 0.001 / step))
            if fit[i] or validation[i]:
                j = int(row["label_archive_row_zero_based"])
                layers = [
                    physical_layers(frame) for frame in labels["label3"][..., j].T
                ]
                if len({len(f) for f in layers}) != 1 or len(layers[0]) not in (1, 2):
                    raise ValueError(
                        f"{key} does not have a constant one- or two-layer label."
                    )
                d = np.median(
                    [
                        [(end - start) * (1 + 0.845 * rho) for start, end, rho in frame]
                        for frame in layers
                    ],
                    axis=0,
                )
                expected[i, : len(d)] = d
                thickness[i, : len(d)] = np.median(
                    [[end - start for start, end, _ in frame] for frame in layers],
                    axis=0,
                )
                density[i, : len(d)] = np.median(
                    [[rho for _, _, rho in frame] for frame in layers], axis=0
                )
    return bands, expected, shift, thickness, density


def observed_features(
    record, event, gaps, shift, fallback, *, tof_width_bins=7.0, tof_gain=1.0
):
    """Represent observed interface locations on the existing D4 feature grid."""
    drift, bins_per_mm = shift
    registered_surface = event[0] * bins_per_mm - drift
    phase = projection(record["phase"], registered_surface, bins_per_mm, 0)
    scatter = projection(record["E"], registered_surface, bins_per_mm, 0).clip(0)
    if not gaps[0]:
        return fallback, None, None
    centers = SURFACE + np.r_[0.0, np.cumsum(gaps[gaps > 0])]
    # Gaussian rasterization of registered delays keeps fractional-bin timing.
    pulses = np.exp(-0.5 * ((GRID[None, :] - centers[:, None]) / 7.0) ** 2).max(0)
    phase_support = 0.12 * np.tanh(np.abs(phase)) + 0.7 * pulses[None, :]
    phase_support += 0.38
    delay_pulses = pulses
    if tof_width_bins != 7.0:
        delay_pulses = np.exp(
            -0.5 * ((GRID[None, :] - centers[:, None]) / tof_width_bins) ** 2
        ).max(0)
    delay = np.broadcast_to(tof_gain * delay_pulses, phase.shape)
    base = {"phase": phase_support.astype(np.float32), "tau": delay.astype(np.float32)}
    return base, scatter, centers


def stacked(data, rows):
    return {
        name: np.stack([data[row["key"]][name] for row in rows])
        for name in FINGERPRINTS
    }


def readout_settings(path, calibration):
    model, _ = load_model(path, "cpu")
    old = EchoPathConfig(**model.core.config.echo_path)
    settings = replace(
        old,
        minimum_gap_bins=max(35, calibration.options.minimum_gap - 6),
        maximum_single_span_bins=calibration.maximum_single_span * 1.2,
        maximum_double_span_bins=calibration.maximum_double_span * 1.2,
        inner_fraction=calibration.inner_fraction,
        fraction_spread=max(0.12, calibration.options.inner_spread),
        minimum_relative_phase=0.35,
        minimum_tof_support=0.055,
        inner_threshold=0.39,
    ).validate()
    return asdict(settings)


def selected_score(value):
    return sum(value[key] for key in TARGET) / len(TARGET)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "match-dir",
        "native-dir",
        "shared-dir",
        "legacy-dir",
        "split-manifest",
        "deployment-report",
        "fp32",
        "int8",
        "output",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(
        "--controlled-thickness-prior",
        action="store_true",
        help="Learn specimen spacing from fitting labels for controlled OPF captures.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf8"))
    deployment = json.loads(args.deployment_report.read_text(encoding="utf8"))
    matching = json.loads(
        (args.match_dir / "match_summary.json").read_text(encoding="utf8")
    )
    with (args.match_dir / "matches.csv").open(encoding="utf8", newline="") as file:
        rows = [
            row for row in csv.DictReader(file) if row["status"] == "unique_identifier"
        ]
    group = manifest["deployment_groups"]
    structure_ids = [protocol_key(row, matching["legacy_date"]) for row in rows]
    part = np.array([group[key] for key in structure_ids])
    fit, validation = part >= 2, part == 1
    if not np.any(part == 0):
        raise ValueError("No structure-held-out observations.")
    train_path = args.legacy_dir / "train.h5"
    test_path = args.legacy_dir / "test_opf.h5"
    for path in (train_path, test_path):
        digest = sha256_file(path)
        if (
            digest != matching["source_sha256"][path.name]
            or digest != manifest["source_sha256"][path.name]
        ):
            raise ValueError(f"Legacy input hash mismatch for {path.name}.")
    args.output.mkdir(parents=True, exist_ok=False)
    bands, expected, shift, thickness, density = read_inputs(
        args, rows, fit, validation
    )
    calibration = fit_echo_paths(bands, expected, fit, validation)
    events, delays, kind, valid = decode_paths(bands, calibration)
    delays[~valid] = 0
    measured_delays = delays.copy()
    prior = None
    if args.controlled_thickness_prior:
        prior = fit_thickness_prior(thickness, density, structure_ids, fit)
        delays = apply_thickness_prior(delays, prior, 0.845)
    settings = readout_settings(args.fp32, calibration)

    fallback, registered, same_grid_scatter = {}, {}, {}
    with h5py.File(args.shared_dir / "all_raw_recovered_features.h5") as base:
        index = {key: i for i, key in enumerate(base["key"].asstr()[:])}
        for i, row in enumerate(rows):
            key = row["key"]
            fallback[key] = {
                name: base[name][..., index[key]].T for name in FINGERPRINTS
            }
            record = finite_native(args.native_dir / f"{key}.mat")
            registered[key], scatter, bounds = observed_features(
                record, events[i], delays[i], shift[i], fallback[key]
            )
            if valid[i]:
                same_grid_scatter[key] = (scatter, bounds)

    # Preserve a single shared gain, estimated only from fitting windows.
    with h5py.File(train_path) as train:
        amplitudes = []
        for i in np.flatnonzero(fit & valid):
            key = rows[i]["key"]
            scatter, bounds = same_grid_scatter[key]
            roi = slice(SURFACE, round(bounds[-1]))
            old = train["E"][..., int(rows[i]["train_row_zero_based"])].T
            if np.quantile(scatter[:, roi], 0.75) > 1e-8:
                amplitudes.append(
                    np.quantile(old[:, roi], 0.75) / np.quantile(scatter[:, roi], 0.75)
                )
        gain = float(np.median(amplitudes)) if amplitudes else 1.0
    modes = {"registered_scatter": {}, "shared_scatter": {}}
    for row in rows:
        key = row["key"]
        base = registered[key]
        modes["shared_scatter"][key] = dict(base, E=fallback[key]["E"])
        if key in same_grid_scatter:
            value = np.asarray(same_grid_scatter[key][0] * gain, dtype=np.float32)
        else:
            value = fallback[key]["E"]
        modes["registered_scatter"][key] = dict(base, E=value)

    test_rows = sorted(
        (row for row in rows if int(row["test_row_zero_based"]) >= 0),
        key=lambda row: int(row["test_row_zero_based"]),
    )
    train_rows = sorted(rows, key=lambda row: int(row["train_row_zero_based"]))
    validate_names = {
        row["key"] for row, flag in zip(rows, validation, strict=True) if flag
    }
    validate_indices = [
        i for i, row in enumerate(test_rows) if row["key"] in validate_names
    ]
    holdout_names = set(deployment["held_out_record_names"])
    holdout_indices = [
        i for i, row in enumerate(test_rows) if row["test_filename"] in holdout_names
    ]
    if len(holdout_indices) != deployment["held_out_records"] or not validate_indices:
        raise ValueError(
            "Deployment report or validation structures disagree with matched records."
        )
    old = supervised_arrays(
        read_legacy(test_path, grid_step_m=0.001, require_axes=True)
    )
    profiles = {key: stacked(value, test_rows) for key, value in modes.items()}
    validation_metrics = replay(
        args.fp32, old, validate_indices, profiles, echo_path=settings
    )
    selected = min(
        modes, key=lambda key: selected_score(validation_metrics[key]["metrics"])
    )
    chosen = modes[selected]
    network = {}
    for name, path in (("fp32", args.fp32), ("int8", args.int8)):
        network[name] = replay(
            path,
            old,
            holdout_indices,
            {"ordered_v6": profiles[selected]},
            echo_path=settings,
        )

    with h5py.File(train_path) as train, h5py.File(test_path) as test:
        write_mirror(
            args.output / "train_matched_ordered.h5",
            train_rows,
            train,
            chosen,
            "train_row_zero_based",
        )
        write_mirror(
            args.output / "test_opf_ordered.h5",
            test_rows,
            test,
            chosen,
            "test_row_zero_based",
        )
        similarity = {}
        for name, mask in (
            ("fit", fit),
            ("validation", validation),
            ("holdout", part == 0),
        ):
            records = [row for row, use in zip(rows, mask, strict=True) if use]
            scores = {}
            for field in FINGERPRINTS:
                values = [
                    (
                        chosen[row["key"]][field],
                        train[field][..., int(row["train_row_zero_based"])].T,
                    )
                    for row in records
                ]
                scores[f"{field}_pearson"] = float(
                    np.mean([correlation(a, b) for a, b in values])
                )
                scores[f"{field}_relative_rmse"] = float(
                    np.mean([relative_rmse(a, b) for a, b in values])
                )
            similarity[name] = scores

    with (args.output / "observed_paths.csv").open(
        "w", encoding="utf8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "key",
                "split",
                "accepted",
                "band",
                "interfaces_native_mm",
                "calibrated_delays_mm",
                "pre_prior_delays_mm",
            )
        )
        for i, row in enumerate(rows):
            count = int(np.count_nonzero(delays[i]))
            edges = events[i, [0, 2]] if count == 1 else events[i]
            writer.writerow(
                (
                    row["key"],
                    int(part[i]),
                    bool(valid[i]),
                    "low" if kind[i] < 2 else "high",
                    edges.tolist() if valid[i] else [],
                    delays[i, :count].tolist(),
                    measured_delays[i, :count].tolist(),
                )
            )

    acceptance = {}
    for name, profiles in network.items():
        metric = profiles["ordered_v6"]["metrics"]
        acceptance[name] = {}
        for key, target in TARGET.items():
            difference = float(metric[key] - target)
            acceptance[name][key] = {
                "signed_difference_points": difference,
                "within_two_points": abs(difference) <= 2,
            }
    report = {
        "scope": "development comparison with structure-disjoint held-out observations",
        "fitting_records": int(fit.sum()),
        "validation_records": int(validation.sum()),
        "network_held_out_records": len(holdout_indices),
        "network_held_out_frames": len(holdout_indices) * 10,
        "calibration": asdict(calibration),
        "controlled_thickness_prior_requested": args.controlled_thickness_prior,
        "controlled_thickness_prior": asdict(prior) if prior is not None else None,
        "corrected_delay_windows": int(np.any(delays != measured_delays, axis=1).sum()),
        "echo_path_settings": settings,
        "selected_scattering": selected,
        "scattering_gain": gain,
        "validation_network": validation_metrics,
        "held_out_network": network,
        "similarity": similarity,
        "target_percent": TARGET,
        "target_comparison": acceptance,
    }
    write_json(args.output / "ordered_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
