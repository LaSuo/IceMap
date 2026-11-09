"""Fit a shared D3 feature calibration and replay matched OPF observations."""

import argparse
import asyncio
import csv
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import re

import torch
import h5py
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.signal import find_peaks

from evaluate_quantized_opf import evaluate_async
from inference import load_model
from match_opf_records import FINGERPRINTS, LABELS, legacy_names
from metrics import MineMetrics
from protocol import sha256_file
from retraining_data import read_legacy, supervised_arrays
from train_opf_cv import TARGET, write_json


SAMPLES = 2200
SURFACE = 200
GRID = np.arange(SAMPLES, dtype=np.float32)
TIME = np.arange(SAMPLES, dtype=np.float64) * (0.002 / 299792458.0)
RANGE = np.arange(SAMPLES, dtype=np.float64) * 0.001
NATIVE_FIELDS = (
    *FINGERPRINTS,
    "time_s",
    "surface_track",
    "low_envelope",
    "high_envelope",
    "phase_coherence",
    "scattering_coherence",
)


def finite_native(path):
    record = loadmat(path, squeeze_me=True, variable_names=NATIVE_FIELDS)
    missing = set(NATIVE_FIELDS) - set(record)
    if missing:
        raise ValueError(f"{path.name} is missing {sorted(missing)}.")
    arrays = [record[k] for k in NATIVE_FIELDS if k not in ("time_s", "surface_track")]
    if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError(f"{path.name} has inconsistent [frames, bins] arrays.")
    if any(not np.isfinite(record[k]).all() for k in NATIVE_FIELDS):
        raise ValueError(f"{path.name} contains nonfinite values.")
    if arrays[0].shape[0] != 10 or len(record["time_s"]) != arrays[0].shape[-1]:
        raise ValueError(f"{path.name} does not contain a complete 10-frame window.")
    return record


def anchors(record):
    distance = record["time_s"] * 299792458.0 / 2
    region = np.flatnonzero((distance >= 0.60) & (distance <= 1.18))
    if len(region) < 8:
        raise ValueError(
            "The native record has no samples in the surface search range."
        )
    out = {"d1": float(record["surface_track"][0])}
    envelopes = {}
    for field in ("low_envelope", "high_envelope"):
        signal = gaussian_filter1d(np.median(record[field], axis=0)[region], 3)
        signal = signal / max(float(signal.max()), 1e-10)
        envelopes[field] = signal
        peaks, _ = find_peaks(signal, distance=20, height=0.32, prominence=0.04)
        out[field + "_first"] = float(
            region[peaks[0] if len(peaks) else signal.argmax()]
        )
        out[field + "_max"] = float(region[signal.argmax()])
    joint = np.sqrt(envelopes["low_envelope"] * envelopes["high_envelope"])
    peaks, _ = find_peaks(
        joint, distance=20, height=0.45 * joint.max(), prominence=0.02
    )
    out["joint_first"] = float(region[peaks[0] if len(peaks) else joint.argmax()])
    return out


def projection(value, anchor, scale, offset, grid=GRID):
    source = anchor + (grid - SURFACE) * scale + offset
    native = np.arange(value.shape[-1])
    return np.stack(
        [np.interp(source, native, frame, left=0, right=0) for frame in value]
    ).astype(np.float32)


def correlation(x, y):
    x, y = np.ravel(x).astype(np.float64), np.ravel(y).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-15 else 0.0


def relative_rmse(x, y):
    return float(
        np.linalg.norm(np.ravel(x - y)) / max(np.linalg.norm(np.ravel(y)), 1e-12)
    )


def amplitude(value):
    return np.log1p(4 * value / max(float(np.quantile(value, 0.75)), 1e-6))


def select_projection(native, reference, fitting, field):
    target_grid = GRID[::8]
    tau = field == "tau"
    region = (target_grid >= (120 if tau else 205)) & (
        target_grid <= (1450 if tau else 2090)
    )
    grid = target_grid[region]
    anchor_values = {key: anchors(native[key]) for key in fitting}
    names = ("tau", "low_envelope") if tau else ("E", "high_envelope")
    prepared, targets = {}, {}
    for key in fitting:
        prepared[key] = {
            name: amplitude(gaussian_filter1d(np.median(native[key][name], axis=0), 4))
            for name in names
        }
        targets[key] = np.log1p(
            6 * np.median(reference[key][field], axis=0)[grid.astype(int)]
        )
    best = None
    for anchor, scale, offset in product(
        next(iter(anchor_values.values())),
        (0.65, 0.82, 1.0, 1.18, 1.4, 1.65),
        (-65, -32, 0, 32, 65),
    ):
        values = []
        for key in fitting:
            center = anchor_values[key][anchor]
            value = max(
                correlation(
                    projection(signal[None, :], center, scale, offset, grid)[0],
                    targets[key],
                )
                for signal in prepared[key].values()
            )
            values.append(value)
        score = np.mean(values) - 0.022 * abs(np.log(scale)) - 0.00022 * abs(offset)
        if best is None or score > best[0]:
            best = (score, anchor, scale, offset)
    return {
        "anchor": best[1],
        "scale": best[2],
        "offset": best[3],
        "fit_score": float(best[0]),
    }


def warp_field(record, anchor_values, field, calibration):
    return projection(
        record[field],
        anchor_values[calibration["anchor"]],
        calibration["scale"],
        calibration["offset"],
    )


def design_bases(record, maps):
    center = anchors(record)
    high, low = maps["high"], maps["low"]
    p = warp_field(record, center, "phase", high)
    e = warp_field(record, center, "E", high)
    coherence = warp_field(record, center, "scattering_coherence", high)
    h = warp_field(record, center, "high_envelope", high)
    t = warp_field(record, center, "tau", low)
    envelope = warp_field(record, center, "low_envelope", low)
    p_abs = abs(p)
    high_log = amplitude(h)
    e_log = amplitude(e)
    l_log = amplitude(envelope)
    local_peak = np.maximum(
        gaussian_filter1d(l_log, 5, axis=-1) - gaussian_filter1d(l_log, 28, axis=-1), 0
    )
    high_peak = np.maximum(
        gaussian_filter1d(high_log, 5, axis=-1)
        - gaussian_filter1d(high_log, 28, axis=-1),
        0,
    )
    ones = np.ones_like(p)
    x = (GRID - SURFACE) / SAMPLES
    surface = np.exp(-0.5 * ((GRID - SURFACE) / 10) ** 2) * ones
    surface_wide = np.exp(-0.5 * ((GRID - SURFACE) / 22) ** 2) * ones
    front_response = np.max(t[:, 155:245], axis=1, keepdims=True)
    t = t / np.maximum(front_response, 0.3 * np.max(t, axis=1, keepdims=True) + 1e-6)
    # Derive the finite support from registered low-band measurements and depth.
    window = np.exp(-np.maximum(GRID - 520, 0) / 380) * (GRID >= 75)
    local_peak = local_peak * window
    t = t * window
    depth = [ones, x * ones, x**2 * ones]
    return {
        "phase": np.stack(
            (
                *depth,
                p,
                p_abs,
                gaussian_filter1d(p_abs, 6, axis=-1),
                high_log,
                high_peak,
                local_peak,
                t,
                surface,
                surface_wide,
            ),
            axis=-1,
        ),
        "E": np.stack(
            (
                *depth,
                e_log,
                high_log,
                high_peak,
                coherence,
                gaussian_filter1d(high_log, 9, axis=-1),
                local_peak,
            ),
            axis=-1,
        ),
        "tau": np.stack(
            (t, gaussian_filter1d(t, 7, axis=-1), local_peak, surface, surface_wide),
            axis=-1,
        ),
    }


def fit_amplitude(native, reference, fitting, maps):
    matrix = {key: [] for key in FINGERPRINTS}
    targets = {key: [] for key in FINGERPRINTS}
    for key in fitting:
        bases = design_bases(native[key], maps)
        for feature in FINGERPRINTS:
            region = slice(SURFACE, None, 5) if feature == "E" else slice(None, None, 5)
            x = bases[feature][::2, region, :]
            y = reference[key][feature][::2, region]
            matrix[feature].append(x.reshape(-1, x.shape[-1]))
            targets[feature].append(y.reshape(-1))
    weights = {}
    for feature in FINGERPRINTS:
        x, y = np.concatenate(matrix[feature]).astype(np.float64), np.concatenate(
            targets[feature]
        ).astype(np.float64)
        if feature == "tau":
            coefficients, _ = nnls(x, y)
        else:
            regularizer = 0.0007 * len(x) * np.eye(x.shape[1])
            regularizer[0, 0] = 0
            coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        weights[feature] = coefficients.tolist()
    return weights


def calibrated_features(record, maps, weights):
    bases = design_bases(record, maps)
    output = {
        key: np.maximum(0, bases[key] @ np.asarray(weights[key], dtype=np.float32))
        for key in FINGERPRINTS
    }
    output["E"][:, :SURFACE] = 0
    output["tau"][:, :75] = 0
    return output


def baseline_features(record):
    origin = anchors(record)["d1"]
    return {key: projection(record[key], origin, 1, 0) for key in FINGERPRINTS}


def protocol_key(row, date):
    match = re.match(r"(e\d+r\d+_t\d+)b", row["key"])
    if match is None:
        raise ValueError(f"Invalid recording key {row['key']}.")
    return f"{date}_{match[1]}"


def write_mirror(path, rows, source, reconstructed, source_column):
    indices = [int(row[source_column]) for row in rows]
    names = legacy_names(source)
    name_values = [names[index].encode("utf8") for index in indices]
    name_codes = np.zeros(
        (max(map(len, name_values)), len(name_values)), dtype=np.uint8
    )
    for j, name in enumerate(name_values):
        name_codes[: len(name), j] = np.frombuffer(name, dtype=np.uint8)
    with h5py.File(path, "x") as h:
        h.attrs["scope"] = (
            "matched raw D1-D3 feature recovery; original labels; development evaluation"
        )
        h.attrs["confirmed_legacy_grid_step_m"] = 0.001
        h["time_s"], h["depth_m"], h["file_names"] = TIME, RANGE, name_codes
        for feature in FINGERPRINTS:
            h.create_dataset(
                feature,
                data=np.stack(
                    [reconstructed[row["key"]][feature].T for row in rows], -1
                ),
                compression="lzf",
            )
        for name in LABELS:
            h.create_dataset(
                name,
                data=np.stack([source[name][..., i] for i in indices], -1),
                compression="lzf",
            )


def replay(weights, data, indices, profiles, *, echo_path=None):
    loaded, bundle = load_model(weights, "cpu")
    if echo_path is not None:
        loaded.core.config = replace(loaded.core.config, echo_path=echo_path).validate()
    output = {}
    for name, override in profiles.items():
        current = dict(data)
        if override is not None:
            for target, key in (("phase", "phase"), ("scatter", "E"), ("tof", "tau")):
                current[target] = override[key]
            current["scatter_valid"] = current["scatter"] > 0
        meter = MineMetrics()
        info = asyncio.run(
            evaluate_async(loaded, bundle, current, indices, "cpu", (meter,))
        )
        output[name] = {"metrics": meter.summary(), "replay": info}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--deployment-report", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf8"))
    reported = json.loads(args.deployment_report.read_text(encoding="utf8"))
    match_summary = json.loads(
        (args.match_dir / "match_summary.json").read_text(encoding="utf8")
    )
    date = match_summary["legacy_date"]
    with (args.match_dir / "matches.csv").open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matched = [row for row in rows if row["status"] == "unique_identifier"]
    groups = manifest["deployment_groups"]
    parts = {row["key"]: groups[protocol_key(row, date)] for row in matched}
    fit = [key for key, split in parts.items() if split >= 2]
    if not fit or not any(split == 0 for split in parts.values()):
        raise ValueError("Need fitting and group-held-out records.")
    native = {
        row["key"]: finite_native(args.native_dir / (row["key"] + ".mat"))
        for row in rows
        if row["raw_filename"]
    }
    train_path = args.legacy_dir / "train.h5"
    test_path = args.legacy_dir / "test_opf.h5"
    for path in (train_path, test_path):
        digest = sha256_file(path)
        if (
            digest != manifest["source_sha256"][path.name]
            or digest != match_summary["source_sha256"][path.name]
        ):
            raise ValueError(f"Source hash mismatch for {path.name}.")
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(train_path) as train, h5py.File(test_path) as test:
        reference = {}
        for row in matched:
            index = int(row["test_row_zero_based"])
            source = test if index >= 0 else train
            if index < 0:
                index = int(row["train_row_zero_based"])
            reference[row["key"]] = {
                name: source[name][..., index].T for name in FINGERPRINTS
            }
        maps = {
            "low": select_projection(native, reference, fit, "tau"),
            "high": select_projection(native, reference, fit, "E"),
        }
        coefficients = fit_amplitude(native, reference, fit, maps)
        write_json(
            args.output / "shared_calibration.json",
            {
                "maps": maps,
                "amplitude_coefficients": coefficients,
                "fitting_keys": fit,
                "source_sha256": manifest["source_sha256"],
            },
        )
        baseline = {key: baseline_features(native[key]) for key in parts}
        aligned = {
            key: calibrated_features(record, maps, coefficients)
            for key, record in native.items()
        }
        train_rows = sorted(
            (row for row in matched if int(row["train_row_zero_based"]) >= 0),
            key=lambda row: int(row["train_row_zero_based"]),
        )
        test_rows = sorted(
            (row for row in matched if int(row["test_row_zero_based"]) >= 0),
            key=lambda row: int(row["test_row_zero_based"]),
        )
        write_mirror(
            args.output / "train_matched_recovered.h5",
            train_rows,
            train,
            aligned,
            "train_row_zero_based",
        )
        write_mirror(
            args.output / "test_opf_recovered.h5",
            test_rows,
            test,
            aligned,
            "test_row_zero_based",
        )
    with h5py.File(args.output / "all_raw_recovered_features.h5", "x") as h:
        h.attrs["scope"] = (
            "shared calibrated D1-D3 fingerprints without attached labels"
        )
        h["time_s"], h["depth_m"] = TIME, RANGE
        keys = list(aligned)
        h.create_dataset("key", data=keys, dtype=h5py.string_dtype("utf-8"))
        for feature in FINGERPRINTS:
            h.create_dataset(
                feature,
                data=np.stack([aligned[key][feature].T for key in keys], -1),
                compression="lzf",
            )
    comparisons = []
    for key in parts:
        item = {
            "key": key,
            "split": (
                "holdout"
                if parts[key] == 0
                else "validation" if parts[key] == 1 else "fit"
            ),
        }
        for variant, dataset in (
            ("baseline", baseline),
            ("shared_calibrated", aligned),
        ):
            for feature in FINGERPRINTS:
                value, truth = dataset[key][feature], reference[key][feature]
                item[f"{variant}_{feature}_pearson"] = correlation(value, truth)
                item[f"{variant}_{feature}_relative_rmse"] = relative_rmse(value, truth)
        comparisons.append(item)
    with (args.output / "feature_similarity.csv").open(
        "w", encoding="utf8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    similarity = {}
    for split in ("fit", "validation", "holdout"):
        subset = [item for item in comparisons if item["split"] == split]
        similarity[split] = {
            "records": len(subset),
            **{
                key: float(np.mean([row[key] for row in subset]))
                for key in subset[0]
                if key not in ("key", "split")
            },
        }
    old = read_legacy(test_path, grid_step_m=0.001, require_axes=True)
    old_arrays = supervised_arrays(old)
    indices = [
        i
        for i, row in enumerate(test_rows)
        if row["test_filename"] in reported["held_out_record_names"]
    ]
    if len(indices) != reported["held_out_records"]:
        raise ValueError("Recovered test rows do not reproduce the deployment holdout.")
    profiles = {"legacy": None}
    for variant, dataset in (("baseline", baseline), ("shared_calibrated", aligned)):
        profiles[variant] = {
            feature: np.stack([dataset[row["key"]][feature] for row in test_rows])
            for feature in FINGERPRINTS
        }
    network = {
        name: replay(path, old_arrays, indices, profiles)
        for name, path in (("fp32", args.fp32), ("int8", args.int8))
    }
    summary = {
        "matched_records": len(matched),
        "raw_features_recovered": len(aligned),
        "similarity": similarity,
        "network_held_out_records": len(indices),
        "network_held_out_frames": len(indices) * 10,
        "network": network,
        "target_percent": TARGET,
    }
    write_json(args.output / "recovery_report.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
