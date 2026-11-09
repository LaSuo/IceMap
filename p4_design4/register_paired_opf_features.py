"""Reference-assisted registration for matched OPF windows.

This diagnostic compares paired fingerprints and uses the reference observation.
The standalone raw-data evaluation uses run_opf_review.py.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.signal import fftconvolve, find_peaks

from align_opf_features import (
    GRID,
    SURFACE,
    correlation,
    finite_native,
    projection,
    relative_rmse,
    replay,
    write_mirror,
)
from match_opf_records import FINGERPRINTS
from protocol import sha256_file
from retraining_data import read_legacy, supervised_arrays
from train_opf_cv import write_json


def log_signal(value):
    return np.log1p(value / max(1e-6, float(np.quantile(value, 0.7))))


def alignment(signals, target, start, stop, transform):
    """Choose a translation and scale by normalized paired signal correlation."""
    best = None
    y = transform(np.median(target, axis=0))[start:stop].astype(np.float64)
    for key, value in signals.items():
        trace = transform(np.median(value, axis=0)).astype(np.float64)
        sums = np.r_[0, np.cumsum(trace)]
        squares = np.r_[0, np.cumsum(trace**2)]
        for scale in np.arange(0.55, 1.86, 0.05):
            size = round((len(y) - 1) * scale) + 1
            if size >= len(trace):
                continue
            template = np.interp(np.arange(size) / scale, np.arange(len(y)), y)
            template -= template.mean()
            norm = np.linalg.norm(template)
            if norm < 1e-9:
                continue
            cov = fftconvolve(trace, template[::-1], mode="valid")
            var = (
                squares[size:]
                - squares[:-size]
                - (sums[size:] - sums[:-size]) ** 2 / size
            )
            score = cov / (norm * np.sqrt(np.maximum(var, 1e-12)))
            anchors = np.arange(len(score)) + (SURFACE - start) * scale
            score[(anchors < 520) | (anchors > 1470)] = -np.inf
            i = int(score.argmax())
            value = float(score[i] - 0.015 * abs(np.log(scale)))
            if best is None or value > best[0]:
                best = value, key, float(anchors[i]), float(scale)
    if best is None or not np.isfinite(best[0]):
        raise ValueError("No finite paired alignment was possible.")
    score, key, origin, scale = best
    return {
        "source": key,
        "origin": origin,
        "scale": scale,
        "score": score,
        "warped": projection(signals[key], origin, scale, 0),
    }


def measured_pulses(observed, minimum_bin, maximum_bin, count, distance):
    median = np.median(observed, axis=0)
    indices, _ = find_peaks(median, distance=distance)
    indices = indices[(indices >= minimum_bin) & (indices <= maximum_bin)]
    return indices[np.argsort(median[indices])[-count:]]


def pulse_response(centers, sigma):
    return np.exp(-0.5 * ((GRID[:, None] - centers[None, :]) / sigma) ** 2)


def frame_gain(observed, indices):
    smoothed = gaussian_filter1d(observed, 4, axis=-1)
    gain = smoothed[:, indices]
    return (gain / np.maximum(np.median(gain, axis=0, keepdims=True), 1e-6)).clip(
        0.55, 1.65
    )


def reconstruct_tau(record, old, radius):
    reference = np.median(old, axis=0)
    support = np.flatnonzero(reference > 0.002 * max(reference.max(), 1e-6))
    if len(support) < 5:
        raise ValueError("Legacy ToF reference lacks a usable peak window.")
    start = max(0, int(support[0]) - 35)
    stop = min(len(GRID), int(support[-1]) + 36)
    signals = {key: record[key] for key in ("tau", "low_envelope", "raw_low_envelope")}
    fit = alignment(
        signals, old, start, stop, lambda x: gaussian_filter1d(log_signal(x), 4)
    )
    observed = fit["warped"]
    peak_source = gaussian_filter1d(observed, 5, axis=-1)
    centers = measured_pulses(peak_source, start, stop, 34, 13)
    if not len(centers):
        return observed, fit, centers
    mapped = centers.copy()
    if radius:
        reference_peaks, _ = find_peaks(
            reference, distance=14, prominence=0.006 * reference.max()
        )
        available = list(range(len(centers)))
        landmarks = []
        for peak in reference_peaks[np.argsort(reference[reference_peaks])[::-1]]:
            if not available:
                break
            j = min(available, key=lambda index: abs(centers[index] - peak))
            if abs(centers[j] - peak) <= radius:
                available.remove(j)
                mapped[j] = peak
                landmarks.append((int(centers[j]), int(peak)))
        fit["tof_landmarks_native_projection_to_reference"] = landmarks
    best = None
    for sigma in (7, 11, 15, 20, 27):
        design = pulse_response(mapped, sigma)
        weights, _ = nnls(design, reference)
        candidate = (frame_gain(peak_source, centers) * weights) @ design.T
        error = relative_rmse(candidate, old)
        if best is None or error < best[0]:
            best = (
                error,
                candidate.astype(np.float32),
                sigma,
                mapped[weights > 0.018 * max(weights.max(), 1e-8)],
            )
    fit["pulse_sigma_bins"] = best[2]
    return best[1], fit, best[3]


def reconstruct_e(record, old):
    signals = {key: record[key] for key in ("E", "high_envelope", "raw_high_envelope")}
    fit = alignment(
        signals,
        old,
        215,
        1880,
        lambda x: gaussian_filter1d(log_signal(x), 5),
    )
    observed = np.stack([log_signal(frame) for frame in fit["warped"]])
    blurred = gaussian_filter1d(observed, 8, axis=-1)
    ones = np.ones_like(observed)
    x = np.stack((ones, observed, blurred, ones * GRID / len(GRID)), -1)
    design = x[:, SURFACE:, :].reshape(-1, x.shape[-1])
    coefficients = np.linalg.lstsq(design, old[:, SURFACE:].reshape(-1), rcond=None)[0]
    values = (x @ coefficients).clip(0).astype(np.float32)
    values[:, :SURFACE] = 0
    return values, fit


def reconstruct_phase(record, old, centers, radius):
    signals = {
        "phase_magnitude": abs(record["phase"]),
        "high_envelope": record["high_envelope"],
        "phase_coherence": record["phase_coherence"],
    }

    def contrast(x):
        return gaussian_filter1d(x, 3) - gaussian_filter1d(x, 32)

    fit = alignment(signals, old, 100, 1800, contrast)
    observed = fit["warped"]
    x = gaussian_filter1d(observed, 3, axis=-1)
    x = np.maximum(x - gaussian_filter1d(observed, 28, axis=-1), 0)
    high_centers = measured_pulses(x, 20, len(GRID) - 20, 45, 14)
    centers = np.unique(np.r_[high_centers, centers, SURFACE]).astype(int)
    if radius:
        reference = np.median(old, axis=0)
        prominence = gaussian_filter1d(reference, 2) - gaussian_filter1d(reference, 16)
        reference_peaks, _ = find_peaks(
            prominence, distance=15, height=max(0.065, 0.35 * prominence.max())
        )
        available = list(centers)
        registered = []
        for peak in reference_peaks[np.argsort(prominence[reference_peaks])[::-1]]:
            if not available:
                break
            closest = min(available, key=lambda bin_index: abs(bin_index - peak))
            if abs(closest - peak) <= radius:
                available.remove(closest)
                registered.append((int(closest), int(peak)))
        centers = np.unique(np.r_[available, [new for _, new in registered]]).astype(
            int
        )
        fit["phase_landmarks_native_projection_to_reference"] = registered
    design = pulse_response(centers, 6)
    position = GRID / len(GRID)
    depth = np.stack((np.ones_like(GRID), position, position**2), -1)
    matrix = np.column_stack((depth, design))
    reg = np.diag(np.r_[np.zeros(depth.shape[1]), np.full(len(centers), 0.7)])
    mean = np.median(old, axis=0)
    coefficients = np.linalg.solve(matrix.T @ matrix + reg, matrix.T @ mean)
    median = matrix @ coefficients
    # Preserve the observed frame variation instead of copying reference frames.
    variability = x - np.median(x, axis=0)
    rescale = float(np.std(old) / max(np.std(variability), 1e-7))
    output = (
        (median + min(0.22 * rescale, 0.18) * variability).clip(0).astype(np.float32)
    )
    return output, fit


def reconstruct_pair(record, reference, shared, phase_radius, tof_radius):
    tof, timing_fit, peaks = reconstruct_tau(record, reference["tau"], tof_radius)
    scatter, scatter_fit = reconstruct_e(record, reference["E"])
    phase, phase_fit = reconstruct_phase(
        record, reference["phase"], peaks, phase_radius
    )
    output = {"phase": phase, "E": scatter, "tau": tof}
    stats = {"phase": phase_fit, "E": scatter_fit, "tau": timing_fit}
    for key in FINGERPRINTS:
        stats[key].pop("warped")
        if relative_rmse(output[key], reference[key]) > relative_rmse(
            shared[key], reference[key]
        ):
            output[key] = shared[key]
            stats[key]["selected"] = "shared"
        else:
            stats[key]["selected"] = "paired"
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--deployment-report", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-landmark-radius-bins", type=int, default=22)
    parser.add_argument("--tof-landmark-radius-bins", type=int, default=0)
    args = parser.parse_args()
    if not all(
        0 <= radius <= 50
        for radius in (args.phase_landmark_radius_bins, args.tof_landmark_radius_bins)
    ):
        parser.error("Landmark radii must be between 0 and 50 bins.")
    torch.set_num_threads(4)
    with (args.match_dir / "matches.csv").open(encoding="utf8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "unique_identifier"]
    train_rows = sorted(
        (r for r in rows if int(r["train_row_zero_based"]) >= 0),
        key=lambda r: int(r["train_row_zero_based"]),
    )
    test_rows = sorted(
        (r for r in rows if int(r["test_row_zero_based"]) >= 0),
        key=lambda r: int(r["test_row_zero_based"]),
    )
    train_path, test_path = (
        args.legacy_dir / "train.h5",
        args.legacy_dir / "test_opf.h5",
    )
    match_summary = json.loads(
        (args.match_dir / "match_summary.json").read_text(encoding="utf8")
    )
    for path in (train_path, test_path):
        if sha256_file(path) != match_summary["source_sha256"][path.name]:
            raise ValueError(f"Source hash mismatch for {path.name}.")
    args.output.mkdir(parents=True, exist_ok=False)
    recovered, settings, comparisons = {}, {}, []
    with (
        h5py.File(train_path) as train,
        h5py.File(test_path) as test,
        h5py.File(args.shared_dir / "all_raw_recovered_features.h5") as shared,
    ):
        lookup = {name: i for i, name in enumerate(shared["key"].asstr()[...])}
        for row in rows:
            key = row["key"]
            index = int(row["test_row_zero_based"])
            source = test if index >= 0 else train
            if index < 0:
                index = int(row["train_row_zero_based"])
            reference = {name: source[name][..., index].T for name in FINGERPRINTS}
            baseline = {name: shared[name][..., lookup[key]].T for name in FINGERPRINTS}
            record = finite_native(args.native_dir / (key + ".mat"))
            record["raw_low_envelope"] = load_raw_envelope(
                args.native_dir, key, "raw_low_envelope"
            )
            record["raw_high_envelope"] = load_raw_envelope(
                args.native_dir, key, "raw_high_envelope"
            )
            recovered[key], settings[key] = reconstruct_pair(
                record,
                reference,
                baseline,
                args.phase_landmark_radius_bins,
                args.tof_landmark_radius_bins,
            )
            item = {"key": key}
            for version, values in (("shared", baseline), ("paired", recovered[key])):
                for name in FINGERPRINTS:
                    item[f"{version}_{name}_pearson"] = correlation(
                        values[name], reference[name]
                    )
                    item[f"{version}_{name}_relative_rmse"] = relative_rmse(
                        values[name], reference[name]
                    )
            comparisons.append(item)
            if len(comparisons) % 15 == 0:
                print(
                    f"Registered {len(comparisons)} / {len(rows)} paired recordings.",
                    flush=True,
                )
        for name, items, original, column in (
            ("train_matched_paired.h5", train_rows, train, "train_row_zero_based"),
            ("test_opf_paired.h5", test_rows, test, "test_row_zero_based"),
        ):
            path = args.output / name
            write_mirror(path, items, original, recovered, column)
            with h5py.File(path, "r+") as output:
                output.attrs["scope"] = (
                    "reference-assisted per-record alignment; not an independent reconstruction"
                )
    with (args.output / "paired_similarity.csv").open(
        "w", newline="", encoding="utf8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    write_json(args.output / "paired_alignment_settings.json", settings)
    original = read_legacy(test_path, grid_step_m=0.001, require_axes=True)
    data = supervised_arrays(original)
    prior = json.loads(args.deployment_report.read_text(encoding="utf8"))
    indices = [
        i
        for i, row in enumerate(test_rows)
        if row["test_filename"] in prior["held_out_record_names"]
    ]
    if len(indices) != prior["held_out_records"]:
        raise ValueError("Paired observations do not reproduce the deployment holdout.")
    profiles = {
        "paired": {
            name: np.stack([recovered[row["key"]][name] for row in test_rows])
            for name in FINGERPRINTS
        }
    }
    network = {
        name: replay(path, data, indices, profiles)
        for name, path in (("fp32", args.fp32), ("int8", args.int8))
    }
    report = {
        "scope": "Each alignment used its own legacy fingerprints. Inference is retrospective.",
        "records": len(rows),
        "network_records": len(indices),
        "network_frames": len(indices) * 10,
        "phase_landmark_radius_bins": args.phase_landmark_radius_bins,
        "tof_landmark_radius_bins": args.tof_landmark_radius_bins,
        "mean_similarity": {
            k: float(np.mean([row[k] for row in comparisons]))
            for k in comparisons[0]
            if k != "key"
        },
        "network": network,
    }
    write_json(args.output / "paired_report.json", report)
    print(json.dumps(report, indent=2))


def load_raw_envelope(directory, key, field):
    from scipy.io import loadmat

    return loadmat(directory / (key + ".mat"), variable_names=[field])[field]


if __name__ == "__main__":
    main()
