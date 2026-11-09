"""Build a labeled D4 dataset from native OPF fingerprints."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from align_opf_echo_paths import observed_features
from align_opf_features import (
    GRID,
    RANGE,
    TIME,
    calibrated_features,
    finite_native,
)
from match_opf_records import FINGERPRINTS, LABELS
from opf_band_paths import BandPathOptions, EchoPathFit, decode_paths, median_bands
from opf_echo_witness import EchoWitnessOptions, decode_with_echo_witness
from opf_thickness_prior import ThicknessPrior, apply_thickness_prior
from prepare_opf_review import CALIBRATION_FORMAT, TEXT, read_json, read_label_index


DATASET_FORMAT = "icemap.opf.review.dataset.v1"
C0_M_S = 299792458.0
DIELECTRIC_COEFFICIENT = 0.845


def load_calibration(path):
    cfg = read_json(path)
    if cfg.get("format") != CALIBRATION_FORMAT:
        raise ValueError("Unsupported OPF calibration format.")
    if cfg["raw"]["design_versions"] != ["v2", "v3", "v6"]:
        raise ValueError("This dataset builder expects D1 v2, D2 v3, and D3 v6.")
    gain = cfg["scattering_gain"]
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("The scattering gain must be finite and positive.")
    raster = cfg.get("tof_raster", {})
    if not set(raster) <= {"tof_width_bins", "tof_gain"} or any(
        not np.isfinite(value) or value <= 0 for value in raster.values()
    ):
        raise ValueError("The ToF raster width and gain must be finite and positive.")
    return cfg


def native_observation(path, raw):
    fields = (
        "raw_low_envelope",
        "raw_high_envelope",
        "shift_bins",
        "key",
        "design1_version",
        "design2_version",
        "design3_version",
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing D1-D3 fingerprint: {path}")
    record = finite_native(path)
    extra = loadmat(path, squeeze_me=True, variable_names=fields)
    if not set(fields) <= extra.keys():
        raise ValueError(f"{path.name} is missing native v6 observation fields.")
    versions = [str(extra[f"design{i}_version"]) for i in (1, 2, 3)]
    if versions != raw["design_versions"] or str(extra["key"]) != path.stem:
        raise ValueError(f"{path.name} does not match the requested key and versions.")
    dt = np.diff(record["time_s"])
    step_m = float(dt[0] * C0_M_S / 2)
    if (
        step_m <= 0
        or not np.isclose(step_m, raw["grid_step_m"], rtol=1e-6, atol=0)
        or not np.allclose(dt, dt[0], rtol=1e-6, atol=0)
    ):
        raise ValueError(f"{path.name} does not use the calibrated range step.")
    shift = extra["shift_bins"]
    shape = record["phase"].shape
    low, high = extra["raw_low_envelope"], extra["raw_high_envelope"]
    if shift.shape != (shape[0],) or not np.isfinite(shift).all():
        raise ValueError(f"{path.name} has an invalid motion shift track.")
    if low.shape != shape or high.shape != shape:
        raise ValueError(
            f"{path.name} raw envelopes do not share the fingerprint grid."
        )
    bands = median_bands(low, high, step_m)
    return record, bands, (float(np.median(shift)), 0.001 / step_m)


def build_dataset(
    native_dir, label_path, calibration_path, destination, *, selection="original-test"
):
    """Compute every feature from observations, then attach labels by key."""
    cfg = load_calibration(calibration_path)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + ".partial")
    with h5py.File(label_path, "r") as labels:
        keys, structures, roles, test_rows = read_label_index(labels)
        if read_source_hashes(labels) != cfg["source_sha256"]:
            raise ValueError("Label and calibration provenance do not agree.")
        if not np.isclose(labels.attrs["label3_bin_step_m"], 0.001):
            raise ValueError("This OPF release expects 1 mm physical label bins.")
        expected_shape = (len(GRID), 10, len(keys))
        if any(labels[name].shape != expected_shape for name in ("label1", "label3")):
            raise ValueError("Boundary and profile labels must share the D4 grid.")
        if selection == "original-test":
            chosen = np.flatnonzero(test_rows >= 0)
            chosen = chosen[np.argsort(test_rows[chosen])]
            if not len(chosen) or not np.array_equal(
                test_rows[chosen], np.arange(len(chosen))
            ):
                raise ValueError("The original test record index must be complete.")
        elif selection == "matched":
            chosen = np.arange(len(keys))
        else:
            raise ValueError(f"Unsupported OPF dataset selection {selection!r}.")
        keys = [keys[i] for i in chosen]
        structures = [structures[i] for i in chosen]
        roles, test_rows = roles[chosen], test_rows[chosen]
        observations = [
            native_observation(native_dir / f"{key}.mat", cfg["raw"]) for key in keys
        ]
        path_parameters = dict(cfg["echo_calibration"])
        path_parameters["options"] = BandPathOptions(**path_parameters["options"])
        path_fit = EchoPathFit(**path_parameters)
        bands = [observation[1] for observation in observations]
        prior = ThicknessPrior(**cfg["controlled_thickness_prior"])
        witness_settings = cfg.get("echo_witness")
        if witness_settings is None:
            event_bins, measured, _, valid = decode_paths(bands, path_fit)
            measured[~valid] = 0
            delays = apply_thickness_prior(measured, prior, DIELECTRIC_COEFFICIENT)
        else:
            options = EchoWitnessOptions(**witness_settings)
            event_bins, measured, delays, valid = decode_with_echo_witness(
                bands, path_fit, prior, options, DIELECTRIC_COEFFICIENT
            )
        features = []
        for i, (record, _, shift) in enumerate(observations):
            if valid[i]:
                base, scatter, _ = observed_features(
                    record,
                    event_bins[i],
                    delays[i],
                    shift,
                    None,
                    **cfg.get("tof_raster", {}),
                )
                item = dict(base, E=scatter * cfg["scattering_gain"])
            else:
                item = calibrated_features(
                    record, cfg["fallback_maps"], cfg["fallback_coefficients"]
                )
            if any(
                item[name].shape != (10, len(GRID)) or not np.isfinite(item[name]).all()
                for name in FINGERPRINTS
            ):
                raise ValueError(f"{keys[i]} produced an invalid D4 fingerprint.")
            features.append(item)
        with h5py.File(partial, "x") as h:
            h.attrs["format"] = DATASET_FORMAT
            h.attrs["scope"] = (
                "controlled OPF observation reconstruction with fixed calibration"
            )
            h.attrs["label3_bin_step_m"] = labels.attrs["label3_bin_step_m"]
            h.attrs["source_sha256"] = labels.attrs["source_sha256"]
            h.attrs["split_codes"] = labels.attrs["split_codes"]
            h.attrs["selection"] = selection
            if witness_settings is not None:
                h.attrs["echo_witness"] = json.dumps(witness_settings, sort_keys=True)
            h.attrs["review_evaluation"] = json.dumps(
                {
                    "design_versions": cfg["raw"]["design_versions"],
                    "source_sha256": cfg["source_sha256"],
                    "weight_sha256": cfg["weight_sha256"],
                    "echo_path_settings": cfg["echo_path_settings"],
                },
                sort_keys=True,
            )
            if "tof_raster" in cfg:
                h.attrs["tof_raster"] = json.dumps(cfg["tof_raster"], sort_keys=True)
            h["time_s"], h["depth_m"] = TIME, RANGE
            for name, values in (
                ("key", keys),
                ("file_names", keys),
                ("structure", structures),
            ):
                h.create_dataset(name, data=values, dtype=TEXT)
            h["split"] = roles
            h["original_test_row"] = test_rows
            h["ordered_echo_available"] = valid
            h["observed_path_bins_mm"] = np.where(valid[:, None], event_bins, -1)
            h["measured_delays_mm"] = measured
            h["calibrated_delays_mm"] = delays
            for name in FINGERPRINTS:
                h.create_dataset(
                    name,
                    data=np.stack([item[name].T for item in features], axis=-1),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                )
            for name in LABELS:
                h.create_dataset(
                    name,
                    data=labels[name][:][..., chosen],
                    compression="gzip",
                    compression_opts=4,
                )
    partial.rename(destination)
    return {
        "selection": selection,
        "records": len(keys),
        "frames": len(keys) * 10,
        "ordered_echo_records": int(valid.sum()),
        "original_test_records": int((test_rows >= 0).sum()),
        "deployment_holdout_records": int(((roles == 0) & (test_rows >= 0)).sum()),
    }


def read_source_hashes(labels):
    return json.loads(labels.attrs["source_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("native-dir", "labels", "calibration", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(
        "--selection", choices=("original-test", "matched"), default="original-test"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.native_dir,
                args.labels,
                args.calibration,
                args.output,
                selection=args.selection,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
