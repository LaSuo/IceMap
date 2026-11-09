"""Prepare portable OPF labels and a raw input manifest for review."""

import argparse
import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np

from match_opf_records import LABELS, RAW_PATTERN


LABEL_FORMAT = "icemap.opf.review.labels.v1"
CALIBRATION_FORMAT = "icemap.opf.review.calibration.v1"
RECORD_KEY = re.compile(r"^(e\d+r\d+_t\d+)b\d+_sub\d+$")
TEXT = h5py.string_dtype("utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_label_index(labels):
    if labels.attrs.get("format") != LABEL_FORMAT:
        raise ValueError("Unsupported OPF label archive.")
    keys = list(labels["key"].asstr()[:])
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("The label archive needs unique recording keys.")
    structures = list(labels["structure"].asstr()[:])
    for key, group in zip(keys, structures, strict=True):
        match = RECORD_KEY.fullmatch(key)
        if match is None or match[1] != group:
            raise ValueError(f"Inconsistent recording key or structure for {key!r}.")
    roles = labels["split"][:]
    test_rows = labels["original_test_row"][:]
    if roles.shape != (len(keys),) or not np.isin(roles, (0, 1, 2)).all():
        raise ValueError("Invalid OPF split codes.")
    if test_rows.shape != roles.shape or not np.issubdtype(test_rows.dtype, np.integer):
        raise ValueError("Invalid original test record index.")
    if (test_rows < -1).any() or len(set(test_rows[test_rows >= 0])) != sum(
        test_rows >= 0
    ):
        raise ValueError("Invalid or duplicate original test record index.")
    structure_roles = {}
    for structure, role in zip(structures, roles, strict=True):
        if structure_roles.setdefault(structure, role) != role:
            raise ValueError(f"Structure appears in more than one split: {structure}")
    if any(labels[field].shape[-1] != len(keys) for field in LABELS):
        raise ValueError("An OPF label record axis does not match the key table.")
    return keys, structures, roles, test_rows


def make_manifest(raw_dir, label_path, output, *, include_unlabeled=False):
    """Index raw acquisition keys without decoding the specimen suffix."""
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    with h5py.File(label_path, "r") as labels:
        keys, _, _, _ = read_label_index(labels)
    labeled = set(keys)
    raw = {}
    unlabeled = 0
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".h5":
            continue
        match = RAW_PATTERN.match(path.stem)
        if match is None and include_unlabeled:
            raise ValueError(f"Cannot derive a recording key from {path.name}.")
        key = f"{match[1].lower()}_sub{int(match[2])}" if match else None
        if key not in labeled:
            unlabeled += 1
            if not include_unlabeled:
                continue
        if key in raw or "|" in path.name:
            raise ValueError(f"Ambiguous raw recording for {key}.")
        raw[key] = path.name
    missing = sorted(labeled - raw.keys())
    if missing:
        raise ValueError(
            f"Missing {len(missing)} labeled raw recordings: {missing[:5]}."
        )
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("key", "raw_filename", "status"))
        for key in sorted(raw):
            status = "unique_identifier" if key in labeled else "raw_only"
            writer.writerow((key, raw[key], status))
    return {"matched": len(keys), "unlabeled_raw": unlabeled, "scheduled": len(raw)}


def package_assets(args):
    """Package exact legacy labels and independently fitted inference settings."""
    match_summary = read_json(args.match_dir / "match_summary.json")
    split = read_json(args.split_manifest)
    deploy = read_json(args.deployment_report)
    ordered = read_json(args.ordered_report)
    fallback = read_json(args.fallback_calibration)
    if not (
        match_summary["source_sha256"]
        == split["source_sha256"]
        == deploy["source_sha256"]
        == fallback["source_sha256"]
    ):
        raise ValueError("The source hashes of the calibration inputs differ.")
    if ordered["selected_scattering"] != "registered_scatter":
        raise ValueError("This review pack expects surface-registered scattering.")
    if ordered["controlled_thickness_prior"] is None:
        raise ValueError("The controlled-specimen calibration is missing.")
    calibration = {
        "format": CALIBRATION_FORMAT,
        "raw": {
            "low_step_m": 0.001,
            "high_step_m": 0.001,
            "grid_step_m": 0.001,
            "chirp_stride": 8,
            "tof_min_peak_distance_bins": 20,
            "design_versions": ["v2", "v3", "v6"],
        },
        "echo_calibration": ordered["calibration"],
        "controlled_thickness_prior": ordered["controlled_thickness_prior"],
        "echo_path_settings": ordered["echo_path_settings"],
        "scattering_gain": ordered["scattering_gain"],
        "fallback_maps": fallback["maps"],
        "fallback_coefficients": fallback["amplitude_coefficients"],
        "source_sha256": match_summary["source_sha256"],
        "weight_sha256": {
            "fp32": deploy["fp32"]["checkpoint_sha256"],
            "int8": deploy["rpi_int8"]["checkpoint_sha256"],
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    text = ("key", "structure")
    with h5py.File(args.match_dir / "labels_matched.h5", "r") as source:
        if source.attrs.get("format") != "icemap.opf.legacy_labels.v1":
            raise ValueError("Unsupported source label archive.")
        keys = list(source["key"].asstr()[:])
        structures = [RECORD_KEY.fullmatch(key)[1] for key in keys]
        roles = [
            split["deployment_groups"][f'{match_summary["legacy_date"]}_{s}']
            for s in structures
        ]
        test_rows = source["test_row_zero_based"][:]
        selected_names = {
            name
            for name, role, row in zip(
                source["test_filename"].asstr()[:], roles, test_rows, strict=True
            )
            if role == 0 and row >= 0
        }
        if selected_names != set(deploy["held_out_record_names"]):
            raise ValueError("The holdout does not match the deployment report.")
        fitting_keys = {key for key, role in zip(keys, roles, strict=True) if role == 2}
        if set(fallback["fitting_keys"]) != fitting_keys:
            raise ValueError(
                "Fallback calibration and fitting labels have different groups."
            )
        if len(keys) != match_summary["unique_raw_labels_extracted"]:
            raise ValueError(
                "The source label count disagrees with the matching summary."
            )
        with h5py.File(args.output / "opf_labels.h5", "x") as output:
            output.attrs["format"] = LABEL_FORMAT
            output.attrs["split_codes"] = "0=holdout, 1=validation, 2=fit"
            output.attrs["label3_bin_step_m"] = 0.001
            output.attrs["physical_truth"] = (
                "contiguous positive density runs in label3"
            )
            output.attrs["source_sha256"] = json.dumps(
                match_summary["source_sha256"], sort_keys=True
            )
            for field in LABELS:
                output.create_dataset(
                    field, data=source[field][:], compression="gzip", compression_opts=4
                )
            for field, values in zip(text, (keys, structures), strict=True):
                output.create_dataset(field, data=values, dtype=TEXT)
            output["split"] = np.array(roles, dtype=np.int8)
            output["original_test_row"] = test_rows
            read_label_index(output)
    (args.output / "opf_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    for option in (
        "match-dir",
        "split-manifest",
        "deployment-report",
        "ordered-report",
        "fallback-calibration",
        "output",
    ):
        package.add_argument(f"--{option}", type=Path, required=True)
    manifest = sub.add_parser("manifest")
    for option in ("raw-dir", "labels", "output"):
        manifest.add_argument(f"--{option}", type=Path, required=True)
    manifest.add_argument(
        "--all-raw", action="store_true", help="Schedule unlabeled raw recordings too."
    )
    args = parser.parse_args()
    if args.command == "package":
        package_assets(args)
    else:
        result = make_manifest(
            args.raw_dir, args.labels, args.output, include_unlabeled=args.all_raw
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
