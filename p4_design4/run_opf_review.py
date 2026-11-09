"""Run the complete OPF review pipeline or evaluate an existing OPF dataset."""

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import h5py
import numpy as np
import torch

from build_opf_dataset import (
    DATASET_FORMAT,
    DIELECTRIC_COEFFICIENT,
    build_dataset,
    load_calibration,
    read_source_hashes,
)
from evaluate_quantized_opf import evaluate_async
from inference import load_model
from metrics import MineMetrics
from prepare_opf_review import make_manifest
from protocol import sha256_file
from retraining_data import read_legacy, supervised_arrays
from train_opf_cv import TARGET


MODEL_FILES = {"fp32": "spai_fp32.pt", "int8": "spai_rpi_int8.pt"}


def matlab_path(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def extract_features(matlab, raw_dir, manifest, native_dir, raw, log_path):
    root = Path(__file__).resolve().parents[1]
    versions = raw["design_versions"]
    grid = {
        "LowStepM": raw["low_step_m"],
        "HighStepM": raw["high_step_m"],
        "GridStepM": raw["grid_step_m"],
        "ChirpStride": raw["chirp_stride"],
        "MinPeakDistance": raw["tof_min_peak_distance_bins"],
    }
    for key, value in grid.items():
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid MATLAB setting {key}.")
    for key in ("ChirpStride", "MinPeakDistance"):
        if grid[key] != int(grid[key]):
            raise ValueError(f"MATLAB setting {key} must be integral.")
    command = (
        f"addpath({matlab_path(root)}); "
        f"processing=struct('design1_version','{versions[0]}',"
        f"'design2_version','{versions[1]}','design3_version','{versions[2]}',"
        f"'tof_min_peak_distance_bins',{grid['MinPeakDistance']}); "
        f"opts=struct('LowStepM',{grid['LowStepM']},'HighStepM',{grid['HighStepM']},"
        f"'GridStepM',{grid['GridStepM']},'ChirpStride',{grid['ChirpStride']},"
        "'MatchedOnly',false,'Processing',processing); "
        f"run_opf_raw_features({matlab_path(raw_dir)},{matlab_path(manifest)},"
        f"{matlab_path(native_dir)},opts);"
    )
    print(f"Running D1-D3 in MATLAB. Log: {log_path}", flush=True)
    with log_path.open("xb") as log:
        result = subprocess.run(
            [matlab, "-batch", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"MATLAB exited with status {result.returncode}. See {log_path}."
        )


def evaluate(dataset, assets, scope, model_paths):
    with h5py.File(dataset, "r") as file:
        if file.attrs.get("format") != DATASET_FORMAT:
            raise ValueError("Unsupported OPF review dataset.")
        if "review_evaluation" in file.attrs:
            cfg = json.loads(file.attrs["review_evaluation"])
        else:
            cfg = load_calibration(assets / "opf_calibration.json")
            cfg["design_versions"] = cfg["raw"]["design_versions"]
        if read_source_hashes(file) != cfg["source_sha256"]:
            raise ValueError("Dataset and calibration provenance do not agree.")
        roles = file["split"][:]
        original = file["original_test_row"][:] >= 0
        keys = file["key"].asstr()[:]
        if roles.shape != keys.shape or original.shape != keys.shape:
            raise ValueError("Invalid evaluation split in the OPF dataset.")
        if not np.isin(roles, (0, 1, 2)).all():
            raise ValueError("Invalid OPF split codes in the dataset.")
    candidates = {
        "structure_holdout": np.flatnonzero((roles == 0) & original).tolist(),
        "original_test_mixed_splits": np.flatnonzero(original).tolist(),
    }
    selected = (
        candidates
        if scope == "both"
        else {key: value for key, value in candidates.items() if key.startswith(scope)}
    )
    if not selected or any(not x for x in selected.values()):
        raise ValueError("No OPF records in the selected evaluation split.")
    input_data = read_legacy(dataset, grid_step_m=0.001, require_axes=True)
    if not np.array_equal(input_data["names"], keys):
        raise ValueError("D4 data order differs from the recording key table.")
    data = supervised_arrays(input_data)
    results = {
        key: {
            "records": len(indices),
            "frames": len(indices) * data["phase"].shape[1],
            "record_keys": keys[indices].tolist(),
            "models": {},
        }
        for key, indices in selected.items()
    }
    for name, weights in model_paths.items():
        if sha256_file(weights) != cfg["weight_sha256"][name]:
            raise ValueError(f"Weight hash mismatch for {weights.name}.")
        model, bundle = load_model(weights, "cpu")
        original_config = model.core.config
        if original_config.dielectric_density_coefficient != DIELECTRIC_COEFFICIENT:
            raise ValueError(
                "Model and OPF timing calibration use different dielectrics."
            )
        for route, settings in (
            ("opf_ordered_echo", cfg["echo_path_settings"]),
            ("network_learned_timing", None),
        ):
            model.core.config = replace(original_config, echo_path=settings).validate()
            for part, indices in selected.items():
                meter = MineMetrics()
                replay = asyncio.run(
                    evaluate_async(model, bundle, data, indices, "cpu", (meter,))
                )
                metrics = meter.summary()
                results[part]["models"].setdefault(name, {})[route] = {
                    "metrics": metrics,
                    "replay": replay,
                    "within_two_percentage_points": {
                        key: abs(metrics[key] - target) <= 2
                        for key, target in TARGET.items()
                    },
                }
    return {
        "format": "icemap.opf.review.results.v1",
        "design_versions": cfg["design_versions"],
        "target_percent": TARGET,
        "weight_sha256": cfg["weight_sha256"],
        "scopes": results,
    }


def write_report(path, report):
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    for scope, value in report["scopes"].items():
        for precision, routes in value["models"].items():
            for route, result in routes.items():
                metrics = {key: round(result["metrics"][key], 4) for key in TARGET}
                print(scope, precision, route, metrics, flush=True)
    print(f"Results: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    complete = sub.add_parser(
        "run", help="Run MATLAB, build the dataset and evaluate D4."
    )
    complete.add_argument("--raw-dir", type=Path, required=True)
    complete.add_argument(
        "--matlab", default="matlab", help="Path to the MATLAB executable."
    )
    review = sub.add_parser("evaluate", help="Evaluate a previously built dataset.")
    review.add_argument("--dataset", type=Path, required=True)
    complete.add_argument("--assets", type=Path, required=True)
    review.add_argument(
        "--assets",
        type=Path,
        help="Directory containing the two review weights. Defaults to the dataset directory.",
    )
    for precision in MODEL_FILES:
        review.add_argument(
            f"--{precision}",
            type=Path,
            help=f"Evaluate with the specified {precision.upper()} checkpoint.",
        )
    for cmd in (complete, review):
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument(
            "--scope",
            choices=("structure", "original", "both"),
            default="original",
            help="The original-test scope includes structures used for calibration.",
        )
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "evaluate":
        assets = args.assets or args.dataset.parent
        weights = {
            name: getattr(args, name) for name in MODEL_FILES if getattr(args, name)
        }
        if not weights:
            weights = {
                name: assets / filename for name, filename in MODEL_FILES.items()
            }
        write_report(args.output, evaluate(args.dataset, assets, args.scope, weights))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    calibration_path = args.assets / "opf_calibration.json"
    labels = args.assets / "opf_labels.h5"
    cfg = load_calibration(calibration_path)
    manifest = args.output / "raw_manifest.csv"
    native = args.output / "native"
    counts = make_manifest(args.raw_dir, labels, manifest, include_unlabeled=True)
    extract_features(
        args.matlab,
        args.raw_dir,
        manifest,
        native,
        cfg["raw"],
        args.output / "matlab.log",
    )
    processed = len(list(native.glob("*.mat")))
    if processed != counts["scheduled"]:
        raise RuntimeError(
            f"MATLAB exported {processed} of {counts['scheduled']} scheduled recordings."
        )
    counts["processed"] = processed
    dataset = args.output / "opf_test.h5"
    built = build_dataset(native, labels, calibration_path, dataset)
    weights = {name: args.assets / filename for name, filename in MODEL_FILES.items()}
    report = evaluate(dataset, args.assets, args.scope, weights)
    report["raw_manifest"] = counts
    report["dataset"] = built
    write_report(args.output / "opf_metrics.json", report)


if __name__ == "__main__":
    main()
