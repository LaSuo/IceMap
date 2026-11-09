"""Fit a calibrated echo-path readout and evaluate it on the original OPF folds."""

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from echo_path import EchoPathConfig, echo_path, extract_echo_candidates
from inference import load_model
from metrics import MineMetrics
from network import ModelConfig
from protocol import sha256_file
from train_opf_cv import (
    PROTOCOL,
    TARGET,
    load_data,
    score,
    split_fold,
    write_json,
)


def consensus_truth(bounds, counts, index):
    count = int(counts[index, 0])
    if count not in (1, 2) or not np.all(counts[index] == count):
        raise ValueError("Calibration expects constant one- or two-layer observations.")
    return np.rint(np.median(bounds[index, :, : count + 1], axis=0)).astype(int)


def initial_config(train, truths, candidates, time_step, c0, gap_cap_m):
    surfaces = np.array([truths[i][0] for i in train])
    surface = float(np.median(surfaces))
    scale = max(12, 2 * 1.48 * float(np.median(abs(surfaces - surface))))
    gap_cap = max(1, round(2 * gap_cap_m / (c0 * time_step)))
    min_gap = min(
        gap_cap, max(8, int(0.62 * min(np.diff(truths[i]).min() for i in train)))
    )
    spans = {1: [], 2: []}
    fractions, strengths, phase = [], [], []
    for i in train:
        truth = truths[i]
        spans[len(truth) - 1].append(truth[-1] - truth[0])
        if len(truth) == 3:
            fractions.append((truth[1] - truth[0]) / (truth[-1] - truth[0]))
        observed = candidates[i]
        if observed is not None:
            closest = np.abs(observed.bins[:, None] - truth[None, :]).argmin(axis=0)
            strengths.extend(observed.support[closest[1:]])
            phase.extend(
                observed.prominence[closest[1:]]
                / max(0.06, observed.prominence[closest[0]])
            )
    if not all(spans.values()) or not strengths:
        raise ValueError(
            "The calibration fold needs both layer counts and observed echoes."
        )
    return EchoPathConfig(
        native_step_s=time_step,
        surface_bin=surface,
        surface_scale_bins=scale,
        minimum_gap_bins=min_gap,
        maximum_single_span_bins=float(1.45 * max(spans[1])),
        maximum_double_span_bins=float(1.45 * max(spans[2])),
        inner_fraction=float(np.median(fractions)),
        minimum_tof_support=max(1e-5, 0.22 * float(np.quantile(strengths, 0.05))),
        minimum_relative_phase=0.45 * float(np.quantile(phase, 0.1)),
    ).validate()


def geometry_error(path, truth):
    if path is None:
        return 3.0
    inferred, expected = np.diff(path), np.diff(truth)
    total_error = abs(inferred.sum() - expected.sum()) / expected.sum()
    count_error = abs(len(inferred) - len(expected))
    layer_error = (
        np.mean(abs(inferred - expected) / expected) if not count_error else 0.0
    )
    return 1.15 * count_error + 0.45 * total_error + 0.8 * layer_error


def refine_config(initial, train, validation, truths, candidates):
    grid = product(
        (0.1, 0.25, 0.45),
        (0.4, 0.85, 1.4),
        (False, True),
        (0.012, 0.02, 0.032, 0.05),
        (0.38, 0.53, 0.68, 0.8, 0.9, 1.0, 1.12),
        (1.5, 2.8, 4.0),
    )
    best = None
    for tof, future, joint, spread, threshold, bonus in grid:
        config = replace(
            initial,
            tof_weight=tof,
            future_weight=future,
            joint_future=joint,
            fraction_spread=spread,
            inner_threshold=threshold,
            inner_bonus=bonus,
        )
        fit = np.mean(
            [geometry_error(echo_path(candidates[i], config), truths[i]) for i in train]
        )
        val = np.mean(
            [
                geometry_error(echo_path(candidates[i], config), truths[i])
                for i in validation
            ]
        )
        regularizer = 0.004 * (
            abs(tof - 0.25)
            + abs(future - 0.85)
            + 25 * abs(spread - 0.02)
            + abs(threshold - 0.55)
            + abs(bonus - 2.8)
        )
        objective = 0.45 * fit + 0.55 * val + regularizer
        if best is None or objective < best[0]:
            best = objective, config, float(fit), float(val)
    return best[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-grid-mm", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--resolution-gap-cap-mm", type=float, default=50.0)
    args = parser.parse_args()
    if (
        not np.isfinite(args.legacy_grid_mm)
        or args.legacy_grid_mm <= 0
        or not np.isfinite(args.resolution_gap_cap_mm)
        or args.resolution_gap_cap_mm <= 0
        or args.input_stride < 1
    ):
        parser.error("Use positive grid sizes, stride and gap cap.")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    source_summary = json.loads(
        (args.checkpoint_run / "cv_summary.json").read_text(encoding="utf8")
    )
    if (
        source_summary.get("protocol") != PROTOCOL
        or source_summary.get("folds_completed") != 5
    ):
        raise ValueError(
            "Echo path calibration requires a completed source five-fold run."
        )
    source_manifest = source_summary
    if not {"source_sha256", "structure_folds"} <= source_manifest.keys():
        source_manifest = json.loads(
            (args.checkpoint_run / "experiment.json").read_text(encoding="utf8")
        )
    data, names, groups, opf, pool, assignments, inventory = load_data(
        args.data_dir, args.legacy_grid_mm, args.input_stride, device, args.seed
    )
    if (
        source_manifest["protocol"] != PROTOCOL
        or source_manifest["source_sha256"] != inventory
        or source_manifest["structure_folds"] != assignments
    ):
        raise ValueError(
            "Source weights must match the supplied H5 inventory and folds."
        )
    phase = data["phase"][opf].cpu().numpy()
    tof = data["tof"][opf].cpu().numpy()
    candidates = {
        index: extract_echo_candidates(phase[i], tof[i]) for i, index in enumerate(opf)
    }
    bounds = data["native_boundary_bins"].cpu().numpy()
    counts = data["layer_count"].cpu().numpy()
    time_step = float(data["time_s"][opf[0]].diff().mean())
    args.output.mkdir(parents=True, exist_ok=False)
    pooled = MineMetrics()
    rows, evaluated = [], set()
    for fold in range(5):
        train, val, test = split_fold(fold, groups, opf, pool, assignments)
        source = args.checkpoint_run / f"fold_{fold}"
        split = json.loads((source / "split.json").read_text(encoding="utf8"))
        for role, indices in (("train", train), ("validation", val), ("test", test)):
            if list(names[indices]) != [item["record"] for item in split[role]]:
                raise ValueError(
                    "Source checkpoint split does not match the original partition."
                )
        base_path = source / "best_spai.pt"
        _, bundle = load_model(base_path, "cpu")
        config = ModelConfig(**bundle["model_config"])
        if (
            bundle.get("fold") != fold
            or bundle.get("evaluation_role") != PROTOCOL
            or bundle.get("quantization")
            or config.input_stride != args.input_stride
        ):
            raise ValueError(
                "Expected an unquantized source model for this fold and stride."
            )
        cohort = np.intersect1d(train, opf)
        truths = {i: consensus_truth(bounds, counts, i) for i in np.r_[cohort, val]}
        initial = initial_config(
            cohort,
            truths,
            candidates,
            time_step,
            config.c0_m_s,
            args.resolution_gap_cap_mm / 1000,
        )
        path_config, fit_error, val_error = refine_config(
            initial, cohort, val, truths, candidates
        )
        bundle["model_config"] = asdict(
            replace(config, echo_path=asdict(path_config)).validate()
        )
        bundle["echo_path_fit"] = {
            "source_sha256": sha256_file(base_path),
            "train_observations": len(cohort),
            "train_geometry_error": fit_error,
            "validation_geometry_error": val_error,
        }
        directory = args.output / f"fold_{fold}"
        directory.mkdir()
        path = directory / "best_spai.pt"
        torch.save(bundle, path)
        write_json(directory / "split.json", split)
        deployed, _ = load_model(path, device)
        meter = MineMetrics()
        predictions = score(
            deployed.core, data, test, 12, device, [meter, pooled], names
        )
        keys = {(row["record"], row["frame"]) for row in predictions}
        if len(keys) != len(predictions) or keys & evaluated:
            raise ValueError("Duplicate held-out prediction.")
        evaluated.update(keys)
        row = {
            "fold": fold,
            "frames": len(predictions),
            "echo_path": asdict(path_config),
            "calibration": bundle["echo_path_fit"],
            **meter.summary(),
        }
        rows.append(row)
        write_json(directory / "held_out_predictions.json", predictions)
        write_json(directory / "held_out_metrics.json", row)
        print(
            json.dumps({"fold": fold, **{key: row[key] for key in TARGET}}), flush=True
        )
    if len(evaluated) != len(opf) * data["phase"].shape[1]:
        raise RuntimeError("Incomplete OPF evaluation.")
    summary = pooled.summary()
    write_json(
        args.output / "cv_summary.json",
        {
            "protocol": PROTOCOL,
            "source_sha256": inventory,
            "structure_folds": assignments,
            "site_ids_available": False,
            "source_run": str(args.checkpoint_run),
            "resolution_gap_cap_mm": args.resolution_gap_cap_mm,
            "folds_completed": len(rows),
            "evaluated_frames": len(evaluated),
            "pooled_held_out": summary,
            "reference_percent": TARGET,
            "delta_points": {k: summary[k] - v for k, v in TARGET.items()},
            "within_two_points": all(
                abs(summary[k] - v) <= 2 for k, v in TARGET.items()
            ),
            "no_worse_than_reference_plus_two": all(
                summary[k] <= v + 2 for k, v in TARGET.items()
            ),
            "folds": rows,
        },
    )
    print(json.dumps({key: summary[key] for key in TARGET}), flush=True)


if __name__ == "__main__":
    main()
