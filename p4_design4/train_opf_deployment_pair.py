"""Train one SPAI model and export FP32 and portable mixed-precision weights."""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from calibrate_echo_path_opf import consensus_truth, initial_config, refine_config
from echo_path import extract_echo_candidates
from evaluate_quantized_opf import MODEL_BUDGET_BYTES, evaluate_async
from inference import load_model
from losses import TrainingConfig
from metrics import MineMetrics
from network import ModelConfig
from protocol import sha256_file
from quantized_features import CONTEXT, FINGERPRINTS
from quantized_weights import compact_weight_bundle
from train_opf_cv import TARGET, load_data, split_fold, train_fold, write_json


PAIR_PROTOCOL = "icemap.opf.deployment_pair.structure_holdout.v1"


def deployment_split(groups, opf, pool, cv_groups):
    # Merge the two lowest-numbered source folds for a single shared holdout.
    # This choice depends on the seeded group assignment alone.
    assignments = {
        group: (0 if fold < 2 else 1 if fold == 2 else 2)
        for group, fold in cv_groups.items()
    }
    train, validation, held_out = split_fold(0, groups, opf, pool, assignments)
    group_sets = [set(groups[ids]) for ids in (train, validation, held_out)]
    if any(a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1 :]):
        raise ValueError("Deployment train, validation and holdout groups overlap.")
    return assignments, train, validation, held_out


def calibrated_config(base, data, training_opf, validation, gap_cap_m):
    fit_indices = np.concatenate((training_opf, validation))
    phase = data["phase"][fit_indices].cpu().numpy()
    tof = data["tof"][fit_indices].cpu().numpy()
    candidates = {
        i: extract_echo_candidates(phase[j], tof[j]) for j, i in enumerate(fit_indices)
    }
    bounds = data["native_boundary_bins"].cpu().numpy()
    counts = data["layer_count"].cpu().numpy()
    truths = {i: consensus_truth(bounds, counts, i) for i in fit_indices}
    time_step = float(data["time_s"][training_opf[0]].diff().mean())
    initial = initial_config(
        training_opf, truths, candidates, time_step, base.c0_m_s, gap_cap_m
    )
    calibrated, fit_error, val_error = refine_config(
        initial, training_opf, validation, truths, candidates
    )
    return replace(base, echo_path=asdict(calibrated)).validate(), {
        "train_observations": len(training_opf),
        "validation_observations": len(validation),
        "train_geometry_error": fit_error,
        "validation_geometry_error": val_error,
    }


def evaluate_weights(path, evaluation_data, tolerance):
    model, bundle = load_model(path, "cpu")
    meter = MineMetrics()
    ids = list(range(len(evaluation_data["layer_count"])))
    replay = asyncio.run(
        evaluate_async(model, bundle, evaluation_data, ids, "cpu", (meter,))
    )
    metrics = meter.summary()
    differences = {key: metrics[key] - target for key, target in TARGET.items()}
    return {
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "feature_profile": bundle["feature_profile"],
        "replay": replay,
        "metrics": metrics,
        "delta_points": differences,
        "within_tolerance": all(
            abs(delta) <= tolerance for delta in differences.values()
        ),
        "no_worse_than_target_plus_tolerance": all(
            delta <= tolerance for delta in differences.values()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-grid-mm", type=float, required=True)
    parser.add_argument("--allow-relative-density-proxy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--stage-epochs", type=int, nargs=3, default=(16, 16, 48))
    parser.add_argument("--interval-density-supervision", type=float, default=1.5)
    parser.add_argument("--target-tolerance-points", type=float, default=2.0)
    args = parser.parse_args()
    if not args.allow_relative_density_proxy:
        parser.error(
            "Explicit opt-in to the legacy relative-density proxy is required."
        )
    if (
        min(args.stage_epochs) < 1
        or args.seed < 0
        or not np.isfinite(args.legacy_grid_mm)
        or args.legacy_grid_mm <= 0
        or any(
            not np.isfinite(value) or value < 0
            for value in (
                args.interval_density_supervision,
                args.target_tolerance_points,
            )
        )
    ):
        parser.error(
            "Supply positive grid and epoch sizes and finite supervision and tolerance settings."
        )
    if args.output_dir.exists() or args.work_dir.exists():
        parser.error("Use fresh output and work directories.")
    if args.output_dir.resolve() == args.work_dir.resolve():
        parser.error("Use separate output and work directories.")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    base = ModelConfig(
        cnn_channels=96,
        gru_hidden=64,
        unet_base=40,
        delay_channels=48,
        input_stride=4,
        timing_fingerprint_weight=2.0,
        minimum_layers=1,
        timing_learned_weight=0.35,
        phase_residual_blocks=2,
        delay_residual_blocks=1,
        compact_calibration=True,
        phase_contrast=True,
    ).validate()
    data, names, groups, opf, pool, cv_groups, inventory = load_data(
        args.data_dir, args.legacy_grid_mm, base.input_stride, device, args.seed
    )
    assignments, train, validation, held_out = deployment_split(
        groups, opf, pool, cv_groups
    )
    config, calibration = calibrated_config(
        base, data, np.intersect1d(train, opf), validation, gap_cap_m=0.05
    )
    training = TrainingConfig(stage_epochs=tuple(args.stage_epochs))
    recipe = SimpleNamespace(
        output_dir=args.work_dir,
        seed=args.seed,
        lr=0.0015,
        fp32=False,
        timing_supervision=0.16,
        opf_training_fraction=0.8,
        balance_opf_counts=True,
        batch_size=12,
        records_per_epoch=144,
        train_frames=4,
        geometry_jitter=0.3,
        geometry_keep=0.45,
        ordinal_supervision=0.6,
        calibration_count_loss=1.5,
        interval_density_supervision=args.interval_density_supervision,
        association_supervision=0.2,
        validate_every=5,
        validation_only=True,
    )
    args.work_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": PAIR_PROTOCOL,
        "seed": args.seed,
        "source_sha256": inventory,
        "cv_group_assignment": cv_groups,
        "deployment_groups": assignments,
        "held_out_cv_folds": [0, 1],
        "validation_cv_folds": [2],
        "training": asdict(training),
        "model": asdict(config),
        "recipe": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(recipe).items()
        },
        "echo_path_fit": calibration,
    }
    write_json(args.work_dir / "experiment.json", manifest)
    train_fold(
        recipe,
        0,
        config,
        training,
        data,
        names,
        groups,
        opf,
        pool,
        assignments,
        device,
        MineMetrics(),
    )
    core, source = load_model(args.work_dir / "fold_0" / "best_spai.pt", "cpu")
    source = dict(
        source,
        evaluation_role=PAIR_PROTOCOL,
        feature_profile="fp32",
        source_sha256=inventory,
        held_out_cv_folds=[0, 1],
        validation_cv_folds=[2],
        echo_path_fit=calibration,
    )
    del source["fold"]
    compact = compact_weight_bundle(deepcopy(core.core), source, "all")
    compact.update(
        feature_profile="companded8",
        source_sha256=inventory,
        held_out_cv_folds=[0, 1],
        validation_cv_folds=[2],
        echo_path_fit=calibration,
        supplementary_calibration_parameters=source[
            "supplementary_calibration_parameters"
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    float_path = args.output_dir / "spai_fp32.pt"
    compact_path = args.output_dir / "spai_rpi_int8.pt"
    torch.save(source, float_path)
    compact["float32_source_sha256"] = sha256_file(float_path)
    torch.save(compact, compact_path)
    if compact_path.stat().st_size > MODEL_BUDGET_BYTES:
        raise RuntimeError("Mixed-precision file exceeds the model size budget.")

    evaluation_keys = (
        FINGERPRINTS
        + CONTEXT
        + ("layer_count", "layer_thickness_m_gt", "layer_density_gt")
    )
    evaluation_data = {
        key: data[key][held_out].cpu().numpy() for key in evaluation_keys
    }
    report = {
        "protocol": PAIR_PROTOCOL,
        "source_sha256": inventory,
        "target_percent": TARGET,
        "target_tolerance_points": args.target_tolerance_points,
        "train_records": len(train),
        "validation_records": len(validation),
        "held_out_records": len(held_out),
        "held_out_frames": len(held_out) * data["phase"].shape[1],
        "held_out_groups": sorted(set(groups[held_out])),
        "held_out_record_names": [str(names[i]) for i in held_out],
        "selected_training_epoch": source["epoch"],
        "total_parameters": sum(p.numel() for p in core.parameters()),
        "supplementary_calibration_parameters": source[
            "supplementary_calibration_parameters"
        ],
        "fp32": evaluate_weights(
            float_path, evaluation_data, args.target_tolerance_points
        ),
        "rpi_int8": evaluate_weights(
            compact_path, evaluation_data, args.target_tolerance_points
        ),
    }
    write_json(args.work_dir / "deployment_report.json", report)
    print(
        json.dumps({k: report[k]["metrics"] for k in ("fp32", "rpi_int8")}), flush=True
    )


if __name__ == "__main__":
    main()
