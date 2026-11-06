"""Fresh HDF5 inference. Model input never contains labels or filenames."""

import argparse
import json
import csv
import hashlib
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from inference import load_model
from legacy_data import physical_layers
from metrics import MineMetrics
from train_opf import TARGET
from legacy_grid import legacy_axes


def predict_file(
    model,
    bundle,
    model_path,
    data_path,
    output_dir,
    batch_size=4,
    targets=None,
    grid_step_m=None,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "r") as f:
        features = {
            key: f[name][...].transpose(2, 1, 0).copy()
            for key, name in [("phase", "phase"), ("tof", "tau"), ("scatter", "E")]
        }
        truth = f["label3"][...].transpose(2, 1, 0).copy()
        names = [
            bytes(f["file_names"][:, i]).decode().strip()
            for i in range(f["file_names"].shape[1])
        ]
        time_s, depth_m, grid_provenance = legacy_axes(f, grid_step_m)
    S, T, N = features["phase"].shape
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    c0 = bundle.get("model_config", bundle.get("backbone_config", {})).get(
        "c0_m_s", 299792458.0
    )
    coefficient = bundle.get("model_config", bundle.get("backbone_config", {})).get(
        "dielectric_density_coefficient", 0.845
    )
    dz = float(np.diff(depth_m).mean())
    max_range = float((time_s[-1] - time_s[0]) * c0 / 2)
    saved = {}
    start = time.time()
    for begin in range(0, S, batch_size):
        end = min(S, begin + batch_size)
        B = end - begin
        inputs = {
            key: torch.from_numpy(value[begin:end]).to(device)
            for key, value in features.items()
        }
        inputs["time_s"] = torch.from_numpy(np.broadcast_to(time_s, (B, N)).copy()).to(
            device
        )
        inputs["depth_m"] = torch.from_numpy(
            np.broadcast_to(depth_m, (B, N)).copy()
        ).to(device)
        inputs["scatter_valid"] = inputs["scatter"] > 0
        with torch.no_grad():
            prediction = model(inputs)
        for key, value in prediction.items():
            saved.setdefault(key, []).append(value.cpu().numpy())
    pred = {key: np.concatenate(values) for key, values in saved.items()}
    meter = MineMetrics()
    rows = []
    max_consistency = 0.0
    unscorable = []
    max_total_range = 0.0
    range_violations = 0
    for b, name in enumerate(names):
        for t in range(T):
            P = int(pred["layer_count"][b, t])
            hp = pred["layer_thickness_m"][b, t, :P]
            rp = pred["layer_density"][b, t, :P]
            tau = pred["layer_delays_s"][b, t, :P]
            expected = c0 * tau / (2 * (1 + coefficient * rp))
            total_range = float(np.sum(tau) * c0 / 2)
            max_total_range = max(max_total_range, total_range)
            range_violations += int(total_range > max_range + 1e-6)
            max_consistency = max(
                max_consistency, float(np.max(abs(hp - expected))) if P else 0.0
            )
            np.testing.assert_allclose(hp, expected, rtol=1e-5, atol=1e-7)
            segments = physical_layers(truth[b, t])
            hg = np.asarray([(z - a) * dz for a, z, _ in segments])
            rg = np.asarray([r for _, _, r in segments])
            if len(hg):
                meter.update(hp, rp, hg, rg)
            else:
                unscorable.append({"record_index": b, "record_name": name, "frame": t})
            rows.append(
                {
                    "record_index": b,
                    "record_name": name,
                    "frame": t,
                    "pred_count": P,
                    "gt_count": len(hg),
                    "pred_h_m": hp.tolist(),
                    "gt_h_m": hg.tolist(),
                    "pred_rho_g_cm3": rp.tolist(),
                    "gt_rho_g_cm3": rg.tolist(),
                    "pred_two_way_delays_s": tau.tolist(),
                }
            )
    policy = bundle.get("evaluation_policies", {}).get(Path(data_path).stem, {})
    status = (
        "complete"
        if not unscorable
        else (
            "partial_gt_ablation_failure_accepted"
            if policy.get("empty_label3") == "accepted_ablation_failure"
            else "partial_ground_truth_pending_confirmation"
        )
    )
    summary = meter.summary()
    summary.update(
        samples=S,
        frames=T * S,
        source_file=str(Path(data_path).resolve()),
        scored_frames=T * S - len(unscorable),
        unscorable_empty_gt_frames=unscorable,
        evaluation_status=status,
        evaluation_policy=policy,
        model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        evaluation_role=bundle["evaluation_role"],
        model_kind=bundle["model_kind"],
        inference_seconds=time.time() - start,
        max_physics_consistency_error_m=max_consistency,
        targets_percent=targets,
        within_one_percentage_point=(
            all(
                summary[k] is not None and abs(summary[k] - v) <= 1
                for k, v in targets.items()
            )
            if targets and not unscorable
            else None
        ),
        no_worse_than_target_plus_one=(
            all(
                summary[k] is not None and summary[k] <= v + 1
                for k, v in targets.items()
            )
            if targets and not unscorable
            else None
        ),
        labels_supplied_to_model=False,
        held_out_independence_verified=False,
        temporal_consensus=(bundle.get("reference_alignment") or {}).get(
            "regression_temporal_consensus", "none"
        ),
        reference_alignment=bundle.get("reference_alignment"),
    )
    summary.update(
        quality_expert_config=bundle.get("quality_expert_config"),
        secondary_quality_expert_config=bundle.get("secondary_quality_expert_config"),
        post_calibration=bundle.get("post_calibration"),
        quality_gate_usage={
            k: {
                "fraction_active": float(np.mean(pred[k] > 0)),
                "mean_weight": float(np.mean(pred[k])),
            }
            for k in [
                "quality_expert_gate",
                "primary_quality_expert_gate",
                "secondary_quality_expert_gate",
            ]
            if k in pred
        },
        max_active_total_range_m=max_total_range,
        max_observation_range_m=max_range,
        grid_provenance=grid_provenance,
        observed_range_violation_frames=range_violations,
    )
    np.savez_compressed(output / "predictions.npz", **pred)
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (output / "per_layer_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "record",
                "frame",
                "layer",
                "pred_count",
                "gt_count",
                "pred_h_m",
                "gt_h_m",
                "pred_rho",
                "gt_rho",
            ]
        )
        for row in rows:
            for l in range(max(row["pred_count"], row["gt_count"])):
                get = lambda key: row[key][l] if l < len(row[key]) else ""
                writer.writerow(
                    [
                        row["record_name"],
                        row["frame"],
                        l + 1,
                        row["pred_count"],
                        row["gt_count"],
                        get("pred_h_m"),
                        get("gt_h_m"),
                        get("pred_rho_g_cm3"),
                        get("gt_rho_g_cm3"),
                    ]
                )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--targets-json",
        help="Explicit metric-name -> percent targets for this experiment; OPF defaults apply only to test_opf.h5.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--legacy-grid-mm", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    model, bundle = load_model(args.model, args.device)
    targets = (
        json.loads(Path(args.targets_json).read_text())
        if args.targets_json
        else (
            bundle.get("experiment_reference_targets", TARGET)
            if Path(args.data).stem == "test_opf"
            else None
        )
    )
    print(
        json.dumps(
            predict_file(
                model,
                bundle,
                args.model,
                args.data,
                args.output_dir,
                args.batch,
                targets,
                None if args.legacy_grid_mm is None else args.legacy_grid_mm / 1000,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
