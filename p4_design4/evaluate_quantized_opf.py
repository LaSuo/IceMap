"""Export and evaluate mixed-precision SPAI on the original OPF structure folds."""

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from async_inference import AsyncSPAI, DEFAULT_OUTPUTS
from inference import SPAIDeployment, load_model
from metrics import MineMetrics
from protocol import sha256_file
from quantized_features import (
    CONTEXT,
    FEATURE_PROFILES,
    FINGERPRINTS,
    pack_feature_window,
)
from quantized_weights import compact_weight_bundle
from retraining_data import read_legacy, supervised_arrays
from train_opf_cv import PROTOCOL, TARGET, structure, validation_score, write_json


MODEL_BUDGET_BYTES = 279_000_000
WEIGHT_PROFILES = ("all", "density_delay", "density", "delay")


def inputs_for(data, index, device="cpu"):
    return {
        key: torch.from_numpy(data[key][index]).to(device)
        for key in FINGERPRINTS + CONTEXT
    }


def prediction_for(model, packet, device):
    with torch.inference_mode():
        output = model(packet.unpack(device))
        return {key: output[key].cpu() for key in DEFAULT_OUTPUTS}


def record_metrics(prediction, data, indices, *meters):
    for i, record in enumerate(indices):
        for t, layers in enumerate(prediction["layer_count"][i].tolist()):
            truth_count = data["layer_count"][record, t]
            hp = prediction["layer_thickness_m"][i, t, :layers].numpy()
            rp = prediction["layer_density"][i, t, :layers].numpy()
            hg = data["layer_thickness_m_gt"][record, t, :truth_count]
            rg = data["layer_density_gt"][record, t, :truth_count]
            for meter in meters:
                meter.update(hp, rp, hg, rg)


def validation_metrics(model, packet, data, indices, device):
    meter = MineMetrics()
    record_metrics(prediction_for(model, packet, device), data, indices, meter)
    return meter.summary()


async def evaluate_async(model, bundle, data, indices, device, meters):
    """Replay actual observation windows through the production queue."""
    cfg = model.core.config
    output_keys = (
        (*DEFAULT_OUTPUTS, "echo_path_used") if cfg.echo_path else DEFAULT_OUTPUTS
    )
    async with AsyncSPAI(
        model, device=device, max_in_flight=2, output_keys=output_keys
    ) as endpoint:
        pending = []
        sizes = []
        start = perf_counter()
        for index in indices:
            packet = pack_feature_window(
                inputs_for(data, [index]), bundle["feature_profile"]
            )
            sizes.append(packet.payload_bytes)
            job = await endpoint.submit(packet, observation_id=index)
            pending.append(job)
        results = await asyncio.gather(*pending)
        elapsed = perf_counter() - start
    path_frames = 0
    for result, index in zip(results, indices, strict=True):
        if result.observation_id != index:
            raise RuntimeError("Asynchronous result observation ID mismatch.")
        out = result.outputs
        if cfg.echo_path:
            path_frames += int(out["echo_path_used"].sum())
        if (out["boundary_times_s"].diff(dim=-1) < 0).any():
            raise RuntimeError("Decoded boundaries are out of order.")
        expected = (
            cfg.c0_m_s
            * out["layer_delays_s"]
            / (2 * (1 + cfg.dielectric_density_coefficient * out["layer_density"]))
        )
        torch.testing.assert_close(out["layer_thickness_m"], expected)
        record_metrics(out, data, [index], *meters)
    summary = {
        "windows": len(results),
        "payload_bytes_per_window": sizes,
        "wall_seconds_including_packet_encoding": elapsed,
    }
    if cfg.echo_path:
        summary["echo_path_used_frames"] = path_frames
    return summary


def source_split(run, fold, names, assignments):
    directory = run / f"fold_{fold}"
    saved = json.loads((directory / "split.json").read_text(encoding="utf8"))
    lookup = {str(name): i for i, name in enumerate(names)}
    split_groups = {
        key: {row["group"] for row in value} for key, value in saved.items()
    }
    roles = ("train", "validation", "test")
    for i, role in enumerate(roles):
        if any(split_groups[role] & split_groups[other] for other in roles[i + 1 :]):
            raise ValueError("Source split has a group overlap.")
    out = []
    for role, expected_fold in (("validation", (fold + 1) % 5), ("test", fold)):
        items = saved[role]
        expected = {
            str(name) for name in names if assignments[structure(name)] == expected_fold
        }
        actual = {row["record"] for row in items}
        if (
            len(items) != len(actual)
            or actual != expected
            or any(
                row["source"] != "test_opf.h5"
                or structure(row["record"]) != row["group"]
                for row in items
            )
        ):
            raise ValueError(
                "Fold manifest does not match the supplied OPF recording names."
            )
        out.append([lookup[row["record"]] for row in items])
    return directory / "best_spai.pt", out[0], out[1], saved


def select_precision(base_metrics, candidates, tolerance):
    def excess(item):
        return max(item["validation"][key] - base_metrics[key] for key in TARGET)

    eligible = [item for item in candidates if excess(item) <= tolerance]
    if eligible:
        return (
            min(
                eligible,
                key=lambda item: (
                    validation_score(item["validation"]),
                    item["model_tensor_bytes"],
                    item["feature_payload_bytes"],
                ),
            ),
            True,
        )
    return (
        min(
            candidates,
            key=lambda item: (excess(item), validation_score(item["validation"])),
        ),
        False,
    )


def tensor_bytes(state):
    return sum(
        t.numel() * t.element_size() for t in state.values() if torch.is_tensor(t)
    )


def run_fold(args, fold, data, names, assignments, pooled):
    source, validation, held_out, split = source_split(
        args.checkpoint_run, fold, names, assignments
    )
    destination = args.output / f"fold_{fold}"
    destination.mkdir()
    write_json(destination / "split.json", split)
    base, metadata = load_model(source, args.device)
    if (
        not isinstance(base, SPAIDeployment)
        or metadata.get("quantization")
        or metadata.get("fold") != fold
        or metadata.get("evaluation_role") != PROTOCOL
    ):
        raise ValueError(
            "Expected unquantized SPAI weights for the corresponding fold."
        )
    val_inputs = inputs_for(data, validation)
    held_inputs = inputs_for(data, held_out)
    val_packets = {
        profile: pack_feature_window(val_inputs, profile)
        for profile in FEATURE_PROFILES
    }
    held_fp32 = pack_feature_window(held_inputs, "fp32")
    base_metrics = validation_metrics(
        base, val_packets["fp32"], data, validation, args.device
    )
    baseline = MineMetrics()
    record_metrics(
        prediction_for(base, held_fp32, args.device),
        data,
        held_out,
        baseline,
        pooled["fp32"],
    )
    candidates = []
    fixed_metrics = MineMetrics()
    fixed_path = destination / "model_all_int8.pt"
    for profile in WEIGHT_PROFILES:
        core = deepcopy(base.core)
        package = compact_weight_bundle(core, metadata, profile)
        model = SPAIDeployment(core).eval()
        for feature_profile in FEATURE_PROFILES[1:]:
            packet = val_packets[feature_profile]
            candidates.append(
                {
                    "weight_profile": profile,
                    "feature_profile": feature_profile,
                    "model_tensor_bytes": tensor_bytes(package["model"]),
                    "feature_payload_bytes": packet.payload_bytes,
                    "validation": validation_metrics(
                        model, packet, data, validation, args.device
                    ),
                }
            )
        if profile == "all":
            package["feature_profile"] = "companded8"
            package["float32_source_sha256"] = sha256_file(source)
            torch.save(package, fixed_path)
            fixed_model, fixed_bundle = load_model(fixed_path, args.device)
            fixed_replay = asyncio.run(
                evaluate_async(
                    fixed_model,
                    fixed_bundle,
                    data,
                    held_out,
                    args.device,
                    (fixed_metrics, pooled["all_conv_companded8"]),
                )
            )
    chosen, eligible = select_precision(
        base_metrics, candidates, args.max_validation_increase
    )
    package = compact_weight_bundle(
        deepcopy(base.core), metadata, chosen["weight_profile"]
    )
    package["feature_profile"] = chosen["feature_profile"]
    package["float32_source_sha256"] = sha256_file(source)
    path = destination / "model_int8.pt"
    torch.save(package, path)
    model_bytes = path.stat().st_size
    fixed_bytes = fixed_path.stat().st_size
    if max(model_bytes, fixed_bytes) > MODEL_BUDGET_BYTES:
        raise RuntimeError(
            f"Fold {fold} exceeds the {MODEL_BUDGET_BYTES}-byte model budget."
        )
    reloaded, bundle = load_model(path, args.device)
    selected = MineMetrics()
    replay = asyncio.run(
        evaluate_async(
            reloaded,
            bundle,
            data,
            held_out,
            args.device,
            (selected, pooled["selected_int8"]),
        )
    )
    row = {
        "fold": fold,
        "held_out_records": len(held_out),
        "held_out_frames": len(held_out) * data["phase"].shape[1],
        "selected": chosen,
        "within_validation_tolerance": eligible,
        "validation_fp32": base_metrics,
        "validation_candidates": candidates,
        "fp32_checkpoint_bytes": source.stat().st_size,
        "quantized_checkpoint_bytes": model_bytes,
        "quantized_sha256": sha256_file(path),
        "all_conv_checkpoint_bytes": fixed_bytes,
        "all_conv_checkpoint_sha256": sha256_file(fixed_path),
        "all_conv_payload_bytes_per_window": pack_feature_window(
            inputs_for(data, held_out[:1]), "companded8"
        ).payload_bytes,
        "fp32_payload_bytes_per_window": pack_feature_window(
            inputs_for(data, held_out[:1]), "fp32"
        ).payload_bytes,
        "async_replay": replay,
        "all_conv_async_replay": fixed_replay,
        "held_out": {
            "fp32": baseline.summary(),
            "all_conv_companded8": fixed_metrics.summary(),
            "selected_int8": selected.summary(),
        },
    }
    write_json(destination / "metrics.json", row)
    return row, held_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-run", type=Path, required=True)
    parser.add_argument("--opf-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-grid-mm", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-validation-increase", type=float, default=0.5)
    parser.add_argument("--target-tolerance-points", type=float, default=1.0)
    args = parser.parse_args()
    if (
        args.threads < 1
        or not np.isfinite(args.legacy_grid_mm)
        or args.legacy_grid_mm <= 0
        or not np.isfinite(args.max_validation_increase)
        or args.max_validation_increase < 0
        or not np.isfinite(args.target_tolerance_points)
        or args.target_tolerance_points < 0
    ):
        parser.error("Supply valid thread, grid and tolerance settings.")
    torch.set_num_threads(args.threads)
    manifest_path = args.checkpoint_run / "cv_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest["protocol"] != PROTOCOL or manifest["folds_completed"] != 5:
        raise ValueError("Expected a completed five-fold OPF source run.")
    digest = sha256_file(args.opf_data)
    if digest != manifest["source_sha256"]["test_opf.h5"]:
        raise ValueError("OPF H5 checksum differs from the source run.")
    source = read_legacy(
        args.opf_data, grid_step_m=args.legacy_grid_mm / 1000, require_axes=True
    )
    data = supervised_arrays(source)
    names = source["names"]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate OPF recording names.")
    assignments = manifest["structure_folds"]
    args.output.mkdir(parents=True, exist_ok=False)
    pooled = {
        key: MineMetrics() for key in ("fp32", "all_conv_companded8", "selected_int8")
    }
    rows, evaluated = [], set()
    for fold in range(5):
        row, held_out = run_fold(args, fold, data, names, assignments, pooled)
        if evaluated & set(held_out):
            raise ValueError("A held-out record was evaluated twice.")
        evaluated.update(held_out)
        rows.append(row)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "weights": row["selected"]["weight_profile"],
                    "features": row["selected"]["feature_profile"],
                    "model_bytes": row["quantized_checkpoint_bytes"],
                    "errors": {
                        key: row["held_out"]["selected_int8"][key] for key in TARGET
                    },
                }
            ),
            flush=True,
        )
    if len(evaluated) != len(names):
        raise ValueError("Not every OPF observation was evaluated.")
    summary = {key: meter.summary() for key, meter in pooled.items()}
    comparison = {
        name: {
            "within_target_tolerance": all(
                metrics[key] is not None
                and abs(metrics[key] - target) <= args.target_tolerance_points
                for key, target in TARGET.items()
            ),
            "no_worse_than_target_plus_tolerance": all(
                metrics[key] is not None
                and metrics[key] <= target + args.target_tolerance_points
                for key, target in TARGET.items()
            ),
        }
        for name, metrics in summary.items()
    }
    report = {
        "protocol": PROTOCOL,
        "source_sha256": manifest["source_sha256"],
        "structure_folds": assignments,
        "device": args.device,
        "threads": args.threads,
        "model_budget_bytes": MODEL_BUDGET_BYTES,
        "all_models_within_budget": True,
        "evaluated_frames": sum(row["held_out_frames"] for row in rows),
        "evaluated_observations": len(evaluated),
        "reference_percent": TARGET,
        "target_tolerance_points": args.target_tolerance_points,
        "target_comparison": comparison,
        "max_validation_increase_points": args.max_validation_increase,
        "selection": "Lowest mean validation error among mixed profiles within tolerance; otherwise least maximum degradation.",
        "pooled_held_out": summary,
        "delta_points_from_fp32": {
            key: summary["selected_int8"][key] - summary["fp32"][key] for key in TARGET
        },
        "within_one_point": all(
            abs(summary["selected_int8"][key] - target) <= 1
            for key, target in TARGET.items()
        ),
        "folds": rows,
    }
    write_json(args.output / "cv_summary.json", report)
    print(json.dumps(report["pooled_held_out"]), flush=True)


if __name__ == "__main__":
    main()
