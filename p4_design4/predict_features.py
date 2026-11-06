"""CPU/GPU inference on prepared C-order records, with optional held-out truth.

Do not use the legacy scorer for calibrated icemap.features.v1 records.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from inference import load_model
from main import H5IceDataset, collate_records, move_batch
from metrics import MineMetrics
from network import ModelConfig
from protocol import sha256_file, validate_record
from losses import TrainingConfig
from feature_io import read_delay_events


@torch.inference_mode()
def predict_records(
    model,
    bundle,
    files,
    output,
    with_labels=False,
    observed_event_tolerance_bins=None,
    allow_debug_data=False,
):
    config = ModelConfig(
        **bundle.get("model_config", bundle.get("backbone_config", {}))
    )
    if with_labels and not allow_debug_data:
        for file in files:
            with h5py.File(file, "r") as f:
                validate_record(
                    f, config, TrainingConfig(**bundle.get("training_config", {}))
                )
    dataset = H5IceDataset(
        files,
        config.max_layers,
        with_labels=with_labels,
        model_config=config,
        allow_debug_data=allow_debug_data,
    )
    device = next(model.parameters()).device
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metric = MineMetrics()
    inventory = []
    for index in range(len(dataset)):
        batch = move_batch(collate_records([dataset[index]]), device)
        # Keep the model's input boundary independent of labels/identifiers.
        inputs = {
            key: batch[key]
            for key in ("phase", "tof", "scatter", "scatter_valid", "time_s", "depth_m")
        }
        if observed_event_tolerance_bins is not None:
            if bundle.get("model_kind") != "spai_six_loss":
                raise ValueError(
                    "Observed event association supports SPAI, not enhanced development bundles."
                )
            if observed_event_tolerance_bins <= 0:
                raise ValueError("Event tolerance must be positive.")
            events = read_delay_events(files[index])
            inputs.update(
                observed_delay_events_s=torch.as_tensor(
                    events["time_s"][None], device=device
                ),
                observed_delay_event_valid=torch.as_tensor(
                    events["valid"][None], device=device
                ),
                event_tolerance_s=float(batch["time_s"].diff().mean())
                * observed_event_tolerance_bins,
            )
        prediction = model(inputs)
        np.savez_compressed(
            output / f"record_{index:05d}.npz",
            **{key: value.cpu().numpy() for key, value in prediction.items()},
        )
        inventory.append(
            {
                "record": Path(files[index]).name,
                "sha256": sha256_file(files[index]),
                "prediction": f"record_{index:05d}.npz",
            }
        )
        if with_labels:
            for t in range(batch["phase"].shape[1]):
                P = int(prediction["layer_count"][0, t])
                G = int(batch["layer_count"][0, t])
                metric.update(
                    *(
                        a.cpu().numpy()
                        for a in (
                            prediction["layer_thickness_m"][0, t, :P],
                            prediction["layer_density"][0, t, :P],
                            batch["layer_thickness_m_gt"][0, t, :G],
                            batch["layer_density_gt"][0, t, :G],
                        )
                    )
                )
    digests = {item["sha256"] for item in inventory}
    split = bundle.get("data_split_sha256", {})
    held_out = (
        bool(digests)
        and digests <= set(split.get("test", []))
        and not digests.intersection(
            split.get("train", []) + split.get("validation", [])
        )
    )
    debug = allow_debug_data or bundle.get("evaluation_role") == "pipeline_debug"
    report = {
        "records": inventory,
        "model_kind": bundle.get("model_kind"),
        "evaluation_role": (
            "pipeline_debug"
            if debug
            else (
                "held_out_checkpoint_test_partition"
                if held_out
                else "user_supplied_records_independence_unverified"
            )
        ),
        "held_out_independence_verified": bool(held_out and not debug),
        "training_evaluation_role": bundle.get("evaluation_role"),
        "ablation_case": bundle.get("ablation_case", "full"),
        "parameter_matching": bundle.get("parameter_matching"),
        "labels_supplied_to_model": False,
        "device": str(device),
        "implementation_contract": bundle.get("implementation_contract"),
        "metrics": metric.summary() if with_labels else None,
        "delay_event_mode": (
            "learned_dense_map"
            if observed_event_tolerance_bins is None
            else "observed_d3_monotone_association"
        ),
        "observed_event_tolerance_bins": observed_event_tolerance_bins,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--with-labels", action="store_true")
    p.add_argument(
        "--observed-event-tolerance-bins",
        type=float,
        help="Explicit inference alternative using exported D3 peaks; calibrated tolerance required.",
    )
    p.add_argument("--allow-debug-data", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    source = Path(a.data)
    files = sorted(source.rglob("*.h5")) if source.is_dir() else [source]
    model, bundle = load_model(a.model, a.device)
    report = predict_records(
        model,
        bundle,
        files,
        a.output_dir,
        a.with_labels,
        a.observed_event_tolerance_bins,
        a.allow_debug_data,
    )
    print(
        json.dumps(
            {"records": len(report["records"]), "metrics": report["metrics"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
