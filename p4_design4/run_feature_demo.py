"""Run Design4 on nominal feature records exported by run_icemap_demo.m.

This demonstration uses the unchanged SPAI architecture and six losses. A short
training run activates the fully warmed Stage III losses immediately; it does
not execute the formal 20/20/40 schedule or an independent validation protocol.
All targets, metrics and checkpoints retain their pipeline_debug designation.
Use --epochs 0 for forward-only execution, including unlabelled exports.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import MineLoss, TrainingConfig
from main import H5IceDataset, collate_records, evaluate, forward_batch, move_batch
from mine_ablations import build_ablation
from network import ModelConfig
from protocol import sha256_file


def text_attribute(value):
    return value.decode("utf8") if isinstance(value, bytes) else str(value)


def inspect_records(directory):
    """Require explicitly marked demonstration exports and consistent labels."""
    files = sorted(directory.rglob("*.h5"))
    if not files:
        raise ValueError(f"No prepared HDF5 feature records under {directory}.")
    labelled = []
    inventory = []
    for path in files:
        with h5py.File(path, "r") as record:
            if (
                text_attribute(record.attrs.get("icemap_schema", ""))
                != "icemap.features.v1"
            ):
                raise ValueError(
                    f"{path}: expected exported feature HDF5, not raw echoes."
                )
            if (
                text_attribute(record.attrs.get("workflow_mode", ""))
                != "pipeline_debug"
            ):
                raise ValueError(
                    f"{path}: this entry accepts pipeline_debug records only."
                )
            has_labels = bool(record.attrs.get("has_labels", 0))
            labelled.append(has_labels)
            inventory.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "has_nominal_labels": has_labels,
                    "calibration_provenance": json.loads(
                        text_attribute(
                            record.attrs.get("calibration_provenance_json", "{}")
                        )
                    ),
                    "experiment_metadata": json.loads(
                        text_attribute(
                            record.attrs.get("experiment_metadata_json", "{}")
                        )
                    ),
                }
            )
    if len(set(labelled)) != 1:
        raise ValueError("Labelled and unlabelled records cannot be mixed in one demo.")
    return files, labelled[0], inventory


def finite_predictions(output):
    for key in ("layer_density", "layer_thickness_m", "physical_profile"):
        if key not in output or not torch.isfinite(output[key]).all():
            raise FloatingPointError(f"Nonfinite or missing network output: {key}.")


@torch.no_grad()
def prediction_summary(model, loader, device):
    model.eval()
    records = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = forward_batch(model, batch)
        finite_predictions(output)
        for b, name in enumerate(batch["record_id"]):
            frames = []
            for t in range(batch["phase"].shape[1]):
                count = int(output["layer_count"][b, t])
                frames.append(
                    {
                        "layer_count": count,
                        "thickness_m": output["layer_thickness_m"][b, t, :count]
                        .cpu()
                        .tolist(),
                        "density_g_cm3": output["layer_density"][b, t, :count]
                        .cpu()
                        .tolist(),
                    }
                )
            records.append({"record_id": name, "frames": frames})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.epochs < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or args.threads < 1
    ):
        parser.error("Require epochs >= 0, positive finite lr and threads >= 1.")
    if not args.data_dir.is_dir():
        parser.error("--data-dir must name an exported feature directory.")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        parser.error("--output-dir must be outside the feature data directory.")
    code_root = Path(__file__).resolve().parent.parent
    if output_dir == code_root or code_root in output_dir.parents:
        parser.error("--output-dir must be outside the submission code directory.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(
            "Choose a new or empty output directory; results are not overwritten."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable. Use --device cpu or auto.")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    files, has_labels, inventory = inspect_records(data_dir)
    if args.epochs and not has_labels:
        parser.error(
            "Training needs nominal labels; use --epochs 0 for unlabelled inputs."
        )
    config = ModelConfig().validate()
    training = TrainingConfig()
    dataset = H5IceDataset(
        files, with_labels=has_labels, allow_debug_data=True, model_config=config
    )
    # One recording per batch also supports heterogeneous frame counts/grids.
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_records
    )
    model = build_ablation(config, "full").to(device)
    criterion = MineLoss(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    full_loss_epoch = sum(training.stage_epochs) - 1
    history = []
    label_description = "nominal labels" if has_labels else "no labels"
    print(
        f"DEMONSTRATION ONLY: {len(files)} record(s), {device}, "
        f"{args.epochs} epoch(s); {label_description} and no independent accuracy estimate.",
        flush=True,
    )
    # Validate all record contracts before creating any output artifacts.
    for i in range(len(dataset)):
        dataset[i]
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        sums = {}
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, batch)
            finite_predictions(output)
            loss, parts, weights = criterion(output, batch, full_loss_epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite demonstration loss; inspect ideal calibration and echoes."
                )
            loss.backward()
            for name, parameter in model.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(
                        f"Nonfinite demonstration gradient in {name}."
                    )
            optimizer.step()
            for name, value in {"total": loss, **parts}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
        row = {
            "epoch": epoch + 1,
            "losses": {k: v / len(loader) for k, v in sums.items()},
        }
        history.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    predictions = prediction_summary(model, loader, device)
    report = {
        "workflow_mode": "pipeline_debug",
        "evaluation_role": "demonstration_only; training-record resubstitution with nominal targets",
        "accuracy_claim": False,
        "epochs_completed": args.epochs,
        "training_schedule": "short demo at fully warmed Stage III; formal 20/20/40 schedule not executed",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "model_config": asdict(config),
        "canonical_training_config": asdict(training),
        "input_inventory": inventory,
        "history": history,
        "predictions": predictions,
    }
    if has_labels:
        report["nominal_target_resubstitution_metrics"] = evaluate(
            model, loader, criterion, device
        )
    if args.epochs:
        checkpoint = output_dir / "demo_spai.pt"
        torch.save(
            {
                "checkpoint_format": "icemap.spai.v1",
                "model_kind": "spai_six_loss",
                "ablation_case": "full",
                "evaluation_role": "pipeline_debug",
                "workflow_mode": "pipeline_debug",
                "accuracy_claim": False,
                "epoch": args.epochs,
                "demo_loss_schedule_epoch": full_loss_epoch,
                "demo_training_schedule": report["training_schedule"],
                "model_config": asdict(config),
                "training_config": asdict(training),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": args.seed,
            },
            checkpoint,
        )
        report["demonstration_checkpoint"] = str(checkpoint)
    destination = output_dir / "demo_results.json"
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf8"
    )
    print(f"Demo completed; saved {destination}", flush=True)


if __name__ == "__main__":
    main()
