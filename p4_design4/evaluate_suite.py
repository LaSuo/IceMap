"""Score every supplied HDF5 with one model; preserve raw predictions/coverage."""

import argparse
import csv
import json
from pathlib import Path

import torch

from inference import load_model
from predict_opf import predict_file
from train_opf import TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--legacy-grid-mm", type=float)
    parser.add_argument(
        "--targets-map", help="JSON mapping file stems to confirmed metric targets"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    model, bundle = load_model(args.model, args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_map = (
        json.loads(Path(args.targets_map).read_text())
        if args.targets_map
        else {"test_opf": bundle.get("experiment_reference_targets", TARGET)}
    )
    reports = []
    for file in sorted(Path(args.data_dir).glob("test*.h5")):
        result = predict_file(
            model,
            bundle,
            args.model,
            file,
            output / file.stem,
            args.batch,
            target_map.get(file.stem),
            None if args.legacy_grid_mm is None else args.legacy_grid_mm / 1000,
        )
        result["dataset"] = file.stem
        reports.append(result)
        (output / "all_dataset_metrics.json").write_text(json.dumps(reports, indent=2))
        print(
            json.dumps(
                {
                    "dataset": file.stem,
                    "metrics": {
                        k: v for k, v in result.items() if k.endswith("_percent")
                    },
                    "evaluation_status": result["evaluation_status"],
                }
            ),
            flush=True,
        )
    columns = [
        "dataset",
        "samples",
        "frames",
        "scored_frames",
        "evaluation_status",
        "LER_percent",
        "LE_h_percent",
        "LE_rho_percent",
        "AE_h_percent",
        "AE_rho_percent",
        "within_one_percentage_point",
    ]
    with (output / "all_dataset_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps({"completed": True, "datasets": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
