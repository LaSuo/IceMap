"""Train/evaluate the corrected SPAI on prepared feature records.

No dataset is opened at import. Run only after dataset calibration and labels
are supplied. The HDF5 feature contract is enforced by H5IceDataset below.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from network import ModelConfig, SPAIIceInversionNet
from losses import TrainingConfig, MineLoss, stage_weights
from metrics import MineMetrics
from protocol import (
    PROTOCOL,
    implementation_contract,
    sha256_file,
    validate_matlab_config,
    validate_record,
)
from mine_ablations import CASES, build_ablation
from environment_snapshot import capture_environment, code_fingerprints
from implementation_settings import (
    TrainingInitials,
    settings_manifest,
    audit_partition_metadata,
)

SCHEMA = "icemap.features.v1"


def _text(value):
    return value.decode("utf8") if isinstance(value, bytes) else str(value)


class H5IceDataset(Dataset):
    """One HDF5 file per recording. All shapes use h5py/C axis order."""

    def __init__(
        self,
        files,
        max_layers=10,
        with_labels=True,
        allow_debug_data=False,
        model_config=None,
    ):
        self.files = [Path(p) for p in files]
        self.max_layers = max_layers
        self.with_labels = with_labels
        self.allow_debug_data = allow_debug_data
        self.model_config = model_config or ModelConfig(max_layers=max_layers)
        if not self.files:
            raise ValueError("Empty dataset partition.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        with h5py.File(filename, "r") as f:
            if _text(f.attrs.get("icemap_schema", "")) != SCHEMA:
                raise ValueError(
                    f"{filename}: prepare calibrated feature records with schema {SCHEMA}; raw radar is not a feature record."
                )
            if (
                _text(f.attrs.get("workflow_mode", "")) == "pipeline_debug"
                and not self.allow_debug_data
            ):
                raise ValueError(
                    f"{filename}: simulated metadata/nominal labels; use allow_debug_data=True only for pipeline debugging."
                )
            keys = ["phase", "E", "tau", "time_s", "depth_m", "scatter_valid"]
            if "acquisition_time_s" in f:
                keys.append("acquisition_time_s")
            validate_matlab_config(f.attrs.get("matlab_config_json"), self.model_config)
            if self.with_labels:
                keys += [
                    "boundary_target",
                    "layer_count",
                    "relative_density_target",
                    "layer_density_gt",
                    "layer_thickness_m_gt",
                    "density_profile_gt",
                ]
                if "boundary_times_s_gt" in f:
                    keys.append("boundary_times_s_gt")
            sample = {
                k: torch.as_tensor(
                    np.asarray(f[k]),
                    dtype=(
                        torch.float64
                        if k
                        in (
                            "time_s",
                            "depth_m",
                            "boundary_times_s_gt",
                            "acquisition_time_s",
                        )
                        else torch.float32
                    ),
                )
                for k in keys
            }
            if self.with_labels and not _text(
                f.attrs.get("relative_density_target_method", "")
            ):
                raise ValueError(
                    f"{filename}: document relative_density_target_method; the reference specification does not define its construction."
                )
            sample["site_id"] = _text(f.attrs["site_id"])
            sample["record_id"] = _text(f.attrs.get("record_id", filename.stem))
            sample["source_file"] = str(filename.resolve())
            sample["boundary_target_method"] = _text(
                f.attrs.get("boundary_target_method", "unspecified")
            )
        sample["scatter"] = sample.pop("E")
        sample["tof"] = sample.pop("tau")
        self._validate(sample, filename)
        sample["scatter_valid"] = sample["scatter_valid"].bool()
        if self.with_labels:
            sample["layer_count"] = sample["layer_count"].long()
        return sample

    def _validate(self, s, filename):
        fail = lambda message: ValueError(f"{filename}: {message}")
        if s["phase"].ndim != 2:
            raise fail("phase must be [T,N].")
        T, N = s["phase"].shape
        if T < 2 or N < 8:
            raise fail("Need T>=2 and N>=8.")
        for key in ("scatter", "tof", "scatter_valid"):
            if s[key].shape != (T, N):
                raise fail(f"{key} must match [T,N].")
        if "acquisition_time_s" in s and (
            s["acquisition_time_s"].shape != (T,)
            or not (s["acquisition_time_s"].diff() > 0).all()
        ):
            raise fail("Acquisition timestamps must increase, one per actual frame.")
        if not ((s["scatter_valid"] == 0) | (s["scatter_valid"] == 1)).all():
            raise fail("scatter_valid must be a binary mask.")
        if s["time_s"].shape != (N,) or not (s["time_s"].diff() > 0).all():
            raise fail("time_s must be increasing [N] in seconds.")
        dt = s["time_s"].diff()
        if not torch.allclose(dt, dt.mean().expand_as(dt), rtol=1e-4, atol=0):
            raise fail("time_s must be uniform.")
        if (
            s["depth_m"].ndim != 1
            or not (s["depth_m"].diff() > 0).all()
            or s["depth_m"][0] < 0
        ):
            raise fail(
                "depth_m must be increasing, nonnegative [Nd], measured from the ice surface."
            )
        for key, value in s.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise fail(f"{key} contains nonfinite values.")
        if not self.with_labels:
            return
        K = self.max_layers
        if (
            s["layer_count"].shape != (T,)
            or not (
                (s["layer_count"] >= 1)
                & (s["layer_count"] <= K)
                & (s["layer_count"] == s["layer_count"].round())
            ).all()
        ):
            raise fail("layer_count must be integer [T] in 1..max_layers.")
        for key in (
            "relative_density_target",
            "layer_density_gt",
            "layer_thickness_m_gt",
        ):
            if s[key].shape != (T, K):
                raise fail(f"{key} must be [T,K], zero padded.")
        mask = torch.arange(K) < s["layer_count"].unsqueeze(-1)
        if (
            not (s["layer_density_gt"][mask] > 0).all()
            or not (s["layer_thickness_m_gt"][mask] > 0).all()
        ):
            raise fail("Ground-truth ice density and thickness must be positive.")
        if (s["relative_density_target"][mask] < 0).any():
            raise fail("Relative-density targets must be nonnegative.")
        for key in (
            "relative_density_target",
            "layer_density_gt",
            "layer_thickness_m_gt",
        ):
            if (s[key][~mask] != 0).any():
                raise fail(f"{key}: padding must be zero and excluded from loss.")
        if (
            s["boundary_target"].shape != (T, N)
            or not ((s["boundary_target"] == 0) | (s["boundary_target"] == 1)).all()
        ):
            raise fail("boundary_target must be binary [T,N].")
        if not torch.equal(s["boundary_target"].sum(-1), s["layer_count"] + 1):
            raise fail(
                "Boundary targets must contain exactly L+1 single-bin physical interfaces per frame."
            )
        if s["density_profile_gt"].shape != (T, s["depth_m"].numel()):
            raise fail(
                "density_profile_gt must be [T,Nd] on depth_m; do not expand it using mean density."
            )
        if "boundary_times_s_gt" in s:
            if s["boundary_times_s_gt"].shape != (T, K + 1):
                raise fail("boundary_times_s_gt must be [T,K+1], zero padded.")
            for t, n in enumerate(s["layer_count"].long()):
                times = s["boundary_times_s_gt"][t, : int(n) + 1]
                if (
                    not (times.diff() > 0).all()
                    or times[0] < s["time_s"][0]
                    or times[-1] > s["time_s"][-1]
                    or (s["boundary_times_s_gt"][t, int(n) + 1 :] != 0).any()
                ):
                    raise fail(
                        "Invalid registered measured/projected boundary times or padding."
                    )


def scan_records(directory, allow_debug_data=False):
    records = []
    for p in sorted(Path(directory).rglob("*.h5")):
        with h5py.File(p, "r") as f:
            if _text(f.attrs.get("icemap_schema", "")) != SCHEMA:
                raise ValueError(f"{p}: not a prepared {SCHEMA} feature record.")
            if (
                _text(f.attrs.get("workflow_mode", "")) == "pipeline_debug"
                and not allow_debug_data
            ):
                raise ValueError(
                    f"{p}: this is pipeline-debug data, not an accuracy benchmark; explicitly pass --allow-debug-data for debugging."
                )
            site = _text(f.attrs.get("site_id", ""))
            if not site:
                raise ValueError(
                    f"{p}: independent site_id is required for leakage-free folds."
                )
            records.append((p, site))
    if not records:
        raise ValueError("No feature .h5 records found.")
    return records


def make_site_split(records, fold, seed, validation_fraction):
    sites = sorted({site for _, site in records})
    if len(sites) < 5:
        raise ValueError(
            "Site-based five-fold CV requires at least five distinct sites."
        )
    rng = random.Random(seed)
    rng.shuffle(sites)
    assignments = {site: i % 5 for i, site in enumerate(sites)}
    remaining = [s for s in sites if assignments[s] != fold]
    rng = random.Random(seed + fold + 1)
    rng.shuffle(remaining)
    nval = max(1, round(len(remaining) * validation_fraction))
    if nval >= len(remaining):
        raise ValueError("No training sites remain after validation split.")
    val_sites = set(remaining[:nval])
    splits = {"train": [], "validation": [], "test": []}
    for p, site in records:
        role = (
            "test"
            if assignments[site] == fold
            else ("validation" if site in val_sites else "train")
        )
        splits[role].append(p)
    return splits, assignments


def collate_records(records):
    # Shape mismatch is an explicit import/calibration error, never silently
    # interpolated; preparation determines the common grid and frame windows.
    keys = records[0].keys()
    if any(set(r) != set(keys) for r in records):
        raise ValueError(
            "Mixed optional record schemas inside one batch; align exports before training."
        )
    return {
        key: (
            torch.stack([r[key] for r in records])
            if torch.is_tensor(records[0][key])
            else [r[key] for r in records]
        )
        for key in keys
    }


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model, batch, stage=3):
    return model(
        tof=batch["tof"],
        phase=batch["phase"],
        scatter=batch["scatter"],
        scatter_valid=batch["scatter_valid"],
        time_s=batch["time_s"],
        depth_m=batch["depth_m"],
        need_layer=stage >= 2,
        need_physical=stage >= 3,
    )


def train_epoch(model, loader, optimizer, criterion, epoch, device):
    stage, _ = stage_weights(epoch, criterion.config)
    model.train()
    totals, batches = {}, 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = forward_batch(model, batch, stage)
        loss, parts, weights = criterion(out, batch, epoch)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Nonfinite training loss; inspect calibrated features."
            )
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
        optimizer.step()
        for key, value in {"total": loss, **parts}.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        batches += 1
    return {
        **{k: v / batches for k, v in totals.items()},
        "stage": stage,
        "weights": weights,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, matching_min_iou=0.0):
    model.eval()
    metric = MineMetrics(matching_min_iou)
    total, frames = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = forward_batch(model, batch)
        # Model selection uses all six fixed weights, after Stage III warmup.
        loss, _, _ = criterion(out, batch, sum(criterion.config.stage_epochs) - 1)
        frame_count = batch["phase"].shape[0] * batch["phase"].shape[1]
        total += float(loss) * frame_count
        frames += frame_count
        for b in range(batch["phase"].shape[0]):
            for t in range(batch["phase"].shape[1]):
                P = int(out["layer_count"][b, t])
                G = int(batch["layer_count"][b, t])
                args = (
                    out["layer_thickness_m"][b, t, :P],
                    out["layer_density"][b, t, :P],
                    batch["layer_thickness_m_gt"][b, t, :G],
                    batch["layer_density_gt"][b, t, :G],
                )
                metric.update(*(x.cpu().numpy() for x in args))
    return {"loss": total / frames, **metric.summary()}


def _write_json(filename, obj):
    Path(filename).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8"
    )


def main():
    initials = TrainingInitials().validate()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-debug-data",
        action="store_true",
        help="Allow nominal calibration/labels for development only.",
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--lr",
        type=float,
        default=initials.learning_rate,
        help="Provisional engineering initial; override explicitly when calibrated.",
    )
    parser.add_argument("--batch-size", type=int, default=initials.batch_size)
    parser.add_argument("--seed", type=int, default=initials.seed)
    parser.add_argument("--weight-decay", type=float, default=initials.weight_decay)
    parser.add_argument("--adam-beta1", type=float, default=initials.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=initials.adam_beta2)
    parser.add_argument("--adam-epsilon", type=float, default=initials.adam_epsilon)
    parser.add_argument("--workers", type=int, default=initials.workers)
    parser.add_argument(
        "--validation-fraction", type=float, default=initials.validation_fraction
    )
    parser.add_argument("--matching-min-iou", type=float, default=0.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--model-config", help="JSON overriding implementation architecture choices."
    )
    parser.add_argument("--ablation", choices=CASES, default="full")
    parser.add_argument("--focal-alpha", type=float, default=initials.focal_alpha)
    parser.add_argument("--focal-gamma", type=float, default=initials.focal_gamma)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Record and enforce supported deterministic operators.",
    )
    args = parser.parse_args()
    TrainingInitials(
        args.lr,
        args.batch_size,
        args.seed,
        args.weight_decay,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_epsilon,
        args.validation_fraction,
        args.workers,
        args.focal_alpha,
        args.focal_gamma,
    ).validate()
    if (
        not np.isfinite(args.lr)
        or args.lr <= 0
        or args.batch_size < 1
        or not 0 < args.validation_fraction < 1
    ):
        parser.error("Invalid learning rate, batch size or validation fraction.")
    records = scan_records(args.data_dir, args.allow_debug_data)
    config = (
        ModelConfig(
            **json.loads(Path(args.model_config).read_text(encoding="utf-8-sig"))
        )
        if args.model_config
        else ModelConfig()
    )
    training = TrainingConfig(
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma
    )
    inventory = []
    inventory_by_path = {}
    seen = {}
    for path, site in records:
        with h5py.File(path, "r") as f:
            if not args.allow_debug_data:
                validate_record(f, config, training)
            digest = hashlib.sha256()
            for key in ("phase", "E", "tau"):
                digest.update(np.asarray(f[key]).tobytes())
            fingerprint = digest.hexdigest()
            if fingerprint in seen:
                raise ValueError(
                    f"Duplicate input recordings: {path} and {seen[fingerprint]}."
                )
            seen[fingerprint] = path
            methods = {
                k: _text(f.attrs.get(k, ""))
                for k in (
                    "relative_density_target_method",
                    "boundary_target_method",
                    "workflow_mode",
                )
            }
        inventory.append(
            {
                "record": path.name,
                "site_id": site,
                "sha256": sha256_file(path),
                "input_sha256": fingerprint,
                **methods,
            }
        )
        inventory_by_path[path] = inventory[-1]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folds = range(5) if args.all_folds else [args.fold]
    summaries = []
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for fold in folds:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = args.deterministic
        torch.use_deterministic_algorithms(args.deterministic)
        directory = output / f"fold_{fold}"
        directory.mkdir(exist_ok=False)
        splits, assignments = make_site_split(
            records, fold, args.seed, args.validation_fraction
        )
        partition_details = audit_partition_metadata(splits)
        loaders = {
            role: DataLoader(
                H5IceDataset(
                    files,
                    config.max_layers,
                    allow_debug_data=args.allow_debug_data,
                    model_config=config,
                ),
                batch_size=args.batch_size,
                shuffle=role == "train",
                num_workers=args.workers,
                collate_fn=collate_records,
                generator=torch.Generator().manual_seed(args.seed),
            )
            for role, files in splits.items()
        }
        provenance = {
            "protocol": PROTOCOL,
            "implementation_contract": implementation_contract(config, args.ablation),
            "implementation_settings": settings_manifest(),
            "data_inventory": inventory,
            "arguments": vars(args),
            "model": asdict(config),
            "training": asdict(training),
            "site_fold": assignments,
            "splits": {k: list(map(str, v)) for k, v in splits.items()},
            "partition_metadata_audit": partition_details,
            "optimizer": {
                "name": "Adam",
                "betas": [args.adam_beta1, args.adam_beta2],
                "epsilon": args.adam_epsilon,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "learning_rate_schedule": "constant",
            },
            "normalization": "identity; physical fingerprints are not standardized using test records",
            "source_code": code_fingerprints(Path(__file__).parent.parent),
            "runtime": capture_environment(),
            "torch_version": torch.__version__,
        }
        model = build_ablation(config, args.ablation).to(device)
        provenance["parameter_matching"] = model.ablation_metadata
        _write_json(directory / "experiment.json", provenance)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        criterion = MineLoss(training)
        best = float("inf")
        history = []
        for epoch in range(sum(training.stage_epochs)):
            train = train_epoch(
                model, loaders["train"], optimizer, criterion, epoch, device
            )
            stage, _ = stage_weights(epoch, training)
            row = {
                "epoch": epoch + 1,
                "train": train,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if (
                stage == 3
                and epoch >= sum(training.stage_epochs[:2]) + training.warmup_epochs - 1
            ):
                validation = evaluate(
                    model,
                    loaders["validation"],
                    criterion,
                    device,
                    args.matching_min_iou,
                )
                row["validation"] = validation
                if validation["loss"] < best:
                    best = validation["loss"]
                    torch.save(
                        {
                            "checkpoint_format": "icemap.spai.v1",
                            "model_kind": (
                                "spai_six_loss"
                                if args.ablation == "full"
                                else "cnn_ablation_six_loss"
                            ),
                            "implementation_contract": implementation_contract(
                                config, args.ablation
                            ),
                            "fold": fold,
                            "ablation_case": args.ablation,
                            "parameter_matching": model.ablation_metadata,
                            "data_split_sha256": {
                                role: [inventory_by_path[p]["sha256"] for p in paths]
                                for role, paths in splits.items()
                            },
                            "source_code_sha256": provenance["source_code"][
                                "code_sha256"
                            ],
                            "runtime": provenance["runtime"],
                            "evaluation_role": (
                                "site_held_out"
                                if not args.allow_debug_data
                                else "pipeline_debug"
                            ),
                            "epoch": epoch + 1,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_state": {
                                "torch": torch.get_rng_state(),
                                "numpy": np.random.get_state(),
                                "python": random.getstate(),
                                "cuda": (
                                    torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available()
                                    else []
                                ),
                            },
                            "model_config": asdict(config),
                            "training_config": asdict(training),
                            "best_validation_loss": best,
                        },
                        directory / "best_spai.pt",
                    )
            history.append(row)
            _write_json(directory / "history.json", history)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        checkpoint = torch.load(
            directory / "best_spai.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        result = evaluate(
            model, loaders["test"], criterion, device, args.matching_min_iou
        )
        result.update(
            protocol=PROTOCOL,
            model_sha256=sha256_file(directory / "best_spai.pt"),
            evaluation_role=(
                "site_held_out" if not args.allow_debug_data else "pipeline_debug"
            ),
        )
        _write_json(directory / "test_metrics.json", result)
        summaries.append({"fold": fold, **result})
    _write_json(output / "fold_metrics.json", summaries)
    if args.all_folds:
        keys = [
            "LER_percent",
            "LE_h_percent",
            "LE_rho_percent",
            "AE_h_percent",
            "AE_rho_percent",
        ]
        summary = {
            key: (
                {
                    "mean": float(
                        np.mean([r[key] for r in summaries if r[key] is not None])
                    ),
                    "std": float(
                        np.std([r[key] for r in summaries if r[key] is not None])
                    ),
                    "valid_folds": sum(r[key] is not None for r in summaries),
                }
                if any(r[key] is not None for r in summaries)
                else {"mean": None, "std": None, "valid_folds": 0}
            )
            for key in keys
        }
        _write_json(output / "cross_validation_summary.json", summary)


if __name__ == "__main__":
    main()
