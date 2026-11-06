"""Development-only shared model training from legacy train.h5.

Example: python p4_design4/train.py --data-dir DATA --output-dir RUN --profile fresh --legacy-grid-mm 1 --allow-relative-density-proxy
Use --smoke for a short end-to-end execution check, not an accuracy experiment.
The historical profile preserves recoverable development settings; its deleted
warm-start backbone cannot be reconstructed exactly from the surviving files.
For site-held-out five-fold experiments on calibrated records use main.py.
Both profiles add auxiliary objectives, input calibration and a residual head;
fresh disables historical test-selected settings but is not pure six-loss SPAI.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from calibration import ResidualPhysicalCalibration, calibration_features
from inference import load_model
from losses import MineLoss, TrainingConfig, stage_weights
from metrics import MineMetrics
from network import ModelConfig
from quality_calibration import QualityPhysicalExpert
from quality_features import build_quality_features
from raw_shape_features import raw_shape_features
from retraining_data import (
    ArrayDataset,
    make_split,
    masked_inputs,
    read_legacy,
    supervised_arrays,
)
from train_opf import CalibratedSPAI, auxiliary_losses, to_device
from options_io import load_mapping
from model_catalog import VARIANTS, ALIASES, ArchitectureConfig
from robustness import AugmentationConfig, perturb_inputs
from scene import Scene, encode_context, CONTEXT_DIM
from training_config import (
    RetrainingConfig,
    calibration_config,
    dense_expert_config,
    historical_post_config,
    historical_reference_config,
    weak_expert_config,
)

METRICS = [
    "LER_percent",
    "LE_h_percent",
    "LE_rho_percent",
    "AE_h_percent",
    "AE_rho_percent",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def score(metrics):
    return sum(metrics[k] if metrics[k] is not None else 1000.0 for k in METRICS)


def metric_from_prediction(out, counts, h, rho):
    meter = MineMetrics()
    for b in range(counts.shape[0]):
        for t in range(counts.shape[1]):
            g, p = int(counts[b, t]), int(out["layer_count"][b, t])
            if g:
                meter.update(
                    out["layer_thickness_m"][b, t, :p].detach().cpu().numpy(),
                    out["layer_density"][b, t, :p].detach().cpu().numpy(),
                    h[b, t, :g].detach().cpu().numpy(),
                    rho[b, t, :g].detach().cpu().numpy(),
                )
    return meter.summary()


@torch.no_grad()
def backbone_validation(model, loader, device, fingerprints):
    model.eval()
    meter = MineMetrics()
    for values in loader:
        batch = to_device(values, device)
        out = model(masked_inputs(batch, fingerprints))
        for b in range(batch["phase"].shape[0]):
            for t in range(batch["phase"].shape[1]):
                p, g = int(out["layer_count"][b, t]), int(batch["layer_count"][b, t])
                meter.update(
                    out["layer_thickness_m"][b, t, :p].cpu().numpy(),
                    out["layer_density"][b, t, :p].cpu().numpy(),
                    batch["layer_thickness_m_gt"][b, t, :g].cpu().numpy(),
                    batch["layer_density_gt"][b, t, :g].cpu().numpy(),
                )
    return meter.summary()


def fit_backbone(
    arrays,
    split,
    cfg,
    folder,
    device,
    fingerprints,
    use_erf,
    checkpoint=None,
    variant="mine",
    research=None,
    augmentation=None,
):
    seed_all(cfg.seed)
    model = CalibratedSPAI(ModelConfig(use_erf=use_erf), variant, research).to(device)
    if checkpoint:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved.get("backbone_state", saved.get("model")))
        if saved.get("backbone_config", saved.get("config", {}).get("model")) != asdict(
            model.core.config
        ):
            raise ValueError("Warm-start backbone configuration differs from this run.")
        if (
            saved.get("backbone_variant", "mine") != model.variant_name
            or saved.get("architecture_config", model.architecture_config)
            != model.architecture_config
        ):
            raise ValueError("Warm-start research variant/configuration differs.")
        return model
    ti, vi = split["train_indices"], split["validation_indices"]
    counts = arrays["layer_count"][ti, 0]
    frequencies = {int(k): int((counts == k).sum()) for k in np.unique(counts)}
    sampler = WeightedRandomSampler(
        torch.tensor([1 / frequencies[int(k)] for k in counts], dtype=torch.double),
        len(ti),
        replacement=True,
    )
    train = DataLoader(
        ArrayDataset(arrays, ti), batch_size=cfg.backbone_batch, sampler=sampler
    )
    validation = DataLoader(ArrayDataset(arrays, vi), batch_size=cfg.cache_batch)
    N = arrays["phase"].shape[-1]
    prior = float(arrays["layer_count"][ti].mean() + 1) / N
    torch.nn.init.constant_(
        model.core.skeleton.head.bias, math.log(prior / (1 - prior))
    )
    criterion = MineLoss(TrainingConfig(stage_epochs=cfg.stage_epochs))
    aux = SimpleNamespace(
        **{
            k: getattr(cfg, k)
            for k in [
                "aux_bce",
                "aux_location",
                "aux_time",
                "aux_thickness",
                "aux_density",
                "sigma",
            ]
        }
    )
    axis = torch.arange(-cfg.sigma * 3, cfg.sigma * 3 + 1, device=device).float()
    aux.kernel = torch.exp(-axis.square() / (2 * cfg.sigma**2)).view(1, 1, -1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.backbone_lr, weight_decay=cfg.backbone_weight_decay
    )
    best, selected = float("inf"), None
    for epoch in range(sum(cfg.stage_epochs)):
        model.train()
        stage, weights = stage_weights(epoch, criterion.config)
        lr = cfg.backbone_lr * (
            0.2
            if epoch >= sum(cfg.stage_epochs[:2]) + int(cfg.stage_epochs[2] * 0.7)
            else 0.5 if stage == 3 else 1.0
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = []
        for values in train:
            batch = to_device(values, device)
            if augmentation is not None:
                batch = perturb_inputs(batch, augmentation)
            out = model(masked_inputs(batch, fingerprints), stage)
            mine, _, _ = criterion(out, batch, epoch)
            extra, terms = auxiliary_losses(out, batch, stage, aux)
            if stage == 2:
                extra -= (
                    aux.aux_density
                    * terms["pre_density_relative"]
                    * (1 - weights["pre_rho"] / 0.5)
                )
            if stage == 3:
                extra -= (
                    aux.aux_thickness * terms["thickness_relative"]
                    + aux.aux_density * terms["density_relative"]
                ) * (1 - weights["h"] / 2)
            loss = mine + extra
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite backbone loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            ):
                raise FloatingPointError("Nonfinite backbone gradient.")
            optimizer.step()
            losses.append(float(loss.detach()))
        checkpoint = {
            "model": cpu_state(model),
            "backbone_config": asdict(model.core.config),
            "backbone_variant": model.variant_name,
            "architecture_config": model.architecture_config,
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "recipe": asdict(cfg),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        torch.save(checkpoint, folder / "backbone_latest.pt")
        if stage == 3 and ((epoch + 1) % 5 == 0 or epoch + 1 == sum(cfg.stage_epochs)):
            metrics = backbone_validation(model, validation, device, fingerprints)
            if score(metrics) < best:
                best, selected = score(metrics), cpu_state(model)
                checkpoint["validation"] = metrics
                torch.save(checkpoint, folder / "backbone_best.pt")
        print(
            json.dumps(
                {
                    "stage": "backbone",
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "lr": lr,
                }
            ),
            flush=True,
        )
    if selected is None:
        raise RuntimeError("No stage III backbone validation checkpoint.")
    model.load_state_dict(selected)
    return model.eval()


@torch.no_grad()
def feature_cache(
    backbone, arrays, indices, cfg, device, fingerprints, expert_cfg=None, style=0
):
    backbone.eval()
    parts = {}
    for begin in range(0, len(indices), cfg.cache_batch):
        ids = indices[begin : begin + cfg.cache_batch]
        batch = {
            k: torch.from_numpy(np.array(v[ids], copy=True)).to(device)
            for k, v in arrays.items()
        }
        if style:
            batch = quality_augmentation(batch, style)
        B, T, N = batch["phase"].shape
        if "time_s" not in batch or "depth_m" not in batch:
            raise ValueError("Feature caching requires explicit calibrated axes.")
        inputs = masked_inputs(batch, fingerprints)
        base = (
            backbone(inputs)
            if expert_cfg is None or expert_cfg.feature_mode == "backbone"
            else None
        )
        vectors = None
        if base is not None:
            vectors = torch.cat(
                [
                    calibration_features(base, cfg.pool_bins),
                    raw_shape_features(inputs, cfg.raw_shape_bins),
                ],
                -1,
            )
        features = (
            vectors
            if expert_cfg is None
            else build_quality_features(inputs, vectors, expert_cfg)
        )
        result = {
            "features": features,
            "layer_count": batch["layer_count"],
            "h_gt": batch["layer_thickness_m_gt"],
            "rho_gt": batch["layer_density_gt"],
        }
        if expert_cfg is None:
            result.update(
                base_density=base["layer_density"],
                base_range_m=base["layer_delays_s"] * 299792458 / 2,
            )
        for key, value in result.items():
            parts.setdefault(key, []).append(value.cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def quality_augmentation(batch, style):
    out = dict(batch)
    B, T, N = out["tof"].shape
    device = out["tof"].device
    if style in [4, 5, 7, 8]:
        mask = out["tof"].abs() > 0.001 * out["tof"].abs().amax(-1, keepdim=True)
        bins = torch.arange(N, device=device)[None, None]
        left = torch.where(mask, bins, N).amin((1, 2))
        right = torch.where(mask, bins, 0).amax((1, 2))
        low = -torch.minimum((left - 8).clamp_min(0), torch.full_like(left, 180))
        high = torch.minimum((N - 8 - right).clamp_min(0), torch.full_like(right, 180))
        shifts = torch.round(
            low.float() + torch.rand(B, device=device) * (high - low).float()
        )
        query = torch.arange(N, device=device)[None].float() - shifts[:, None]
        grid = torch.stack([query / (N - 1) * 2 - 1, torch.zeros_like(query)], -1)[
            :, None
        ]
        for k in ["phase", "scatter", "tof"]:
            out[k] = F.grid_sample(
                out[k][:, :, None], grid, align_corners=True, padding_mode="zeros"
            ).squeeze(2)
    for key in ["phase", "scatter", "tof"]:
        v = out[key]
        gain = torch.exp(
            torch.randn(B, 1, 1, device=device) * (1.1 if style in [1, 5, 8] else 0.35)
        )
        if style in [1, 5, 8]:
            gain *= 0.25
        out[key] = v * gain
    if style in [2, 3, 4, 6, 7, 8]:
        for key in ["phase", "scatter", "tof"]:
            v = out[key]
            peak = v.abs().amax(-1, keepdim=True).clamp_min(1e-5)
            if style in [3, 6, 7]:
                noise = F.avg_pool1d(
                    torch.randn(B * T, 1, N, device=device), 31, stride=1, padding=15
                ).reshape(B, T, N)
                v = v + 0.08 * peak * noise
            v = (
                v
                - v.amin(-1, keepdim=True)
                + (0.01 + 0.15 * torch.rand(B, 1, 1, device=device)) * peak
            )
            if style in [2, 3, 4, 7]:
                v = v / v.amax(-1, keepdim=True).clamp_min(1e-6)
            out[key] = v
    out["scatter_valid"] = out["scatter"] > 0
    return out


def fit_physical_head(data, ti, vi, cfg, folder, device):
    seed_all(cfg.seed)
    a = {k: v.to(device) for k, v in data.items()}
    x = a["features"][ti].flatten(0, 1)
    model = ResidualPhysicalCalibration(
        x.shape[-1], x.mean(0), x.std(0).clamp_min(0.01), calibration_config()
    ).to(device)
    return fit_cached_model(
        model,
        a,
        ti,
        vi,
        cfg.head_epochs,
        cfg.head_lr,
        cfg.head_batch,
        cfg.head_weight_decay,
        0.5,
        3.0,
        cfg,
        folder / "calibration_best.pt",
        "calibration",
        False,
    )


def fit_cached_model(
    model,
    a,
    ti,
    vi,
    epochs,
    lr,
    batch_size,
    weight_decay,
    weight_power,
    h_weight,
    cfg,
    destination,
    kind,
    expert,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    data = {k: v[ti].flatten(0, 1) for k, v in a.items()}
    L = data["layer_count"]
    freq = torch.bincount(L, minlength=11).float().clamp_min(1)
    weights = freq[L].pow(-weight_power)
    weights /= weights.mean()
    best, selected = float("inf"), None
    history = []
    for epoch in range(epochs):
        model.train()
        scheduled_lr = lr * (
            (0.1 if expert else 0.05)
            + (0.9 if expert else 0.95) * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        )
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr
        perm = torch.randperm(len(L), device=L.device)
        losses = []
        for begin in range(0, len(L), batch_size):
            idx = perm[begin : begin + batch_size]
            x = data["features"][idx]
            if expert and model.config.feature_mode == "peak_spectral":
                x = x + 0.02 * model.feature_std * torch.randn_like(x)
            out = (
                model(x)
                if expert
                else model(x, data["base_density"][idx], data["base_range_m"][idx])
            )
            counts = L[idx]
            mask = torch.arange(10, device=L.device) < counts[:, None]
            eh = (out["layer_thickness_m"] - data["h_gt"][idx]) / data["h_gt"][
                idx
            ].clamp_min(0.005 if expert else 0.01)
            er = (out["layer_density"] - data["rho_gt"][idx]) / data["rho_gt"][
                idx
            ].clamp_min(0.1)
            h = (
                F.smooth_l1_loss(eh, torch.zeros_like(eh), beta=0.05, reduction="none")
                * mask
            ).sum(-1) / counts
            r = (
                F.smooth_l1_loss(er, torch.zeros_like(er), beta=0.03, reduction="none")
                * mask
            ).sum(-1) / counts
            ce = F.cross_entropy(out["count_logits"], counts - 1, reduction="none")
            loss = ((ce + h_weight * h + 2 * r) * weights[idx]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite " + kind + " objective.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if (
            epoch == 0
            or (epoch + 1) % (20 if expert else 10) == 0
            or epoch + 1 == epochs
        ):
            model.eval()
            with torch.no_grad():
                x = a["features"][vi]
                S, T, _ = x.shape
                raw = (
                    model(x.flatten(0, 1))
                    if expert
                    else model(
                        x.flatten(0, 1),
                        a["base_density"][vi].flatten(0, 1),
                        a["base_range_m"][vi].flatten(0, 1),
                    )
                )
                out = {k: v.reshape(S, T, *v.shape[1:]) for k, v in raw.items()}
                out["layer_density"] = torch.quantile(
                    out["layer_density"], 0.5, dim=1, keepdim=True
                ).expand_as(out["layer_density"])
                out["layer_delays_s"] = torch.quantile(
                    out["layer_delays_s"], 0.5, dim=1, keepdim=True
                ).expand_as(out["layer_delays_s"])
                if expert:
                    out["layer_count"] = (
                        out["count_logits"].mean(1, keepdim=True).argmax(-1) + 1
                    ).expand(S, T)
                out["layer_thickness_m"] = (
                    299792458
                    * out["layer_delays_s"]
                    / (2 * (1 + 0.845 * out["layer_density"]))
                )
                metrics = metric_from_prediction(
                    out, a["layer_count"][vi], a["h_gt"][vi], a["rho_gt"][vi]
                )
            row = {
                "stage": kind,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "validation": metrics,
            }
            history.append(row)
            if score(metrics) < best:
                best, selected = score(metrics), cpu_state(model)
                torch.save(
                    {
                        "state": selected,
                        "epoch": epoch + 1,
                        "config": asdict(model.config),
                        "validation": metrics,
                        "optimizer": optimizer.state_dict(),
                    },
                    destination,
                )
            print(json.dumps(row), flush=True)
    destination.with_suffix(".json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(selected)
    return model.eval()


def fit_expert(
    backbone, arrays, split, expert_cfg, cfg, folder, device, fingerprints, kind
):
    seed_all(cfg.expert_seed)
    eligible = sorted(split["eligible_indices"])
    parts = [
        feature_cache(
            backbone, arrays, eligible, cfg, device, fingerprints, expert_cfg, style
        )
        for style in range(cfg.augmentation_styles)
    ]
    a = {k: torch.cat([p[k] for p in parts]).to(device) for k in parts[0]}
    source_ids = np.tile(eligible, cfg.augmentation_styles)
    ti = np.flatnonzero(np.isin(source_ids, split["train_indices"])).tolist()
    vi = np.flatnonzero(np.isin(source_ids, split["validation_indices"])).tolist()
    x = a["features"][ti].flatten(0, 1)
    seed_all(cfg.expert_seed)
    model = QualityPhysicalExpert(
        x.shape[-1], x.mean(0), x.std(0).clamp_min(0.01), expert_cfg
    ).to(device)
    return fit_cached_model(
        model,
        a,
        ti,
        vi,
        cfg.expert_epochs,
        cfg.expert_lr,
        cfg.expert_batch,
        cfg.expert_weight_decay,
        0.75 if kind == "dense_expert" else 0.5,
        4.0,
        cfg,
        folder / (kind + "_best.pt"),
        kind,
        True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--profile",
        choices=["historical", "fresh"],
        default="fresh",
        help="historical restores disclosed test-selected global/gate settings; fresh omits them",
    )
    p.add_argument(
        "--allow-test-selected-history",
        action="store_true",
        help="Explicitly permit test-selected development settings; never an independent main-method evaluation.",
    )
    p.add_argument(
        "--legacy-grid-mm",
        type=float,
        help="Explicitly confirmed legacy physical-depth and air-range bin step, in mm.",
    )
    p.add_argument(
        "--allow-relative-density-proxy",
        action="store_true",
        help="Opt in to the legacy absolute-density proxy; this is a development-only target.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--max-source-gib",
        type=float,
        default=8.0,
        help="Maximum decoded arrays per H5 file before loading",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit CLI seed overrides recipe JSON; otherwise recipe/default 431",
    )
    p.add_argument(
        "--backbone-checkpoint",
        type=Path,
        help="Optional recovered backbone; omitted means fresh initialization",
    )
    p.add_argument(
        "--fingerprints",
        nargs="+",
        choices=["phase", "scatter", "tof"],
        default=["phase", "scatter", "tof"],
    )
    p.add_argument("--erf", choices=["on", "off"], default="on")
    p.add_argument("--network-variant", choices=(*VARIANTS, *ALIASES), default="mine")
    p.add_argument(
        "--architecture-config",
        type=Path,
        help="JSON partial ArchitectureConfig overrides",
    )
    p.add_argument(
        "--recipe-json", type=Path, help="JSON partial RetrainingConfig overrides"
    )
    p.add_argument(
        "--scene-config", type=Path, help="Optional SI-unit scene metadata JSON"
    )
    p.add_argument(
        "--augment-config", type=Path, help="Optional receiver perturbation JSON"
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Four records and short budgets; never use its output for reported accuracy",
    )
    args = p.parse_args()
    if args.profile == "historical" and not args.allow_test_selected_history:
        p.error(
            "Historical settings used test labels. Explicit --allow-test-selected-history is required for development reconstruction. Use main.py --all-folds for independent six-loss SPAI evaluation."
        )
    if not args.allow_relative_density_proxy:
        p.error(
            "Legacy labels do not provide calibrated relative density. Use main.py with calibrated targets, or explicitly opt in with --allow-relative-density-proxy for development only."
        )
    args.network_variant = ALIASES.get(args.network_variant, args.network_variant)
    if not (args.data_dir / "train.h5").is_file():
        p.error("--data-dir must contain train.h5")
    recipe = load_mapping(args.recipe_json) if args.recipe_json else {}
    if args.seed is not None:
        recipe["seed"] = args.seed
    cfg = RetrainingConfig(**recipe).validate()
    research = ArchitectureConfig(
        **(load_mapping(args.architecture_config) if args.architecture_config else {})
    ).validate()
    scene = (
        Scene.from_dict(load_mapping(args.scene_config))
        if args.scene_config
        else Scene()
    )
    augmentation = AugmentationConfig(
        **(load_mapping(args.augment_config) if args.augment_config else {})
    ).validate()
    if research.context_dim and (
        args.network_variant == "mine"
        or research.context_dim != CONTEXT_DIM
        or args.scene_config is None
    ):
        p.error(
            "Context requires a research variant, scene-config, and context_dim=32."
        )
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if not math.isfinite(args.max_source_gib) or args.max_source_gib <= 0:
        p.error("--max-source-gib must be positive and finite.")
    source = read_legacy(
        args.data_dir / "train.h5",
        max_bytes=args.max_source_gib * 2**30,
        grid_step_m=None if args.legacy_grid_mm is None else args.legacy_grid_mm / 1000,
        require_axes=True,
    )
    split = make_split(
        source,
        args.data_dir,
        cfg.seed,
        cfg.validation_fraction,
        max_bytes=args.max_source_gib * 2**30,
    )
    if args.smoke:
        split["train_indices"] = split["train_indices"][:2]
        split["validation_indices"] = split["validation_indices"][:2]
        split["eligible_indices"] = split["train_indices"] + split["validation_indices"]
        cfg = replace(
            cfg,
            stage_epochs=(1, 1, 2),
            backbone_batch=2,
            head_epochs=1,
            expert_epochs=1,
            augmentation_styles=2,
            cache_batch=2,
        )
    arrays = supervised_arrays(source)
    del source
    context = encode_context(scene) if research.context_dim else None
    if context is not None:
        arrays["context"] = np.broadcast_to(
            np.array(context, dtype=np.float32),
            (*arrays["phase"].shape[:2], len(context)),
        ).copy()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=False)
    run = {
        "recipe": asdict(cfg),
        "split": split,
        "profile": args.profile,
        "legacy_grid_mm": args.legacy_grid_mm,
        "network_variant": args.network_variant,
        "architecture_config": asdict(research),
        "scene_metadata": scene.to_dict(),
        "receiver_augmentation": asdict(augmentation),
        "fingerprints": args.fingerprints,
        "erf": args.erf,
        "smoke_only": args.smoke,
        "previous_weights_recovered": False,
        "historical_metrics_guaranteed": False,
        "known_missing_state": "deleted multi-stage warm-start backbone and optimizer/RNG lineage",
        "selection": "train-side validation only in this execution",
        "historical_parameter_provenance": "historical global posterior and gates previously selected with test-labelled development data",
        "relative_density_supervision": "absolute density proxy, since the reference specification does not specify the legacy target construction",
        "evaluation_role": "development reconstruction; no independent reproduction claim",
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
    }
    (folder / "run_config.json").write_text(json.dumps(run, indent=2))
    backbone = fit_backbone(
        arrays,
        split,
        cfg,
        folder,
        device,
        args.fingerprints,
        args.erf == "on",
        args.backbone_checkpoint,
        args.network_variant,
        asdict(research),
        augmentation,
    )
    data = feature_cache(
        backbone,
        arrays,
        list(range(len(arrays["phase"]))),
        cfg,
        device,
        args.fingerprints,
    )
    head = fit_physical_head(
        data, split["train_indices"], split["validation_indices"], cfg, folder, device
    )
    bundle = {
        "backbone_state": cpu_state(backbone),
        "backbone_config": asdict(backbone.core.config),
        "backbone_variant": backbone.variant_name,
        "architecture_config": backbone.architecture_config,
        "default_context": context,
        "scene_metadata": scene.to_dict(),
        "calibration_state": cpu_state(head),
        "calibration_config": asdict(head.config),
        "input_dim": int(data["features"].shape[-1]),
        "pool_bins": cfg.pool_bins,
        "auxiliary_features": {"kind": "raw_shape_v1", "bins": cfg.raw_shape_bins},
        "input_fingerprints": args.fingerprints,
        "reference_alignment": (
            asdict(historical_reference_config())
            if args.profile == "historical"
            else None
        ),
        "post_calibration": (
            asdict(historical_post_config())
            if args.profile == "historical"
            else {"enforce_observed_range": True}
        ),
        "model_kind": "SPAI + explicit supplementary physical calibration",
        "evaluation_role": run["evaluation_role"],
        "retraining_config": run,
        "evaluation_policies": {
            "test_d1_a1": {"empty_label3": "accepted_ablation_failure"}
        },
    }
    if args.profile == "historical":
        maximum = (arrays["phase"].shape[-1] - 1) * 0.001
        for name, expert_cfg, prefix in [
            ("dense_expert", dense_expert_config(maximum), "quality_expert"),
            ("weak_expert", weak_expert_config(maximum), "secondary_quality_expert"),
        ]:
            model = fit_expert(
                backbone,
                arrays,
                split,
                expert_cfg,
                cfg,
                folder,
                device,
                args.fingerprints,
                name,
            )
            bundle[prefix + "_state"] = cpu_state(model)
            bundle[prefix + "_config"] = asdict(model.config)
            bundle[prefix + "_input_dim"] = len(model.feature_mean)
            del model
    destination = folder / "model.pt"
    torch.save(bundle, destination)
    loaded, _ = load_model(destination, device=device)
    sample = to_device(
        next(
            iter(
                DataLoader(
                    ArrayDataset(arrays, split["validation_indices"][:1]), batch_size=1
                )
            )
        ),
        device,
    )
    # Labels are stripped before inference; the serialized model uses input only.
    inputs = {
        k: sample[k]
        for k in ["phase", "scatter", "tof", "scatter_valid", "time_s", "depth_m"]
    }
    with torch.no_grad():
        result = loaded(inputs)
    if not all(
        torch.isfinite(result[k]).all()
        for k in ["layer_density", "layer_thickness_m", "layer_delays_s"]
    ):
        raise FloatingPointError("Serialized model produced nonfinite output.")
    print(
        json.dumps(
            {
                "completed": True,
                "model": str(destination),
                "smoke_only": args.smoke,
                "historical_metrics_guaranteed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
