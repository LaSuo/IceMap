"""Train the SPAI backbone on legacy H5 with five held-out OPF structure folds."""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from losses import MineLoss, TrainingConfig, stage_weights
from main import forward_batch
from metrics import MineMetrics
from network import ModelConfig, SPAIIceInversionNet, pool_layers
from protocol import sha256_file
from retraining_data import read_legacy, signature, supervised_arrays
from train_opf import TARGET


PROTOCOL = "icemap.opf5fold.structure_proxy.v1"


def structure(name):
    match = re.match(r"^(\d+_e\d+r\d+_t\d+)b\d+(?:_|$)", str(name))
    if match is None:
        raise ValueError(f"Missing acquisition/structure/batch identifier: {name!r}")
    return match.group(1)


def load_data(data_dir, grid_mm, stride, device, seed):
    paths = [data_dir / name for name in ("train.h5", "test_opf.h5")]
    sources = [
        read_legacy(p, grid_step_m=grid_mm / 1000, require_axes=True) for p in paths
    ]
    left, right = sources
    if left["phase"].shape[1:] != right["phase"].shape[1:] or any(
        not np.array_equal(left[k], right[k]) for k in ("time_s", "depth_m")
    ):
        raise ValueError(
            "Training and OPF windows must share frame sizes and calibrated axes."
        )
    names = np.concatenate([s["names"] for s in sources])
    groups = np.array([structure(name) for name in names])
    first_opf = len(left["names"])
    opf = np.arange(first_opf, len(names))
    train_pool = []
    seen = {}
    opf_names = set(names[opf])
    if len(opf_names) != len(opf):
        raise ValueError("OPF record names must be unique.")
    for source, offset in ((right, first_opf), (left, 0)):
        for i, name in enumerate(source["names"]):
            j = offset + i
            digest = signature(source, i)
            if digest in seen:
                other = seen[digest]
                if groups[j] != groups[other] or offset == first_opf:
                    raise ValueError(
                        f"Unresolved duplicate recordings: {name} and {names[other]}."
                    )
                continue
            seen[digest] = j
            if offset == 0 and name not in opf_names:
                train_pool.append(j)
    opf_groups = sorted(set(groups[opf]))
    if len(opf_groups) < 5:
        raise ValueError(
            "Five folds require at least five distinct OPF structure groups."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(opf_groups)
    assignments = {group: i % 5 for i, group in enumerate(opf_groups)}
    arrays = [supervised_arrays(source) for source in sources]
    for arr in arrays:
        original = arr["boundary_target"]
        counts = (arr["layer_count"] + 1).ravel()
        rec, frame, sample = np.nonzero(original)
        rank = np.arange(len(sample)) - np.repeat(np.cumsum(counts) - counts, counts)
        native_bins = np.zeros((*original.shape[:2], 11), dtype=np.int32)
        native_bins[rec, frame, rank] = sample
        arr["native_boundary_bins"] = native_bins
        compact = np.maximum.reduceat(
            original, np.arange(0, original.shape[-1], stride), axis=-1
        )
        if not np.array_equal(compact.sum(-1), arr["layer_count"] + 1):
            raise ValueError(
                "input_stride merges distinct interfaces; select a smaller stride."
            )
        arr["boundary_target"] = compact
    data = {
        key: torch.from_numpy(np.concatenate([a[key] for a in arrays])).to(device)
        for key in arrays[0]
    }
    inventory = {p.name: sha256_file(p) for p in paths}
    return (
        data,
        names,
        groups,
        opf,
        np.asarray(train_pool, dtype=int),
        assignments,
        inventory,
    )


def split_fold(fold, groups, opf, train_pool, assignments):
    test_groups = {g for g, f in assignments.items() if f == fold}
    val_groups = {g for g, f in assignments.items() if f == (fold + 1) % 5}
    test = opf[np.isin(groups[opf], sorted(test_groups))]
    val = opf[np.isin(groups[opf], sorted(val_groups))]
    candidates = np.concatenate([train_pool, opf])
    train = candidates[~np.isin(groups[candidates], sorted(test_groups | val_groups))]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "Fold must have nonempty training, validation and test partitions."
        )
    return train, val, test


def batch_for(data, indices, device, frames=None, rng=None):
    ix = torch.as_tensor(indices, device=device, dtype=torch.long)
    t = data["phase"].shape[1]
    frame_ix = None
    if frames is not None and frames < t:
        first = rng.integers(0, t - frames + 1, len(indices))
        frame_ix = torch.as_tensor(first[:, None] + np.arange(frames), device=device)
    return {
        k: (
            v[ix[:, None], frame_ix]
            if frame_ix is not None and v.ndim >= 2 and k not in {"time_s", "depth_m"}
            else v.index_select(0, ix)
        )
        for k, v in data.items()
    }


def minibatches(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def stretch_ice(batch, stride, log_range, keep_probability=0.0):
    """Apply one post-surface geometric scale per training recording."""
    native = batch["native_boundary_bins"]
    size = batch["phase"].shape[-1]
    surface = native[..., :1]
    valid = torch.arange(native.shape[-1], device=native.device) <= batch[
        "layer_count"
    ].unsqueeze(-1)
    scale = (
        torch.empty(native.shape[0], 1, 1, device=native.device)
        .uniform_(-log_range, log_range)
        .exp()
    )
    if keep_probability:
        keep = torch.rand_like(scale) < keep_probability
        scale = torch.where(keep, 1.0, scale)

    def project(factor):
        return (surface + (native - surface) * factor).round().long()

    projected = project(scale)
    indices = projected // stride
    merged = ((indices.diff(dim=-1) <= 0) & valid[..., 1:]).any(dim=(1, 2))
    clipped = ((projected >= size) & valid).any(dim=(1, 2))
    scale = torch.where((merged | clipped)[:, None, None], 1.0, scale)
    projected = project(scale).masked_fill(~valid, 0)
    indices = projected // stride
    target = torch.zeros_like(batch["boundary_target"])
    target.scatter_add_(-1, indices, valid.to(target.dtype))

    x = torch.arange(size, device=native.device) - surface
    source = surface + torch.minimum(x, torch.zeros_like(x)) + x.clamp_min(0) / scale
    outside = source > size - 1
    source = source.clamp(0, size - 1)
    left = source.floor().long()
    right = (left + 1).clamp_max(size - 1)
    mix = source - left
    result = dict(
        batch,
        boundary_target=target,
        native_boundary_bins=projected,
        layer_thickness_m_gt=batch["layer_thickness_m_gt"] * scale,
    )
    for name in ("phase", "scatter", "tof"):
        values = batch[name]
        moved = torch.lerp(values.gather(-1, left), values.gather(-1, right), mix)
        result[name] = moved if name == "phase" else moved.masked_fill(outside, 0)
    result["scatter_valid"] = (
        batch["scatter_valid"].gather(-1, source.round().long()) & ~outside
    )
    profile = batch["density_profile_gt"]
    old_depth = (
        (torch.arange(profile.shape[-1], device=native.device) / scale).floor().long()
    )
    result["density_profile_gt"] = profile.gather(
        -1, old_depth.expand_as(profile).clamp_max(profile.shape[-1] - 1)
    ).masked_fill(old_depth >= profile.shape[-1], 0)
    return result


@torch.no_grad()
def score(model, data, indices, batch_size, device, meters, names=None):
    model.eval()
    rows = []
    for block in minibatches(indices, batch_size):
        batch = batch_for(data, block, device)
        out = forward_batch(model, batch)
        keys = ("layer_count", "layer_thickness_m", "layer_density")
        pred = [out[k].cpu().numpy() for k in keys]
        truth = [
            batch[k].cpu().numpy()
            for k in ("layer_count", "layer_thickness_m_gt", "layer_density_gt")
        ]
        for b, i in enumerate(block):
            for t in range(pred[0].shape[1]):
                count, expected = pred[0][b, t], truth[0][b, t]
                layers = (
                    pred[1][b, t, :count],
                    pred[2][b, t, :count],
                    truth[1][b, t, :expected],
                    truth[2][b, t, :expected],
                )
                for meter in meters:
                    meter.update(*layers)
                if names is not None:
                    rows.append(
                        {
                            "record": str(names[i]),
                            "frame": t,
                            **{
                                k: v.tolist()
                                for k, v in zip(
                                    ("pred_h_m", "pred_rho", "true_h_m", "true_rho"),
                                    layers,
                                )
                            },
                        }
                    )
    return rows


def validation_score(metrics):
    # Validation only. Equal weight on the five physical errors, with no target matching.
    return sum(metrics[k] if metrics[k] is not None else 100 for k in TARGET) / len(
        TARGET
    )


def sparse_boundary_loss(logits, boundary_target, kernel, ordinal_weight=0.0):
    """Unit-mass soft labels with a full penalty for false positive mass."""
    smoothed = (
        F.conv1d(
            boundary_target.flatten(0, 1).unsqueeze(1),
            kernel,
            padding=kernel.numel() // 2,
        )
        .reshape_as(boundary_target)
        .float()
    )
    n_events = boundary_target.sum(dim=-1, keepdim=True)
    smoothed = (smoothed * n_events / smoothed.sum(dim=-1, keepdim=True)).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits, smoothed, reduction="none")
    loss = (bce.sum(-1) / n_events.squeeze(-1)).mean()
    if ordinal_weight:
        mass_error = (logits.sigmoid() - smoothed).cumsum(-1) / n_events
        loss = loss + ordinal_weight * mass_error.abs().mean()
    return loss


def association_loss(logits, batch, stride):
    """Supervise each ordered event with probability mass near its interface."""
    posterior = logits.transpose(-1, -2).log_softmax(-1)
    offsets = torch.arange(-2, 3, device=logits.device)
    centers = batch["native_boundary_bins"].long() // stride
    indices = centers[..., None] + offsets
    outside = (indices < 0) | (indices >= posterior.shape[-1])
    local = posterior.gather(-1, indices.clamp(0, posterior.shape[-1] - 1))
    loss = -local.masked_fill(outside, -torch.inf).logsumexp(dim=-1)
    valid = torch.arange(centers.shape[-1], device=logits.device) <= batch[
        "layer_count"
    ].unsqueeze(-1)
    return loss.masked_fill(~valid, 0).sum() / valid.sum()


def interface_role_loss(logits, batch, stride):
    """Supervise local surface, interior, and bottom evidence against clutter."""
    bins = logits.shape[-2]
    center = batch["native_boundary_bins"].long() // stride
    rank = torch.arange(center.shape[-1], device=logits.device)
    count = batch["layer_count"].unsqueeze(-1)
    role = torch.where(rank == 0, 0, torch.where(rank == count, 2, 1))
    offset = torch.arange(-2, 3, device=logits.device)
    local = center.unsqueeze(-1) + offset
    valid = (rank <= count).unsqueeze(-1) & (local >= 0) & (local < bins)
    weight = logits.new_tensor((0.15, 0.6, 1.0, 0.6, 0.15)) * valid
    indices = role.unsqueeze(-1) * bins + local.clamp(0, bins - 1)
    target = logits.new_zeros((*logits.shape[:2], 3 * bins))
    target.scatter_add_(-1, indices.flatten(-2), weight.flatten(-2))
    target = target.reshape(*logits.shape[:2], 3, bins).clamp_max(1)
    evidence = logits.transpose(-1, -2)
    positive = (F.softplus(-evidence) * target).sum((0, 1, 3))
    positive = (positive / target.sum((0, 1, 3)).clamp_min(1)).mean()
    negative = F.softplus(evidence) * (1 - target)
    return positive + 0.5 * (
        negative.mean() + negative.topk(min(8, bins), dim=-1).values.mean()
    )


def interval_density_loss(model, out, batch):
    """Train density in physical intervals while the timing branch is still learning."""
    stride = model.config.input_stride
    evidence = out["density_evidence"]
    encoded_bins = torch.arange(evidence.shape[2], device=evidence.device) * stride
    bounds = batch["native_boundary_bins"].long()
    layer_count = batch["layer_count"]
    active = torch.arange(
        bounds.shape[-1] - 1, device=bounds.device
    ) < layer_count.unsqueeze(-1)
    coords = encoded_bins[None, None, :, None]
    inside = (
        (coords >= bounds[..., :-1].unsqueeze(2))
        & (coords < bounds[..., 1:].unsqueeze(2))
        & active.unsqueeze(2)
    )
    valid = batch["scatter_valid"]
    if stride > 1:
        valid = (
            F.max_pool1d(
                valid.flatten(0, 1).unsqueeze(1).float(), stride, ceil_mode=True
            )
            .reshape(*valid.shape[:2], -1)
            .bool()
        )
    density = pool_layers(evidence, (inside & valid.unsqueeze(-1)).float())
    delay = pool_layers(out["delay_sample_features"], inside.float())
    axis = batch["time_s"].unsqueeze(1).expand(-1, bounds.shape[1], -1)
    tau = axis.gather(-1, bounds).diff(dim=-1).clamp_min(0).float()
    rho_tilde = model.g_rho(density).squeeze(-1)
    rho0, _, _, rho, _, _ = model.invert_layer_delays(rho_tilde, delay, tau)
    target = batch["layer_density_gt"]
    error = (rho - target).abs() + 0.3 * (rho0 - target).abs()
    relative = error / target.clamp_min(0.1)
    return ((relative * active).sum(-1) / layer_count.clamp_min(1)).mean()


@torch.no_grad()
def refine_boundaries(
    model,
    data,
    val,
    batch_size,
    device,
    local_peak=False,
    ordered_peaks=False,
    temporal=False,
    density=False,
    observed=False,
    count_bias=False,
):
    """Select the boundary bias and label-free ToF readout on validation."""
    initial = model.config
    bias = model.skeleton.head.bias.clone()
    best = None
    for weight in dict.fromkeys(
        (initial.timing_fingerprint_weight, 0.0, 1.0, 4.0, 8.0)
    ):
        model.config = replace(initial, timing_fingerprint_weight=weight)
        for shift in (0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45):
            model.skeleton.head.bias.copy_(bias + shift)
            meter = MineMetrics()
            score(model, data, val, batch_size, device, [meter])
            validation = meter.summary()
            candidate = (validation_score(validation), shift, weight, validation)
            if best is None or candidate[0] < best[0]:
                best = candidate
    value, shift, weight, validation = best
    model.skeleton.head.bias.copy_(bias + shift)
    radius = initial.timing_peak_radius_bins
    for candidate_radius in (0, 2, 5, 10) if local_peak else ():
        model.config = replace(
            initial,
            timing_fingerprint_weight=weight,
            timing_peak_radius_bins=candidate_radius,
        )
        meter = MineMetrics()
        score(model, data, val, batch_size, device, [meter])
        candidate_metrics = meter.summary()
        candidate_value = validation_score(candidate_metrics)
        if candidate_value < value:
            value, radius, validation = (
                candidate_value,
                candidate_radius,
                candidate_metrics,
            )
    model.config = replace(
        initial, timing_fingerprint_weight=weight, timing_peak_radius_bins=radius
    )
    selected = model.config
    if ordered_peaks:
        # Spacing is measured on the encoded propagation grid. Keep the
        # original soft decoder in the comparison for low-evidence recordings.
        n = data["boundary_target"].shape[-1]
        for gap in (2, 4, 8, 12):
            if n <= gap * initial.max_layers:
                continue
            for timing_weight in dict.fromkeys((weight, 2.0, 6.0)):
                model.config = replace(
                    selected,
                    timing_fingerprint_weight=timing_weight,
                    timing_min_separation_bins=gap,
                )
                meter = MineMetrics()
                score(model, data, val, batch_size, device, [meter])
                metrics = meter.summary()
                candidate = validation_score(metrics)
                if candidate < value:
                    value, selected, validation = candidate, model.config, metrics
    if temporal:
        initial_readout = selected
        for window in (1, 3, 5):
            for conditioned in (False, True):
                model.config = replace(
                    initial_readout,
                    boundary_context_frames=window,
                    timing_condition_on_count=conditioned,
                )
                meter = MineMetrics()
                score(model, data, val, batch_size, device, [meter])
                metrics = meter.summary()
                candidate = validation_score(metrics)
                if candidate < value:
                    value, selected, validation = candidate, model.config, metrics
    if observed:
        n = data["boundary_target"].shape[-1]
        initial_readout = selected
        for gap in (2, 4, 8, 12):
            if n <= gap * initial.max_layers:
                continue
            model.config = replace(
                initial_readout,
                timing_observed_peaks_only=True,
                timing_min_separation_bins=gap,
            )
            meter = MineMetrics()
            score(model, data, val, batch_size, device, [meter])
            metrics = meter.summary()
            candidate = validation_score(metrics)
            if candidate < value:
                value, selected, validation = candidate, model.config, metrics
    count_tilt = 0.0
    if initial.compact_calibration:
        for prior in (0.0, 0.5, 1.0, 2.0):
            model.config = replace(selected, calibration_phase_prior=prior)
            meter = MineMetrics()
            score(model, data, val, batch_size, device, [meter])
            metrics = meter.summary()
            candidate = validation_score(metrics)
            if candidate < value:
                value, selected, validation = candidate, model.config, metrics
        count_output_bias = model.interface_calibration.count[-1].bias
        count_original = count_output_bias.clone()
        class_index = torch.arange(count_output_bias.numel(), device=device)
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
            model.config = replace(selected, calibration_strength=strength)
            tilts = (0.0, -0.4, 0.4, -0.8, 0.8) if count_bias and strength else (0.0,)
            for tilt in tilts:
                count_output_bias.copy_(count_original + tilt * class_index)
                meter = MineMetrics()
                score(model, data, val, batch_size, device, [meter])
                metrics = meter.summary()
                candidate = validation_score(metrics)
                if candidate < value:
                    value, selected, validation = candidate, model.config, metrics
                    count_tilt = tilt
        count_output_bias.copy_(count_original + count_tilt * class_index)
        for learned_weight in (0.0, 0.15, 0.35, 0.75):
            model.config = replace(selected, timing_learned_weight=learned_weight)
            meter = MineMetrics()
            score(model, data, val, batch_size, device, [meter])
            metrics = meter.summary()
            candidate = validation_score(metrics)
            if candidate < value:
                value, selected, validation = candidate, model.config, metrics
    model.config = selected
    density_shift = 0.0
    if density:
        # This is the scalar bias already present in SPAI's g_ref layer.
        # Recompute analytic thickness for every candidate density.
        density_bias = model.g_ref[-2].bias
        original_density_bias = density_bias.clone()
        for candidate_shift in (-0.04, -0.02, 0.02, 0.04):
            density_bias.copy_(original_density_bias + candidate_shift)
            meter = MineMetrics()
            score(model, data, val, batch_size, device, [meter])
            metrics = meter.summary()
            candidate = validation_score(metrics)
            if candidate < value:
                value, density_shift, validation = candidate, candidate_shift, metrics
        density_bias.copy_(original_density_bias + density_shift)
    return {
        "bias_shift": shift,
        "density_refinement_bias_shift": density_shift,
        "timing_fingerprint_weight": selected.timing_fingerprint_weight,
        "timing_learned_weight": selected.timing_learned_weight,
        "timing_peak_radius_bins": radius,
        "timing_min_separation_bins": selected.timing_min_separation_bins,
        "minimum_layers": selected.minimum_layers,
        "boundary_context_frames": selected.boundary_context_frames,
        "timing_condition_on_count": selected.timing_condition_on_count,
        "timing_observed_peaks_only": selected.timing_observed_peaks_only,
        "calibration_phase_prior": selected.calibration_phase_prior,
        "calibration_strength": selected.calibration_strength,
        "calibration_count_bias_tilt": count_tilt,
        "metrics": validation,
    }


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


@torch.no_grad()
def refine_training_spacing(model, data, train, val, batch_size, device):
    """Compare ordered decoders with spacing priors from training labels."""
    initial = model.config
    stride = initial.input_stride
    cap = (data["boundary_target"].shape[-1] - 1) // initial.max_layers
    delta = data["native_boundary_bins"][train].diff(dim=-1)
    slots = torch.arange(delta.shape[-1], device=device)
    gaps = delta[slots < data["layer_count"][train].unsqueeze(-1)].float() / stride
    quantiles = (
        torch.quantile(gaps, gaps.new_tensor([0.0, 0.05, 0.10])).tolist()
        if gaps.numel()
        else []
    )

    configs = [initial]
    if cap:
        spacing_bins = sorted({max(1, min(cap, int(q))) for q in quantiles})
        for gap in spacing_bins:
            for observed in (False, True):
                for weight in dict.fromkeys(
                    (initial.timing_fingerprint_weight, 1.0, 2.0, 4.0)
                ):
                    configs.append(
                        replace(
                            initial,
                            timing_min_separation_bins=gap,
                            timing_fingerprint_weight=weight,
                            timing_observed_peaks_only=observed,
                        )
                    )

    best = None
    for config in configs:
        model.config = config
        meter = MineMetrics()
        score(model, data, val, batch_size, device, [meter])
        metrics = meter.summary()
        value = validation_score(metrics)
        if best is None or value < best[0]:
            best = value, config, metrics
    value, selected, metrics = best
    model.config = selected
    return {
        "training_gap_quantiles_bins": quantiles,
        "timing_min_separation_bins": selected.timing_min_separation_bins,
        "timing_observed_peaks_only": selected.timing_observed_peaks_only,
        "timing_fingerprint_weight": selected.timing_fingerprint_weight,
        "changed": selected != initial,
        "validation_score": value,
        "validation": metrics,
    }


@torch.no_grad()
def refine_interface_roles(model, data, val, batch_size, device, gaps=None):
    """Select joint decoding parameters using only the fold's validation data."""
    initial = model.config
    if not initial.compact_interface_roles:
        raise ValueError("Interface role refinement requires a trained role head.")

    def measure(config):
        model.config = config
        meter = MineMetrics()
        score(model, data, val, batch_size, device, [meter])
        metrics = meter.summary()
        return validation_score(metrics), config, metrics

    best = measure(initial)
    gaps = gaps or (initial.timing_min_separation_bins,)
    for gap in gaps:
        for penalty in (0.0, 1.0, 2.0, 3.0, 4.0):
            for prior in (0.25, 1.0):
                candidate = measure(
                    replace(
                        initial,
                        timing_min_separation_bins=gap,
                        interface_role_penalty=penalty,
                        interface_role_count_prior=prior,
                    )
                )
                if candidate[0] < best[0]:
                    best = candidate
    selected = best[1]
    for weight in (0.0, 0.25, 0.5, 1.0):
        for frames in (1, 3, 7):
            for pool in (True, False):
                candidate = measure(
                    replace(
                        selected,
                        interface_role_tof_weight=weight,
                        boundary_context_frames=frames,
                        interface_role_interval_pooling=pool,
                    )
                )
                if candidate[0] < best[0]:
                    best = candidate
    selected = best[1]
    for phase_weight in (0.0, 2.0, 4.0, 8.0):
        for radius in (0, 4, 8):
            candidate = measure(
                replace(
                    selected,
                    interface_role_phase_weight=phase_weight,
                    interface_role_evidence_radius_bins=radius,
                )
            )
            if candidate[0] < best[0]:
                best = candidate
    value, model.config, metrics = best
    return {"validation_score": value, "metrics": metrics}


def train_fold(
    args,
    fold,
    config,
    training,
    data,
    names,
    groups,
    opf,
    pool,
    assignments,
    device,
    pooled,
):
    directory = args.output_dir / f"fold_{fold}"
    directory.mkdir(exist_ok=False)
    train, val, test = split_fold(fold, groups, opf, pool, assignments)
    write_json(
        directory / "split.json",
        {
            role: [
                {
                    "record": str(names[i]),
                    "group": str(groups[i]),
                    "source": "test_opf.h5" if i >= opf[0] else "train.h5",
                }
                for i in ids
            ]
            for role, ids in (("train", train), ("validation", val), ("test", test))
        },
    )
    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    model = SPAIIceInversionNet(config).to(device)
    with torch.no_grad():
        bins = data["boundary_target"].shape[-1]
        prior = (
            float(
                data["layer_count"]
                .index_select(0, torch.as_tensor(train, device=device))
                .float()
                .mean()
            )
            + 1
        )
        prior_logit = math.log(prior / (bins - prior))
        model.skeleton.head.bias.fill_(prior_logit)
        if args.timing_supervision:
            model.tof_encoder.output.bias[-1] = prior_logit
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = MineLoss(training)
    x = torch.arange(-6, 7, device=device).float()
    kernel = torch.exp(-0.5 * (x / 2).square())
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    # Oversample the OPF acquisition cohort using group metadata only.
    relevant = np.isin(groups[train], list(assignments))
    weights = np.ones(len(train))
    if relevant.any() and (~relevant).any():
        weights[relevant] = args.opf_training_fraction / relevant.sum()
        weights[~relevant] = (1 - args.opf_training_fraction) / (~relevant).sum()
        if args.balance_opf_counts:
            idx = torch.as_tensor(train[relevant], device=device)
            counts = data["layer_count"].index_select(0, idx).amax(dim=-1).cpu().numpy()
            labels, frequency = np.unique(counts, return_inverse=True)
            weights[relevant] = args.opf_training_fraction / (
                len(labels) * np.bincount(frequency)[frequency]
            )
    weights /= weights.sum()
    calibration_parameters = 0
    if model.interface_calibration is not None:
        calibration_parameters = sum(
            p.numel() for p in model.interface_calibration.parameters()
        )
        layers = data["layer_count"].index_select(
            0, torch.as_tensor(train, device=device)
        )
        freq = (
            F.one_hot(layers, config.max_layers + 1).float().mean(1)
            * torch.as_tensor(weights, device=device, dtype=torch.float32)[:, None]
        ).sum(0)
        with torch.no_grad():
            model.interface_calibration.count[-1].bias.copy_((freq + 0.005).log())
    history = []
    best = float("inf")
    epochs = sum(training.stage_epochs)
    print(
        f"fold={fold} train={len(train)} validation={len(val)} held_out={len(test)}",
        flush=True,
    )
    for epoch in range(epochs):
        stage, loss_weights = stage_weights(epoch, training)
        lr = args.lr * (
            1.0
            if stage == 1
            else (
                0.8
                if stage == 2
                else 0.15
                + 0.55
                * (
                    1
                    + math.cos(
                        math.pi
                        * (epoch - sum(training.stage_epochs[:2]))
                        / training.stage_epochs[2]
                    )
                )
                / 2
            )
        )
        optimizer.param_groups[0]["lr"] = lr
        model.train()
        sampled = rng.choice(
            train, size=args.records_per_epoch, replace=True, p=weights
        )
        total = torch.zeros((), device=device)
        for ids in minibatches(sampled, args.batch_size):
            batch = batch_for(data, ids, device, args.train_frames, rng)
            if args.geometry_jitter:
                batch = stretch_ice(
                    batch, config.input_stride, args.geometry_jitter, args.geometry_keep
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                out = forward_batch(model, batch, stage)
                loss, _, _ = criterion(out, batch, epoch)
                loss = loss + 0.75 * sparse_boundary_loss(
                    out["boundary_logits"],
                    batch["boundary_target"],
                    kernel,
                    args.ordinal_supervision,
                )
                if stage == 3:
                    if args.interval_density_supervision:
                        loss = (
                            loss
                            + args.interval_density_supervision
                            * interval_density_loss(model, out, batch)
                        )
                    if config.compact_interface_roles:
                        loss = loss + args.interface_role_loss * interface_role_loss(
                            out["interface_role_logits"], batch, config.input_stride
                        )
                    if model.interface_calibration is not None:
                        loss = loss + args.calibration_count_loss * F.cross_entropy(
                            out["calibrated_count_logits"].flatten(0, 1),
                            batch["layer_count"].flatten(),
                        )
                    mask = torch.arange(config.max_layers, device=device) < batch[
                        "layer_count"
                    ].unsqueeze(-1)
                    relative_h = (
                        out["layer_thickness_m"] - batch["layer_thickness_m_gt"]
                    ).abs()
                    relative_h = relative_h / batch["layer_thickness_m_gt"].clamp_min(
                        0.01
                    )
                    loss = (
                        loss
                        + 0.5
                        * ((relative_h * mask).sum(-1) / batch["layer_count"]).mean()
                    )
                    if args.timing_supervision:
                        loss = loss + args.timing_supervision * sparse_boundary_loss(
                            out["tof_node_logits"], batch["boundary_target"], kernel
                        )
                    if args.association_supervision:
                        loss = loss + args.association_supervision * association_loss(
                            out["event_association_logits"], batch, config.input_stride
                        )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite training objective in fold {fold}, epoch {epoch + 1}."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=not use_amp
            )
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach() * len(ids)
        row = {
            "epoch": epoch + 1,
            "stage": stage,
            "lr": lr,
            "loss": float(total) / len(sampled),
            "loss_weights": loss_weights,
        }
        if stage == 3 and (
            (epoch + 1) % args.validate_every == 0 or epoch + 1 == epochs
        ):
            validation = MineMetrics()
            score(model, data, val, args.batch_size, device, [validation])
            row["validation"] = validation.summary()
            value = validation_score(row["validation"])
            if value < best:
                best = value
                torch.save(
                    {
                        "checkpoint_format": "icemap.spai.v1",
                        "model_kind": (
                            "spai_opf_compact"
                            if calibration_parameters
                            else "spai_opf_main"
                        ),
                        "evaluation_role": PROTOCOL,
                        "fold": fold,
                        "epoch": epoch + 1,
                        "model_config": asdict(config),
                        "training_config": asdict(training),
                        "supplementary_calibration_parameters": calibration_parameters,
                        "model": model.state_dict(),
                        "validation": row["validation"],
                    },
                    directory / "best_spai.pt",
                )
        history.append(row)
        write_json(directory / "history.json", history)
        if epoch == 0 or (epoch + 1) % args.validate_every == 0 or epoch + 1 == epochs:
            print(json.dumps({"fold": fold, **row}), flush=True)
    if args.validation_only:
        return None
    return finish_fold(args, directory, data, names, train, val, test, device, pooled)


def finish_fold(
    args, directory, data, names, train, val, test, device, pooled, trained_path=None
):
    path = directory / "best_spai.pt"
    checkpoint = torch.load(
        trained_path or path, map_location=device, weights_only=True
    )
    if checkpoint["fold"] != int(directory.name.split("_")[-1]):
        raise ValueError("Checkpoint fold does not match the held-out partition.")
    config = ModelConfig(**checkpoint["model_config"])
    if args.known_ice:
        config = replace(config, minimum_layers=1)
    model = SPAIIceInversionNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    if args.refine_boundaries:
        refinement = refine_boundaries(
            model,
            data,
            val,
            args.batch_size,
            device,
            args.refine_local_peak,
            args.refine_ordered_peaks,
            args.refine_temporal,
            args.refine_density,
            args.refine_observed_peaks,
            args.refine_count_bias,
        )
        checkpoint["validation_refinement"] = refinement
    if args.refine_training_spacing:
        checkpoint["spacing_selection"] = refine_training_spacing(
            model, data, train, val, args.batch_size, device
        )
    if args.refine_interface_roles:
        cap = (data["boundary_target"].shape[-1] - 1) // model.config.max_layers
        gaps = tuple(gap for gap in (8, 12, 16, 22) if gap <= cap)
        checkpoint["interface_role_selection"] = refine_interface_roles(
            model, data, val, args.batch_size, device, gaps
        )
    if (
        args.refine_boundaries
        or args.refine_training_spacing
        or args.refine_interface_roles
    ):
        checkpoint.update(model_config=asdict(model.config), model=model.state_dict())
        torch.save(checkpoint, path)
    meter = MineMetrics()
    predictions = score(
        model, data, test, args.batch_size, device, [meter, pooled], names
    )
    result = {
        "fold": checkpoint["fold"],
        "best_epoch": checkpoint["epoch"],
        "frames": len(predictions),
        **meter.summary(),
    }
    write_json(directory / "held_out_predictions.json", predictions)
    write_json(directory / "held_out_metrics.json", result)
    print(json.dumps({"held_out": result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-grid-mm", type=float, required=True)
    parser.add_argument("--allow-relative-density-proxy", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--stage-epochs", type=int, nargs=3, default=(20, 20, 40))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--records-per-epoch", type=int, default=96)
    parser.add_argument("--train-frames", type=int, default=4)
    parser.add_argument("--input-stride", type=int, default=4)
    parser.add_argument(
        "--widths",
        type=int,
        nargs=4,
        default=(64, 64, 32, 32),
        metavar=("CNN", "GRU", "UNET", "RESNET"),
    )
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument(
        "--phase-depth",
        type=int,
        default=0,
        help="Additional phase CNN residual blocks.",
    )
    parser.add_argument(
        "--delay-depth", type=int, default=0, help="Additional ToF residual blocks."
    )
    parser.add_argument(
        "--compact-calibration",
        action="store_true",
        help="Train a compact count and local timing calibration head in stage III.",
    )
    parser.add_argument(
        "--phase-contrast",
        action="store_true",
        help="Retain within-bin phase variation in the skeleton CNN.",
    )
    parser.add_argument("--calibration-count-loss", type=float, default=1.5)
    parser.add_argument("--interval-density-supervision", type=float, default=0.0)
    parser.add_argument(
        "--interface-roles",
        action="store_true",
        help="Train surface, interior and bottom evidence in the compact calibration head.",
    )
    parser.add_argument("--interface-role-loss", type=float, default=1.0)
    parser.add_argument("--interface-role-local-kernel", type=int, default=0)
    parser.add_argument("--timing-fingerprint-weight", type=float, default=2.0)
    parser.add_argument("--timing-supervision", type=float, default=0.0)
    parser.add_argument("--timing-learned-weight", type=float, default=0.0)
    parser.add_argument("--association-supervision", type=float, default=0.0)
    parser.add_argument("--ordinal-supervision", type=float, default=0.0)
    parser.add_argument(
        "--geometry-jitter",
        type=float,
        default=0.0,
        help="Training-only log scale of post-surface range and physical labels.",
    )
    parser.add_argument(
        "--known-ice",
        action="store_true",
        help="Admit at least one physical layer when the task guarantees ice.",
    )
    parser.add_argument(
        "--geometry-keep",
        type=float,
        default=0.0,
        help="Fraction of training recordings left unstretched.",
    )
    parser.add_argument("--opf-training-fraction", type=float, default=0.65)
    parser.add_argument("--balance-opf-counts", action="store_true")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--refine-boundaries",
        action="store_true",
        help="Select the existing boundary bias and ToF weight using each fold's validation data.",
    )
    parser.add_argument(
        "--refine-local-peak",
        action="store_true",
        help="Also compare local ToF peak centroids on each fold's validation data.",
    )
    parser.add_argument(
        "--trained-run",
        type=Path,
        help="Reuse the recorded training splits and checkpoints for validation refinement.",
    )
    parser.add_argument(
        "--refine-ordered-peaks",
        action="store_true",
        help="Compare distinct ordered ToF peak paths on validation.",
    )
    parser.add_argument(
        "--refine-temporal",
        action="store_true",
        help="Compare local frame consensus and count-conditioned timing on validation.",
    )
    parser.add_argument(
        "--refine-density",
        action="store_true",
        help="Compare small shifts to the existing g_ref bias on validation.",
    )
    parser.add_argument(
        "--refine-observed-peaks",
        action="store_true",
        help="Also compare paths constrained to distinct observed ToF maxima.",
    )
    parser.add_argument(
        "--refine-count-bias",
        action="store_true",
        help="Tune the compact count head's existing output bias on validation.",
    )
    parser.add_argument(
        "--refine-training-spacing",
        action="store_true",
        help="Compare spacing constraints derived from training interface gaps on validation.",
    )
    parser.add_argument(
        "--refine-interface-roles",
        action="store_true",
        help="Select the joint role decoder's spacing, penalty and evidence weights on validation.",
    )
    args = parser.parse_args()
    if not args.allow_relative_density_proxy:
        parser.error(
            "Legacy H5 lacks relative-density targets. Explicit proxy opt-in is required."
        )
    if (
        min(
            *args.stage_epochs,
            args.batch_size,
            args.records_per_epoch,
            args.train_frames,
            args.input_stride,
            args.validate_every,
        )
        < 1
        or args.seed < 0
        or any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.legacy_grid_mm))
    ):
        parser.error("Invalid epoch, batch, grid, seed or learning-rate configuration.")
    if args.train_frames < 2:
        parser.error("ERF training needs at least two adjacent frames.")
    if (
        not 0 <= args.opf_training_fraction <= 1
        or not math.isfinite(args.timing_supervision)
        or args.timing_supervision < 0
    ):
        parser.error(
            "OPF fraction must be in [0,1] and timing supervision must be finite and nonnegative."
        )
    if args.trained_run and (
        not (
            args.refine_boundaries
            or args.refine_training_spacing
            or args.refine_interface_roles
        )
        or args.validation_only
    ):
        parser.error(
            "--trained-run requires a refinement option and a held-out evaluation."
        )
    if (
        args.refine_local_peak
        or args.refine_ordered_peaks
        or args.refine_temporal
        or args.refine_density
        or args.refine_observed_peaks
        or args.refine_count_bias
    ) and not args.refine_boundaries:
        parser.error("Peak refinement requires --refine-boundaries.")
    if not 0 <= args.geometry_jitter <= 1 or any(
        not math.isfinite(w) or w < 0
        for w in (
            args.ordinal_supervision,
            args.association_supervision,
            args.calibration_count_loss,
            args.interface_role_loss,
            args.interval_density_supervision,
        )
    ):
        parser.error("Invalid geometry jitter or ordinal supervision weight.")
    if args.interface_roles and not args.compact_calibration:
        parser.error("--interface-roles requires --compact-calibration.")
    if args.refine_interface_roles and not (args.interface_roles or args.trained_run):
        parser.error(
            "--refine-interface-roles requires --interface-roles or --trained-run."
        )
    if not 0 <= args.geometry_keep <= 1:
        parser.error("Geometry keep probability must be in [0,1].")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    config = ModelConfig(
        cnn_channels=args.widths[0],
        gru_hidden=args.widths[1],
        unet_base=args.widths[2],
        delay_channels=args.widths[3],
        input_stride=args.input_stride,
        timing_fingerprint_weight=args.timing_fingerprint_weight,
        minimum_layers=int(args.known_ice),
        timing_learned_weight=args.timing_learned_weight,
        phase_residual_blocks=args.phase_depth,
        delay_residual_blocks=args.delay_depth,
        compact_calibration=args.compact_calibration,
        compact_interface_roles=args.interface_roles,
        interface_role_local_kernel=args.interface_role_local_kernel,
        timing_min_separation_bins=8 if args.interface_roles else 0,
        interface_role_penalty=2.0 if args.interface_roles else 1.0,
        phase_contrast=args.phase_contrast,
    ).validate()
    training = TrainingConfig(stage_epochs=tuple(args.stage_epochs))
    data, names, groups, opf, pool, assignments, inventory = load_data(
        args.data_dir, args.legacy_grid_mm, args.input_stride, device, args.seed
    )
    splits = {}
    if args.trained_run:
        original = json.loads(
            (args.trained_run / "experiment.json").read_text(encoding="utf8")
        )
        if (
            original["protocol"] != PROTOCOL
            or original["source_sha256"] != inventory
            or original["structure_folds"] != assignments
            or original["model"]["input_stride"] != args.input_stride
            or original["arguments"]["legacy_grid_mm"] != args.legacy_grid_mm
        ):
            raise ValueError(
                "Trained run does not match this H5 inventory, folds or stride."
            )
        config = ModelConfig(**original["model"])
        training = TrainingConfig(**original["training"])
        for fold in range(5) if args.fold is None else [args.fold]:
            partitions = split_fold(fold, groups, opf, pool, assignments)
            directory = args.trained_run / f"fold_{fold}"
            saved = json.loads((directory / "split.json").read_text(encoding="utf8"))
            for role, ids in zip(("train", "validation", "test"), partitions):
                actual = [
                    (str(names[i]), "test_opf.h5" if i >= opf[0] else "train.h5")
                    for i in ids
                ]
                if actual != [(row["record"], row["source"]) for row in saved[role]]:
                    raise ValueError(
                        f"Trained run fold {fold} has different {role} records."
                    )
            if not (directory / "best_spai.pt").is_file():
                raise FileNotFoundError(f"Missing trained checkpoint for fold {fold}.")
            splits[fold] = (directory, saved, partitions)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = SPAIIceInversionNet(config)
    counts = {
        k: sum(p.numel() for p in m.parameters()) for k, m in model.named_children()
    }
    write_json(
        args.output_dir / "experiment.json",
        {
            "protocol": PROTOCOL,
            "source_sha256": inventory,
            "structure_folds": assignments,
            "site_ids_available": False,
            "model": asdict(config),
            "training": asdict(training),
            "parameters_by_component": counts,
            "total_parameters": sum(counts.values()),
            "supplementary_calibration_parameters": counts.get(
                "interface_calibration", 0
            ),
            "label_provenance": "label3 physical density runs, label1 surface, dielectric-projected interfaces",
            "relative_density_target": "absolute label3 layer density proxy",
            "training_only_auxiliary_weights": (
                original["training_only_auxiliary_weights"]
                if args.trained_run
                else {
                    "unit_mass_sparse_boundary_bce": 0.75,
                    "relative_thickness": 0.5,
                    "tof_peak": args.timing_supervision,
                    "ordinal_mass": 0.75 * args.ordinal_supervision,
                    "ordered_interface_likelihood": args.association_supervision,
                    "calibrated_count": (
                        args.calibration_count_loss if args.compact_calibration else 0
                    ),
                    "interface_roles": (
                        args.interface_role_loss if args.interface_roles else 0
                    ),
                    "interval_density": args.interval_density_supervision,
                }
            ),
            "validation_selection": "mean of five validation errors",
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        },
    )
    pooled = MineMetrics()
    results = []
    for fold in range(5) if args.fold is None else [args.fold]:
        if args.trained_run:
            trained, saved, (train, val, test) = splits[fold]
            directory = args.output_dir / f"fold_{fold}"
            directory.mkdir(exist_ok=False)
            write_json(directory / "split.json", saved)
            result = finish_fold(
                args,
                directory,
                data,
                names,
                train,
                val,
                test,
                device,
                pooled,
                trained / "best_spai.pt",
            )
        else:
            result = train_fold(
                args,
                fold,
                config,
                training,
                data,
                names,
                groups,
                opf,
                pool,
                assignments,
                device,
                pooled,
            )
        if result is not None:
            results.append(result)
            write_json(args.output_dir / "fold_metrics.json", results)
    if results:
        summary = pooled.summary()
        delta = {
            k: (summary[k] - target if summary[k] is not None else None)
            for k, target in TARGET.items()
        }
        write_json(
            args.output_dir / "cv_summary.json",
            {
                "protocol": PROTOCOL,
                "source_sha256": inventory,
                "structure_folds": assignments,
                "folds_completed": len(results),
                "evaluated_frames": sum(row["frames"] for row in results),
                "pooled_held_out": summary,
                "reference_percent": TARGET,
                "delta_points": delta,
                "within_one_point": all(
                    v is not None and abs(v) <= 1 for v in delta.values()
                ),
                "fold_mean_std": {
                    k: (
                        {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        if (v := [r[k] for r in results if r[k] is not None])
                        else None
                    )
                    for k in TARGET
                },
            },
        )


if __name__ == "__main__":
    main()
