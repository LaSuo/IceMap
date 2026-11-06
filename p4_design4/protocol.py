"""Versioned evaluation and cross-language configuration contracts."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

PROTOCOL = "icemap.site5fold.six_loss.v1"
METRICS = "icemap.monotone_iou.full_unmatched_mass.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_matlab_config(raw, model, training=None):
    """Reject disagreement instead of ignoring the MATLAB-exported constants.

    Empty metadata remains readable for older files, but formal CLI training
    additionally requires the calibrated export contract (see validate_record).
    """
    if raw is None:
        return
    if isinstance(raw, bytes):
        raw = raw.decode("utf8")
    cfg = json.loads(raw)
    fields = (
        "max_layers",
        "assignment_temperature",
        "c0_m_s",
        "dielectric_density_coefficient",
        "erf_windows",
        "erf_beta",
    )
    normalize = lambda x: json.loads(json.dumps(x))
    for name in fields:
        if name not in cfg or normalize(cfg[name]) != normalize(getattr(model, name)):
            raise ValueError(f"MATLAB/Python configuration mismatch: {name}")
    if training is not None:
        pairs = {
            "stage_epochs": training.stage_epochs,
            "loss_warmup_epochs": training.warmup_epochs,
            "loss_weights": [
                training.lambda_b,
                training.lambda_count,
                training.lambda_pre_rho,
                training.lambda_rho,
                training.lambda_h,
                training.lambda_profile,
            ],
        }
        for name, value in pairs.items():
            if name not in cfg or normalize(cfg[name]) != normalize(value):
                raise ValueError(f"MATLAB/Python training mismatch: {name}")


def validate_record(file, model, training):
    mode = file.attrs.get("workflow_mode", "")
    if isinstance(mode, bytes):
        mode = mode.decode("utf8")
    if mode != "calibrated_experiment":
        raise ValueError(
            "Formal training requires workflow_mode=calibrated_experiment; unverified/synthetic records need --allow-debug-data."
        )
    if "matlab_config_json" not in file.attrs:
        raise ValueError(
            "Formal training requires matlab_config_json with calibrated preprocessing settings."
        )
    validate_matlab_config(file.attrs["matlab_config_json"], model, training)
    cfg = json.loads(file.attrs["matlab_config_json"])
    if cfg.get("calibration_status") == "engineering_initial":
        raise ValueError(
            "Approximate engineering initials are not measured experimental calibration; use --allow-debug-data for development or provide measured calibration."
        )
    meta = json.loads(file.attrs.get("experiment_metadata_json", "{}"))
    import numpy as np

    for name, fields in (
        ("radar_frequency_hz", ("frequency_low_hz", "frequency_high_hz")),
        ("radar_bandwidth_hz", ("bandwidth_low_hz", "bandwidth_high_hz")),
    ):
        if name in meta:
            actual = np.asarray(meta[name], dtype=float)
            expected = np.asarray([cfg.get(k, np.nan) for k in fields], dtype=float)
            if (
                actual.shape != (2,)
                or not np.isfinite(actual).all()
                or not np.isfinite(expected).all()
                or not np.allclose(actual, expected, rtol=1e-6, atol=0)
            ):
                raise ValueError(
                    "Declared acquisition frequencies/bandwidths disagree with preprocessing calibration; resolve dataset/reference discrepancy explicitly."
                )
    required = (
        "surface_range_m",
        "shallow_end_s",
        "psf_sigma_grid_bins",
        "lambda_attenuation",
        "lambda_deconvolution",
        "spatial_sigma_m",
        "spatial_epsilon_m",
        "boundary_phase_threshold_rad",
        "boundary_dilation_bins",
        "tof_min_peak_distance_bins",
        "tof_min_prominence",
    )
    import numpy as np

    for key in required:
        value = np.asarray(cfg.get(key, []), dtype=float)
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"Missing/nonfinite preprocessing calibration: {key}")
    for key in ("relative_density_target_method", "boundary_target_method"):
        text = file.attrs.get(key, "")
        if isinstance(text, bytes):
            text = text.decode("utf8")
        if not text or text == "unspecified" or "proxy" in text.lower():
            raise ValueError(f"Formal evaluation needs documented non-proxy {key}.")
    if cfg.get("fingerprint_input_mode", "dual_band") != "dual_band":
        raise ValueError(
            "Fused fingerprints are a separate experiment, outside the main protocol."
        )
    for name, expected in (
        ("design1_version", "v2"),
        ("design2_version", "v3"),
        ("design3_version", "v1"),
    ):
        if cfg.get(name) != expected:
            raise ValueError(
                f"{name} is a separate-method experiment; main protocol requires {expected}."
            )
    return cfg


def implementation_contract(model, ablation_case="full"):
    """Algorithmic choices not uniquely specified by the article."""
    contract = {
        "protocol": PROTOCOL,
        "metrics_version": METRICS,
        "ablation_case": ablation_case,
        "layer_indexing": "q=1+cumsum(b); M_mine diagnostic; ice pooling soft(q-1) with support/presence/mask",
        "boundary_decoder": "soft ordinal half-bin ranks with learned ToF evidence; rounded soft-count at inference",
        "density_erf": "samplewise before masked layer pooling",
        "erf_alignment": "surface registered; no internal-boundary warping",
        "delay_input": "dense D3 event map; explicit D3 event list exported for inspection, not forced as ground truth",
        "network_config": asdict(model),
        "temporal_future_frames": (
            max(w[1] // 2 for w in model.erf_windows) if model.use_erf else 0
        ),
        "architecture_widths": "explicit implementation choices; original trained widths unavailable",
    }
    if ablation_case == "direct_fusion":
        contract.update(
            layer_indexing="independent categorical count; no soft layer assignment",
            boundary_decoder="per-sample CNN logits; categorical layer count",
            density_erf="absent",
            erf_alignment="absent",
            delay_input="dense D3 map in fused CNN input; h/rho directly regressed, reported delays derived from h/rho",
            temporal_future_frames=0,
        )
    contract["physical_inversion"] = (
        "analytic h with one density refinement"
        if ablation_case not in ("plain_delay", "direct_fusion")
        else "independent CNN h/rho regression; no analytic thickness constraint"
    )
    return contract
