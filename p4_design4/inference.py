"""Standalone deployment loader for SPAI + optional residual calibration."""

from pathlib import Path
from dataclasses import asdict
import torch
from torch import nn
from calibration import (
    ResidualPhysicalCalibration,
    CalibrationConfig,
    calibration_features,
)
from train_opf import CalibratedSPAI
from network import ModelConfig, SPAIIceInversionNet
from raw_shape_features import raw_shape_features
from quality_calibration import (
    QualityPhysicalExpert,
    QualityExpertConfig,
    combine_quality_expert,
)
from quality_features import build_quality_features
from post_calibration import PostCalibrationConfig, apply_post_calibration
from reference_alignment import ReferenceAlignmentConfig, align_prediction


class EnhancedIceInversion(nn.Module):
    def __init__(
        self,
        backbone,
        calibration,
        pool_bins=16,
        reference_alignment=None,
        input_fingerprints=None,
        auxiliary_features=None,
        quality_expert=None,
        post_calibration=None,
        secondary_quality_expert=None,
        default_context=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.calibration = calibration
        self.pool_bins = pool_bins
        self.reference_alignment = reference_alignment
        self.input_fingerprints = input_fingerprints or ["phase", "scatter", "tof"]
        self.auxiliary_features = auxiliary_features
        self.quality_expert = quality_expert
        self.post_calibration = post_calibration
        self.secondary_quality_expert = secondary_quality_expert
        self.register_buffer(
            "default_context",
            (
                None
                if default_context is None
                else torch.as_tensor(default_context, dtype=torch.float32)
            ),
            persistent=False,
        )

    def forward(self, inputs):
        inputs = dict(inputs)
        if "context" not in inputs and self.default_context is not None:
            B, T = inputs["phase"].shape[:2]
            inputs["context"] = self.default_context.to(inputs["phase"])[
                None, None
            ].expand(B, T, -1)
        for name in ["phase", "scatter", "tof"]:
            if name not in self.input_fingerprints:
                inputs[name] = torch.zeros_like(inputs[name])
        if "scatter" not in self.input_fingerprints:
            inputs["scatter_valid"] = torch.zeros_like(inputs["scatter_valid"])
        # Precision settings reproduce the independently cached backbone and
        # calibration training. No labels or record identifiers are accessed.
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            base = self.backbone(inputs)
            features = calibration_features(base, self.pool_bins)
            if self.auxiliary_features:
                if self.auxiliary_features["kind"] != "raw_shape_v1":
                    raise ValueError("Unknown auxiliary feature specification.")
                features = torch.cat(
                    [
                        features,
                        raw_shape_features(inputs, self.auxiliary_features["bins"]),
                    ],
                    dim=-1,
                )
            torch.backends.cuda.matmul.allow_tf32 = True
            B, T, _ = features.shape
            output = self.calibration(
                features.flatten(0, 1),
                base["layer_density"].flatten(0, 1),
                base["layer_delays_s"].flatten(0, 1) * 299792458 / 2,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        shaped = {
            key: value.reshape(B, T, *value.shape[1:]) for key, value in output.items()
        }
        if self.reference_alignment is not None:
            shaped = align_prediction(
                shaped, base, self.reference_alignment, self.calibration.config
            )
        if self.quality_expert is not None:
            quality_features = build_quality_features(
                inputs, features, self.quality_expert.config
            )
            raw = self.quality_expert(quality_features.flatten(0, 1))
            refined = {k: v.reshape(B, T, *v.shape[1:]) for k, v in raw.items()}
            shaped = combine_quality_expert(
                shaped, refined, inputs, self.quality_expert.config
            )
        if self.secondary_quality_expert is not None:
            quality_features = build_quality_features(
                inputs, features, self.secondary_quality_expert.config
            )
            raw = self.secondary_quality_expert(quality_features.flatten(0, 1))
            refined = {k: v.reshape(B, T, *v.shape[1:]) for k, v in raw.items()}
            primary_gate = shaped.get("quality_expert_gate")
            shaped = combine_quality_expert(
                shaped, refined, inputs, self.secondary_quality_expert.config
            )
            if primary_gate is not None:
                shaped["primary_quality_expert_gate"] = primary_gate
            shaped["secondary_quality_expert_gate"] = shaped["quality_expert_gate"]
        if self.post_calibration is not None:
            shaped = apply_post_calibration(
                shaped, self.post_calibration, self.calibration.config, inputs
            )
        return shaped


class SPAIDeployment(nn.Module):
    """Label-free dictionary interface for the six-loss SPAI checkpoint.

    Diagnostic ERF dictionaries and training-only None entries stay inside the
    network; deployment returns tensors with the same interface as extensions.
    """

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, inputs):
        required = ("phase", "scatter", "tof", "scatter_valid", "time_s", "depth_m")
        missing = set(required) - set(inputs)
        if missing:
            raise ValueError(f"Missing inference inputs: {sorted(missing)}")
        optional = (
            "observed_delay_events_s",
            "observed_delay_event_valid",
            "event_tolerance_s",
        )
        output = self.core(
            **{key: inputs[key] for key in (*required, *optional) if key in inputs}
        )
        return {key: value for key, value in output.items() if torch.is_tensor(value)}


def load_model(filename, device=None):
    explicit_device = device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Model file does not exist: {filename}")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; install the GPU runtime or select device=cpu."
        )
    bundle = torch.load(filename, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("Model package must be a mapping.")
    if bundle.get("quantization") and explicit_device is None:
        device = "cpu"
    if {"model", "model_config", "training_config"} <= set(bundle):
        if bundle.get("checkpoint_format", "icemap.spai.v1") != "icemap.spai.v1":
            raise ValueError("Unsupported SPAI checkpoint format.")
        from mine_ablations import build_ablation

        core = build_ablation(
            ModelConfig(**bundle["model_config"]), bundle.get("ablation_case", "full")
        ).to(device)
        if bundle.get("quantization") is not None:
            if (
                bundle["quantization"] != "dynamic_linear_gru_qint8"
                or torch.device(device).type != "cpu"
            ):
                raise ValueError(
                    "This quantized SPAI format requires CPU dynamic Linear/GRU qint8."
                )
            engine = bundle.get("quantization_engine")
            if engine not in torch.backends.quantized.supported_engines:
                raise ValueError(
                    f"Quantization backend {engine!r} is unavailable on this host; re-export for the target backend."
                )
            torch.backends.quantized.engine = engine
            core = torch.ao.quantization.quantize_dynamic(
                core.eval(), {nn.Linear, nn.GRU}, dtype=torch.qint8
            )
        core.load_state_dict(bundle["model"], strict=True)
        bundle = dict(bundle)
        bundle.setdefault("model_kind", "spai_six_loss")
        bundle.setdefault("evaluation_role", "site_held_out; see experiment manifest")
        return SPAIDeployment(core).to(device).eval(), bundle
    required = {
        "backbone_config",
        "backbone_state",
        "calibration_config",
        "calibration_state",
        "input_dim",
        "pool_bins",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model package missing fields: {sorted(missing)}")
    if (
        type(bundle["input_dim"]) is not int
        or bundle["input_dim"] < 1
        or type(bundle["pool_bins"]) is not int
        or bundle["pool_bins"] < 1
    ):
        raise ValueError("Invalid feature dimensions in model package.")
    if "architecture_config" not in bundle and "research_network_config" in bundle:
        bundle["architecture_config"] = bundle["research_network_config"]
    backbone = CalibratedSPAI(
        ModelConfig(**bundle["backbone_config"]),
        bundle.get("backbone_variant", "mine"),
        bundle.get("architecture_config"),
    ).to(device)
    backbone.load_state_dict(bundle["backbone_state"])
    assert (
        asdict(backbone.core.config) == bundle["backbone_config"]
    ), "Backbone configuration differs from the trained bundle."
    state = bundle["calibration_state"]
    config = CalibrationConfig(**bundle["calibration_config"])
    head = ResidualPhysicalCalibration(
        bundle["input_dim"], state["feature_mean"], state["feature_std"], config
    ).to(device)
    head.load_state_dict(state)
    alignment = (
        ReferenceAlignmentConfig(**bundle["reference_alignment"])
        if bundle.get("reference_alignment")
        else None
    )
    expert = None
    if bundle.get("quality_expert_state") is not None:
        state = bundle["quality_expert_state"]
        config = QualityExpertConfig(**bundle["quality_expert_config"])
        expert = QualityPhysicalExpert(
            bundle.get("quality_expert_input_dim", bundle["input_dim"]),
            state["feature_mean"],
            state["feature_std"],
            config,
        ).to(device)
        expert.load_state_dict(state)
    post = (
        PostCalibrationConfig(**bundle["post_calibration"])
        if bundle.get("post_calibration")
        else None
    )
    secondary = None
    if bundle.get("secondary_quality_expert_state") is not None:
        state = bundle["secondary_quality_expert_state"]
        cfg = QualityExpertConfig(**bundle["secondary_quality_expert_config"])
        secondary = QualityPhysicalExpert(
            bundle["secondary_quality_expert_input_dim"],
            state["feature_mean"],
            state["feature_std"],
            cfg,
        ).to(device)
        secondary.load_state_dict(state)
    model = (
        EnhancedIceInversion(
            backbone,
            head,
            bundle["pool_bins"],
            alignment,
            bundle.get("input_fingerprints"),
            bundle.get("auxiliary_features"),
            expert,
            post,
            secondary,
            bundle.get("default_context"),
        )
        .to(device)
        .eval()
    )
    return model, bundle
