"""Export mixed-precision SPAI and compare it against the FP32 source."""

import argparse
import json
from pathlib import Path
import h5py
import torch
from torch import nn
from export_quantized_features import ARCHIVE_FORMAT, iter_feature_archive
from inference import load_model, SPAIDeployment
from main import H5IceDataset, collate_records
from network import ModelConfig
from protocol import sha256_file
from quantized_features import FEATURE_PROFILES, pack_feature_window
from quantized_weights import compact_weight_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--calibration-data",
        required=True,
        help="Unlabelled prepared feature record or FP32 feature archive for a numerical comparison.",
    )
    p.add_argument(
        "--engine",
        choices=[x for x in torch.backends.quantized.supported_engines if x != "none"],
    )
    p.add_argument("--format", choices=("dynamic", "conv_int8"), default="dynamic")
    p.add_argument(
        "--weight-profile",
        choices=("all", "density_delay", "density", "delay"),
        default="all",
    )
    p.add_argument("--feature-profile", choices=FEATURE_PROFILES, default="fp32")
    p.add_argument("--allow-debug-data", action="store_true")
    a = p.parse_args()
    if a.format == "dynamic" and a.engine is None:
        p.error("Dynamic quantization requires --engine.")
    output = Path(a.output)
    if output.exists() or Path(str(output) + ".json").exists():
        p.error("Output already exists.")
    torch.set_num_threads(4)
    original, bundle = load_model(a.model, "cpu")
    if bundle.get("quantization") or not isinstance(original, SPAIDeployment):
        p.error("Export an unquantized SPAI checkpoint.")
    cfg = ModelConfig(**bundle["model_config"])
    with h5py.File(a.calibration_data, "r") as calibration_file:
        is_archive = calibration_file.attrs.get("format") == ARCHIVE_FORMAT
    if is_archive:
        with_archive = iter_feature_archive(a.calibration_data)
        try:
            _, source_packet = next(with_archive)
        finally:
            with_archive.close()
        if source_packet.profile != "fp32":
            p.error(
                "Supply an FP32 feature archive to establish the numerical reference."
            )
        inputs = source_packet.unpack()
    else:
        batch = collate_records(
            [
                H5IceDataset(
                    [a.calibration_data],
                    cfg.max_layers,
                    with_labels=False,
                    model_config=cfg,
                    allow_debug_data=a.allow_debug_data,
                )[0]
            ]
        )
        inputs = {
            k: batch[k]
            for k in ("phase", "scatter", "tof", "scatter_valid", "time_s", "depth_m")
        }
    packet = pack_feature_window(inputs, a.feature_profile)
    deployed_inputs = packet.unpack()
    with torch.inference_mode():
        ref = original(inputs)
    if a.format == "conv_int8":
        quantized = original.core
        package = compact_weight_bundle(quantized, bundle, a.weight_profile)
        scope = "Selected Conv1d kernels stored as INT8; arithmetic remains FP32."
    else:
        torch.backends.quantized.engine = a.engine
        quantized = torch.ao.quantization.quantize_dynamic(
            original.core, {nn.Linear, nn.GRU}, dtype=torch.qint8, inplace=False
        )
        package = {
            key: bundle[key]
            for key in (
                "model_config",
                "training_config",
                "implementation_contract",
                "evaluation_role",
                "ablation_case",
                "parameter_matching",
                "data_split_sha256",
                "source_code_sha256",
            )
            if key in bundle
        }
        package.update(
            checkpoint_format="icemap.spai.v1",
            model_kind=bundle["model_kind"],
            model=quantized.state_dict(),
            quantization="dynamic_linear_gru_qint8",
            quantization_engine=a.engine,
        )
        scope = "Linear/GRU dynamic INT8; convolution and inversion remain FP32."
    with torch.inference_mode():
        candidate = SPAIDeployment(quantized).eval()(deployed_inputs)
    comparison = {
        key: float((candidate[key] - ref[key]).abs().max())
        for key in ("layer_density", "layer_thickness_m", "layer_delays_s")
    }
    comparison["layer_count_disagreement_frames"] = int(
        (candidate["layer_count"] != ref["layer_count"]).sum()
    )
    report = {
        "source_sha256": sha256_file(a.model),
        "comparison_data_sha256": sha256_file(a.calibration_data),
        "quantization": package["quantization"],
        "quantization_engine": package.get("quantization_engine"),
        "feature_profile": a.feature_profile,
        "feature_payload_bytes": packet.payload_bytes,
        "max_absolute_difference": comparison,
        "accuracy_verified": False,
        "scope": scope,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    package.update(
        feature_profile=a.feature_profile,
        quantization_report=report,
    )
    torch.save(package, output)
    reloaded, _ = load_model(output, "cpu")
    with torch.inference_mode():
        actual = reloaded(deployed_inputs)
    for key in ("layer_thickness_m", "layer_density", "layer_delays_s"):
        torch.testing.assert_close(actual[key], candidate[key])
    report.update(
        serialized_checkpoint_bytes=output.stat().st_size, sha256=sha256_file(output)
    )
    Path(str(output) + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
