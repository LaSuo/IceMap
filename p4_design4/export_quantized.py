"""Export measured mixed FP32/dynamic INT8 SPAI; validate before deployment."""

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from inference import load_model, SPAIDeployment
from main import H5IceDataset, collate_records
from network import ModelConfig
from protocol import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--calibration-data",
        required=True,
        help="Unlabelled prepared feature record for a numerical comparison, not an accuracy certificate.",
    )
    p.add_argument(
        "--engine",
        required=True,
        choices=[x for x in torch.backends.quantized.supported_engines if x != "none"],
    )
    p.add_argument("--allow-debug-data", action="store_true")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists() or Path(str(output) + ".json").exists():
        p.error("Output already exists.")
    torch.set_num_threads(4)
    original, bundle = load_model(a.model, "cpu")
    if bundle.get("quantization") or bundle.get("model_kind") != "spai_six_loss":
        p.error("Export an unquantized six-loss SPAI checkpoint.")
    cfg = ModelConfig(**bundle["model_config"])
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
    torch.backends.quantized.engine = a.engine
    quantized = torch.ao.quantization.quantize_dynamic(
        original.core, {nn.Linear, nn.GRU}, dtype=torch.qint8, inplace=False
    )
    with torch.inference_mode():
        ref = original(inputs)
        candidate = SPAIDeployment(quantized).eval()(inputs)
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
        "quantization": "dynamic_linear_gru_qint8",
        "quantization_engine": a.engine,
        "max_absolute_difference": comparison,
        "accuracy_verified": False,
        "scope": "Linear/GRU int8; convolution and physical inversion remain floating point",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
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
        model_kind="spai_six_loss",
        model=quantized.state_dict(),
        quantization=report["quantization"],
        quantization_engine=a.engine,
        quantization_report=report,
    )
    torch.save(package, output)
    reloaded, _ = load_model(output, "cpu")
    with torch.inference_mode():
        actual = reloaded(inputs)
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
