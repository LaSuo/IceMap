"""Inspect the existing model/settings; no dataset training or new experiment."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from network import ModelConfig, SPAIIceInversionNet
from losses import TrainingConfig, stage_weights
from implementation_settings import settings_manifest
from protocol import implementation_contract


def describe(config=None):
    cfg = config or ModelConfig()
    model = SPAIIceInversionNet(cfg)
    modules = []
    for name, module in model.named_modules():
        row = {"name": name, "class": type(module).__name__}
        if isinstance(module, torch.nn.Conv1d):
            row.update(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=list(module.kernel_size),
                stride=list(module.stride),
                padding=list(module.padding),
                dilation=list(module.dilation),
                groups=module.groups,
                bias=module.bias is not None,
            )
        elif isinstance(module, torch.nn.BatchNorm1d):
            row.update(
                channels=module.num_features,
                epsilon=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )
        elif isinstance(module, torch.nn.GRU):
            row.update(
                input_size=module.input_size,
                hidden_size=module.hidden_size,
                num_layers=module.num_layers,
                bidirectional=module.bidirectional,
                dropout=module.dropout,
                batch_first=module.batch_first,
            )
        elif isinstance(module, torch.nn.Linear):
            row.update(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        elif isinstance(module, torch.nn.LayerNorm):
            row.update(
                normalized_shape=list(module.normalized_shape),
                epsilon=module.eps,
                elementwise_affine=module.elementwise_affine,
            )
        else:
            continue
        modules.append(row)
    training = TrainingConfig()
    return {
        "model_config": asdict(cfg),
        "training_config": asdict(training),
        "implementation_contract": implementation_contract(cfg),
        "settings": settings_manifest(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "staged_loss_weights": [
            {
                "epoch": epoch + 1,
                "stage": stage_weights(epoch, training)[0],
                "weights": stage_weights(epoch, training)[1],
            }
            for epoch in range(sum(training.stage_epochs))
        ],
        "modules": modules,
        "tensor_layout": {
            "fingerprints": "[batch,frames,range_bins]",
            "layer_properties": "[batch,frames,max_layers]",
            "interfaces": "L+1 for L physical layers; zero padding excluded from losses",
            "time_s": "seconds, calibrated two-way propagation",
            "depth_m": "meters from registered ice surface",
            "density": "g/cm^3",
        },
        "architecture_width_source": "explicit current implementation choices; not proof of the original released checkpoint architecture",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config")
    p.add_argument("--output")
    p.add_argument("--feature-metadata")
    a = p.parse_args()
    cfg = (
        ModelConfig(**json.loads(Path(a.model_config).read_text(encoding="utf-8-sig")))
        if a.model_config
        else ModelConfig()
    )
    value = describe(cfg)
    if a.feature_metadata:
        import h5py

        with h5py.File(a.feature_metadata, "r") as f:
            value["exported_processing_metadata"] = {
                key: json.loads(f.attrs[key])
                for key in (
                    "matlab_config_json",
                    "parameter_catalog_json",
                    "calibration_provenance_json",
                    "execution_environment_json",
                    "sync_metadata_json",
                    "sensor_registration_json",
                )
                if key in f.attrs
            }
    encoded = json.dumps(value, indent=2)
    if a.output:
        output = Path(a.output)
        if output.exists():
            p.error("Output already exists.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
