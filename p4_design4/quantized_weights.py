"""Portable per-output-channel INT8 storage for SPAI convolutions."""

import torch
from torch import nn
from torch.nn import functional as F


CONV_INT8_FORMAT = "conv1d_weight_only_int8_v1"


class Int8Conv1d(nn.Module):
    """Store INT8 kernels and reconstruct only the active kernel for FP32 convolution."""

    def __init__(self, conv, *, encode):
        super().__init__()
        if conv.padding_mode != "zeros":
            raise ValueError("INT8 Conv1d requires zero padding.")
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        weight = conv.weight.detach().float()
        if encode:
            if not torch.isfinite(weight).all():
                raise ValueError("Cannot quantize a nonfinite convolution kernel.")
            scale = weight.abs().amax(dim=(1, 2), keepdim=True)
            scale = torch.where(scale > 0, (scale / 127).clamp_min(1e-38), 1.0)
            codes = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        else:
            codes = torch.zeros_like(weight, dtype=torch.int8)
            scale = weight.new_ones((weight.shape[0], 1, 1))
        self.register_buffer("weight_int8", codes)
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "bias", None if conv.bias is None else conv.bias.detach().float().clone()
        )

    def forward(self, inputs):
        weight = self.weight_int8.float() * self.weight_scale
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return F.conv1d(
            inputs,
            weight.to(inputs.dtype),
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def quantizable_convolutions(core, profile="all"):
    branches = {
        "all": ("skeleton", "density_unet", "tof_encoder", "interface_calibration"),
        "density_delay": ("density_unet", "tof_encoder"),
        "density": ("density_unet",),
        "delay": ("tof_encoder",),
    }
    if profile not in branches:
        raise ValueError(f"Unknown convolution profile: {profile!r}.")
    return [
        name
        for name, module in core.named_modules()
        if isinstance(module, nn.Conv1d) and name.split(".")[0] in branches[profile]
    ]


def replace_convolutions(core, names, *, encode):
    if (
        not isinstance(names, (list, tuple))
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Expected a nonempty list of unique convolution names.")
    modules = dict(core.named_modules())
    if any(not isinstance(modules.get(name), nn.Conv1d) for name in names):
        raise ValueError("INT8 state names must point to Conv1d modules.")
    for name in names:
        parent, _, child = name.rpartition(".")
        setattr(modules[parent], child, Int8Conv1d(modules[name], encode=encode))
    return core


def compact_weight_bundle(core, source, profile):
    """Encode a disposable inference core and return its compact checkpoint."""
    names = quantizable_convolutions(core, profile)
    replace_convolutions(core, names, encode=True)
    fields = (
        "model_config",
        "training_config",
        "implementation_contract",
        "evaluation_role",
        "ablation_case",
        "data_split_sha256",
        "source_code_sha256",
        "fold",
        "epoch",
    )
    package = {key: source[key] for key in fields if key in source}
    return dict(
        package,
        checkpoint_format="icemap.spai.v1",
        model_kind=source.get("model_kind", "spai_six_loss"),
        model=core.state_dict(),
        quantization=CONV_INT8_FORMAT,
        quantized_convolutions=names,
        weight_profile=profile,
    )
