"""Observation-level eight-bit transport for Design3 fingerprints."""

from dataclasses import dataclass

import torch


FINGERPRINTS = ("phase", "scatter", "tof")
CONTEXT = ("scatter_valid", "time_s", "depth_m")
OPTIONAL_CONTEXT = (
    "observed_delay_events_s",
    "observed_delay_event_valid",
    "event_tolerance_s",
)
FEATURE_PROFILES = ("fp32", "linear8", "companded8", "phase_fp32")


@dataclass(frozen=True)
class QuantizedField:
    values: torch.Tensor
    scale: torch.Tensor | None = None
    square: bool = False

    def unpack(self, device):
        value = self.values.to(device=device, dtype=torch.float32)
        if self.scale is not None:
            value = value * self.scale.to(device=device)
        return value.square() if self.square else value


@dataclass(frozen=True)
class FeaturePacket:
    """Owned CPU tensors for one complete [B,T,N] observation window."""

    fields: dict
    context: dict
    profile: str

    def unpack(self, device="cpu"):
        inputs = {
            name: value.to(device=device) if torch.is_tensor(value) else value
            for name, value in self.context.items()
        }
        inputs.update(
            (name, field.unpack(device)) for name, field in self.fields.items()
        )
        return inputs

    @property
    def payload_bytes(self):
        tensors = [x for x in self.context.values() if torch.is_tensor(x)]
        for field in self.fields.values():
            tensors.append(field.values)
            if field.scale is not None:
                tensors.append(field.scale)
        return sum(x.numel() * x.element_size() for x in tensors)


def _cpu_owned(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def pack_feature_window(inputs, profile="companded8"):
    """Quantize each frame independently while preserving calibrated axes and validity."""
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {profile!r}.")
    missing = set(FINGERPRINTS + CONTEXT) - set(inputs)
    if missing:
        raise ValueError(f"Missing feature inputs: {sorted(missing)}")
    fields = {}
    shape = None
    for name in FINGERPRINTS:
        raw = inputs[name]
        if not torch.is_tensor(raw) or not raw.is_floating_point() or raw.ndim != 3:
            raise ValueError(f"{name} must be a floating-point [B,T,N] tensor.")
        value = _cpu_owned(raw).float()
        if (
            min(value.shape) < 1
            or value.shape[-1] < 8
            or (shape is not None and value.shape != shape)
        ):
            raise ValueError("All fingerprints must share a nonempty [B,T,N>=8] shape.")
        shape = value.shape
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values.")
        if profile == "fp32" or (name == "phase" and profile == "phase_fp32"):
            fields[name] = QuantizedField(value)
            continue
        signed = name == "phase"
        if not signed and (value < 0).any():
            raise ValueError(f"{name} must be nonnegative for unsigned transport.")
        square = not signed and profile in ("companded8", "phase_fp32")
        if square:
            value = value.sqrt()
        limit = 127 if signed else 255
        scale = (value.abs().amax(dim=-1, keepdim=True) / limit).clamp_min(1e-38)
        codes = (value / scale).round().clamp(-limit if signed else 0, limit)
        dtype = torch.int8 if signed else torch.uint8
        fields[name] = QuantizedField(codes.to(dtype), scale, square)
    valid = inputs["scatter_valid"]
    if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != shape:
        raise ValueError(
            "scatter_valid must be a Boolean tensor matching the fingerprints."
        )
    context = {}
    for name in CONTEXT + OPTIONAL_CONTEXT:
        if name in inputs:
            value = inputs[name]
            if name != "event_tolerance_s" and not torch.is_tensor(value):
                raise ValueError(f"{name} must be a tensor.")
            context[name] = _cpu_owned(value) if torch.is_tensor(value) else value
    return FeaturePacket(fields, context, profile)
