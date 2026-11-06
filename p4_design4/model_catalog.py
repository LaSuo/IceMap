"""Stable network registry; method implementations live under models/."""

from network import ModelConfig, SPAIIceInversionNet
from models.config import ArchitectureConfig
from models.composed_spai import ComposedSPAI

VARIANTS = (
    "mine",
    "tcn",
    "transformer",
    "convnext",
    "multiscale",
    "cnn_lstm",
    "spectral",
    "attention_fusion",
    "cross_modal",
    "hybrid",
)
ALIASES = {f"v{i+1}": name for i, name in enumerate(VARIANTS)}


def build_network(config=None, variant="mine", options=None):
    variant = ALIASES.get(variant, variant)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown network {variant!r}; choose from {VARIANTS}.")
    cfg = config or ModelConfig()
    settings = (
        options
        if isinstance(options, ArchitectureConfig)
        else ArchitectureConfig(**(options or {}))
    ).validate()
    if variant == "mine":
        if settings.context_dim:
            raise ValueError("The main network does not consume context.")
        return SPAIIceInversionNet(cfg)
    return ComposedSPAI(cfg, variant, settings)
