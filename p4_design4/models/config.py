from dataclasses import dataclass
import math


@dataclass
class ArchitectureConfig:
    tcn_depth: int = 4
    tcn_kernel: int = 5
    dropout: float = 0.1
    transformer_depth: int = 2
    transformer_heads: int = 4
    transformer_patch_bins: int = 8
    transformer_max_tokens: int = 512
    transformer_ff_multiplier: int = 4
    convnext_depth: int = 4
    convnext_kernel: int = 7
    expansion: int = 4
    multiscale_kernels: tuple = (3, 7, 15)
    temporal_hidden: int = 32
    context_dim: int = 0
    context_gain_limit: float = 0.1
    context_phase_bias_limit_rad: float = 0.1
    lstm_depth: int = 2
    spectral_depth: int = 4
    spectral_modes: int = 16
    crossmodal_width: int = 32
    crossmodal_heads: int = 4
    crossmodal_depth: int = 2
    crossmodal_patch_bins: int = 8
    fusion_residual_limit: float = 0.2
    hybrid_skeleton: str = "attention_fusion"
    hybrid_density: str = "spectral"
    hybrid_delay: str = "multiscale"
    hybrid_crossmodal: bool = True

    def validate(self):
        ints = [
            "tcn_depth",
            "tcn_kernel",
            "transformer_depth",
            "transformer_heads",
            "transformer_patch_bins",
            "transformer_max_tokens",
            "transformer_ff_multiplier",
            "convnext_depth",
            "convnext_kernel",
            "expansion",
            "temporal_hidden",
            "lstm_depth",
            "spectral_depth",
            "spectral_modes",
            "crossmodal_width",
            "crossmodal_heads",
            "crossmodal_depth",
            "crossmodal_patch_bins",
        ]
        if any(
            not isinstance(getattr(self, k), int) or getattr(self, k) < 1 for k in ints
        ):
            raise ValueError("Research sizes must be positive integers.")
        if self.tcn_kernel % 2 == 0 or self.convnext_kernel % 2 == 0:
            raise ValueError("Same-length convolutions require odd kernels.")
        if not self.multiscale_kernels or any(
            not isinstance(k, int) or k < 1 or k % 2 == 0
            for k in self.multiscale_kernels
        ):
            raise ValueError("Multiscale kernels must be positive odd integers.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout.")
        if not isinstance(self.context_dim, int) or self.context_dim < 0:
            raise ValueError("context_dim must be nonnegative.")
        if any(
            not math.isfinite(v) or v < 0
            for v in [self.context_gain_limit, self.context_phase_bias_limit_rad]
        ):
            raise ValueError("Invalid context modulation bounds.")
        if self.crossmodal_width % self.crossmodal_heads:
            raise ValueError("crossmodal_width must divide crossmodal_heads.")
        if (
            not math.isfinite(self.fusion_residual_limit)
            or not 0 <= self.fusion_residual_limit <= 1
        ):
            raise ValueError("fusion_residual_limit must be in [0,1].")
        if self.hybrid_skeleton not in [
            "mine",
            "tcn",
            "transformer",
            "cnn_lstm",
            "attention_fusion",
        ]:
            raise ValueError("Unknown hybrid skeleton.")
        if self.hybrid_density not in ["unet", "convnext", "spectral"]:
            raise ValueError("Unknown hybrid density.")
        if self.hybrid_delay not in ["resnet", "multiscale"]:
            raise ValueError("Unknown hybrid delay.")
        if not isinstance(self.hybrid_crossmodal, bool):
            raise ValueError("hybrid_crossmodal must be Boolean.")
        return self
