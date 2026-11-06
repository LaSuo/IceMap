from torch import nn
from network import SPAIIceInversionNet
from .tcn import TCNSkeleton
from .patch_transformer import TransformerSkeleton
from .cnn_lstm import CNNLSTMSkeleton
from .convnext import ConvNeXtDensity
from .multiscale import MultiscaleTemporalToF
from .spectral import SpectralDensity
from .gated_fusion import GatedSkeletonFusion
from .cross_modal import CrossModalAdapter


SKELETONS = {
    "tcn": TCNSkeleton,
    "transformer": TransformerSkeleton,
    "cnn_lstm": CNNLSTMSkeleton,
    "attention_fusion": GatedSkeletonFusion,
}
DENSITIES = {"convnext": ConvNeXtDensity, "spectral": SpectralDensity}


class ComposedSPAI(SPAIIceInversionNet):
    """Composition only; all encoder algorithms are implemented separately."""

    def __init__(self, cfg, variant, options):
        super().__init__(cfg)
        self.variant = variant
        self.options = options
        skeleton = variant if variant in SKELETONS else "mine"
        density = variant if variant in DENSITIES else "unet"
        delay = "multiscale" if variant == "multiscale" else "resnet"
        cross = variant == "cross_modal"
        if variant == "hybrid":
            skeleton = options.hybrid_skeleton
            density = options.hybrid_density
            delay = options.hybrid_delay
            cross = options.hybrid_crossmodal
        if skeleton != "mine":
            self.skeleton = SKELETONS[skeleton](cfg, options)
        if density != "unet":
            self.density_unet = DENSITIES[density](cfg.unet_base, options)
        if delay != "resnet":
            self.tof_encoder = MultiscaleTemporalToF(cfg.delay_channels, options)
        self.crossmodal = CrossModalAdapter(options) if cross else None
        self.context_adapter = None
        if options.context_dim:
            self.context_adapter = nn.Sequential(
                nn.Linear(options.context_dim, 32), nn.GELU(), nn.Linear(32, 4)
            )
            nn.init.zeros_(self.context_adapter[-1].weight)
            nn.init.zeros_(self.context_adapter[-1].bias)

    def forward(
        self,
        tof,
        phase,
        scatter,
        time_s=None,
        depth_m=None,
        scatter_valid=None,
        need_layer=True,
        need_physical=True,
        context=None,
    ):
        if self.context_adapter is not None:
            if (
                context is None
                or context.shape != (*phase.shape[:2], self.options.context_dim)
                or not context.is_floating_point()
                or not context.isfinite().all()
            ):
                raise ValueError(
                    "Context must be finite floating [batch,frames,context_dim]."
                )
            z = self.context_adapter(context).tanh()
            phase = (
                phase + self.options.context_phase_bias_limit_rad * z[..., 0, None]
            ) * (self.options.context_gain_limit * z[..., 1, None]).exp()
            scatter = (
                scatter * (self.options.context_gain_limit * z[..., 2, None]).exp()
            )
            tof = tof * (self.options.context_gain_limit * z[..., 3, None]).exp()
        elif context is not None:
            raise ValueError("Context supplied but context_dim is zero.")
        if self.crossmodal is not None:
            phase, scatter, tof = self.crossmodal(phase, scatter, tof)
        return super().forward(
            tof,
            phase,
            scatter,
            time_s,
            depth_m,
            scatter_valid,
            need_layer,
            need_physical,
        )
