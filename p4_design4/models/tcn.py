from torch import nn
from network import ConvBlock, EntropyReliabilityRefinement
from .blocks import DilatedResidual


class TCNSkeleton(nn.Module):
    def __init__(self, cfg, options):
        super().__init__()
        width = 2 * cfg.gru_hidden
        self.stem = ConvBlock(1, width, 7)
        self.encoder = nn.Sequential(
            *(
                DilatedResidual(width, options.tcn_kernel, 2**i, options.dropout)
                for i in range(options.tcn_depth)
            )
        )
        self.erf = (
            EntropyReliabilityRefinement(width, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )
        self.head = nn.Linear(width, 1)

    def forward(self, phase):
        B, T, N = phase.shape
        z = (
            self.encoder(self.stem(phase.reshape(B * T, 1, N)))
            .transpose(1, 2)
            .reshape(B, T, N, -1)
        )
        diagnostic = {}
        if self.erf is not None:
            z, diagnostic = self.erf(z)
        return self.head(z).squeeze(-1).float(), z, diagnostic
