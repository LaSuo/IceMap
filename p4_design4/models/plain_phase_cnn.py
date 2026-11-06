from torch import nn
from network import EntropyReliabilityRefinement
from .parameter_matching import matched_plain_cnn


class PlainPhaseCNN(nn.Module):
    """Replace CNN/BiGRU by a parameter-matched depth CNN; retain phase ERF."""

    def __init__(self, cfg, target_parameters):
        super().__init__()
        channels = 2 * cfg.gru_hidden
        self.encoder, self.parameter_match = matched_plain_cnn(
            1, channels, target_parameters, fixed_parameters=channels + 1
        )
        self.head = nn.Linear(channels, 1)
        self.erf = (
            EntropyReliabilityRefinement(channels, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )

    def forward(self, phase):
        B, T, N = phase.shape
        z = (
            self.encoder(phase.reshape(B * T, 1, N))
            .transpose(1, 2)
            .reshape(B, T, N, -1)
        )
        reliability = {}
        if self.erf is not None:
            z, reliability = self.erf(z)
        return self.head(z).squeeze(-1), z, reliability
