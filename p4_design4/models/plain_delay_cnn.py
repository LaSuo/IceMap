from torch import nn
from .parameter_matching import matched_plain_cnn


class PlainDelayCNN(nn.Module):
    """Plain non-residual CNN replacing the eight-layer delay encoder."""

    def __init__(self, cfg, target_parameters):
        super().__init__()
        self.encoder, self.parameter_match = matched_plain_cnn(
            1, cfg.delay_channels + 1, target_parameters
        )

    def forward(self, tof):
        B, T, N = tof.shape
        return (
            self.encoder(tof.reshape(B * T, 1, N)).transpose(1, 2).reshape(B, T, N, -1)
        )
