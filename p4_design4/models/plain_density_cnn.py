from torch import nn
from .parameter_matching import matched_plain_cnn


class PlainDensityCNN(nn.Module):
    """Plain depth CNN in place of the multiscale encoder/decoder U-Net."""

    def __init__(self, cfg, target_parameters):
        super().__init__()
        self.encoder, self.parameter_match = matched_plain_cnn(
            1, cfg.unet_base, target_parameters
        )

    def forward(self, scatter):
        B, T, N = scatter.shape
        return (
            self.encoder(scatter.reshape(B * T, 1, N))
            .transpose(1, 2)
            .reshape(B, T, N, -1)
        )
