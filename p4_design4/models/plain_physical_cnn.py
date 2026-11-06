import torch
from torch import nn


class PlainPhysicalCNN(nn.Module):
    """Pointwise layer CNN directly regressing rho/h without analytic coupling."""

    def __init__(self, delay_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(delay_channels + 1, 32, 1),
            nn.GELU(),
            nn.Conv1d(32, 2, 1),
            nn.Softplus(),
        )

    def forward(self, rho_tilde, delay_features):
        x = torch.cat((rho_tilde.unsqueeze(-1), delay_features), -1)
        B, T, K, C = x.shape
        out = (
            self.layers(x.reshape(B * T, K, C).transpose(1, 2))
            .transpose(1, 2)
            .reshape(B, T, K, 2)
        )
        return out[..., 0], out[..., 1]
