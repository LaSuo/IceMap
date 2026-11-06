import torch
from torch import nn
from network import ConvBlock
from .blocks import DilatedResidual


class MultiscaleTemporalToF(nn.Module):
    def __init__(self, width, options):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(1, width, k),
                    DilatedResidual(width, 3, 2, options.dropout),
                )
                for k in options.multiscale_kernels
            ]
        )
        self.projection = nn.Conv1d(width * len(options.multiscale_kernels), width, 1)
        self.temporal = nn.GRU(
            width, options.temporal_hidden, batch_first=True, bidirectional=True
        )
        self.head = nn.Linear(2 * options.temporal_hidden, width + 1)

    def forward(self, tof):
        B, T, N = tof.shape
        x = tof.reshape(B * T, 1, N)
        x = self.projection(torch.cat([branch(x) for branch in self.branches], 1))
        x = x.reshape(B, T, -1, N).permute(0, 3, 1, 2).reshape(B * N, T, -1)
        z, _ = self.temporal(x)
        return self.head(z).reshape(B, N, T, -1).permute(0, 2, 1, 3)
