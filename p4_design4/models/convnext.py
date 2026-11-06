import torch
from torch import nn
from network import ConvBlock
from .blocks import DilatedResidual


class ConvNeXtBlock(nn.Module):
    def __init__(self, width, options):
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            options.convnext_kernel,
            padding=options.convnext_kernel // 2,
            groups=width,
        )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * options.expansion),
            nn.GELU(),
            nn.Dropout(options.dropout),
            nn.Linear(width * options.expansion, width),
        )
        self.scale = nn.Parameter(torch.full((width,), 1e-6))

    def forward(self, x):
        y = self.depthwise(x).transpose(1, 2)
        return x + (self.mlp(self.norm(y)) * self.scale).transpose(1, 2)


class ConvNeXtDensity(nn.Module):
    def __init__(self, width, options):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, width, 7),
            *(ConvNeXtBlock(width, options) for _ in range(options.convnext_depth)),
        )

    def forward(self, scatter):
        B, T, N = scatter.shape
        return (
            self.net(scatter.reshape(B * T, 1, N)).transpose(1, 2).reshape(B, T, N, -1)
        )
