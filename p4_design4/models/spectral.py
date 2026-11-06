import torch
from torch import nn
from torch.nn import functional as F
from network import ConvBlock


class SpectralBlock(nn.Module):
    """Low-mode complex spectral operator plus local residual channel mixing."""

    def __init__(self, width, modes, dropout):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, 2) / (width**0.5))
        self.local = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if not x.is_floating_point() or x.ndim != 3:
            raise ValueError("Spectral input must be real [batch,channels,bins].")
        n = x.shape[-1]
        spectrum = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
        m = min(self.modes, spectrum.shape[-1])
        output = torch.zeros_like(spectrum)
        weight = torch.view_as_complex(self.weight.contiguous())
        output[..., :m] = torch.einsum(
            "bim,iom->bom", spectrum[..., :m], weight[..., :m]
        )
        transformed = torch.fft.irfft(output, n=n, dim=-1, norm="ortho").to(x.dtype)
        return x + self.drop(F.gelu(self.norm(transformed + self.local(x))))


class SpectralDensity(nn.Module):
    def __init__(self, width, options):
        super().__init__()
        self.stem = ConvBlock(1, width, 7)
        self.blocks = nn.Sequential(
            *(
                SpectralBlock(width, options.spectral_modes, options.dropout)
                for _ in range(options.spectral_depth)
            )
        )

    def forward(self, scatter):
        B, T, N = scatter.shape
        return (
            self.blocks(self.stem(scatter.reshape(B * T, 1, N)))
            .transpose(1, 2)
            .reshape(B, T, N, -1)
        )
