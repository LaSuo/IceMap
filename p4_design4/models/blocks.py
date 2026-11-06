from torch import nn


class DilatedResidual(nn.Module):
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                width,
                width,
                kernel,
                padding=dilation * (kernel // 2),
                dilation=dilation,
            ),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return x + self.net(x)
