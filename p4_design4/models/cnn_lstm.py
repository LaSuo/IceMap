from torch import nn
from network import ConvBlock, EntropyReliabilityRefinement


class CNNLSTMSkeleton(nn.Module):
    """Local convolutions + depth-sequence bidirectional LSTM."""

    def __init__(self, cfg, options):
        super().__init__()
        width = 2 * cfg.gru_hidden
        self.stem = nn.Sequential(
            ConvBlock(1, cfg.cnn_channels, 7),
            ConvBlock(cfg.cnn_channels, cfg.cnn_channels, 3),
        )
        self.recurrent = nn.LSTM(
            cfg.cnn_channels,
            cfg.gru_hidden,
            num_layers=options.lstm_depth,
            batch_first=True,
            bidirectional=True,
            dropout=options.dropout if options.lstm_depth > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        self.erf = (
            EntropyReliabilityRefinement(width, *cfg.erf_windows[0], cfg.erf_beta[0])
            if cfg.use_erf
            else None
        )

    def forward(self, phase):
        B, T, N = phase.shape
        x = self.stem(phase.reshape(B * T, 1, N)).transpose(1, 2)
        x, _ = self.recurrent(x)
        z = self.norm(x).reshape(B, T, N, -1)
        info = {}
        if self.erf is not None:
            z, info = self.erf(z)
        return self.head(z).squeeze(-1).float(), z, info
