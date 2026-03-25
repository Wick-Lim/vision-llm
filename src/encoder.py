"""1D-CNN condition vector encoder for vector path sequences."""

import torch
import torch.nn as nn

from .vectorizer import TENSOR_DIM


class PathEncoder(nn.Module):
    """Encode a vector path tensor into a fixed-size condition vector.

    Input:  [B, max_len, 8] (path command sequence)
    Output: [B, cond_dim] (condition vector)
    """

    def __init__(self, cond_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            # [B, 8, max_len] after transpose
            nn.Conv1d(TENSOR_DIM, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, 256, 1]
        )
        self.fc = nn.Linear(256, cond_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len, 8]
        x = x.transpose(1, 2)  # [B, 8, max_len]
        x = self.conv(x)       # [B, 256, 1]
        x = x.squeeze(-1)      # [B, 256]
        return self.fc(x)      # [B, cond_dim]
