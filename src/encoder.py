"""1D-CNN condition vector encoder with masked pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import TENSOR_DIM


class PathEncoder(nn.Module):
    """Encode a vector path tensor into a fixed-size condition vector.

    Uses masked average pooling to ignore padding rows,
    so only actual content contributes to the condition embedding.

    Input:  [B, max_len, 8] (path command sequence)
    Output: [B, cond_dim] (condition vector)
    """

    def __init__(self, cond_dim: int = 256):
        super().__init__()
        # Conv stack without pooling — stride=2 per layer → total 8x downsample
        self.conv = nn.Sequential(
            nn.Conv1d(TENSOR_DIM, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, cond_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len, 8]
        # Content mask: cmd > -0.45 means non-padding
        mask = (x[:, :, 0] > -0.45).float()  # [B, L]

        feat = self.conv(x.transpose(1, 2))  # [B, 256, L/8]

        # Downsample mask to match conv output (8x downsample)
        mask_down = F.max_pool1d(mask.unsqueeze(1), kernel_size=8).squeeze(1)  # [B, L/8]

        # Masked average pooling: only content positions
        masked_feat = feat * mask_down.unsqueeze(1)  # [B, 256, L/8]
        denom = mask_down.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        pooled = masked_feat.sum(dim=-1) / denom  # [B, 256]

        return self.fc(pooled)
