"""1D-CNN encoder: feature sequence + cmd prediction. 2x downsample only."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import NUM_CMDS, TENSOR_DIM


class PathEncoder(nn.Module):
    """Encode path tensor into feature sequence with minimal downsampling.

    Only 2x downsample (was 8x) → preserves 32 positions from 64-length input.
    More context positions = stronger cross-attention signal.

    Input:  [B, L, 8]
    Output: (feat_seq [B, L/2, 256], cmd_logits [B, L, NUM_CMDS])
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        # Only 1 stride-2 layer (2x downsample instead of 8x)
        self.conv = nn.Sequential(
            nn.Conv1d(TENSOR_DIM, 64, kernel_size=3, stride=2, padding=1),   # L → L/2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),          # L/2 → L/2
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feat_dim, kernel_size=3, stride=1, padding=1),    # L/2 → L/2
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        self.cmd_head = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, NUM_CMDS, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        feat = self.conv(x.transpose(1, 2))  # [B, 256, L/2]
        feat_seq = feat.transpose(1, 2)       # [B, L/2, 256]

        cmd_feat = self.cmd_head(feat)
        cmd_logits = F.interpolate(cmd_feat, size=L, mode="nearest").transpose(1, 2)

        return feat_seq, cmd_logits
