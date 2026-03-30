"""1D-CNN encoder: feature sequence + cmd prediction from clean input."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import NUM_CMDS, TENSOR_DIM


class PathEncoder(nn.Module):
    """Encode path tensor into feature SEQUENCE (not single vector).

    Returns the full conv feature sequence for cross-attention in UNet,
    plus cmd logits for command classification.

    Input:  [B, L, 8]
    Output: (feat_seq [B, L/8, 256], cmd_logits [B, L, NUM_CMDS])
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(TENSOR_DIM, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        # Cmd prediction head
        self.cmd_head = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, NUM_CMDS, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape

        feat = self.conv(x.transpose(1, 2))  # [B, 256, L/8]

        # Feature sequence for cross-attention (keep full spatial info)
        feat_seq = feat.transpose(1, 2)  # [B, L/8, 256]

        # Cmd prediction
        cmd_feat = self.cmd_head(feat)  # [B, NUM_CMDS, L/8]
        cmd_logits = F.interpolate(cmd_feat, size=L, mode="nearest")
        cmd_logits = cmd_logits.transpose(1, 2)  # [B, L, NUM_CMDS]

        return feat_seq, cmd_logits
