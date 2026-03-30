"""1D-CNN encoder: condition vector + cmd sequence prediction from clean input."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import NUM_CMDS, TENSOR_DIM


class PathEncoder(nn.Module):
    """Encode path tensor into condition vector AND predict cmd sequence.

    The cmd predictor works on CLEAN input (not noisy coords),
    so it can accurately classify cmd types.

    Input:  [B, max_len, 8]
    Output: (cond [B, cond_dim], cmd_logits [B, max_len, NUM_CMDS])
    """

    def __init__(self, cond_dim: int = 256):
        super().__init__()
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

        # Cmd prediction head — operates on conv features, upsampled to full length
        self.cmd_head = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, NUM_CMDS, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        mask = (x[:, :, 0] > -0.45).float()

        feat = self.conv(x.transpose(1, 2))  # [B, 256, L/8]

        # Condition vector via masked pooling
        mask_down = F.avg_pool1d(mask.unsqueeze(1), kernel_size=8).squeeze(1)
        masked_feat = feat * mask_down.unsqueeze(1)
        denom = mask_down.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = masked_feat.sum(dim=-1) / denom
        cond = self.fc(pooled)

        # Cmd prediction — upsample features to original length
        cmd_feat = self.cmd_head(feat)  # [B, NUM_CMDS, L/8]
        cmd_logits = F.interpolate(cmd_feat, size=L, mode="nearest")  # [B, NUM_CMDS, L]
        cmd_logits = cmd_logits.transpose(1, 2)  # [B, L, NUM_CMDS]

        return cond, cmd_logits
