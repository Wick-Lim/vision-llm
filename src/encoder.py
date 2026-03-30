"""Encoder + Decoder for latent diffusion on vector paths."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import NUM_CMDS, TENSOR_DIM, COORD_DIM


class PathEncoder(nn.Module):
    """Encode path tensor → latent sequence [B, L/2, latent_dim]."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(TENSOR_DIM, 64, 3, stride=2, padding=1),  # L → L/2
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, latent_dim, 3, padding=1),
            nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )
        # Cmd prediction head
        self.cmd_head = nn.Sequential(
            nn.Conv1d(latent_dim, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, NUM_CMDS, 1),
        )

    def forward(self, x):
        """x: [B, L, 8] → (latent [B, L/2, latent_dim], cmd_logits [B, L, NUM_CMDS])"""
        B, L, _ = x.shape
        feat = self.conv(x.transpose(1, 2))  # [B, latent_dim, L/2]
        latent = feat.transpose(1, 2)  # [B, L/2, latent_dim]

        cmd_feat = self.cmd_head(feat)
        cmd_logits = F.interpolate(cmd_feat, size=L, mode="nearest").transpose(1, 2)

        return latent, cmd_logits


class PathDecoder(nn.Module):
    """Decode latent sequence → coordinates [B, L, COORD_DIM]."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, 4, stride=2, padding=1),  # L/2 → L
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, COORD_DIM, 3, padding=1),
            nn.Tanh(),  # output in [-1, 1], will scale to [-0.5, 0.5]
        )

    def forward(self, z):
        """z: [B, L/2, latent_dim] → coords [B, L, COORD_DIM]"""
        out = self.deconv(z.transpose(1, 2))  # [B, COORD_DIM, L]
        return out.transpose(1, 2) * 0.5  # scale to [-0.5, 0.5]
