"""CNN Autoencoder with Think layer for vision-based text processing."""

import torch
import torch.nn as nn

from .renderer import IMG_H, IMG_W


class Encoder(nn.Module):
    """Grayscale image → latent vector."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Input: [B, 1, 64, 256]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # → [B, 32, 32, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → [B, 64, 16, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # → [B, 128, 8, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# → [B, 256, 4, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.flat_dim = 256 * 4 * 16  # 16384
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ThinkLayer(nn.Module):
    """MLP that transforms latent representations — the 'reasoning' step."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            d_in = latent_dim if i == 0 else hidden_dim
            d_out = latent_dim if i == depth - 1 else hidden_dim
            layers.extend([nn.Linear(d_in, d_out), nn.ReLU()])
        layers.pop()  # remove last ReLU
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) + z  # residual connection


class Decoder(nn.Module):
    """Latent vector → grayscale image."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.flat_dim = 256 * 4 * 16
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # → [B, 128, 8, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # → [B, 64, 16, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # → [B, 32, 32, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # → [B, 1, 64, 256]
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 16)
        return self.deconv(h)


class VisionLLM(nn.Module):
    """Full pipeline: Encode → Think → Decode."""

    def __init__(self, latent_dim: int = 256, think_depth: int = 3):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.think = ThinkLayer(latent_dim, depth=think_depth)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.think(z)
        return self.decoder(z)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
