"""1D U-Net diffusion model for vector path generation.

Key design: diffusion operates ONLY on coordinates (dims 1-7).
Command types (dim 0) are predicted via classification head, not diffused.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import NUM_CMDS, TENSOR_DIM

# Coordinate dimensions (x, y, cx1, cy1, cx2, cy2, flag)
COORD_DIM = TENSOR_DIM - 1  # 7


# ============================================================
# Noise scheduler (cosine)
# ============================================================


class NoiseScheduler:
    """Cosine beta noise schedule (Nichol & Dhariwal, 2021)."""

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

        # Cosine schedule
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).float()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion on coordinates only."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t.cpu()].to(x0.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(x0.device)

        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape_coords: tuple,
        cond: torch.Tensor,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> torch.Tensor:
        """DDIM sampling. Returns coords [B, L, 7]."""
        B, L, _ = shape_coords
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()

        x = torch.randn(shape_coords, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, cond)

            alpha_t = self.alphas_cumprod[t].to(device)
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0_pred = x0_pred.clamp(-0.5, 0.5)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev].to(device)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = x0_pred

        return x


# ============================================================
# Time embedding
# ============================================================


class TimeMLPEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ============================================================
# 1D U-Net blocks
# ============================================================


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        # Position-dependent conditioning: expand + conv for per-position signal
        self.cond_fc = nn.Linear(cond_dim, out_ch)
        self.cond_conv = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_emb):
        # Expand condition to sequence length, then conv for position-specific signal
        cond_signal = self.cond_fc(cond_emb).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        cond_signal = self.cond_conv(cond_signal)  # [B, out_ch, L] — different per position
        h = self.conv(x) + self.time_proj(t_emb).unsqueeze(-1) + cond_signal
        return h + self.res_conv(x)


class SelfAttention1d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2)


class Downsample1d(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


# ============================================================
# 1D U-Net with separate coord/cmd heads
# ============================================================


class UNet1d(nn.Module):
    """1D U-Net for coordinate-only diffusion.

    Predicts noise on coordinates (7 dims).
    Cmd prediction is handled by the encoder, not the UNet.

    Input:  noisy coords [B, L, 7]
    Output: pred_noise [B, L, 7]
    """

    def __init__(self, cond_dim: int = 256, model_dim: int = 128, time_dim: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_embed = TimeMLPEmbedding(time_dim)

        # Input: coords only (7 dims)
        self.input_proj = nn.Conv1d(COORD_DIM, model_dim, 1)

        # Down
        self.down1 = ConvBlock1d(model_dim, model_dim, time_dim, cond_dim)
        self.down_sample1 = Downsample1d(model_dim)
        self.down2 = ConvBlock1d(model_dim, model_dim * 2, time_dim, cond_dim)
        self.down_sample2 = Downsample1d(model_dim * 2)
        self.down3 = ConvBlock1d(model_dim * 2, model_dim * 4, time_dim, cond_dim)
        self.down_sample3 = Downsample1d(model_dim * 4)

        # Mid
        self.mid1 = ConvBlock1d(model_dim * 4, model_dim * 4, time_dim, cond_dim)
        self.mid_attn = SelfAttention1d(model_dim * 4)
        self.mid2 = ConvBlock1d(model_dim * 4, model_dim * 4, time_dim, cond_dim)

        # Up
        self.up_sample3 = Upsample1d(model_dim * 4)
        self.up3 = ConvBlock1d(model_dim * 4 * 2, model_dim * 2, time_dim, cond_dim)
        self.up_sample2 = Upsample1d(model_dim * 2)
        self.up2 = ConvBlock1d(model_dim * 2 * 2, model_dim, time_dim, cond_dim)
        self.up_sample1 = Upsample1d(model_dim)
        self.up1 = ConvBlock1d(model_dim * 2, model_dim, time_dim, cond_dim)

        # Coord-only output head
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_dim), nn.SiLU(),
            nn.Conv1d(model_dim, COORD_DIM, 1),
        )

    def forward(self, x, t, cond):
        """x: [B, L, 7] noisy coords. Returns pred_noise [B, L, 7]."""
        B, L, _ = x.shape
        t_emb = self.time_embed(t)

        x = x.transpose(1, 2)
        x = self.input_proj(x)

        h1 = self.down1(x, t_emb, cond)
        h1d = self.down_sample1(h1)
        h2 = self.down2(h1d, t_emb, cond)
        h2d = self.down_sample2(h2)
        h3 = self.down3(h2d, t_emb, cond)
        h3d = self.down_sample3(h3)

        m = self.mid1(h3d, t_emb, cond)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb, cond)

        u3 = _pad_to_match(self.up_sample3(m), h3)
        u3 = self.up3(torch.cat([u3, h3], dim=1), t_emb, cond)
        u2 = _pad_to_match(self.up_sample2(u3), h2)
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb, cond)
        u1 = _pad_to_match(self.up_sample1(u2), h1)
        u1 = self.up1(torch.cat([u1, h1], dim=1), t_emb, cond)
        return self.output_proj(u1).transpose(1, 2)  # [B, L, 7]


def _pad_to_match(x, target):
    diff = target.shape[-1] - x.shape[-1]
    if diff > 0:
        x = F.pad(x, (0, diff))
    elif diff < 0:
        x = x[..., :target.shape[-1]]
    return x


# ============================================================
# EMA helper
# ============================================================


def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    """Update EMA weights."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
