"""1D U-Net diffusion model with cross-attention conditioning.

Diffusion operates ONLY on coordinates (dims 1-7).
Conditioning via cross-attention to encoder feature sequence.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import TENSOR_DIM

COORD_DIM = TENSOR_DIM - 1  # 7


# ============================================================
# Noise scheduler (cosine)
# ============================================================


class NoiseScheduler:
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def add_noise(self, x0, noise, t):
        sa = self.sqrt_alphas_cumprod[t.cpu()].to(x0.device)
        sb = self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(x0.device)
        while sa.dim() < x0.dim():
            sa = sa.unsqueeze(-1)
            sb = sb.unsqueeze(-1)
        return sa * x0 + sb * noise

    @torch.no_grad()
    def ddim_sample(self, model, shape_coords, context, num_steps=100, device="cpu"):
        """DDIM sampling. context: [B, S, 256] encoder feature sequence."""
        B, L, _ = shape_coords
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()
        x = torch.randn(shape_coords, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, context)
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
# Building blocks
# ============================================================


class TimeMLPEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class ConvBlock1d(nn.Module):
    """Conv block with time embedding (no condition — condition via cross-attn)."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        return self.conv(x) + self.time_proj(t_emb).unsqueeze(-1) + self.res_conv(x)


class CrossAttention1d(nn.Module):
    """Cross-attention: UNet features attend to encoder sequence."""

    def __init__(self, dim, context_dim=256, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.proj_q = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, kdim=context_dim, vdim=context_dim, batch_first=True)

    def forward(self, x, context):
        # x: [B, C, L], context: [B, S, context_dim]
        h = self.norm(x).transpose(1, 2)  # [B, L, C]
        h = self.proj_q(h)
        h, _ = self.attn(h, context, context)
        return x + h.transpose(1, 2)


class SelfAttention1d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2)


class Downsample1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


# ============================================================
# 1D U-Net with cross-attention
# ============================================================


class UNet1d(nn.Module):
    """1D U-Net for coordinate diffusion with cross-attention to encoder features.

    Input:  noisy coords [B, L, 7]
    Output: pred_noise [B, L, 7]
    Context: encoder feature sequence [B, S, context_dim]
    """

    def __init__(self, context_dim=256, model_dim=128, time_dim=128):
        super().__init__()
        D = model_dim
        self.time_embed = TimeMLPEmbedding(time_dim)
        self.input_proj = nn.Conv1d(COORD_DIM, D, 1)

        # Down
        self.down1 = ConvBlock1d(D, D, time_dim)
        self.down_sample1 = Downsample1d(D)
        self.down2 = ConvBlock1d(D, D * 2, time_dim)
        self.down_sample2 = Downsample1d(D * 2)
        self.down3 = ConvBlock1d(D * 2, D * 4, time_dim)
        self.down_sample3 = Downsample1d(D * 4)

        # Mid: self-attention + cross-attention
        self.mid1 = ConvBlock1d(D * 4, D * 4, time_dim)
        self.mid_self_attn = SelfAttention1d(D * 4)
        self.mid_cross_attn = CrossAttention1d(D * 4, context_dim)
        self.mid2 = ConvBlock1d(D * 4, D * 4, time_dim)

        # Up with cross-attention at each scale
        self.up_sample3 = Upsample1d(D * 4)
        self.up3 = ConvBlock1d(D * 4 * 2, D * 2, time_dim)
        self.up_cross3 = CrossAttention1d(D * 2, context_dim)

        self.up_sample2 = Upsample1d(D * 2)
        self.up2 = ConvBlock1d(D * 2 * 2, D, time_dim)
        self.up_cross2 = CrossAttention1d(D, context_dim)

        self.up_sample1 = Upsample1d(D)
        self.up1 = ConvBlock1d(D * 2, D, time_dim)
        self.up_cross1 = CrossAttention1d(D, context_dim)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, D), nn.SiLU(), nn.Conv1d(D, COORD_DIM, 1),
        )

    def forward(self, x, t, context):
        """x: [B,L,7], t: [B], context: [B,S,256]. Returns [B,L,7]."""
        t_emb = self.time_embed(t)
        x = self.input_proj(x.transpose(1, 2))

        h1 = self.down1(x, t_emb)
        h1d = self.down_sample1(h1)
        h2 = self.down2(h1d, t_emb)
        h2d = self.down_sample2(h2)
        h3 = self.down3(h2d, t_emb)
        h3d = self.down_sample3(h3)

        m = self.mid1(h3d, t_emb)
        m = self.mid_self_attn(m)
        m = self.mid_cross_attn(m, context)
        m = self.mid2(m, t_emb)

        u3 = _pad(self.up_sample3(m), h3)
        u3 = self.up3(torch.cat([u3, h3], 1), t_emb)
        u3 = self.up_cross3(u3, context)

        u2 = _pad(self.up_sample2(u3), h2)
        u2 = self.up2(torch.cat([u2, h2], 1), t_emb)
        u2 = self.up_cross2(u2, context)

        u1 = _pad(self.up_sample1(u2), h1)
        u1 = self.up1(torch.cat([u1, h1], 1), t_emb)
        u1 = self.up_cross1(u1, context)

        return self.output_proj(u1).transpose(1, 2)


def _pad(x, target):
    d = target.shape[-1] - x.shape[-1]
    return F.pad(x, (0, d)) if d > 0 else x[..., :target.shape[-1]] if d < 0 else x


def ema_update(ema, model, decay=0.9999):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(p.data, alpha=1 - decay)
