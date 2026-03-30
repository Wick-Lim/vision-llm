"""Latent diffusion: 1D U-Net operates in encoder's latent space.

No cross-attention needed — UNet input IS the latent representation.
Predicts x0 directly in latent space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        ac = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = torch.clip(1 - ac[1:] / ac[:-1], 0.0001, 0.9999).float()
        self.alphas_cumprod = torch.cumprod(1.0 - betas, 0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1mac = torch.sqrt(1.0 - self.alphas_cumprod)
        self.t_weights = 1.0 / torch.sqrt(torch.arange(num_timesteps, dtype=torch.float) + 1)
        self.t_weights = self.t_weights / self.t_weights.sum()

    def add_noise(self, x0, noise, t):
        sa = self.sqrt_ac[t.cpu()].to(x0.device)
        sb = self.sqrt_1mac[t.cpu()].to(x0.device)
        while sa.dim() < x0.dim(): sa = sa.unsqueeze(-1); sb = sb.unsqueeze(-1)
        return sa * x0 + sb * noise

    def sample_timesteps(self, B, device):
        return torch.multinomial(self.t_weights.to(device), B, replacement=True)

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond_latent, num_steps=200, device="cpu"):
        """DDIM in latent space. cond_latent: encoder output [B, S, D]."""
        B = shape[0]
        ts = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()
        x = torch.randn(shape, device=device)
        for i, t in enumerate(ts):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            x0_pred = model(x, tt, cond_latent).clamp(-3, 3)  # wider range in latent space
            if i < len(ts) - 1:
                at = self.alphas_cumprod[t].to(device)
                ap = self.alphas_cumprod[ts[i+1]].to(device)
                noise = (x - torch.sqrt(at) * x0_pred) / torch.sqrt(1 - at).clamp(min=1e-8)
                x = torch.sqrt(ap) * x0_pred + torch.sqrt(1 - ap) * noise
            else:
                x = x0_pred
        return x


class TimeMLPEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = t.float().unsqueeze(-1) * f.unsqueeze(0)
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))


class ConvBlock1d(nn.Module):
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


class SelfAttention1d(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2)


class Downsample1d(nn.Module):
    def __init__(self, ch): super().__init__(); self.c = nn.Conv1d(ch, ch, 3, 2, 1)
    def forward(self, x): return self.c(x)

class Upsample1d(nn.Module):
    def __init__(self, ch): super().__init__(); self.c = nn.ConvTranspose1d(ch, ch, 4, 2, 1)
    def forward(self, x): return self.c(x)


class LatentUNet1d(nn.Module):
    """1D U-Net for latent diffusion with additive skip conditioning.

    UNet predicts RESIDUAL. Output = z_cond + residual.
    At t=0 (echo): residual=0 → output=z_cond (trivially correct).
    """

    def __init__(self, latent_dim=256, model_dim=256, time_dim=128):
        super().__init__()
        D = model_dim
        self.time_embed = TimeMLPEmbedding(time_dim)
        # Input: noisy_latent only (no concat bottleneck)
        self.input_proj = nn.Conv1d(latent_dim, D, 1)

        self.down1 = ConvBlock1d(D, D, time_dim)
        self.down_sample1 = Downsample1d(D)
        self.down2 = ConvBlock1d(D, D*2, time_dim)
        self.down_sample2 = Downsample1d(D*2)

        self.mid1 = ConvBlock1d(D*2, D*2, time_dim)
        self.mid_attn = SelfAttention1d(D*2)
        self.mid2 = ConvBlock1d(D*2, D*2, time_dim)

        self.up_sample2 = Upsample1d(D*2)
        self.up2 = ConvBlock1d(D*2*2, D, time_dim)
        self.up_sample1 = Upsample1d(D)
        self.up1 = ConvBlock1d(D*2, D, time_dim)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, D), nn.SiLU(), nn.Conv1d(D, latent_dim, 1),
        )

    def forward(self, z_noisy, t, z_cond):
        """z_noisy: [B,S,D], t: [B], z_cond: [B,S,D]. Returns x0_pred = z_cond + residual."""
        te = self.time_embed(t)
        # Input: noisy latent only (no concat)
        x = self.input_proj(z_noisy.transpose(1, 2))  # [B, model_dim, S]

        h1 = self.down1(x, te)
        h1d = self.down_sample1(h1)
        h2 = self.down2(h1d, te)
        h2d = self.down_sample2(h2)

        m = self.mid1(h2d, te)
        m = self.mid_attn(m)
        m = self.mid2(m, te)

        u2 = _p(self.up_sample2(m), h2)
        u2 = self.up2(torch.cat([u2, h2], 1), te)
        u1 = _p(self.up_sample1(u2), h1)
        u1 = self.up1(torch.cat([u1, h1], 1), te)

        # Additive skip: output = z_cond + learned residual
        residual = self.output_proj(u1).transpose(1, 2)  # [B, S, latent_dim]
        return z_cond + residual


def _p(x, t):
    d = t.shape[-1] - x.shape[-1]
    return F.pad(x, (0, d)) if d > 0 else x[..., :t.shape[-1]] if d < 0 else x


def ema_update(ema, model, decay=0.9999):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(p.data, alpha=1 - decay)
