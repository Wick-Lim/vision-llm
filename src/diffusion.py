"""1D U-Net diffusion with strong cross-attention conditioning.

Cross-attention at ALL scales (down + mid + up).
Learnable residual scale for cross-attention output.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import TENSOR_DIM

COORD_DIM = TENSOR_DIM - 1  # 7


class NoiseScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        ac = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = torch.clip(1 - ac[1:] / ac[:-1], 0.0001, 0.9999).float()
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, 0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1mac = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, noise, t):
        sa = self.sqrt_ac[t.cpu()].to(x0.device)
        sb = self.sqrt_1mac[t.cpu()].to(x0.device)
        while sa.dim() < x0.dim(): sa = sa.unsqueeze(-1); sb = sb.unsqueeze(-1)
        return sa * x0 + sb * noise

    @torch.no_grad()
    def ddim_sample(self, model, shape_coords, context, num_steps=200, device="cpu", guidance_scale=3.0):
        B, L, _ = shape_coords
        ts = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()
        x = torch.randn(shape_coords, device=device)
        null_ctx = torch.zeros_like(context)
        for i, t in enumerate(ts):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            if guidance_scale > 1.0:
                nc = model(x, tt, context)
                nu = model(x, tt, null_ctx)
                pn = nu + guidance_scale * (nc - nu)
            else:
                pn = model(x, tt, context)
            at = self.alphas_cumprod[t].to(device)
            x0 = ((x - torch.sqrt(1 - at) * pn) / torch.sqrt(at)).clamp(-0.5, 0.5)
            if i < len(ts) - 1:
                ap = self.alphas_cumprod[ts[i + 1]].to(device)
                x = torch.sqrt(ap) * x0 + torch.sqrt(1 - ap) * pn
            else:
                x = x0
        return x


class TimeMLPEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
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


class CrossAttention1d(nn.Module):
    """Cross-attention with learnable residual scale."""
    def __init__(self, dim, context_dim=256, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.proj_q = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        # Learnable scale — starts at 1.0, can grow to amplify cross-attn signal
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, context):
        h = self.norm(x).transpose(1, 2)
        h = self.proj_q(h)
        h, _ = self.attn(h, context, context)
        return x + self.scale * h.transpose(1, 2)


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


class UNet1d(nn.Module):
    """1D U-Net with cross-attention at ALL scales."""

    def __init__(self, context_dim=256, model_dim=128, time_dim=128):
        super().__init__()
        D = model_dim
        self.time_embed = TimeMLPEmbedding(time_dim)
        self.input_proj = nn.Conv1d(COORD_DIM, D, 1)

        # Down — each with cross-attention
        self.down1 = ConvBlock1d(D, D, time_dim)
        self.down_cross1 = CrossAttention1d(D, context_dim)
        self.down_sample1 = Downsample1d(D)

        self.down2 = ConvBlock1d(D, D * 2, time_dim)
        self.down_cross2 = CrossAttention1d(D * 2, context_dim)
        self.down_sample2 = Downsample1d(D * 2)

        self.down3 = ConvBlock1d(D * 2, D * 4, time_dim)
        self.down_cross3 = CrossAttention1d(D * 4, context_dim)
        self.down_sample3 = Downsample1d(D * 4)

        # Mid
        self.mid1 = ConvBlock1d(D * 4, D * 4, time_dim)
        self.mid_self_attn = SelfAttention1d(D * 4)
        self.mid_cross_attn = CrossAttention1d(D * 4, context_dim)
        self.mid2 = ConvBlock1d(D * 4, D * 4, time_dim)

        # Up — each with cross-attention
        self.up_sample3 = Upsample1d(D * 4)
        self.up3 = ConvBlock1d(D * 4 * 2, D * 2, time_dim)
        self.up_cross3 = CrossAttention1d(D * 2, context_dim)

        self.up_sample2 = Upsample1d(D * 2)
        self.up2 = ConvBlock1d(D * 2 * 2, D, time_dim)
        self.up_cross2 = CrossAttention1d(D, context_dim)

        self.up_sample1 = Upsample1d(D)
        self.up1 = ConvBlock1d(D * 2, D, time_dim)
        self.up_cross1 = CrossAttention1d(D, context_dim)

        self.output_proj = nn.Sequential(nn.GroupNorm(8, D), nn.SiLU(), nn.Conv1d(D, COORD_DIM, 1))

    def forward(self, x, t, context):
        te = self.time_embed(t)
        x = self.input_proj(x.transpose(1, 2))

        h1 = self.down_cross1(self.down1(x, te), context)
        h1d = self.down_sample1(h1)
        h2 = self.down_cross2(self.down2(h1d, te), context)
        h2d = self.down_sample2(h2)
        h3 = self.down_cross3(self.down3(h2d, te), context)
        h3d = self.down_sample3(h3)

        m = self.mid1(h3d, te)
        m = self.mid_self_attn(m)
        m = self.mid_cross_attn(m, context)
        m = self.mid2(m, te)

        u3 = self.up_cross3(self.up3(torch.cat([_p(self.up_sample3(m), h3), h3], 1), te), context)
        u2 = self.up_cross2(self.up2(torch.cat([_p(self.up_sample2(u3), h2), h2], 1), te), context)
        u1 = self.up_cross1(self.up1(torch.cat([_p(self.up_sample1(u2), h1), h1], 1), te), context)

        return self.output_proj(u1).transpose(1, 2)


def _p(x, t):
    d = t.shape[-1] - x.shape[-1]
    return F.pad(x, (0, d)) if d > 0 else x[..., :t.shape[-1]] if d < 0 else x


def ema_update(ema, model, decay=0.9999):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(p.data, alpha=1 - decay)
