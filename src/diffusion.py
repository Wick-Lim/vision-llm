"""1D U-Net diffusion model for vector path generation.

Generates vector path sequences conditioned on an encoder output.
Uses a 1D U-Net architecture operating on [B, max_len, path_dim] tensors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vectorizer import TENSOR_DIM


# ============================================================
# Noise scheduler
# ============================================================


class NoiseScheduler:
    """Linear beta noise schedule for diffusion."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x0.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].to(x0.device)

        # Reshape for broadcasting: [B] → [B, 1, 1]
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple,
        cond: torch.Tensor,
        num_steps: int = 50,
        device: str = "cpu",
    ) -> torch.Tensor:
        """DDIM deterministic sampling (eta=0)."""
        # Create sub-sequence of timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, cond)

            alpha_t = self.alphas_cumprod[t].to(device)
            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev].to(device)
                # DDIM update (eta=0, deterministic)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = x0_pred

        return x


# ============================================================
# Time embedding
# ============================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeMLPEmbedding(nn.Module):
    """Sinusoidal → MLP for time conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ============================================================
# 1D U-Net blocks
# ============================================================


class ConvBlock1d(nn.Module):
    """Conv1d → GroupNorm → SiLU, with time embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(x) + self.time_proj(t_emb).unsqueeze(-1)
        return h + self.res_conv(x)


class SelfAttention1d(nn.Module):
    """Multi-head self-attention over sequence dimension."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = self.norm(x)
        h = h.transpose(1, 2)  # [B, L, C]
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2)  # [B, C, L]
        return x + h


class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# 1D U-Net
# ============================================================


class UNet1d(nn.Module):
    """1D U-Net for sequence diffusion.

    Input:  noisy paths [B, max_len, path_dim]
    Output: predicted noise [B, max_len, path_dim]

    Conditioning via concatenation: cond vector is expanded
    to sequence length and concatenated with input.
    """

    def __init__(
        self,
        path_dim: int = TENSOR_DIM,
        cond_dim: int = 256,
        model_dim: int = 128,
        time_dim: int = 128,
    ):
        super().__init__()
        self.path_dim = path_dim
        self.cond_dim = cond_dim

        # Time embedding
        self.time_embed = TimeMLPEmbedding(time_dim)

        # Input projection: path_dim + cond_dim → model_dim
        self.input_proj = nn.Conv1d(path_dim + cond_dim, model_dim, 1)

        # Down path
        self.down1 = ConvBlock1d(model_dim, model_dim, time_dim)
        self.down_sample1 = Downsample1d(model_dim)
        self.down2 = ConvBlock1d(model_dim, model_dim * 2, time_dim)
        self.down_sample2 = Downsample1d(model_dim * 2)
        self.down3 = ConvBlock1d(model_dim * 2, model_dim * 4, time_dim)
        self.down_sample3 = Downsample1d(model_dim * 4)

        # Mid
        self.mid1 = ConvBlock1d(model_dim * 4, model_dim * 4, time_dim)
        self.mid_attn = SelfAttention1d(model_dim * 4)
        self.mid2 = ConvBlock1d(model_dim * 4, model_dim * 4, time_dim)

        # Up path
        self.up_sample3 = Upsample1d(model_dim * 4)
        self.up3 = ConvBlock1d(model_dim * 4 * 2, model_dim * 2, time_dim)  # skip concat
        self.up_sample2 = Upsample1d(model_dim * 2)
        self.up2 = ConvBlock1d(model_dim * 2 * 2, model_dim, time_dim)
        self.up_sample1 = Upsample1d(model_dim)
        self.up1 = ConvBlock1d(model_dim * 2, model_dim, time_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_dim),
            nn.SiLU(),
            nn.Conv1d(model_dim, path_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, max_len, path_dim] noisy path sequence
        t:    [B] integer timesteps
        cond: [B, cond_dim] condition vector from encoder
        """
        B, L, _ = x.shape
        t_emb = self.time_embed(t)  # [B, time_dim]

        # Concatenate condition vector with each position
        cond_expanded = cond.unsqueeze(1).expand(B, L, self.cond_dim)  # [B, L, cond_dim]
        x = torch.cat([x, cond_expanded], dim=-1)  # [B, L, path_dim + cond_dim]
        x = x.transpose(1, 2)  # [B, path_dim + cond_dim, L]

        x = self.input_proj(x)  # [B, model_dim, L]

        # Down
        h1 = self.down1(x, t_emb)            # [B, D, L]
        h1d = self.down_sample1(h1)           # [B, D, L/2]
        h2 = self.down2(h1d, t_emb)           # [B, 2D, L/2]
        h2d = self.down_sample2(h2)           # [B, 2D, L/4]
        h3 = self.down3(h2d, t_emb)           # [B, 4D, L/4]
        h3d = self.down_sample3(h3)           # [B, 4D, L/8]

        # Mid
        m = self.mid1(h3d, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)

        # Up (with skip connections + padding to match sizes)
        u3 = self.up_sample3(m)                         # [B, 4D, L/4]
        u3 = _pad_to_match(u3, h3)
        u3 = self.up3(torch.cat([u3, h3], dim=1), t_emb)  # [B, 2D, L/4]

        u2 = self.up_sample2(u3)                         # [B, 2D, L/2]
        u2 = _pad_to_match(u2, h2)
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb)  # [B, D, L/2]

        u1 = self.up_sample1(u2)                         # [B, D, L]
        u1 = _pad_to_match(u1, h1)
        u1 = self.up1(torch.cat([u1, h1], dim=1), t_emb)  # [B, D, L]

        out = self.output_proj(u1)  # [B, path_dim, L]
        return out.transpose(1, 2)  # [B, L, path_dim]


def _pad_to_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pad x along sequence dim to match target size."""
    diff = target.shape[-1] - x.shape[-1]
    if diff > 0:
        x = F.pad(x, (0, diff))
    elif diff < 0:
        x = x[..., :target.shape[-1]]
    return x
