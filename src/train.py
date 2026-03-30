"""Training loop for vector path diffusion model.

Diffusion operates on coordinates only (dims 1-7).
Command types (dim 0) are predicted via cross-entropy classification.
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import GlyphEchoDataset, collate_glyph_batch
from .diffusion import COORD_DIM, NoiseScheduler, UNet1d, ema_update
from .encoder import PathEncoder
from .vectorizer import DEFAULT_MAX_LEN, NUM_CMDS, _denormalize_cmd


def train(
    font_path: str | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    save_dir: str = "checkpoints",
    num_timesteps: int = 1000,
    cond_dim: int = 256,
    model_dim: int = 128,
    max_chars: int | None = None,
):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # Dataset — echo task: same data for train and val (different noise levels)
    dataset = GlyphEchoDataset(font_path=font_path, max_len=max_len, max_chars=max_chars)

    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_glyph_batch)
    val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_glyph_batch)

    # Models
    encoder = PathEncoder(feat_dim=cond_dim).to(device)
    unet = UNet1d(context_dim=cond_dim, model_dim=model_dim).to(device)
    scheduler = NoiseScheduler(num_timesteps=num_timesteps)

    # EMA copies for stable sampling
    ema_encoder = copy.deepcopy(encoder)
    ema_unet = copy.deepcopy(unet)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in unet.parameters())
    print(f"Parameters: encoder={sum(p.numel() for p in encoder.parameters()):,} "
          f"unet={sum(p.numel() for p in unet.parameters()):,} total={total_params:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(unet.parameters()),
        lr=lr,
    )
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        encoder.train()
        unet.train()
        train_loss = 0.0
        n_train = 0

        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            B = src.size(0)

            # Split target into cmd indices and coordinates
            cmd_normalized = tgt[:, :, 0]  # [B, L] normalized cmd values
            coords = tgt[:, :, 1:]         # [B, L, 7] coordinates

            # Convert normalized cmd to integer indices for cross-entropy
            cmd_indices = ((cmd_normalized + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)

            # Content mask from cmd (non-pad positions)
            content_mask = (cmd_indices > 0).float()  # [B, L]

            # Encode: condition vector + cmd prediction from CLEAN input
            context, cmd_logits = encoder(src)

            # Forward diffusion on COORDINATES ONLY (no noise on padding)
            t = torch.randint(0, num_timesteps, (B,), device=device)
            noise = torch.randn_like(coords)
            coord_mask = content_mask.unsqueeze(-1)  # [B, L, 1]
            noise = noise * coord_mask  # zero noise on padding positions
            coords_noisy = scheduler.add_noise(coords, noise, t)

            # UNet predicts coord noise only
            pred_noise = unet(coords_noisy, t, context)

            # Loss 1: MSE on coordinate noise (content only)
            coord_loss = ((pred_noise - noise) ** 2 * coord_mask).sum() / coord_mask.sum().clamp(min=1)

            # Loss 2: Cross-entropy on cmd from encoder (all positions, downweighted)
            cmd_loss = F.cross_entropy(
                cmd_logits.reshape(-1, NUM_CMDS),
                cmd_indices.reshape(-1),
            )

            loss = coord_loss + 0.1 * cmd_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(unet.parameters()), max_norm=1.0
            )
            optimizer.step()

            # EMA update
            ema_update(ema_encoder, encoder)
            ema_update(ema_unet, unet)

            train_loss += loss.item() * B
            n_train += B

        train_loss /= n_train

        # --- Validate ---
        encoder.eval()
        unet.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                B = src.size(0)

                cmd_normalized = tgt[:, :, 0]
                coords = tgt[:, :, 1:]
                cmd_indices = ((cmd_normalized + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)
                content_mask = (cmd_indices > 0).float()

                context, cmd_logits = encoder(src)
                t = torch.randint(0, num_timesteps, (B,), device=device)
                noise = torch.randn_like(coords)
                coord_mask = content_mask.unsqueeze(-1)
                noise = noise * coord_mask
                coords_noisy = scheduler.add_noise(coords, noise, t)
                pred_noise = unet(coords_noisy, t, context)

                coord_loss = ((pred_noise - noise) ** 2 * coord_mask).sum() / coord_mask.sum().clamp(min=1)
                cmd_loss = F.cross_entropy(cmd_logits.reshape(-1, NUM_CMDS), cmd_indices.reshape(-1))

                val_loss += (coord_loss + 0.1 * cmd_loss).item() * B
                n_val += B

        val_loss /= n_val
        scheduler_lr.step()
        elapsed = time.time() - t0

        print(f"[{epoch:3d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": ema_encoder.state_dict(),
                "unet": ema_unet.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, save_path / "best.pt")

    torch.save({
        "encoder": ema_encoder.state_dict(),
        "unet": ema_unet.state_dict(),
        "epoch": epochs,
        "val_loss": val_loss,
    }, save_path / "last.pt")
    print(f"Done. Best val loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train Vector Path Diffusion Model")
    parser.add_argument("--font", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()

    train(
        font_path=args.font,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_chars=args.max_chars,
        num_timesteps=args.timesteps,
    )


if __name__ == "__main__":
    main()
