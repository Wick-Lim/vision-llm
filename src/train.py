"""Training loop for vector path diffusion model."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import GlyphEchoDataset
from .diffusion import NoiseScheduler, UNet1d
from .encoder import PathEncoder
from .vectorizer import DEFAULT_MAX_LEN


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

    # Dataset
    dataset = GlyphEchoDataset(font_path=font_path, max_len=max_len, max_chars=max_chars)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Models
    encoder = PathEncoder(cond_dim=cond_dim).to(device)
    unet = UNet1d(cond_dim=cond_dim, model_dim=model_dim).to(device)
    scheduler = NoiseScheduler(num_timesteps=num_timesteps)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in unet.parameters())
    print(f"Parameters: encoder={sum(p.numel() for p in encoder.parameters()):,} "
          f"unet={sum(p.numel() for p in unet.parameters()):,} total={total_params:,}")

    # Optimizer (joint for encoder + unet)
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

            # Encode condition
            cond = encoder(src)

            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (B,), device=device)

            # Forward diffusion on target
            noise = torch.randn_like(tgt)
            x_t = scheduler.add_noise(tgt, noise, t)

            # Predict noise
            pred_noise = unet(x_t, t, cond)

            # Weighted loss: 10x on content rows, 1x on padding
            content_mask = (tgt[:, :, 0] > -0.45).float()
            weight = (1.0 + 9.0 * content_mask).unsqueeze(-1)
            loss = (weight * (pred_noise - noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(unet.parameters()), max_norm=1.0
            )
            optimizer.step()

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

                cond = encoder(src)
                t = torch.randint(0, num_timesteps, (B,), device=device)
                noise = torch.randn_like(tgt)
                x_t = scheduler.add_noise(tgt, noise, t)
                pred_noise = unet(x_t, t, cond)

                content_mask = (tgt[:, :, 0] > -0.45).float()
                weight = (1.0 + 9.0 * content_mask).unsqueeze(-1)
                val_loss += (weight * (pred_noise - noise) ** 2).mean().item() * B
                n_val += B

        val_loss /= n_val
        scheduler_lr.step()
        elapsed = time.time() - t0

        print(f"[{epoch:3d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.1f}s)")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "unet": unet.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, save_path / "best.pt")

    torch.save({
        "encoder": encoder.state_dict(),
        "unet": unet.state_dict(),
        "epoch": epochs,
        "val_loss": val_loss,
    }, save_path / "last.pt")
    print(f"Done. Best val loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train Vector Path Diffusion Model")
    parser.add_argument("--font", type=str, default=None, help="Path to TTF/OTF font file")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=None, help="Limit number of glyphs (for fast iteration)")
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
