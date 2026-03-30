"""Latent diffusion training: diffuse in encoder's latent space, decode to coords."""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import GlyphEchoDataset, collate_glyph_batch
from .diffusion import LatentUNet1d, NoiseScheduler, ema_update
from .encoder import PathEncoder, PathDecoder
from .vectorizer import DEFAULT_MAX_LEN, NUM_CMDS


def train(
    font_path=None, max_len=DEFAULT_MAX_LEN, epochs=100, batch_size=32,
    lr=1e-3, device=None, save_dir="checkpoints", num_timesteps=1000,
    latent_dim=256, model_dim=256, max_chars=None,
):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = GlyphEchoDataset(font_path=font_path, max_len=max_len, max_chars=max_chars)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_glyph_batch)

    encoder = PathEncoder(latent_dim=latent_dim).to(device)
    decoder = PathDecoder(latent_dim=latent_dim).to(device)
    unet = LatentUNet1d(latent_dim=latent_dim, model_dim=model_dim).to(device)
    scheduler = NoiseScheduler(num_timesteps=num_timesteps)

    # EMA
    ema_enc = copy.deepcopy(encoder)
    ema_dec = copy.deepcopy(decoder)
    ema_unet = copy.deepcopy(unet)

    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(unet.parameters())
    total = sum(p.numel() for p in all_params)
    print(f"Params: enc={sum(p.numel() for p in encoder.parameters()):,} "
          f"dec={sum(p.numel() for p in decoder.parameters()):,} "
          f"unet={sum(p.numel() for p in unet.parameters()):,} total={total:,}")

    optimizer = torch.optim.AdamW(all_params, lr=lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Phase 1: Train encoder-decoder reconstruction (first 20% of epochs) ---
        # --- Phase 2: Train latent diffusion (remaining 80%) ---
        warmup = epoch <= max(1, epochs // 5)

        encoder.train(); decoder.train(); unet.train()
        train_loss = 0.0; n = 0

        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            B = src.size(0)
            cmd_norm = tgt[:, :, 0]
            coords = tgt[:, :, 1:]
            cmd_idx = ((cmd_norm + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)
            content_mask = (cmd_idx > 0).float().unsqueeze(-1)

            # Encode
            z, cmd_logits = encoder(src)  # z: [B, S, 256]

            if warmup:
                # Phase 1: just train encoder-decoder reconstruction
                pred_coords = decoder(z)
                recon_loss = ((pred_coords - coords) ** 2 * content_mask).sum() / content_mask.sum().clamp(min=1)
                cmd_loss = F.cross_entropy(cmd_logits.reshape(-1, NUM_CMDS), cmd_idx.reshape(-1))
                loss = recon_loss + 0.1 * cmd_loss
            else:
                # Phase 2: latent diffusion
                t = scheduler.sample_timesteps(B, device)
                noise = torch.randn_like(z)
                z_noisy = scheduler.add_noise(z.detach(), noise, t)  # detach encoder for stable diffusion

                z_pred = unet(z_noisy, t, z.detach())  # condition on clean latent

                # Latent MSE loss
                latent_loss = F.mse_loss(z_pred, z.detach())

                # Also train decoder on predicted latent
                pred_coords = decoder(z_pred)
                recon_loss = ((pred_coords - coords) ** 2 * content_mask).sum() / content_mask.sum().clamp(min=1)

                cmd_loss = F.cross_entropy(cmd_logits.reshape(-1, NUM_CMDS), cmd_idx.reshape(-1))
                loss = latent_loss + recon_loss + 0.1 * cmd_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            ema_update(ema_enc, encoder)
            ema_update(ema_dec, decoder)
            ema_update(ema_unet, unet)

            train_loss += loss.item() * B
            n += B

        train_loss /= n

        # --- Validate ---
        encoder.eval(); decoder.eval(); unet.eval()
        val_loss = 0.0; nv = 0
        with torch.no_grad():
            for src, tgt in dl:
                src, tgt = src.to(device), tgt.to(device)
                B = src.size(0)
                coords = tgt[:, :, 1:]
                cmd_idx = ((tgt[:, :, 0] + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)
                content_mask = (cmd_idx > 0).float().unsqueeze(-1)

                z, _ = encoder(src)
                if warmup:
                    pred_coords = decoder(z)
                else:
                    # Full pipeline: encode → add noise → denoise → decode
                    t = scheduler.sample_timesteps(B, device)
                    noise = torch.randn_like(z)
                    z_noisy = scheduler.add_noise(z, noise, t)
                    z_pred = unet(z_noisy, t, z)
                    pred_coords = decoder(z_pred)

                mse = ((pred_coords - coords) ** 2 * content_mask).sum() / content_mask.sum().clamp(min=1)
                val_loss += mse.item() * B
                nv += B
        val_loss /= nv
        lr_sched.step()

        elapsed = time.time() - t0
        phase = "warmup" if warmup else "diffusion"
        print(f"[{epoch:5d}/{epochs}] {phase} train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder": ema_enc.state_dict(),
                "decoder": ema_dec.state_dict(),
                "unet": ema_unet.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
            }, save_path / "best.pt")

    torch.save({
        "encoder": ema_enc.state_dict(),
        "decoder": ema_dec.state_dict(),
        "unet": ema_unet.state_dict(),
        "epoch": epochs, "val_loss": val_loss,
    }, save_path / "last.pt")
    print(f"Done. Best val loss: {best_val:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font", default=None)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--max-chars", type=int, default=None)
    a = p.parse_args()
    train(font_path=a.font, max_len=a.max_len, epochs=a.epochs, batch_size=a.batch_size,
          lr=a.lr, device=a.device, max_chars=a.max_chars, num_timesteps=a.timesteps)

if __name__ == "__main__":
    main()
