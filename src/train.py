"""2-phase latent diffusion training.

Phase 1: Encoder-Decoder pretraining (no diffusion)
Phase 2: Latent diffusion with frozen encoder
"""

import argparse, copy, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import GlyphEchoDataset, GlyphPairDataset, collate_glyph_batch
from .diffusion import LatentUNet1d, NoiseScheduler, ema_update
from .encoder import PathEncoder, PathDecoder
from .vectorizer import DEFAULT_MAX_LEN, NUM_CMDS


def train(
    font_path=None, max_len=DEFAULT_MAX_LEN, epochs=5000, batch_size=32,
    lr=1e-3, device=None, save_dir="checkpoints", num_timesteps=1000,
    latent_dim=256, model_dim=256, max_chars=None, mode="echo",
):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Mode: {mode}")

    if mode == "pair":
        dataset = GlyphPairDataset(font_path=font_path, max_len=max_len, max_pairs=max_chars)
    else:
        dataset = GlyphEchoDataset(font_path=font_path, max_len=max_len, max_chars=max_chars)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_glyph_batch)

    encoder = PathEncoder(latent_dim=latent_dim).to(device)
    decoder = PathDecoder(latent_dim=latent_dim).to(device)
    unet = LatentUNet1d(latent_dim=latent_dim, model_dim=model_dim).to(device)
    scheduler = NoiseScheduler(num_timesteps=num_timesteps)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    phase1_epochs = epochs * 2 // 5  # 40% for autoencoder
    phase2_epochs = epochs - phase1_epochs

    # ===== Phase 1: Encoder-Decoder pretraining =====
    print(f"\n=== Phase 1: Encoder-Decoder ({phase1_epochs} epochs) ===")
    opt1 = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=phase1_epochs)
    best_recon = float("inf")

    for epoch in range(1, phase1_epochs + 1):
        encoder.train(); decoder.train()
        train_loss = 0.0; n = 0
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            B = src.size(0)
            coords = tgt[:, :, 1:]
            cmd_idx = ((tgt[:, :, 0] + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)
            mask = (cmd_idx > 0).float().unsqueeze(-1)

            z, cmd_logits = encoder(src)
            pred_coords = decoder(z)

            recon = ((pred_coords - coords) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            cmd_loss = F.cross_entropy(cmd_logits.reshape(-1, NUM_CMDS), cmd_idx.reshape(-1))
            loss = recon + 0.1 * cmd_loss

            opt1.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt1.step()
            train_loss += loss.item() * B; n += B

        # Val
        encoder.eval(); decoder.eval()
        val_loss = 0.0; nv = 0
        with torch.no_grad():
            for src, tgt in dl:
                src, tgt = src.to(device), tgt.to(device)
                coords = tgt[:, :, 1:]
                cmd_idx = ((tgt[:, :, 0] + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)
                mask = (cmd_idx > 0).float().unsqueeze(-1)
                z, _ = encoder(src)
                pred = decoder(z)
                val_loss += ((pred - coords) ** 2 * mask).sum().item() / mask.sum().clamp(min=1).item()
                nv += 1
        val_loss /= nv
        sched1.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"[{epoch:5d}/{phase1_epochs}] recon train={train_loss/n:.6f} val={val_loss:.6f}")
        if val_loss < best_recon:
            best_recon = val_loss

    print(f"Phase 1 done. Best recon: {best_recon:.6f}")

    # ===== Phase 2: Latent Diffusion (encoder frozen) =====
    print(f"\n=== Phase 2: Latent Diffusion ({phase2_epochs} epochs) ===")
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    for p in decoder.parameters(): p.requires_grad_(False)

    # Compute latent stats for normalization (from both src and tgt)
    with torch.no_grad():
        all_z = []
        for src, tgt in dl:
            z_src, _ = encoder(src.to(device))
            z_tgt, _ = encoder(tgt.to(device))
            all_z.extend([z_src, z_tgt])
        all_z = torch.cat(all_z, 0)
        z_mean = all_z.mean()
        z_std = all_z.std()
        print(f"Latent stats: mean={z_mean:.4f}, std={z_std:.4f}")

    ema_unet = copy.deepcopy(unet)
    opt2 = torch.optim.AdamW(unet.parameters(), lr=lr * 0.3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs)
    best_diff = float("inf")

    for epoch in range(1, phase2_epochs + 1):
        unet.train()
        train_loss = 0.0; n = 0
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            B = src.size(0)

            with torch.no_grad():
                z_src, _ = encoder(src)  # condition
                z_tgt, _ = encoder(tgt)  # target to generate
                z_src_norm = (z_src - z_mean) / z_std
                z_tgt_norm = (z_tgt - z_mean) / z_std

            t = scheduler.sample_timesteps(B, device)
            noise = torch.randn_like(z_tgt_norm)
            z_noisy = scheduler.add_noise(z_tgt_norm, noise, t)

            # Condition on SRC latent, predict TGT latent
            z_pred = unet(z_noisy, t, z_src_norm)
            loss = F.mse_loss(z_pred, z_tgt_norm)

            opt2.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt2.step()
            ema_update(ema_unet, unet)
            train_loss += loss.item() * B; n += B

        # Val: full pipeline
        unet.eval()
        val_loss = 0.0; nv = 0
        with torch.no_grad():
            for src, tgt in dl:
                src, tgt = src.to(device), tgt.to(device)
                coords = tgt[:, :, 1:]
                cmd_idx = ((tgt[:, :, 0] + 0.5) * (NUM_CMDS - 1)).round().long().clamp(0, NUM_CMDS - 1)
                mask = (cmd_idx > 0).float().unsqueeze(-1)

                z_src, _ = encoder(src)
                z_src_norm = (z_src - z_mean) / z_std

                # DDIM: generate tgt latent conditioned on src latent
                z_gen = scheduler.ddim_sample(ema_unet, z_src_norm.shape, z_src_norm, num_steps=50, device=device)
                z_gen_denorm = z_gen * z_std + z_mean
                pred_coords = decoder(z_gen_denorm)

                mse = ((pred_coords - coords) ** 2 * mask).sum() / mask.sum().clamp(min=1)
                val_loss += mse.item(); nv += 1
        val_loss /= nv
        sched2.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"[{epoch:5d}/{phase2_epochs}] diffusion train={train_loss/n:.6f} val(coord)={val_loss:.6f}")
        if val_loss < best_diff:
            best_diff = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "unet": ema_unet.state_dict(),
                "z_mean": z_mean, "z_std": z_std,
                "epoch": epoch, "val_loss": val_loss,
                "latent_dim": latent_dim, "model_dim": model_dim,
            }, save_path / "best.pt")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "unet": ema_unet.state_dict(),
        "z_mean": z_mean, "z_std": z_std,
        "epoch": phase1_epochs + phase2_epochs, "val_loss": val_loss,
        "latent_dim": latent_dim, "model_dim": model_dim,
    }, save_path / "last.pt")
    print(f"\nDone. Best recon: {best_recon:.6f}, Best diffusion: {best_diff:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font", default=None)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--mode", choices=["echo", "pair"], default="echo")
    a = p.parse_args()
    train(font_path=a.font, max_len=a.max_len, epochs=a.epochs, batch_size=a.batch_size,
          lr=a.lr, device=a.device, max_chars=a.max_chars, num_timesteps=a.timesteps, mode=a.mode)

if __name__ == "__main__":
    main()
