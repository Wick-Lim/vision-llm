"""Latent diffusion demo with z normalization."""

import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diffusion import LatentUNet1d, NoiseScheduler
from src.encoder import PathEncoder, PathDecoder
from src.renderer import render_tensor
from src.vectorizer import DEFAULT_MAX_LEN, _normalize_cmd, extract_text, load_font


def run_pipeline(text, checkpoint, font_path=None, max_len=DEFAULT_MAX_LEN,
                 ddim_steps=200, device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    font = load_font(font_path)
    print(f"[1] Extracting '{text}'...")
    input_tensor = extract_text(font, text, max_len=max_len)
    n = (input_tensor[:, 0] > -0.4).sum().item()
    render_tensor(input_tensor).save("demo_input.png")
    print(f"    {n} commands")

    print(f"[2] Loading from {checkpoint}...")
    enc = PathEncoder(latent_dim=256).to(device)
    dec = PathDecoder(latent_dim=256).to(device)
    unet = LatentUNet1d(latent_dim=256, model_dim=128).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    unet.load_state_dict(ckpt["unet"])
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)
    enc.eval(); dec.eval(); unet.eval()

    print("[3] Encoding...")
    inp = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        z, _ = enc(inp)
        z_norm = (z - z_mean) / z_std

    print(f"[4] Latent diffusion ({ddim_steps} steps)...")
    sched = NoiseScheduler(1000)
    with torch.no_grad():
        z_gen = sched.ddim_sample(unet, z_norm.shape, z_norm, num_steps=ddim_steps, device=device)
        z_gen_denorm = z_gen * z_std + z_mean

    print("[5] Decoding + rendering...")
    with torch.no_grad():
        coords = dec(z_gen_denorm).squeeze(0).cpu()

    true_cmds = ((input_tensor[:, 0] + 0.5) * 4).round().long().clamp(0, 4)
    cmd_norm = torch.tensor([_normalize_cmd(c.item()) for c in true_cmds])
    output_tensor = torch.cat([cmd_norm.unsqueeze(-1), coords], dim=-1)
    render_tensor(output_tensor).save("demo_output.png")
    print(f"    Generated: {(true_cmds > 0).sum().item()} commands")

    # Also render pure encoder-decoder (no diffusion) for comparison
    with torch.no_grad():
        coords_recon = dec(z).squeeze(0).cpu()
    recon_tensor = torch.cat([cmd_norm.unsqueeze(-1), coords_recon], dim=-1)
    render_tensor(recon_tensor).save("demo_recon.png")
    print("    Also saved: demo_recon.png (encoder-decoder only, no diffusion)")

    print(f"\nSaved: demo_input.png, demo_output.png, demo_recon.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--font", default=None)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--ddim-steps", type=int, default=200)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run_pipeline(a.text, a.checkpoint, a.font, a.max_len, a.ddim_steps, a.device)

if __name__ == "__main__":
    main()
