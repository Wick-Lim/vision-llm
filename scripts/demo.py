"""Latent diffusion demo: encode → diffuse in latent → decode → render."""

import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diffusion import LatentUNet1d, NoiseScheduler
from src.encoder import PathEncoder, PathDecoder
from src.renderer import render_tensor
from src.vectorizer import DEFAULT_MAX_LEN, NUM_CMDS, _normalize_cmd, extract_text, load_font


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
    encoder = PathEncoder(latent_dim=256).to(device)
    decoder = PathDecoder(latent_dim=256).to(device)
    unet = LatentUNet1d(latent_dim=256, model_dim=128).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    unet.load_state_dict(ckpt["unet"])
    encoder.eval(); decoder.eval(); unet.eval()

    print("[3] Encoding...")
    inp = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        z_cond, _ = encoder(inp)

    print(f"[4] Latent diffusion ({ddim_steps} steps)...")
    sched = NoiseScheduler(1000)
    with torch.no_grad():
        z_gen = sched.ddim_sample(unet, z_cond.shape, z_cond, num_steps=ddim_steps, device=device)

    print("[5] Decoding + rendering...")
    with torch.no_grad():
        coords = decoder(z_gen).squeeze(0).cpu()

    # Use true cmds from input
    true_cmds = ((input_tensor[:, 0] + 0.5) * 4).round().long().clamp(0, 4)
    cmd_norm = torch.tensor([_normalize_cmd(c.item()) for c in true_cmds])
    output_tensor = torch.cat([cmd_norm.unsqueeze(-1), coords], dim=-1)

    render_tensor(output_tensor).save("demo_output.png")
    n_gen = (true_cmds > 0).sum().item()
    print(f"    Generated: {n_gen} commands")

    try:
        import pytesseract
        img = render_tensor(output_tensor)
        ocr = pytesseract.image_to_string(img, lang="kor+eng", config="--psm 7").strip()
        print(f"[6] OCR: '{ocr}'")
    except Exception as e:
        print(f"[6] OCR skipped: {e}")

    print(f"\nSaved: demo_input.png, demo_output.png")


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
