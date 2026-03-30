"""End-to-end demo: text → vector paths → diffusion → render → OCR."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diffusion import COORD_DIM, NoiseScheduler, UNet1d
from src.encoder import PathEncoder
from src.renderer import render_paths, render_tensor
from src.vectorizer import (
    DEFAULT_MAX_LEN,
    NUM_CMDS,
    _normalize_cmd,
    extract_text,
    load_font,
    tensor_to_paths,
)


def run_pipeline(
    text: str,
    checkpoint: str,
    font_path: str | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    ddim_steps: int = 200,
    guidance_scale: float = 3.0,
    device: str | None = None,
) -> None:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    font = load_font(font_path)

    # 1. Extract input vector paths
    print(f"[1] Extracting vector paths for '{text}'...")
    input_tensor = extract_text(font, text, max_len=max_len)
    n_cmds = (input_tensor[:, 0] > -0.4).sum().item()

    input_img = render_tensor(input_tensor)
    input_img.save("demo_input.png")
    print(f"    Input: {n_cmds} commands")

    # 2. Load models
    print(f"[2] Loading model from {checkpoint}...")
    encoder = PathEncoder(feat_dim=256).to(device)
    unet = UNet1d(context_dim=256, model_dim=128).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    unet.load_state_dict(ckpt["unet"])
    encoder.eval()
    unet.eval()

    # 3. Encode condition; use input cmds directly (echo task)
    print("[3] Encoding condition...")
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        context, _ = encoder(input_batch)

    # For echo task: use input cmd values directly
    true_cmds = ((input_tensor[:, 0] + 0.5) * 4).round().long().clamp(0, 4)
    cmd_indices = true_cmds

    # 4. Generate coordinates via diffusion
    print(f"[4] Generating coords via DDIM ({ddim_steps} steps, guidance={guidance_scale})...")
    scheduler = NoiseScheduler(num_timesteps=1000)
    with torch.no_grad():
        coords = scheduler.ddim_sample(
            unet,
            shape_coords=(1, max_len, COORD_DIM),
            context=context,
            num_steps=ddim_steps,
            device=device,
            guidance_scale=guidance_scale,
        )

    # 5. Reconstruct full tensor: cmd (from encoder) + coords (from diffusion)
    coords = coords.squeeze(0).cpu()  # [L, 7]
    cmd_normalized = torch.tensor([_normalize_cmd(c.item()) for c in cmd_indices.cpu()])
    output_tensor = torch.cat([cmd_normalized.unsqueeze(-1), coords], dim=-1)  # [L, 8]

    # 6. Render output
    print("[5] Rendering output...")
    output_img = render_tensor(output_tensor)
    output_img.save("demo_output.png")

    # Count generated commands
    n_generated = (cmd_indices > 0).sum().item()
    print(f"    Generated: {n_generated} commands (pad: {max_len - n_generated})")

    # 7. OCR (optional)
    try:
        import pytesseract
        ocr_text = pytesseract.image_to_string(output_img, lang="kor+eng", config="--psm 7").strip()
        print(f"[6] OCR result: '{ocr_text}'")
    except Exception as e:
        print(f"[6] OCR skipped: {e}")

    print(f"\nSaved: demo_input.png, demo_output.png")


def main():
    parser = argparse.ArgumentParser(description="Vision LLM Demo")
    parser.add_argument("text", help="Input text")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--font", default=None)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_pipeline(
        text=args.text,
        checkpoint=args.checkpoint,
        font_path=args.font,
        max_len=args.max_len,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
