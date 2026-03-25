"""End-to-end demo: text → vector paths → diffusion → render → OCR."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diffusion import NoiseScheduler, UNet1d
from src.encoder import PathEncoder
from src.renderer import render_paths, render_tensor
from src.vectorizer import (
    DEFAULT_MAX_LEN,
    extract_glyph,
    extract_text,
    get_glyph_bounds,
    load_font,
    paths_to_tensor,
    tensor_to_paths,
)


def run_pipeline(
    text: str,
    checkpoint: str,
    font_path: str | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    ddim_steps: int = 50,
    device: str | None = None,
) -> None:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    font = load_font(font_path)

    # 1. Extract input vector paths
    print(f"[1] Extracting vector paths for '{text}'...")
    input_tensor = extract_text(font, text, max_len=max_len)

    # Save input rendering
    input_img = render_tensor(input_tensor)
    input_img.save("demo_input.png")
    print(f"    Input: {(input_tensor[:, 0] > 0).sum().item()} commands")

    # 2. Load models
    print(f"[2] Loading model from {checkpoint}...")
    encoder = PathEncoder(cond_dim=256).to(device)
    unet = UNet1d(cond_dim=256, model_dim=128).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    unet.load_state_dict(ckpt["unet"])
    encoder.eval()
    unet.eval()

    # 3. Encode condition
    print("[3] Encoding condition vector...")
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        cond = encoder(input_batch)

    # 4. Generate via diffusion
    print(f"[4] Generating via DDIM ({ddim_steps} steps)...")
    scheduler = NoiseScheduler(num_timesteps=1000)
    with torch.no_grad():
        output_tensor = scheduler.ddim_sample(
            unet,
            shape=(1, max_len, 8),
            cond=cond,
            num_steps=ddim_steps,
            device=device,
        )
    output_tensor = output_tensor.squeeze(0).cpu()

    # 5. Render output
    print("[5] Rendering output...")
    output_img = render_tensor(output_tensor)
    output_img.save("demo_output.png")

    # 6. OCR (optional)
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
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_pipeline(
        text=args.text,
        checkpoint=args.checkpoint,
        font_path=args.font,
        max_len=args.max_len,
        ddim_steps=args.ddim_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
