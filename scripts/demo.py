"""End-to-end demo: text → render → model → OCR → text."""

import argparse
import sys
from pathlib import Path

import torch
import pytesseract

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VisionLLM
from src.renderer import image_to_tensor, render_text, tensor_to_image


def run_pipeline(text: str, checkpoint: str, device: str | None = None) -> str:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Render text to image
    input_img = render_text(text)
    input_tensor = image_to_tensor(input_img).unsqueeze(0).to(device)

    # 2. Run through model
    model = VisionLLM(latent_dim=256, think_depth=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 3. Convert back to image
    output_img = tensor_to_image(output_tensor.squeeze(0))

    # 4. OCR
    ocr_text = pytesseract.image_to_string(output_img, config="--psm 7").strip()

    # Save images for inspection
    input_img.save("demo_input.png")
    output_img.save("demo_output.png")

    return ocr_text


def main():
    parser = argparse.ArgumentParser(description="Vision LLM demo")
    parser.add_argument("text", help="Input text")
    parser.add_argument("--checkpoint", default="checkpoints/best_echo.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"Input:  '{args.text}'")
    result = run_pipeline(args.text, args.checkpoint, args.device)
    print(f"Output: '{result}'")
    print("Saved: demo_input.png, demo_output.png")


if __name__ == "__main__":
    main()
