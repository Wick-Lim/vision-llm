"""Text → grayscale image renderer."""

import torch
from PIL import Image, ImageDraw, ImageFont

# Default render size: 256 wide × 64 tall, enough for ~20 chars
IMG_W = 256
IMG_H = 64


def _get_font(size: int = 28) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace font, falling back to default if unavailable."""
    for path in [
        "/System/Library/Fonts/Menlo.ttc",          # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",  # Arch
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _get_font()


def render_text(text: str, width: int = IMG_W, height: int = IMG_H) -> Image.Image:
    """Render text string to a grayscale PIL Image (white text on black)."""
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=FONT)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=255, font=FONT)
    return img


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to float tensor [1, H, W] in [0, 1]."""
    import torchvision.transforms.functional as F
    return F.to_tensor(img)  # already scales 0-255 → 0-1


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Convert [1, H, W] or [H, W] tensor back to PIL Image."""
    if t.dim() == 3:
        t = t.squeeze(0)
    arr = (t.clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="L")
