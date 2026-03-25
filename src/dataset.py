"""Dataset for vector path diffusion training.

Phase 1: Echo task — reproduce the same glyph's vector paths.
Uses Korean Hangul syllables (가~힣) + ASCII printable characters.
"""

import torch
from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from .vectorizer import (
    DEFAULT_MAX_LEN,
    extract_glyph,
    get_glyph_bounds,
    load_font,
    paths_to_tensor,
)

# Hangul syllable range
HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3  # 11,172 syllables

# ASCII printable range
ASCII_START = 0x21
ASCII_END = 0x7E  # 94 characters


def _chars_in_font(font: TTFont, chars: list[str]) -> list[str]:
    """Filter characters that exist in the font's cmap."""
    cmap = font.getBestCmap()
    return [c for c in chars if ord(c) in cmap]


class GlyphEchoDataset(Dataset):
    """Echo task: input = target = same glyph's vector path tensor.

    Validates that the diffusion model can learn to generate
    vector paths from conditioned noise.
    """

    def __init__(
        self,
        font_path: str | None = None,
        max_len: int = DEFAULT_MAX_LEN,
        hangul: bool = True,
        ascii_chars: bool = True,
        max_chars: int | None = None,
    ):
        self.font = load_font(font_path)
        self.max_len = max_len

        # Collect available characters
        chars = []
        if hangul:
            chars.extend(chr(cp) for cp in range(HANGUL_START, HANGUL_END + 1))
        if ascii_chars:
            chars.extend(chr(cp) for cp in range(ASCII_START, ASCII_END + 1))

        # Limit dataset size for fast iteration
        if max_chars is not None and len(chars) > max_chars:
            import random
            random.Random(42).shuffle(chars)
            chars = chars[:max_chars]

        self.chars = _chars_in_font(self.font, chars)
        print(f"GlyphEchoDataset: {len(self.chars)} glyphs available")

        # Pre-extract all tensors for speed
        self.tensors: list[torch.Tensor] = []
        skipped = 0
        for char in self.chars:
            try:
                paths = extract_glyph(self.font, char)
                bounds = get_glyph_bounds(self.font, char)
                tensor = paths_to_tensor(paths, max_len=max_len, bounds=bounds)
                # Skip empty glyphs
                if tensor.abs().sum() > 0:
                    self.tensors.append(tensor)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        if skipped:
            print(f"  Skipped {skipped} glyphs (empty or error)")
        print(f"  Final dataset size: {len(self.tensors)}")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tensors[idx]
        return t, t  # echo: input = target
