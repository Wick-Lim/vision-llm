"""Dataset for vector path diffusion training.

Phase 1: Echo task — reproduce the same glyph's vector paths.
Uses Korean Hangul syllables (가~힣) + ASCII printable characters.
"""

import torch
import torch.nn.functional as F
from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from .vectorizer import (
    DEFAULT_MAX_LEN,
    _normalize_cmd,
    CMD_PAD,
    TENSOR_DIM,
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

# Minimum tensor length (must be divisible by 8 for U-Net downsampling)
MIN_LEN = 64


def _chars_in_font(font: TTFont, chars: list[str]) -> list[str]:
    """Filter characters that exist in the font's cmap."""
    cmap = font.getBestCmap()
    return [c for c in chars if ord(c) in cmap]


def _trim_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Trim tensor to actual content length + margin, aligned to 8."""
    pad_val = _normalize_cmd(CMD_PAD)
    content_mask = tensor[:, 0] > pad_val + 0.1
    if not content_mask.any():
        return tensor[:MIN_LEN]
    last_content = content_mask.nonzero()[-1].item()
    # Round up to next multiple of 8, with minimum MIN_LEN
    trim_len = max(MIN_LEN, ((last_content + 8) // 8) * 8)
    return tensor[:trim_len]


def collate_glyph_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Custom collate: pad tensors to max length in batch."""
    srcs, tgts = zip(*batch)
    max_len = max(s.shape[0] for s in srcs)
    # Round up to multiple of 8
    max_len = ((max_len + 7) // 8) * 8

    pad_val = _normalize_cmd(CMD_PAD)

    padded_srcs = []
    padded_tgts = []
    for s, t in zip(srcs, tgts):
        pad_rows = max_len - s.shape[0]
        if pad_rows > 0:
            pad = torch.full((pad_rows, TENSOR_DIM), pad_val)
            s = torch.cat([s, pad], dim=0)
            t = torch.cat([t, pad], dim=0)
        padded_srcs.append(s)
        padded_tgts.append(t)

    return torch.stack(padded_srcs), torch.stack(padded_tgts)


class GlyphEchoDataset(Dataset):
    """Echo task: input = target = same glyph's vector path tensor.

    Uses adaptive trimming to reduce padding ratio.
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

        chars = []
        if hangul:
            chars.extend(chr(cp) for cp in range(HANGUL_START, HANGUL_END + 1))
        if ascii_chars:
            chars.extend(chr(cp) for cp in range(ASCII_START, ASCII_END + 1))

        if max_chars is not None and len(chars) > max_chars:
            import random
            random.Random(42).shuffle(chars)
            chars = chars[:max_chars]

        self.chars = _chars_in_font(self.font, chars)
        print(f"GlyphEchoDataset: {len(self.chars)} glyphs available")

        # Pre-extract and trim tensors
        self.tensors: list[torch.Tensor] = []
        skipped = 0
        for char in self.chars:
            try:
                paths = extract_glyph(self.font, char)
                bounds = get_glyph_bounds(self.font, char)
                tensor = paths_to_tensor(paths, max_len=max_len, bounds=bounds)
                has_content = (tensor[:, 0] > -0.4).any()
                if has_content:
                    self.tensors.append(_trim_tensor(tensor))
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                print(f"  Warning: failed to extract '{char}': {e}")

        if skipped:
            print(f"  Skipped {skipped} glyphs")
        avg_len = sum(t.shape[0] for t in self.tensors) / max(len(self.tensors), 1)
        print(f"  Final: {len(self.tensors)} glyphs, avg trimmed len={avg_len:.0f}")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tensors[idx]
        return t, t
