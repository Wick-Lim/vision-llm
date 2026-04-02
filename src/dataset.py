"""Datasets for vector path diffusion training.

GlyphEchoDataset: input = target (Phase 1 validation)
GlyphPairDataset: input ≠ target (Phase 2 generation)
"""

import torch
import torch.nn.functional as F
from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from .vectorizer import (
    DEFAULT_MAX_LEN, _normalize_cmd, CMD_PAD, TENSOR_DIM,
    extract_glyph, get_glyph_bounds, load_font, paths_to_tensor,
    extract_text,
)

HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3
MIN_LEN = 64


def _chars_in_font(font, chars):
    cmap = font.getBestCmap()
    return [c for c in chars if ord(c) in cmap]


def _trim_tensor(tensor):
    pad_val = _normalize_cmd(CMD_PAD)
    mask = tensor[:, 0] > pad_val + 0.1
    if not mask.any():
        return tensor[:MIN_LEN]
    last = mask.nonzero()[-1].item()
    trim_len = max(MIN_LEN, ((last + 8) // 8) * 8)
    return tensor[:trim_len]


def _extract_tensor(font, char, max_len):
    paths = extract_glyph(font, char)
    bounds = get_glyph_bounds(font, char)
    tensor = paths_to_tensor(paths, max_len=max_len, bounds=bounds)
    if not (tensor[:, 0] > -0.4).any():
        return None
    return _trim_tensor(tensor)


def collate_glyph_batch(batch):
    srcs, tgts = zip(*batch)
    max_src = max(s.shape[0] for s in srcs)
    max_tgt = max(t.shape[0] for t in tgts)
    max_len = max(max_src, max_tgt)
    max_len = ((max_len + 7) // 8) * 8
    pad_val = _normalize_cmd(CMD_PAD)

    def pad_to(t, length):
        if t.shape[0] < length:
            return torch.cat([t, torch.full((length - t.shape[0], TENSOR_DIM), pad_val)])
        return t

    return (
        torch.stack([pad_to(s, max_len) for s in srcs]),
        torch.stack([pad_to(t, max_len) for t in tgts]),
    )


class GlyphEchoDataset(Dataset):
    """Echo: input = target."""

    def __init__(self, font_path=None, max_len=DEFAULT_MAX_LEN, max_chars=None, **kw):
        self.font = load_font(font_path)
        chars = [chr(cp) for cp in range(HANGUL_START, HANGUL_END + 1)]
        if max_chars and len(chars) > max_chars:
            import random; random.Random(42).shuffle(chars); chars = chars[:max_chars]
        self.chars = _chars_in_font(self.font, chars)
        self.tensors = []
        for c in self.chars:
            try:
                t = _extract_tensor(self.font, c, max_len)
                if t is not None: self.tensors.append(t)
            except: pass
        print(f"GlyphEchoDataset: {len(self.tensors)} glyphs")

    def __len__(self): return len(self.tensors)
    def __getitem__(self, idx): return self.tensors[idx], self.tensors[idx]


class GlyphPairDataset(Dataset):
    """Glyph pair: input → different target. For generation training."""

    def __init__(self, font_path=None, max_len=DEFAULT_MAX_LEN, max_pairs=None):
        self.font = load_font(font_path)
        # Create sequential pairs: 가→나, 나→다, 다→라, ...
        chars = _chars_in_font(self.font, [chr(cp) for cp in range(HANGUL_START, HANGUL_START + 200)])

        self.pairs = []  # (src_tensor, tgt_tensor)
        skipped = 0
        for i in range(len(chars) - 1):
            try:
                src = _extract_tensor(self.font, chars[i], max_len)
                tgt = _extract_tensor(self.font, chars[i + 1], max_len)
                if src is not None and tgt is not None:
                    self.pairs.append((src, tgt))
            except:
                skipped += 1

        if max_pairs and len(self.pairs) > max_pairs:
            self.pairs = self.pairs[:max_pairs]

        print(f"GlyphPairDataset: {len(self.pairs)} pairs (skipped {skipped})")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


class TextPairDataset(Dataset):
    """Text Q&A pairs rendered as vector paths."""

    def __init__(self, pairs: list[tuple[str, str]], font_path=None, max_len=DEFAULT_MAX_LEN):
        self.font = load_font(font_path)
        self.data = []
        for q, a in pairs:
            try:
                src = _trim_tensor(extract_text(self.font, q, max_len=max_len))
                tgt = _trim_tensor(extract_text(self.font, a, max_len=max_len))
                if (src[:, 0] > -0.4).any() and (tgt[:, 0] > -0.4).any():
                    self.data.append((src, tgt))
            except:
                pass
        print(f"TextPairDataset: {len(self.data)} pairs from {len(pairs)} input")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
