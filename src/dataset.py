"""Dataset generator for Vision LLM training."""

import random
import string

import torch
from torch.utils.data import Dataset

from .renderer import image_to_tensor, render_text


# Word pool for generating random text
WORDS = [
    "hello", "world", "python", "torch", "vision", "model", "train",
    "learn", "data", "text", "image", "pixel", "font", "read", "write",
    "code", "deep", "neural", "layer", "batch", "loss", "adam", "grad",
    "test", "demo", "echo", "open", "save", "load", "run", "fast",
    "slow", "big", "small", "red", "blue", "green", "cat", "dog", "sun",
]


def random_text(min_chars: int = 3, max_chars: int = 16) -> str:
    """Generate a random word or short phrase."""
    mode = random.random()
    if mode < 0.4:
        # Single word from pool
        return random.choice(WORDS)
    elif mode < 0.7:
        # Two words
        return f"{random.choice(WORDS)} {random.choice(WORDS)}"
    else:
        # Random letters
        length = random.randint(min_chars, max_chars)
        return "".join(random.choices(string.ascii_lowercase, k=length))


class EchoDataset(Dataset):
    """Echo task: model must reproduce the input image exactly.

    This is the simplest possible task — identity mapping.
    Proves the encoder-decoder pipeline works before adding transformations.
    """

    def __init__(self, size: int = 10000, seed: int = 42):
        self.size = size
        self.rng = random.Random(seed)
        # Pre-generate texts for reproducibility
        self.texts = [self._random_text() for _ in range(size)]

    def _random_text(self) -> str:
        mode = self.rng.random()
        if mode < 0.4:
            return self.rng.choice(WORDS)
        elif mode < 0.7:
            return f"{self.rng.choice(WORDS)} {self.rng.choice(WORDS)}"
        else:
            length = self.rng.randint(3, 16)
            return "".join(self.rng.choices(string.ascii_lowercase, k=length))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        img = render_text(text)
        t = image_to_tensor(img)
        return t, t  # input = target for echo task


class UppercaseDataset(Dataset):
    """Uppercase task: input is lowercase text, target is UPPERCASE.

    First real 'thinking' task — model must learn the visual
    transformation from lowercase to uppercase letterforms.
    """

    def __init__(self, size: int = 10000, seed: int = 42):
        self.size = size
        self.rng = random.Random(seed)
        self.texts = [self._random_text() for _ in range(size)]

    def _random_text(self) -> str:
        length = self.rng.randint(3, 12)
        return "".join(self.rng.choices(string.ascii_lowercase, k=length))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        src = image_to_tensor(render_text(text))
        tgt = image_to_tensor(render_text(text.upper()))
        return src, tgt
