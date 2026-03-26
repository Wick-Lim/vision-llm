"""Font glyph vector path extraction and tensor conversion.

Extracts Bezier curves from font files using fontTools and converts
them to fixed-size tensors for neural network processing.
All quadratic curves (TrueType) are converted to cubic via fontTools.qu2cu.
"""

import os
from pathlib import Path

import torch
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.pointPen import SegmentToPointPen
from fontTools.ttLib import TTFont

# Command encoding
CMD_PAD = 0
CMD_MOVE = 1
CMD_LINE = 2
CMD_CURVE = 3  # cubic bezier (2 control points)
CMD_CLOSE = 4
NUM_CMDS = 5  # 0..4

# Tensor column layout: [cmd, x, y, cx1, cy1, cx2, cy2, flag]
TENSOR_DIM = 8
DEFAULT_MAX_LEN = 128  # reduced from 512 — most glyphs have <60 commands


def _normalize_cmd(cmd: int) -> float:
    """Map cmd integer [0..4] → [-0.5, 0.5] range."""
    return cmd / (NUM_CMDS - 1) - 0.5


def _denormalize_cmd(val: float) -> int:
    """Map normalized cmd back to integer, clamped to valid range."""
    cmd = round((val + 0.5) * (NUM_CMDS - 1))
    return max(0, min(NUM_CMDS - 1, cmd))


# Font search paths (macOS → Linux)
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(path: str | None = None) -> TTFont:
    """Load a TTFont.

    Priority: explicit path > VISION_LLM_FONT env var > FONT_PATHS search.
    Handles .ttc font collections by trying fontNumber=0.
    """
    def _try_load(p: str) -> TTFont:
        try:
            return TTFont(p)
        except Exception:
            return TTFont(p, fontNumber=0)

    if path:
        return _try_load(path)

    env_font = os.environ.get("VISION_LLM_FONT")
    if env_font and Path(env_font).exists():
        return _try_load(env_font)

    for p in FONT_PATHS:
        if Path(p).exists():
            return _try_load(p)

    raise FileNotFoundError("No suitable font found. Set VISION_LLM_FONT or install a Korean font.")


def extract_glyph(font: TTFont, char: str) -> list[tuple[str, list[tuple[float, float]]]]:
    """Extract vector paths for a single character.

    Returns list of (command, points) tuples with only cubic curves.
    TrueType quadratic curves are automatically converted to cubic
    by fontTools' recording pen.
    """
    cmap = font.getBestCmap()
    cp = ord(char)
    if cp not in cmap:
        raise ValueError(f"Character '{char}' (U+{cp:04X}) not in font cmap")

    glyph_name = cmap[cp]
    glyphset = font.getGlyphSet()
    glyph = glyphset[glyph_name]

    # Use RecordingPen — fontTools handles composite decomposition
    pen = RecordingPen()
    glyph.draw(pen)

    result: list[tuple[str, list[tuple[float, float]]]] = []
    current_point = (0.0, 0.0)

    for cmd, args in pen.value:
        if cmd == "moveTo":
            pt = (float(args[0][0]), float(args[0][1]))
            result.append(("moveTo", [pt]))
            current_point = pt
        elif cmd == "lineTo":
            pt = (float(args[0][0]), float(args[0][1]))
            result.append(("lineTo", [pt]))
            current_point = pt
        elif cmd == "curveTo":
            pts = [(float(p[0]), float(p[1])) for p in args]
            result.append(("curveTo", pts))
            current_point = pts[-1]
        elif cmd == "qCurveTo":
            # Convert each quadratic segment to cubic inline
            q_pts = [(float(p[0]), float(p[1])) for p in args]
            prev = current_point
            _convert_qcurve_to_cubic(prev, q_pts, result)
            current_point = q_pts[-1]
        elif cmd in ("closePath", "endPath"):
            result.append(("closePath", []))

    return result


def _convert_qcurve_to_cubic(
    start: tuple[float, float],
    q_pts: list[tuple[float, float]],
    out: list[tuple[str, list[tuple[float, float]]]],
) -> None:
    """Convert a qCurveTo command (possibly with implicit on-curves) to cubic curves.

    A qCurveTo with N off-curve points + 1 on-curve endpoint creates
    N-1 implicit on-curve midpoints. Each quadratic segment P0→C→P1
    becomes cubic P0→C1→C2→P1 where C1 = P0 + 2/3*(C-P0), C2 = P1 + 2/3*(C-P1).
    """
    if len(q_pts) < 2:
        # Degenerate: single point = line
        if q_pts:
            out.append(("lineTo", [q_pts[0]]))
        return

    on_curve_end = q_pts[-1]
    off_curves = q_pts[:-1]

    # Build list of (on, off, on) segments by inserting implicit midpoints
    segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    prev_on = start

    for i, off in enumerate(off_curves):
        if i < len(off_curves) - 1:
            # Implicit on-curve point between consecutive off-curves
            next_off = off_curves[i + 1]
            next_on = ((off[0] + next_off[0]) / 2, (off[1] + next_off[1]) / 2)
        else:
            next_on = on_curve_end
        segments.append((prev_on, off, next_on))
        prev_on = next_on

    # Convert each quadratic segment to cubic
    for p0, ctrl, p1 in segments:
        c1 = (p0[0] + 2/3 * (ctrl[0] - p0[0]), p0[1] + 2/3 * (ctrl[1] - p0[1]))
        c2 = (p1[0] + 2/3 * (ctrl[0] - p1[0]), p1[1] + 2/3 * (ctrl[1] - p1[1]))
        out.append(("curveTo", [c1, c2, p1]))


def get_glyph_bounds(font: TTFont, char: str) -> tuple[float, float, float, float]:
    """Get bounding box (xMin, yMin, xMax, yMax) for a glyph."""
    cmap = font.getBestCmap()
    glyph_name = cmap[ord(char)]
    glyphset = font.getGlyphSet()
    glyph = glyphset[glyph_name]

    from fontTools.pens.boundsPen import BoundsPen
    bp = BoundsPen(glyphset)
    glyph.draw(bp)
    bounds = bp.bounds
    if bounds is None:
        return (0.0, 0.0, 1.0, 1.0)
    return bounds


def get_advance_width(font: TTFont, char: str) -> float:
    """Get horizontal advance width for a character."""
    cmap = font.getBestCmap()
    glyph_name = cmap[ord(char)]
    return float(font["hmtx"].metrics[glyph_name][0])


def paths_to_tensor(
    paths: list[tuple[str, list[tuple[float, float]]]],
    max_len: int = DEFAULT_MAX_LEN,
    bounds: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """Convert path commands to a fixed-size tensor [max_len, 8].

    All values normalized to [-0.5, 0.5].
    """
    pad_val = _normalize_cmd(CMD_PAD)
    tensor = torch.full((max_len, TENSOR_DIM), pad_val)

    if bounds is None:
        all_coords = [pt for _, pts in paths for pt in pts]
        if not all_coords:
            return tensor
        xs = [p[0] for p in all_coords]
        ys = [p[1] for p in all_coords]
        bounds = (min(xs), min(ys), max(xs), max(ys))

    x_min, y_min, x_max, y_max = bounds
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)

    def norm(x: float, y: float) -> tuple[float, float]:
        return ((x - x_min) / w - 0.5, (y - y_min) / h - 0.5)

    idx = 0
    for cmd, pts in paths:
        if idx >= max_len:
            break
        if cmd == "moveTo":
            nx, ny = norm(*pts[0])
            tensor[idx] = torch.tensor([_normalize_cmd(CMD_MOVE), nx, ny, 0, 0, 0, 0, 0])
        elif cmd == "lineTo":
            nx, ny = norm(*pts[0])
            tensor[idx] = torch.tensor([_normalize_cmd(CMD_LINE), nx, ny, 0, 0, 0, 0, 0])
        elif cmd == "curveTo":
            nx, ny = norm(*pts[2])
            cx1, cy1 = norm(*pts[0])
            cx2, cy2 = norm(*pts[1])
            tensor[idx] = torch.tensor([_normalize_cmd(CMD_CURVE), nx, ny, cx1, cy1, cx2, cy2, 0])
        elif cmd == "closePath":
            tensor[idx] = torch.tensor([_normalize_cmd(CMD_CLOSE), 0, 0, 0, 0, 0, 0, 0])
        idx += 1

    return tensor


def tensor_to_paths(
    tensor: torch.Tensor,
    bounds: tuple[float, float, float, float] = (0, 0, 1000, 1000),
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Convert tensor back to path commands."""
    x_min, y_min, x_max, y_max = bounds
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)

    def denorm(nx: float, ny: float) -> tuple[float, float]:
        return ((nx + 0.5) * w + x_min, (ny + 0.5) * h + y_min)

    paths = []
    for row in tensor:
        cmd = _denormalize_cmd(row[0].item())
        if cmd == CMD_PAD:
            continue
        elif cmd == CMD_MOVE:
            paths.append(("moveTo", [denorm(row[1].item(), row[2].item())]))
        elif cmd == CMD_LINE:
            paths.append(("lineTo", [denorm(row[1].item(), row[2].item())]))
        elif cmd == CMD_CURVE:
            end = denorm(row[1].item(), row[2].item())
            c1 = denorm(row[3].item(), row[4].item())
            c2 = denorm(row[5].item(), row[6].item())
            paths.append(("curveTo", [c1, c2, end]))
        elif cmd == CMD_CLOSE:
            paths.append(("closePath", []))

    return paths


def extract_text(
    font: TTFont,
    text: str,
    max_len: int = DEFAULT_MAX_LEN,
) -> torch.Tensor:
    """Extract vector paths for a text string and return as tensor."""
    upm = font["head"].unitsPerEm
    all_paths: list[tuple[str, list[tuple[float, float]]]] = []
    x_offset = 0.0

    for char in text:
        if char == " ":
            x_offset += upm * 0.3
            continue
        try:
            paths = extract_glyph(font, char)
        except ValueError:
            x_offset += upm * 0.5
            continue
        for cmd, pts in paths:
            offset_pts = [(p[0] + x_offset, p[1]) for p in pts]
            all_paths.append((cmd, offset_pts))
        x_offset += get_advance_width(font, char)

    return paths_to_tensor(all_paths, max_len=max_len)
