"""Font glyph vector path extraction and tensor conversion.

Extracts Bezier curves, lines, and contours from font files
and converts them to fixed-size tensors for neural network processing.
"""

import math
from pathlib import Path

import numpy as np
import torch
from fontTools.pens.recordingPen import RecordingPen
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
DEFAULT_MAX_LEN = 512


def _normalize_cmd(cmd: int) -> float:
    """Map cmd integer [0..4] → [-0.5, 0.5] range."""
    return cmd / (NUM_CMDS - 1) - 0.5  # 0→-0.5, 1→-0.25, 2→0.0, 3→0.25, 4→0.5


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
    """Load a TTFont, searching default paths if none given.

    Handles .ttc font collections by trying fontNumber=0.
    """
    def _try_load(p: str) -> TTFont:
        try:
            return TTFont(p)
        except Exception:
            # TTC font collection — try first font
            return TTFont(p, fontNumber=0)

    if path:
        return _try_load(path)
    for p in FONT_PATHS:
        if Path(p).exists():
            return _try_load(p)
    raise FileNotFoundError("No suitable font found. Install a Korean font or specify --font.")


def _cubic_from_quadratic(qpoints: list[tuple[float, float]]) -> list[tuple[str, list[tuple[float, float]]]]:
    """Convert quadratic bezier segments to cubic.

    TrueType uses quadratic curves (1 control point).
    We convert to cubic (2 control points) for uniform tensor format.
    A quadratic P0→C→P1 becomes cubic P0→C1→C2→P1 where:
        C1 = P0 + 2/3 * (C - P0)
        C2 = P1 + 2/3 * (C - P0)
    """
    if len(qpoints) < 2:
        return []

    results = []
    # qCurveTo can have multiple off-curve points with implicit on-curve between them
    on_curve = qpoints[-1]
    off_curves = qpoints[:-1]

    if len(off_curves) == 0:
        # Degenerate: just a line
        results.append(("lineTo", [on_curve]))
        return results

    # Generate implicit on-curve points between consecutive off-curve points
    points = []
    for i, oc in enumerate(off_curves):
        points.append(("off", oc))
        if i < len(off_curves) - 1:
            # Implicit on-curve point between two off-curves
            next_oc = off_curves[i + 1]
            mid = ((oc[0] + next_oc[0]) / 2, (oc[1] + next_oc[1]) / 2)
            points.append(("on", mid))
    points.append(("on", on_curve))

    return points


def extract_glyph(font: TTFont, char: str) -> list[tuple[str, list[tuple[float, float]]]]:
    """Extract vector paths for a single character.

    Returns list of (command, points) tuples:
        ("moveTo", [(x, y)])
        ("lineTo", [(x, y)])
        ("curveTo", [(cx1, cy1), (cx2, cy2), (x, y)])
        ("closePath", [])
    All quadratic curves are converted to cubic.
    """
    cmap = font.getBestCmap()
    cp = ord(char)
    if cp not in cmap:
        raise ValueError(f"Character '{char}' (U+{cp:04X}) not in font cmap")

    glyph_name = cmap[cp]
    glyphset = font.getGlyphSet()
    glyph = glyphset[glyph_name]

    pen = RecordingPen()
    glyph.draw(pen)

    # Convert all quadratic curves to cubic
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
            # Already cubic
            pts = [(float(p[0]), float(p[1])) for p in args]
            result.append(("curveTo", pts))
            current_point = pts[-1]
        elif cmd == "qCurveTo":
            # Convert quadratic to cubic segments
            q_pts = [(float(p[0]), float(p[1])) for p in args]
            expanded = _cubic_from_quadratic(q_pts)

            # Now walk through expanded points producing cubic curves
            prev = current_point
            i = 0
            while i < len(expanded):
                kind, pt = expanded[i]
                if kind == "on":
                    # Line to this on-curve point
                    result.append(("lineTo", [pt]))
                    prev = pt
                    i += 1
                elif kind == "off":
                    # Off-curve: expect next is on-curve
                    ctrl = pt
                    if i + 1 < len(expanded):
                        _, end = expanded[i + 1]
                    else:
                        end = q_pts[-1]
                    # Quadratic→cubic: C1 = P0 + 2/3*(C-P0), C2 = P1 + 2/3*(C-P1)
                    c1 = (prev[0] + 2/3 * (ctrl[0] - prev[0]),
                           prev[1] + 2/3 * (ctrl[1] - prev[1]))
                    c2 = (end[0] + 2/3 * (ctrl[0] - end[0]),
                           end[1] + 2/3 * (ctrl[1] - end[1]))
                    result.append(("curveTo", [c1, c2, end]))
                    prev = end
                    i += 2
            current_point = prev
        elif cmd == "closePath":
            result.append(("closePath", []))
        elif cmd == "endPath":
            result.append(("closePath", []))

    return result


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

    Coordinates are normalized to [-0.5, 0.5] relative to bounding box.
    cmd values are also normalized to [-0.5, 0.5] so all channels share the same scale.
    Padding rows use _normalize_cmd(CMD_PAD) = -0.5 for all values.
    """
    # Fill with padding value (-0.5) instead of zeros
    pad_val = _normalize_cmd(CMD_PAD)
    tensor = torch.full((max_len, TENSOR_DIM), pad_val)

    # Compute normalization from bounds
    if bounds is None:
        # Compute bounds from paths
        all_coords = []
        for cmd, pts in paths:
            for pt in pts:
                all_coords.append(pt)
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
            nx, ny = norm(*pts[2])  # endpoint
            cx1, cy1 = norm(*pts[0])  # control 1
            cx2, cy2 = norm(*pts[1])  # control 2
            tensor[idx] = torch.tensor([_normalize_cmd(CMD_CURVE), nx, ny, cx1, cy1, cx2, cy2, 0])
        elif cmd == "closePath":
            tensor[idx] = torch.tensor([_normalize_cmd(CMD_CLOSE), 0, 0, 0, 0, 0, 0, 0])
        idx += 1

    return tensor


def tensor_to_paths(
    tensor: torch.Tensor,
    bounds: tuple[float, float, float, float] = (0, 0, 1000, 1000),
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Convert tensor back to path commands (inverse of paths_to_tensor)."""
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
            pt = denorm(row[1].item(), row[2].item())
            paths.append(("moveTo", [pt]))
        elif cmd == CMD_LINE:
            pt = denorm(row[1].item(), row[2].item())
            paths.append(("lineTo", [pt]))
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
    """Extract vector paths for a text string and return as tensor.

    Characters are laid out horizontally using advance widths.
    """
    upm = font["head"].unitsPerEm
    all_paths: list[tuple[str, list[tuple[float, float]]]] = []
    x_offset = 0.0

    for char in text:
        if char == " ":
            x_offset += upm * 0.3  # rough space width
            continue

        try:
            paths = extract_glyph(font, char)
        except ValueError:
            x_offset += upm * 0.5
            continue

        # Offset all coordinates by x_offset
        for cmd, pts in paths:
            offset_pts = [(p[0] + x_offset, p[1]) for p in pts]
            all_paths.append((cmd, offset_pts))

        x_offset += get_advance_width(font, char)

    return paths_to_tensor(all_paths, max_len=max_len)
