"""Vector path renderer — converts path commands to PIL images.

Phase 1: Cubic bezier curves approximated as line segments via de Casteljau.
Phase 2+: Switch to cairo for proper curve rendering.
"""

from PIL import Image, ImageDraw

from .vectorizer import CMD_CLOSE, CMD_CURVE, CMD_LINE, CMD_MOVE, CMD_PAD


def _bezier_points(
    p0: tuple[float, float],
    c1: tuple[float, float],
    c2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 16,
) -> list[tuple[float, float]]:
    """Approximate cubic bezier with line segments via de Casteljau."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * c1[0] + 3 * u * t**2 * c2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * c1[1] + 3 * u * t**2 * c2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def render_paths(
    paths: list[tuple[str, list[tuple[float, float]]]],
    width: int = 256,
    height: int = 256,
    padding: int = 16,
    line_width: int = 2,
) -> Image.Image:
    """Render vector path commands to a grayscale PIL image.

    Coordinates are expected in font units — they are mapped
    to the image bounding box with padding.
    """
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    if not paths:
        return img

    # Collect all coordinates for bounding box
    all_pts = []
    for cmd, pts in paths:
        all_pts.extend(pts)
    if not all_pts:
        return img

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    data_w = max(x_max - x_min, 1)
    data_h = max(y_max - y_min, 1)

    # Map coordinates to image space (with padding, y-flipped)
    draw_w = width - 2 * padding
    draw_h = height - 2 * padding
    scale = min(draw_w / data_w, draw_h / data_h)
    off_x = padding + (draw_w - data_w * scale) / 2
    off_y = padding + (draw_h - data_h * scale) / 2

    def to_img(x: float, y: float) -> tuple[float, float]:
        ix = (x - x_min) * scale + off_x
        iy = height - ((y - y_min) * scale + off_y)  # flip Y
        return (ix, iy)

    # Draw
    current = (0.0, 0.0)
    contour_start = (0.0, 0.0)

    for cmd, pts in paths:
        if cmd == "moveTo":
            current = to_img(*pts[0])
            contour_start = current
        elif cmd == "lineTo":
            end = to_img(*pts[0])
            draw.line([current, end], fill=255, width=line_width)
            current = end
        elif cmd == "curveTo":
            c1 = to_img(*pts[0])
            c2 = to_img(*pts[1])
            end = to_img(*pts[2])
            bezier_pts = _bezier_points(current, c1, c2, end)
            for i in range(len(bezier_pts) - 1):
                draw.line([bezier_pts[i], bezier_pts[i + 1]], fill=255, width=line_width)
            current = end
        elif cmd == "closePath":
            if current != contour_start:
                draw.line([current, contour_start], fill=255, width=line_width)
            current = contour_start

    return img


def render_tensor(
    tensor,
    bounds: tuple[float, float, float, float] = (0, 0, 1000, 1000),
    width: int = 256,
    height: int = 256,
) -> Image.Image:
    """Render a path tensor directly to an image."""
    from .vectorizer import tensor_to_paths
    paths = tensor_to_paths(tensor, bounds=bounds)
    return render_paths(paths, width=width, height=height)
