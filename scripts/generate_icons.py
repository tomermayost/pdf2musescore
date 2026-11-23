"""Generate reproducible multi-resolution app icons for PDF to MuseScore."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "src" / "pdf2musescore" / "assets"
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def _lerp_channel(start: int, end: int, t: float) -> int:
    return int(round(start + (end - start) * t))


def render_master_icon(size: int = 256) -> Image.Image:
    """Render a crisp supersampled 256x256 cobalt-indigo badge with a sheet-music & eighth-note mark."""
    scale = 4
    w = size * scale  # 1024 px canvas
    canvas = Image.new("RGBA", (w, w), (0, 0, 0, 0))

    # Build vertical gradient background (#2563EB -> #4338CA)
    grad = Image.new("RGBA", (w, w))
    px = grad.load()
    c_top = (37, 99, 235)
    c_bot = (67, 56, 202)
    for y in range(w):
        t = y / max(1, w - 1)
        r = _lerp_channel(c_top[0], c_bot[0], t)
        g = _lerp_channel(c_top[1], c_bot[1], t)
        b = _lerp_channel(c_top[2], c_bot[2], t)
        for x in range(w):
            px[x, y] = (r, g, b, 255)

    # Rounded square badge mask with comfortable outer breathing room
    pad = int(w * 0.06)
    radius = int(w * 0.21)
    mask = Image.new("L", (w, w), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((pad, pad, w - pad, w - pad), radius=radius, fill=255)
    canvas.paste(grad, (0, 0), mask)

    # Composite subtle inner highlight & shadow properly so edge alpha is preserved
    overlay = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        (pad + 6, pad + 6, w - pad - 6, w - pad - 6),
        radius=max(10, radius - 6),
        outline=(255, 255, 255, 64),
        width=6,
    )

    # Stylized ivory sheet-music page silhouette with folded top-right corner
    left = int(w * 0.23)
    top = int(w * 0.16)
    right = int(w * 0.77)
    bottom = int(w * 0.84)
    fold = int(w * 0.14)
    sheet_pts = [
        (left, top + 16),
        (left + 16, top),
        (right - fold, top),
        (right, top + fold),
        (right, bottom - 16),
        (right - 16, bottom),
        (left + 16, bottom),
        (left, bottom - 16),
    ]
    shadow_pts = [(x, y + 14) for (x, y) in sheet_pts]
    odraw.polygon(shadow_pts, fill=(15, 23, 42, 72))
    canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)
    draw.polygon(sheet_pts, fill=(248, 250, 252, 255))

    # Folded corner triangle accent
    fold_pts = [
        (right - fold, top),
        (right - fold, top + fold),
        (right, top + fold),
    ]
    draw.polygon(fold_pts, fill=(199, 210, 254, 255))

    # Five subtle staff lines across upper-middle of page
    line_left = left + int(w * 0.07)
    line_right = right - int(w * 0.07)
    for idx in range(5):
        ly = top + int(w * (0.17 + idx * 0.042))
        draw.line([(line_left, ly), (line_right, ly)], fill=(203, 213, 225, 255), width=7)

    # Bold beamed eighth-note mark in rich cobalt-indigo
    ink = (30, 58, 138, 255)
    stem_w = 24
    # Left & right noteheads (tilted ellipses)
    nh1_cx, nh1_cy = int(w * 0.40), int(w * 0.68)
    nh2_cx, nh2_cy = int(w * 0.61), int(w * 0.63)
    rx, ry = 54, 38

    draw.ellipse((nh1_cx - rx, nh1_cy - ry, nh1_cx + rx, nh1_cy + ry), fill=ink)
    draw.ellipse((nh2_cx - rx, nh2_cy - ry, nh2_cx + rx, nh2_cy + ry), fill=ink)

    # Stems rising from right edges of noteheads
    s1_x = nh1_cx + rx - stem_w // 2 - 4
    s2_x = nh2_cx + rx - stem_w // 2 - 4
    b1_y = int(w * 0.37)
    b2_y = int(w * 0.32)
    draw.rectangle((s1_x - stem_w // 2, b1_y, s1_x + stem_w // 2, nh1_cy), fill=ink)
    draw.rectangle((s2_x - stem_w // 2, b2_y, s2_x + stem_w // 2, nh2_cy), fill=ink)

    # Slanted double beam connecting the two stems
    beam_h = 34
    draw.polygon(
        [
            (s1_x - stem_w // 2, b1_y),
            (s2_x + stem_w // 2, b2_y),
            (s2_x + stem_w // 2, b2_y + beam_h),
            (s1_x - stem_w // 2, b1_y + beam_h),
        ],
        fill=ink,
    )
    beam2_off = 52
    beam2_h = 22
    draw.polygon(
        [
            (s1_x - stem_w // 2, b1_y + beam2_off),
            (s2_x + stem_w // 2, b2_y + beam2_off),
            (s2_x + stem_w // 2, b2_y + beam2_off + beam2_h),
            (s1_x - stem_w // 2, b1_y + beam2_off + beam2_h),
        ],
        fill=ink,
    )

    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def generate_icons(output_dir: Path = ASSETS_DIR) -> tuple[Path, Path]:
    """Write ``icon.png`` and multi-size ``icon.ico`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    master = render_master_icon(256)
    png_path = output_dir / "icon.png"
    ico_path = output_dir / "icon.ico"
    master.save(png_path, format="PNG")
    master.save(ico_path, format="ICO", sizes=ICO_SIZES)
    return ico_path, png_path


if __name__ == "__main__":
    ico, png = generate_icons()
    print(f"Generated {ico} and {png}")
