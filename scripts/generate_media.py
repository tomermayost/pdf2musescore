"""Generate reproducible README and GitHub media assets for PDF to MuseScore."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

log = logging.getLogger("pdf2musescore.generate_media")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
DEFAULT_MSCZ = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.mscz"
DEFAULT_ICON = ROOT / "src" / "pdf2musescore" / "assets" / "icon.png"
DEFAULT_SCREENSHOT = ROOT / "docs" / "assets" / "ui-dark.png"
DEFAULT_BEFORE_AFTER_OUT = ROOT / "docs" / "assets" / "before-after-bach.png"
DEFAULT_SOCIAL_OUT = ROOT / "docs" / "assets" / "social-preview.png"
DEFAULT_MEDIA_TMP = ROOT / "tmp" / "media"
DEFAULT_MUSESCORE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load Segoe UI from Windows Fonts when available, falling back to Pillow's default font."""
    candidates = (
        [Path(r"C:\Windows\Fonts\segoeuib.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")]
        if bold
        else [Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")]
    )
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception as exc:  # pragma: no cover - font loading defensive guard
                log.debug("Failed to load font %s: %s", path, exc)
    log.debug("Using Pillow default bitmap font (size=%d, bold=%s)", size, bold)
    return ImageFont.load_default()


def _trim_outer_whitespace(img: Image.Image, pad: int = 20, white_threshold: int = 248) -> Image.Image:
    """Gently trim outer blank margins from a rendered page image and add balanced padding."""
    rgb = img.convert("RGB")
    gray = rgb.convert("L")
    mask = gray.point(lambda p: 255 if p < white_threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    cropped = rgb.crop((left, top, right, bottom))
    return ImageOps.expand(cropped, border=pad, fill=(255, 255, 255))


def _fit_into_box(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize ``img`` proportionally to fit within ``(max_w, max_h)``."""
    src_w, src_h = max(1, img.width), max(1, img.height)
    scale = min(max_w / src_w, max_h / src_h)
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def render_pdf_first_page(pdf_path: Path, out_png: Path, dpi: int = 200) -> Path:
    """Render page 1 of ``pdf_path`` into an RGB PNG at ``out_png``."""
    import pypdfium2 as pdfium

    pdf_path = Path(pdf_path)
    out_png = Path(out_png)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    log.info("Rendering PDF page 1 at %d DPI from %s", dpi, pdf_path)
    doc = pdfium.PdfDocument(str(pdf_path))
    page = None
    try:
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        page = doc[0]
        pil = page.render(scale=dpi / 72.0).to_pil().convert("RGB")
        pil.save(out_png, format="PNG")
        log.debug("Saved PDF page 1 render to %s (%dx%d)", out_png, pil.width, pil.height)
        return out_png
    finally:
        if page is not None:
            page.close()
        doc.close()


def render_mscz_first_page(mscz_path: Path, out_dir: Path, musescore_exe: Path = DEFAULT_MUSESCORE) -> Path:
    """Render ``mscz_path`` to PNG via MuseScore CLI and return the first page PNG."""
    mscz_path = Path(mscz_path)
    out_dir = Path(out_dir)
    musescore_exe = Path(musescore_exe)
    if not mscz_path.is_file():
        raise FileNotFoundError(f"MuseScore file not found: {mscz_path}")
    if not musescore_exe.is_file():
        raise FileNotFoundError(f"MuseScore executable not found: {musescore_exe}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("after*.png"):
        stale.unlink(missing_ok=True)

    target = out_dir / "after.png"
    cmd = [str(musescore_exe), "-r", "180", "-o", str(target), str(mscz_path)]
    log.info("Rendering MuseScore preview via CLI: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"MuseScore CLI failed (code {res.returncode}): {(res.stderr or res.stdout).strip()}")

    pages = sorted(out_dir.glob("after*.png"))
    if not pages:
        raise RuntimeError(f"MuseScore CLI produced no PNG files in {out_dir}")
    log.info("MuseScore rendered %d page(s); using %s", len(pages), pages[0].name)
    return pages[0]


def _draw_right_arrow(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int = 24) -> None:
    """Draw a cobalt badge circle with a crisp white right-pointing vector arrow."""
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(37, 99, 235),
        outline=(29, 78, 216),
        width=2,
    )
    shaft_half_h = 3
    head_w = 10
    left_x = cx - 11
    right_x = cx + 11
    tip_base_x = right_x - head_w
    draw.rectangle((left_x, cy - shaft_half_h, tip_base_x, cy + shaft_half_h), fill=(255, 255, 255))
    draw.polygon(
        [
            (tip_base_x, cy - 9),
            (right_x + 1, cy),
            (tip_base_x, cy + 9),
        ],
        fill=(255, 255, 255),
    )


def compose_before_after(
    before_img: Image.Image,
    after_img: Image.Image,
    out_path: Path,
    width: int = 1400,
) -> Path:
    """Compose side-by-side framed 'Original PDF' -> 'Editable MuseScore 4' comparison PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trimmed_before = _trim_outer_whitespace(before_img, pad=18)
    trimmed_after = _trim_outer_whitespace(after_img, pad=18)

    outer_pad = 32
    gap_w = 72
    card_w = (width - outer_pad * 2 - gap_w) // 2
    header_h = 54
    inner_pad = 16
    slot_w = card_w - inner_pad * 2

    # Compute uniform slot height from page proportions so neither side looks squished or giant
    aspect_b = trimmed_before.height / max(1, trimmed_before.width)
    aspect_a = trimmed_after.height / max(1, trimmed_after.width)
    slot_h = int(round(slot_w * max(aspect_b, aspect_a)))
    slot_h = max(360, min(800, slot_h))

    card_h = header_h + inner_pad * 2 + slot_h
    total_h = outer_pad * 2 + card_h

    canvas = Image.new("RGB", (width, total_h), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    header_font = _load_font(18, bold=True)
    badge_font = _load_font(13, bold=True)

    cards = [
        {
            "x": outer_pad,
            "img": trimmed_before,
            "title": "1. Original PDF Scan",
            "badge": "INPUT PDF",
            "header_bg": (241, 245, 249),
            "title_fg": (15, 23, 42),
            "badge_bg": (226, 232, 240),
            "badge_fg": (51, 65, 85),
            "border": (203, 213, 225),
        },
        {
            "x": outer_pad + card_w + gap_w,
            "img": trimmed_after,
            "title": "2. Editable MuseScore 4 (.mscz)",
            "badge": "LOCAL AI OUTPUT",
            "header_bg": (239, 246, 255),
            "title_fg": (30, 58, 138),
            "badge_bg": (219, 234, 254),
            "badge_fg": (29, 78, 216),
            "border": (147, 197, 253),
        },
    ]

    y0 = outer_pad
    y1 = y0 + card_h

    for card in cards:
        x0 = card["x"]
        x1 = x0 + card_w

        # Subtle drop shadow
        draw.rounded_rectangle((x0 + 2, y0 + 4, x1 + 2, y1 + 4), radius=14, fill=(226, 232, 240))
        # Card surface
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(255, 255, 255), outline=card["border"], width=2)

        # Header band + crisp separator
        draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y0 + header_h), radius=12, fill=card["header_bg"])
        draw.rectangle((x0 + 2, y0 + header_h - 12, x1 - 2, y0 + header_h), fill=card["header_bg"])
        draw.line([(x0 + 2, y0 + header_h), (x1 - 2, y0 + header_h)], fill=card["border"], width=1)

        # Title on left of header
        draw.text((x0 + 18, y0 + 15), card["title"], font=header_font, fill=card["title_fg"])

        # Right pill badge
        bbox = draw.textbbox((0, 0), card["badge"], font=badge_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        bw = tw + 20
        bh = th + 12
        bx1 = x1 - 16
        bx0 = bx1 - bw
        by0 = y0 + (header_h - bh) // 2
        by1 = by0 + bh
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=bh // 2, fill=card["badge_bg"])
        tx = bx0 + (bw - tw) // 2 - bbox[0]
        ty = by0 + (bh - th) // 2 - bbox[1]
        draw.text((tx, ty), card["badge"], font=badge_font, fill=card["badge_fg"])

        # Centered sheet-music render inside slot
        fitted = _fit_into_box(card["img"], slot_w, slot_h)
        paste_x = x0 + inner_pad + (slot_w - fitted.width) // 2
        paste_y = y0 + header_h + inner_pad + (slot_h - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

    # Center arrow badge between the two cards
    arrow_cx = outer_pad + card_w + gap_w // 2
    arrow_cy = y0 + header_h + inner_pad + slot_h // 2
    _draw_right_arrow(draw, arrow_cx, arrow_cy, radius=24)

    canvas.save(out_path, format="PNG")
    log.info("Saved before/after comparison to %s (%dx%d)", out_path, canvas.width, canvas.height)
    return out_path


def _draw_pill_row(
    draw: ImageDraw.ImageDraw,
    pills: Sequence[str],
    start_x: int,
    start_y: int,
    max_right: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw rounded feature pills left-to-right, wrapping to a second row if needed."""
    x = start_x
    y = start_y
    pad_x = 14
    pad_y = 8
    gap_x = 10
    gap_y = 10

    for text in pills:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pw = tw + pad_x * 2
        ph = th + pad_y * 2
        if x + pw > max_right and x > start_x:
            x = start_x
            y += ph + gap_y
        draw.rounded_rectangle(
            (x, y, x + pw, y + ph),
            radius=ph // 2,
            fill=(30, 41, 59),
            outline=(59, 130, 246),
            width=1,
        )
        tx = x + (pw - tw) // 2 - bbox[0]
        ty = y + (ph - th) // 2 - bbox[1]
        draw.text((tx, ty), text, font=font, fill=(226, 232, 240))
        x += pw + gap_x


def generate_social_preview(
    icon_path: Path,
    screenshot_path: Path,
    out_path: Path,
) -> Path:
    """Create a 1280x640 GitHub OpenGraph social preview banner."""
    icon_path = Path(icon_path)
    screenshot_path = Path(screenshot_path)
    out_path = Path(out_path)
    if not icon_path.is_file():
        raise FileNotFoundError(f"Icon not found: {icon_path}")
    if not screenshot_path.is_file():
        raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1280, 640
    canvas = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    # Top cobalt-indigo gradient bar (#2563EB -> #4338CA)
    c_left = (37, 99, 235)
    c_right = (67, 56, 202)
    for x in range(w):
        t = x / max(1, w - 1)
        color = tuple(int(round(c_left[i] + (c_right[i] - c_left[i]) * t)) for i in range(3))
        draw.line([(x, 0), (x, 7)], fill=color)

    # Framed app screenshot neatly fitted inside the right column
    shot = Image.open(screenshot_path).convert("RGB")
    fitted = _fit_into_box(shot, max_w=590, max_h=528)
    card_pad = 16
    card_w = fitted.width + card_pad * 2
    card_h = fitted.height + card_pad * 2
    cx0 = 620 + (612 - card_w) // 2
    cy0 = (h - card_h) // 2
    draw.rounded_rectangle(
        (cx0, cy0, cx0 + card_w, cy0 + card_h),
        radius=18,
        fill=(22, 32, 50),
        outline=(51, 65, 85),
        width=2,
    )

    # Paste app icon on top-left
    icon = Image.open(icon_path).convert("RGBA").resize((88, 88), Image.Resampling.LANCZOS)
    canvas.paste(icon, (60, 74), icon)

    tag_font = _load_font(15, bold=True)
    title_font = _load_font(48, bold=True)
    sub_font = _load_font(23, bold=False)
    desc_font = _load_font(17, bold=False)
    pill_font = _load_font(15, bold=True)

    # Brand tag beside icon
    draw.text((166, 90), "OPEN-SOURCE WINDOWS STUDIO", font=tag_font, fill=(96, 165, 250))
    draw.text((166, 114), "Powered by homr AI · GNU AGPL-3.0", font=desc_font, fill=(148, 163, 184))

    # Main headline & value prop
    draw.text((60, 200), "PDF to MuseScore", font=title_font, fill=(248, 250, 252))
    draw.text((60, 268), "Free, Offline AI Sheet Music Converter", font=sub_font, fill=(226, 232, 240))
    draw.text(
        (60, 314),
        "Turn printed or scanned PDFs into editable\nMuseScore 4 (.mscz) and MusicXML scores.",
        font=desc_font,
        fill=(148, 163, 184),
        spacing=7,
    )

    # Feature pills
    _draw_pill_row(
        draw,
        pills=("100% Local AI", "Direct .mscz & MusicXML", "No Account · No Limits"),
        start_x=60,
        start_y=404,
        max_right=600,
        font=pill_font,
    )

    # Repo footer link
    draw.text((60, 548), "github.com/tomermayost/pdf2musescore", font=desc_font, fill=(148, 163, 184))

    sx = cx0 + card_pad
    sy = cy0 + card_pad
    draw.rounded_rectangle(
        (sx - 2, sy - 2, sx + fitted.width + 2, sy + fitted.height + 2),
        radius=8,
        outline=(71, 85, 105),
        width=2,
    )
    canvas.paste(fitted, (sx, sy))

    canvas.save(out_path, format="PNG")
    log.info("Saved GitHub social preview to %s (%dx%d)", out_path, canvas.width, canvas.height)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate README and GitHub social media assets.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--before-after", action="store_true", help="Generate before/after Bach comparison PNG.")
    parser.add_argument("--social-preview", action="store_true", help="Generate 1280x640 GitHub social preview PNG.")
    parser.add_argument("--all", action="store_true", help="Generate all media assets.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--mscz", type=Path, default=DEFAULT_MSCZ)
    parser.add_argument("--musescore", type=Path, default=DEFAULT_MUSESCORE)
    parser.add_argument("--icon", type=Path, default=DEFAULT_ICON)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_MEDIA_TMP)
    parser.add_argument("--before-after-out", type=Path, default=DEFAULT_BEFORE_AFTER_OUT)
    parser.add_argument("--social-out", type=Path, default=DEFAULT_SOCIAL_OUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    run_ba = args.all or args.before_after or not (args.before_after or args.social_preview)
    run_social = args.all or args.social_preview or not (args.before_after or args.social_preview)

    if run_ba:
        before_png = render_pdf_first_page(args.pdf, args.tmp_dir / "before.png")
        after_png = render_mscz_first_page(args.mscz, args.tmp_dir / "mscz_pages", args.musescore)
        with Image.open(before_png) as b_img, Image.open(after_png) as a_img:
            compose_before_after(b_img, a_img, args.before_after_out)

    if run_social:
        generate_social_preview(args.icon, args.screenshot, args.social_out)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
