"""Generate reproducible README and GitHub media assets for PDF to MuseScore."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import logging
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import tkinter as tk
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageOps

log = logging.getLogger("pdf2musescore.generate_media")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
DEFAULT_MSCZ = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.mscz"
DEFAULT_ICON = ROOT / "src" / "pdf2musescore" / "assets" / "icon.png"
DEFAULT_SCREENSHOT = ROOT / "docs" / "assets" / "ui-dark.png"
DEFAULT_UI_DARK_OUT = ROOT / "docs" / "assets" / "ui-dark.png"
DEFAULT_UI_LIGHT_OUT = ROOT / "docs" / "assets" / "ui-light.png"
DEFAULT_BEFORE_AFTER_OUT = ROOT / "docs" / "assets" / "before-after-bach.png"
DEFAULT_BEFORE_AFTER_FULLSCORE_OUT = ROOT / "docs" / "assets" / "before-after-fullscore.png"
DEFAULT_SOCIAL_OUT = ROOT / "docs" / "assets" / "social-preview.png"
DEFAULT_MEDIA_TMP = ROOT / "tmp" / "media"
DEFAULT_STORE_DIR = ROOT / "tmp" / "store_screenshots"
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
    *,
    before_title: str = "1. Original PDF Scan",
    after_title: str = "2. Editable MuseScore 4 (.mscz)",
    before_badge: str = "INPUT PDF",
    after_badge: str = "LOCAL AI OUTPUT",
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
            "title": before_title,
            "badge": before_badge,
            "header_bg": (241, 245, 249),
            "title_fg": (15, 23, 42),
            "badge_bg": (226, 232, 240),
            "badge_fg": (51, 65, 85),
            "border": (203, 213, 225),
        },
        {
            "x": outer_pad + card_w + gap_w,
            "img": trimmed_after,
            "title": after_title,
            "badge": after_badge,
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
        "Turn sheet music PDFs & photos into editable\nMuseScore 4 (.mscz) and MusicXML scores.",
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


def _capture_window_printwindow(hwnd: int) -> Optional[Image.Image]:
    """Capture a window off-screen via Win32 PrintWindow + DWM extended-frame crop."""
    if sys.platform != "win32":
        return None
    windll = getattr(ctypes, "windll", None)
    if windll is None or not hasattr(windll, "user32") or not hasattr(windll, "gdi32"):
        return None

    user32 = windll.user32
    gdi32 = windll.gdi32
    win_rect = wintypes.RECT()
    if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(win_rect)):
        return None
    win_w = int(win_rect.right - win_rect.left)
    win_h = int(win_rect.bottom - win_rect.top)
    if win_w <= 0 or win_h <= 0:
        return None

    class _BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class _BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

    hwnd_dc = user32.GetWindowDC(wintypes.HWND(hwnd))
    if not hwnd_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    if not mem_dc:
        user32.ReleaseDC(wintypes.HWND(hwnd), hwnd_dc)
        return None

    bmi = _BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = win_w
    bmi.bmiHeader.biHeight = -win_h  # top-down DIB
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0  # BI_RGB

    bits_ptr = ctypes.c_void_p()
    dib = gdi32.CreateDIBSection(mem_dc, ctypes.byref(bmi), 0, ctypes.byref(bits_ptr), None, 0)
    if not dib or not bits_ptr.value:
        if dib:
            gdi32.DeleteObject(dib)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(wintypes.HWND(hwnd), hwnd_dc)
        return None

    old_obj = gdi32.SelectObject(mem_dc, dib)
    try:
        # PW_RENDERFULLCONTENT = 2 captures DWM client + non-client area cleanly
        ok = user32.PrintWindow(wintypes.HWND(hwnd), mem_dc, 2)
        if not ok:
            ok = user32.PrintWindow(wintypes.HWND(hwnd), mem_dc, 0)
        if not ok:
            return None

        raw_bytes = ctypes.string_at(bits_ptr.value, win_w * win_h * 4)
        img = Image.frombuffer("RGBA", (win_w, win_h), raw_bytes, "raw", "BGRA", 0, 1).convert("RGB")

        # Crop invisible Win10/11 drop-shadow margins via DWMWA_EXTENDED_FRAME_BOUNDS (9)
        if hasattr(windll, "dwmapi"):
            vis_rect = wintypes.RECT()
            res = windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd),
                9,
                ctypes.byref(vis_rect),
                ctypes.sizeof(vis_rect),
            )
            if res == 0:
                cl = max(0, int(vis_rect.left - win_rect.left))
                ct = max(0, int(vis_rect.top - win_rect.top))
                cr = min(win_w, int(vis_rect.right - win_rect.left))
                cb = min(win_h, int(vis_rect.bottom - win_rect.top))
                if cr > cl and cb > ct:
                    img = img.crop((cl, ct, cr, cb))
        return img
    finally:
        gdi32.SelectObject(mem_dc, old_obj)
        gdi32.DeleteObject(dib)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(wintypes.HWND(hwnd), hwnd_dc)


def _create_showcase_window() -> tk.Tk | tk.Toplevel:
    """Create a standalone Tk root in CLI runs, or a Toplevel when a Tk root already exists."""
    existing = getattr(tk, "_default_root", None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                win = tk.Toplevel(existing)
                win.withdraw()
                return win
        except Exception:
            pass
    win = tk.Tk()
    win.withdraw()
    return win


def _capture_window_image(window: tk.Misc) -> Image.Image:
    """Capture the exact visible bounds of a Tk window (including Windows DWM titlebar)."""
    window.update_idletasks()
    window.update()
    time.sleep(0.35)
    window.update()

    if sys.platform == "win32":
        try:
            windll = getattr(ctypes, "windll", None)
            if windll is not None and hasattr(windll, "user32"):
                hwnd = int(windll.user32.GetParent(window.winfo_id()) or window.winfo_id())
                pw_img = _capture_window_printwindow(hwnd)
                if pw_img is not None:
                    return pw_img
        except Exception as exc:
            log.debug("PrintWindow capture fallback: %s", exc)

    x0 = window.winfo_rootx()
    y0 = window.winfo_rooty()
    x1 = x0 + window.winfo_width()
    y1 = y0 + window.winfo_height()
    return ImageGrab.grab(bbox=(x0, y0, x1, y1), all_screens=True).convert("RGB")


def _configure_showcase_window(
    root: tk.Tk | tk.Toplevel,
    theme: str,
    dark_titlebar: bool,
) -> None:
    from pdf2musescore.ui_theme import apply_windows_titlebar_theme, get_assets_dir

    assets = get_assets_dir()
    ico_path = assets / "icon.ico"
    png_path = assets / "icon.png"
    try:
        if ico_path.is_file():
            root.iconbitmap(str(ico_path))
        if png_path.is_file():
            from PIL import ImageTk

            icon_photo = ImageTk.PhotoImage(Image.open(png_path), master=root)
            root.iconphoto(True, icon_photo)
            root._app_icon_ref = icon_photo  # type: ignore[attr-defined]
    except Exception as exc:
        log.debug("Showcase window icon setup skipped: %s", exc)

    root.update_idletasks()
    scr_w = max(1024, root.winfo_screenwidth())
    scr_h = max(720, root.winfo_screenheight())
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    width = min(max(860, req_w + 12), max(820, scr_w - 60))
    height = min(max(420, req_h + 8), max(420, scr_h - 80))
    x = max(20, (scr_w - width) // 2)
    y = max(20, (scr_h - height) // 2 - 20)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.deiconify()
    apply_windows_titlebar_theme(root, dark_titlebar)
    try:
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
    except Exception:
        pass
    root.update()


def capture_ui_screenshots(
    pdf_path: Path,
    dark_out: Path,
    light_out: Path,
    *,
    capture_fn: Optional[Callable[[tk.Misc], Image.Image]] = None,
) -> tuple[Path, Path]:
    """Render and capture fresh Dark and Light Studio screenshots for README & docs."""
    pdf_path = Path(pdf_path)
    dark_out = Path(dark_out)
    light_out = Path(light_out)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Sample PDF not found: {pdf_path}")

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from pdf2musescore.config import AppConfig
    from pdf2musescore.gui import App
    from pdf2musescore.ui_theme import enable_high_dpi

    enable_high_dpi()
    grab = _capture_window_image if capture_fn is None else capture_fn
    dark_out.parent.mkdir(parents=True, exist_ok=True)
    light_out.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        tmp_dir = Path(td)

        # 1. Dark Studio: loaded PDF, collapsed optional drawers with live summary badge, completion banner
        root_dark = _create_showcase_window()
        app_dark = App(
            root_dark,
            AppConfig(log_dir=tmp_dir / "logs_d", work_dir=tmp_dir / "work_d", check_updates=False),
            musescore_available=lambda: True,
            theme="dark",
            prefs_path=tmp_dir / "prefs_dark.json",
            enable_dnd=False,
            prefetch_models=False,
            dual_install_checker=lambda: None,
        )
        try:
            app_dark.load_source(pdf_path)
            app_dark.file_meta_var.set(r"C:\Scores\Bach")
            app_dark.pages_var.set("1")
            app_dark.preview_pane.goto_page(1)
            app_dark.instrument_var.set("Cello")
            app_dark.clef_var.set("bass")
            app_dark._set_progress_visible(True)
            app_dark.progress["value"] = 100
            app_dark.pct_var.set("100%")
            app_dark.elapsed_var.set("0:18")
            app_dark.status_var.set("Done! Saved bach_cello_suite_2.mscz")
            app_dark._show_result_banner(Path(r"C:\Scores\Bach\bach_cello_suite_2.mscz"))
            _configure_showcase_window(root_dark, "dark", True)
            img_dark = grab(root_dark)
            img_dark.save(dark_out, format="PNG")
            log.info("Saved Dark Studio UI screenshot to %s (%dx%d)", dark_out, img_dark.width, img_dark.height)
        finally:
            app_dark.destroy()
            root_dark.destroy()

        # 2. Light Studio: loaded PDF, expanded Instrument/Clef/Transpose drawer with live interval hint
        root_light = _create_showcase_window()
        app_light = App(
            root_light,
            AppConfig(log_dir=tmp_dir / "logs_l", work_dir=tmp_dir / "work_l", check_updates=False),
            musescore_available=lambda: True,
            theme="light",
            prefs_path=tmp_dir / "prefs_light.json",
            enable_dnd=False,
            prefetch_models=False,
            dual_install_checker=lambda: None,
        )
        try:
            app_light.load_source(pdf_path)
            app_light.file_meta_var.set(r"C:\Scores\Bach")
            app_light.pages_var.set("1-2")
            app_light.preview_pane.goto_page(1)
            app_light.part_options_section.set_expanded(True)
            app_light.instrument_var.set("Trumpet")
            app_light.clef_var.set("treble")
            app_light.transpose_var.set("+2")
            _configure_showcase_window(root_light, "light", False)
            img_light = grab(root_light)
            img_light.save(light_out, format="PNG")
            log.info("Saved Light Studio UI screenshot to %s (%dx%d)", light_out, img_light.width, img_light.height)
        finally:
            app_light.destroy()
            root_light.destroy()

    return dark_out, light_out


_STORE_LISTING_COPY = """=== MICROSOFT STORE PARTNER CENTER LISTING COPY (PDF2MuseScore v0.6.0) ===

--- SHORT DESCRIPTION ---
Convert sheet music PDFs & photos into editable MuseScore 4 (.mscz) and MusicXML scores — 100% free, local AI, no account or page limits.

--- FULL DESCRIPTION ---
PDF2MuseScore converts printed sheet music PDFs, faded photocopies, and smartphone photos (.png, .jpg, .webp, .bmp, .tif) into editable MuseScore 4 (.mscz) and MusicXML (.musicxml) scores — running 100% locally on your Windows PC with no account, no cloud uploads, and no page limits.

Powered by the open-source homr deep-learning optical music recognition engine and RapidOCR, PDF2MuseScore supports both Single Part and Full Score (Conductor) modes — including a native-scale barline-chunked OMR engine for dense 20-35+ staff band and orchestral scores. It recognizes notes, slurs, ties, tied accidentals, beams, dynamics, compound meters, tempo markings, rehearsal marks, expression text, and percussion staves while sanitizing slurs, tuplets, dynamics collisions, multi-voice measures, and forced line breaks so MuseScore 4 imports cleanly.

Key highlights:
• Drag-and-drop PDFs or phone photos with live page preview and context-aware page range picking for multi-page scores
• One-click scan presets: Clean PDF, Faded Scan (contrast boost + deskew), and Phone Photo (shadow cleanup)
• Single Part & Full Score (Conductor) modes with native-scale barline-chunked OMR, automatic written instrument detection (Piano fallback — no more Voice mixer default), accurate MIDI playback & transpositions, OCR tempo markings, boxed/unboxed rehearsal marks, and expression text
• Optional collapsible Instrument, Clef & Transpose overrides with live interval hints (e.g. +2 for B♭ instruments)
• Clean Dark & Light Studio themes with one-click Open in MuseScore and Show in Folder completion actions

NOTE: On first launch, the app automatically downloads the open-source homr and RapidOCR AI model weights (~180 MB) once into your local user folder for offline use afterward. Exporting directly to .mscz uses MuseScore 4 if installed on your PC; MusicXML (.musicxml) export works standalone without MuseScore.

System requirements: 64-bit Windows 10/11, x86-64 CPU with AVX2 & SSE4.2, 8 GB RAM recommended (4 GB minimum), ~500 MB free disk space.

Open source under GNU AGPL-3.0-or-later (source code: https://github.com/tomermayost/pdf2musescore). PDF2MuseScore is an independent open-source project and is not affiliated with or endorsed by Muse Group or MuseScore.

--- PRODUCT FEATURES (UP TO 200 CHARS EACH) ---
1. 100% local offline AI on your PC — no account, no cloud uploads, no page limits
2. Drag & drop PDFs or photos (.png, .jpg, .webp, .bmp, .tif) with live page preview
3. Single Part & Full Score (Conductor) modes — handles dense 20-35+ staff orchestral & band scores
4. OCR boxed/unboxed rehearsal marks, mid-piece tempo changes, and expression text (rit., tranq.)
5. Pre-export Instrument, Clef & Transpose overrides with live musical interval hints
6. Direct .mscz and .musicxml export with dynamics collision, tuplet, and slur sanitization

--- WHAT'S NEW IN THIS VERSION ---
• Upgraded Full Score (Conductor) mode with native-scale barline-chunked OMR, skew-aware measure alignment, and coordinate-free sequence overlap stitching for dense multi-staff pages.
• Preserved overlap-confirmed tied accidentals and cross-boundary chord ties, plus trimmed cautionary end-of-page key/time stubs so measures never desync across pages.
• Hardened compound meters (12/8, 6/4, 3/4, 2/4), repaired spurious tuplets, and normalized unpitched percussion staves cleanly.
• Improved boxed/unboxed rehearsal mark OCR (including O/P), multi-tempo placement, and non-English / curly-quote instrument label matching.
"""


def _draw_check_bullet(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int = 13) -> None:
    """Draw a cobalt circle with a crisp white vector checkmark for Store feature bullets."""
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(37, 99, 235))
    draw.line(
        [(cx - 5, cy + 1), (cx - 1, cy + 5), (cx + 6, cy - 4)],
        fill=(255, 255, 255),
        width=3,
    )


def compose_store_slide(
    image: Image.Image,
    title: str,
    subtitle: str,
    out_path: Path,
    *,
    badge: str = "",
    bullets: Sequence[str] = (),
    layout: str = "centered",
    dark_canvas: bool = True,
    icon_path: Optional[Path] = DEFAULT_ICON,
    size: tuple[int, int] = (1920, 1080),
) -> Path:
    """Compose a crisp 16:9 Microsoft Store showcase screenshot slide."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = int(size[0]), int(size[1])

    bg = (15, 23, 42) if dark_canvas else (241, 245, 249)
    title_fg = (248, 250, 252) if dark_canvas else (15, 23, 42)
    sub_fg = (148, 163, 184) if dark_canvas else (71, 85, 105)
    bullet_fg = (226, 232, 240) if dark_canvas else (30, 41, 59)
    card_bg = (22, 32, 50) if dark_canvas else (255, 255, 255)
    card_border = (51, 65, 85) if dark_canvas else (203, 213, 225)
    shadow_col = (7, 11, 22) if dark_canvas else (203, 213, 225)
    badge_bg = (30, 58, 104) if dark_canvas else (219, 234, 254)
    badge_fg = (147, 197, 253) if dark_canvas else (29, 78, 216)
    badge_border = (59, 130, 246)

    canvas = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(canvas)

    # Top cobalt-indigo gradient bar (#2563EB -> #4338CA)
    c_left = (37, 99, 235)
    c_right = (67, 56, 202)
    for x in range(w):
        t = x / max(1, w - 1)
        color = tuple(int(round(c_left[i] + (c_right[i] - c_left[i]) * t)) for i in range(3))
        draw.line([(x, 0), (x, 7)], fill=color)

    badge_font = _load_font(15, bold=True)
    cleaned_badge = (badge or "").strip()

    if layout == "split":
        brand_font = _load_font(24, bold=True)
        brand_sub_font = _load_font(16, bold=False)
        title_font = _load_font(46, bold=True)
        sub_font = _load_font(22, bold=False)
        bullet_font = _load_font(21, bold=False)
        foot_font = _load_font(17, bold=False)

        left_x = 76
        cur_y = 76

        # Brand icon + label row
        text_x = left_x
        if icon_path is not None and Path(icon_path).is_file():
            icon = Image.open(icon_path).convert("RGBA").resize((56, 56), Image.Resampling.LANCZOS)
            canvas.paste(icon, (left_x, cur_y), icon)
            text_x = left_x + 74
        draw.text((text_x, cur_y + 2), "PDF to MuseScore", font=brand_font, fill=title_fg)
        draw.text((text_x, cur_y + 34), "Offline Windows Studio", font=brand_sub_font, fill=sub_fg)
        cur_y += 96

        if cleaned_badge:
            bb = draw.textbbox((0, 0), cleaned_badge, font=badge_font)
            bw = (bb[2] - bb[0]) + 28
            bh = (bb[3] - bb[1]) + 14
            draw.rounded_rectangle(
                (left_x, cur_y, left_x + bw, cur_y + bh),
                radius=bh // 2,
                fill=badge_bg,
                outline=badge_border,
                width=1,
            )
            draw.text(
                (
                    left_x + (bw - (bb[2] - bb[0])) // 2 - bb[0],
                    cur_y + (bh - (bb[3] - bb[1])) // 2 - bb[1],
                ),
                cleaned_badge,
                font=badge_font,
                fill=badge_fg,
            )
            cur_y += bh + 28

        tb = draw.multiline_textbbox((0, 0), title, font=title_font, spacing=10)
        th = tb[3] - tb[1]
        draw.multiline_text((left_x, cur_y - tb[1]), title, font=title_font, fill=title_fg, spacing=10)
        cur_y += th + 24

        sb = draw.multiline_textbbox((0, 0), subtitle, font=sub_font, spacing=8)
        sh = sb[3] - sb[1]
        draw.multiline_text((left_x, cur_y - sb[1]), subtitle, font=sub_font, fill=sub_fg, spacing=8)
        cur_y += sh + 42

        for item in bullets:
            _draw_check_bullet(draw, left_x + 13, cur_y + 13, radius=13)
            draw.text((left_x + 40, cur_y), item, font=bullet_font, fill=bullet_fg)
            cur_y += 52

        draw.text((left_x, h - 74), "100% Local CPU AI · GNU AGPL-3.0", font=foot_font, fill=sub_fg)

        # Right column: large near-full-height framed screenshot
        right_x0 = 704
        right_w = w - right_x0 - 64
        right_y0 = 54
        right_h = h - 108
        card_pad = 18
        fitted = _fit_into_box(
            image.convert("RGB"),
            max_w=right_w - card_pad * 2,
            max_h=right_h - card_pad * 2,
        )
        card_w = fitted.width + card_pad * 2
        card_h = fitted.height + card_pad * 2
        cx0 = right_x0 + (right_w - card_w) // 2
        cy0 = right_y0 + (right_h - card_h) // 2
    else:
        title_font = _load_font(40, bold=True)
        sub_font = _load_font(22, bold=False)

        cur_y = 28
        if cleaned_badge:
            bb = draw.textbbox((0, 0), cleaned_badge, font=badge_font)
            bw = (bb[2] - bb[0]) + 28
            bh = (bb[3] - bb[1]) + 14
            bx0 = (w - bw) // 2
            draw.rounded_rectangle(
                (bx0, cur_y, bx0 + bw, cur_y + bh),
                radius=bh // 2,
                fill=badge_bg,
                outline=badge_border,
                width=1,
            )
            draw.text(
                (bx0 + (bw - (bb[2] - bb[0])) // 2 - bb[0], cur_y + (bh - (bb[3] - bb[1])) // 2 - bb[1]),
                cleaned_badge,
                font=badge_font,
                fill=badge_fg,
            )
            cur_y += bh + 14

        tb = draw.textbbox((0, 0), title, font=title_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text(((w - tw) // 2 - tb[0], cur_y - tb[1]), title, font=title_font, fill=title_fg)
        cur_y += th + 12

        sb = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw, sh = sb[2] - sb[0], sb[3] - sb[1]
        draw.text(((w - sw) // 2 - sb[0], cur_y - sb[1]), subtitle, font=sub_font, fill=sub_fg)
        cur_y += sh + 24

        bottom_margin = 30
        card_pad = 16
        avail_w = w - 72 - card_pad * 2
        avail_h = max(200, h - cur_y - bottom_margin - card_pad * 2)
        fitted = _fit_into_box(image.convert("RGB"), max_w=avail_w, max_h=avail_h)

        card_w = fitted.width + card_pad * 2
        card_h = fitted.height + card_pad * 2
        cx0 = (w - card_w) // 2
        cy0 = cur_y + max(0, (h - bottom_margin - cur_y - card_h) // 2)

    draw.rounded_rectangle(
        (cx0 + 4, cy0 + 8, cx0 + card_w + 4, cy0 + card_h + 8),
        radius=20,
        fill=shadow_col,
    )
    draw.rounded_rectangle(
        (cx0, cy0, cx0 + card_w, cy0 + card_h),
        radius=20,
        fill=card_bg,
        outline=card_border,
        width=2,
    )
    sx = cx0 + card_pad
    sy = cy0 + card_pad
    draw.rounded_rectangle(
        (sx - 2, sy - 2, sx + fitted.width + 2, sy + fitted.height + 2),
        radius=8,
        outline=card_border,
        width=2,
    )
    canvas.paste(fitted, (sx, sy))

    canvas.save(out_path, format="PNG")
    log.info("Saved Microsoft Store slide to %s (%dx%d)", out_path, canvas.width, canvas.height)
    return out_path


def capture_empty_dark_screenshot(
    out_path: Path,
    *,
    capture_fn: Optional[Callable[[tk.Misc], Image.Image]] = None,
) -> Path:
    """Capture the clean empty first-launch Dark Studio window."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from pdf2musescore.config import AppConfig
    from pdf2musescore.gui import App
    from pdf2musescore.ui_theme import enable_high_dpi

    enable_high_dpi()
    grab = _capture_window_image if capture_fn is None else capture_fn
    with TemporaryDirectory() as td:
        tmp_dir = Path(td)
        root_empty = _create_showcase_window()
        app_empty = App(
            root_empty,
            AppConfig(log_dir=tmp_dir / "logs_e", work_dir=tmp_dir / "work_e", check_updates=False),
            musescore_available=lambda: True,
            theme="dark",
            prefs_path=tmp_dir / "prefs_empty.json",
            enable_dnd=False,
            prefetch_models=False,
            dual_install_checker=lambda: None,
        )
        try:
            _configure_showcase_window(root_empty, "dark", True)
            img = grab(root_empty)
            img.save(out_path, format="PNG")
            log.info("Saved empty first-launch UI screenshot to %s (%dx%d)", out_path, img.width, img.height)
        finally:
            app_empty.destroy()
            root_empty.destroy()
    return out_path


def generate_store_screenshots(
    out_dir: Path,
    *,
    dark_ui_path: Path = DEFAULT_UI_DARK_OUT,
    light_ui_path: Path = DEFAULT_UI_LIGHT_OUT,
    before_after_path: Path = DEFAULT_BEFORE_AFTER_OUT,
    fullscore_before_after_path: Optional[Path] = DEFAULT_BEFORE_AFTER_FULLSCORE_OUT,
    capture_fn: Optional[Callable[[tk.Misc], Image.Image]] = None,
) -> list[Path]:
    """Generate 1920x1080 Microsoft Store showcase slides and Partner Center copy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for req_path in (dark_ui_path, light_ui_path, before_after_path):
        if not Path(req_path).is_file():
            raise FileNotFoundError(f"Required source asset not found: {req_path}")

    outputs: list[Path] = []
    with Image.open(dark_ui_path) as img_dark:
        outputs.append(
            compose_store_slide(
                img_dark,
                "Convert Sheet Music\nPDFs & Photos Locally",
                "Turn printed scores, scans, and photos\ninto editable MuseScore 4 & MusicXML.",
                out_dir / "01-dark-studio.png",
                badge="MICROSOFT STORE · FREE & OPEN SOURCE",
                bullets=(
                    "100% local CPU AI — zero cloud uploads",
                    "No account required & unlimited pages",
                    "Direct .mscz export & Open in MuseScore",
                ),
                layout="split",
                dark_canvas=True,
            )
        )

    with Image.open(before_after_path) as img_ba:
        outputs.append(
            compose_store_slide(
                img_ba,
                "Real Local AI Conversion: PDF Scan → Editable MuseScore 4",
                "Preserves clefs, key & time signatures, slurs, ties, beams, and flowing measures",
                out_dir / "02-before-after-bach.png",
                badge="BACH CELLO SUITE NO. 2 · CONVERTED ON CPU",
                layout="centered",
                dark_canvas=True,
            )
        )

    with Image.open(light_ui_path) as img_light:
        outputs.append(
            compose_store_slide(
                img_light,
                "Built-In Transpose &\nFlexible Score Setup",
                "Switch between Single Part and Full Score\nwith live musical interval hints.",
                out_dir / "03-light-studio-transpose.png",
                badge="LIGHT & DARK THEMES · MUSICIAN TOOLS",
                bullets=(
                    "Single Part & Full Score (Conductor) modes",
                    "Transpose by half-steps (e.g. +2 for Bb)",
                    "Instrument & clef overrides with live badge",
                ),
                layout="split",
                dark_canvas=False,
            )
        )

    with TemporaryDirectory() as td:
        raw_empty = Path(td) / "ui-empty-dark.png"
        capture_empty_dark_screenshot(raw_empty, capture_fn=capture_fn)
        with Image.open(raw_empty) as img_empty:
            outputs.append(
                compose_store_slide(
                    img_empty,
                    "Simple to Start,\nPowerful When Needed",
                    "Drop a PDF or photo and click Convert —\nadvanced AI knobs stay neatly tucked away.",
                    out_dir / "04-clean-first-launch.png",
                    badge="STREAMLINED STUDIO WORKFLOW",
                    bullets=(
                        "Drag & drop PDF, PNG, JPG, WebP, or TIFF",
                        "One-click Clean PDF, Faded & Photo presets",
                        "Auto-shown page picker for multi-page PDFs",
                    ),
                    layout="split",
                    dark_canvas=True,
                )
            )

    if fullscore_before_after_path is not None and Path(fullscore_before_after_path).is_file():
        with Image.open(fullscore_before_after_path) as img_fs:
            outputs.append(
                compose_store_slide(
                    img_fs,
                    "Full Score Conductor Mode: Dense Orchestral PDF → MuseScore 4",
                    "Aligns 13 staves, preserves keyless Brass/Timpani vs keyed Winds/Strings, tempo, and multi-voice slurs",
                    out_dir / "05-before-after-fullscore.png",
                    badge="BEETHOVEN VIOLIN CONCERTO · FULL SCORE AI",
                    layout="centered",
                    dark_canvas=True,
                )
            )

    copy_path = out_dir / "store_listing_copy.txt"
    copy_path.write_text(_STORE_LISTING_COPY.lstrip(), encoding="utf-8")
    log.info("Saved Microsoft Store Partner Center copy to %s", copy_path)
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate README and GitHub social media assets.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--before-after", action="store_true", help="Generate before/after Bach comparison PNG.")
    parser.add_argument(
        "--before-after-fullscore",
        action="store_true",
        help="Compose before/after Full Score comparison PNG from rendered before/after page images.",
    )
    parser.add_argument("--social-preview", action="store_true", help="Generate 1280x640 GitHub social preview PNG.")
    parser.add_argument("--ui-screenshots", action="store_true", help="Capture refreshed Dark & Light UI PNGs.")
    parser.add_argument("--store-screenshots", action="store_true", help="Generate 1920x1080 Microsoft Store slides.")
    parser.add_argument("--all", action="store_true", help="Generate all media assets.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--mscz", type=Path, default=DEFAULT_MSCZ)
    parser.add_argument("--musescore", type=Path, default=DEFAULT_MUSESCORE)
    parser.add_argument("--icon", type=Path, default=DEFAULT_ICON)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--ui-dark-out", type=Path, default=DEFAULT_UI_DARK_OUT)
    parser.add_argument("--ui-light-out", type=Path, default=DEFAULT_UI_LIGHT_OUT)
    parser.add_argument("--store-out-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_MEDIA_TMP)
    parser.add_argument("--before-after-out", type=Path, default=DEFAULT_BEFORE_AFTER_OUT)
    parser.add_argument("--fullscore-before-png", type=Path, default=DEFAULT_MEDIA_TMP / "fullscore_before.png")
    parser.add_argument("--fullscore-after-png", type=Path, default=DEFAULT_MEDIA_TMP / "fullscore_after.png")
    parser.add_argument("--before-after-fullscore-out", type=Path, default=DEFAULT_BEFORE_AFTER_FULLSCORE_OUT)
    parser.add_argument("--social-out", type=Path, default=DEFAULT_SOCIAL_OUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    explicit = (
        args.before_after
        or args.before_after_fullscore
        or args.social_preview
        or args.ui_screenshots
        or args.store_screenshots
    )
    run_ba = args.all or args.before_after or not explicit
    run_social = args.all or args.social_preview or not explicit

    if args.ui_screenshots:
        capture_ui_screenshots(args.pdf, args.ui_dark_out, args.ui_light_out)

    if run_ba:
        before_png = render_pdf_first_page(args.pdf, args.tmp_dir / "before.png")
        after_png = render_mscz_first_page(args.mscz, args.tmp_dir / "mscz_pages", args.musescore)
        with Image.open(before_png) as b_img, Image.open(after_png) as a_img:
            compose_before_after(b_img, a_img, args.before_after_out)

    if args.before_after_fullscore:
        if not args.fullscore_before_png.is_file():
            raise FileNotFoundError(f"Full score before PNG not found: {args.fullscore_before_png}")
        if not args.fullscore_after_png.is_file():
            raise FileNotFoundError(f"Full score after PNG not found: {args.fullscore_after_png}")
        with Image.open(args.fullscore_before_png) as b_img, Image.open(args.fullscore_after_png) as a_img:
            compose_before_after(
                b_img,
                a_img,
                args.before_after_fullscore_out,
                before_title="1. Original Full Score PDF",
                after_title="2. Editable MuseScore 4 (.mscz)",
                before_badge="FULL SCORE PDF",
                after_badge="LOCAL AI OUTPUT",
            )

    if run_social:
        shot = args.ui_dark_out if args.ui_screenshots else args.screenshot
        generate_social_preview(args.icon, shot, args.social_out)

    if args.store_screenshots:
        fs_ba = args.before_after_fullscore_out if args.before_after_fullscore_out.is_file() else None
        generate_store_screenshots(
            args.store_out_dir,
            dark_ui_path=args.ui_dark_out,
            light_ui_path=args.ui_light_out,
            before_after_path=args.before_after_out,
            fullscore_before_after_path=fs_ba,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
