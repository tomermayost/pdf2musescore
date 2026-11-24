"""Reusable studio desktop UI widgets: ToolTip, CollapsibleSection, and PdfPreviewPane."""

from __future__ import annotations

import logging
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

from PIL import Image, ImageTk

from .engines.homr_engine import is_image_file, load_pil_image_rgb
from .ui_theme import THEMES, ThemePalette, get_assets_dir

log = logging.getLogger(__name__)


def _cancel_after(widget: tk.Misc, after_id: Optional[str]) -> None:
    if after_id:
        try:
            widget.after_cancel(after_id)
        except Exception:
            pass


def render_pdf_page_image(
    pdf: Path,
    page_number: int,
    max_size: tuple[int, int] = (240, 325),
) -> Image.Image:
    """Render a 1-based page from ``pdf`` (or image file) into a bounded RGB PIL Image."""
    path = Path(pdf)
    if is_image_file(path):
        rgb = load_pil_image_rgb(path, page_number=page_number)
        rgb.thumbnail(max_size, Image.Resampling.LANCZOS)
        return rgb

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    page = None
    try:
        total = len(doc)
        if page_number < 1 or page_number > total:
            raise ValueError(f"Page {page_number} out of range (1..{total})")
        page = doc[page_number - 1]
        bitmap = page.render(scale=1.5)
        pil_img = bitmap.to_pil()
        rgb = pil_img.convert("RGB")
        rgb.thumbnail(max_size, Image.Resampling.LANCZOS)
        return rgb
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        try:
            doc.close()
        except Exception:
            pass


class ToolTip:
    """Modern delayed dark-slate hover tooltip with bold title, body copy, and screen clamping."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay_ms: int = 350,
        palette: Optional[ThemePalette] = None,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = max(0, int(delay_ms))
        self.palette = palette or THEMES["light"]
        self.tip_window: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def set_palette(self, palette: ThemePalette) -> None:
        self.palette = palette

    def _schedule(self, _event: Optional[tk.Event] = None) -> None:
        if self.tip_window is not None or not self.text:
            return
        _cancel_after(self.widget, self._after_id)
        if self.delay_ms <= 0:
            self.show()
        else:
            self._after_id = self.widget.after(self.delay_ms, self.show)

    def show(self, _event: Optional[tk.Event] = None) -> None:
        _cancel_after(self.widget, self._after_id)
        self._after_id = None
        if self.tip_window is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except tk.TclError:
            return

        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)

        p = self.palette
        outer = tk.Frame(
            tw,
            background=p.tooltip_bg,
            highlightbackground=p.border_strong,
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        outer.pack()

        lines = self.text.split("\n", 1)
        title_text = lines[0].strip()
        body_text = lines[1].strip() if len(lines) > 1 else ""

        tk.Label(
            outer,
            text=title_text,
            justify="left",
            background=p.tooltip_bg,
            foreground=p.tooltip_fg,
            font=("Segoe UI", 9, "bold"),
            wraplength=360,
            anchor="w",
        ).pack(anchor="w")

        if body_text:
            tk.Label(
                outer,
                text=body_text,
                justify="left",
                background=p.tooltip_bg,
                foreground=p.tooltip_muted,
                font=("Segoe UI", 8),
                wraplength=360,
                anchor="w",
                pady=2,
            ).pack(anchor="w", pady=(3, 0))

        # Clamp tooltip inside current screen bounds so edges are never cut off
        tw.update_idletasks()
        try:
            req_w = max(tw.winfo_reqwidth(), 120)
            req_h = max(tw.winfo_reqheight(), 40)
            scr_w = self.widget.winfo_screenwidth()
            scr_h = self.widget.winfo_screenheight()
            if scr_w > 0:
                x = max(8, min(x, scr_w - req_w - 8))
            if scr_h > 0 and y + req_h > scr_h - 8:
                y = max(8, self.widget.winfo_rooty() - req_h - 6)
        except tk.TclError:
            pass

        tw.wm_geometry(f"+{x}+{y}")
        self.tip_window = tw

    def hide(self, _event: Optional[tk.Event] = None) -> None:
        _cancel_after(self.widget, self._after_id)
        self._after_id = None
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None


class CollapsibleSection(ttk.Frame):
    """Studio accordion drawer with chevron toggle, optional badge pill, and theme support."""

    def __init__(
        self,
        master: tk.Misc,
        title: str,
        expanded: bool = False,
        palette: Optional[ThemePalette] = None,
        card_surface: bool = True,
        on_toggle: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self.palette = palette or THEMES["light"]
        self._card_surface = card_surface
        super().__init__(master, style="Card.TFrame" if card_surface else "App.TFrame")
        self._title = title
        self._expanded = bool(expanded)
        self._on_toggle = on_toggle

        bg = self.palette.card if card_surface else self.palette.bg
        self.header_frame = tk.Frame(self, background=bg, cursor="hand2")
        self.header_frame.pack(fill="x")

        self.chevron_lbl = tk.Label(
            self.header_frame,
            text="▾" if self._expanded else "▸",
            background=bg,
            foreground=self.palette.text_muted,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.chevron_lbl.pack(side="left", padx=(0, 6))

        self.title_lbl = tk.Label(
            self.header_frame,
            text=title,
            background=bg,
            foreground=self.palette.text,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.title_lbl.pack(side="left")

        self.badge_lbl = tk.Label(
            self.header_frame,
            text="",
            background=self.palette.accent_soft,
            foreground=self.palette.accent,
            font=("Segoe UI", 8, "bold"),
            padx=7,
            pady=1,
            cursor="hand2",
        )

        for w in (self.header_frame, self.chevron_lbl, self.title_lbl, self.badge_lbl):
            w.bind("<Button-1>", self._on_click, add="+")

        self.content = ttk.Frame(self, style="Card.TFrame" if card_surface else "App.TFrame")
        if self._expanded:
            self.content.pack(fill="both", expand=True, pady=(8, 0))

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def _on_click(self, _event: Optional[tk.Event] = None) -> None:
        self.toggle()

    def set_expanded(self, expanded: bool) -> None:
        want = bool(expanded)
        if want == self._expanded:
            return
        self._expanded = want
        self.chevron_lbl.configure(text="▾" if self._expanded else "▸")
        if self._expanded:
            self.content.pack(fill="both", expand=True, pady=(8, 0))
        else:
            self.content.pack_forget()
        if self._on_toggle is not None:
            self._on_toggle(self._expanded)

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_badge(self, text: str, tone: str = "accent") -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            self.badge_lbl.pack_forget()
            return
        p = self.palette
        if tone == "danger":
            bg, fg = p.danger_soft, p.danger
        elif tone == "warn":
            bg, fg = p.warn_bg, p.warn_text
        else:
            bg, fg = p.accent_soft, p.accent
        self.badge_lbl.configure(text=cleaned, background=bg, foreground=fg)
        if not self.badge_lbl.winfo_manager():
            self.badge_lbl.pack(side="left", padx=(8, 0))

    def apply_theme(self, palette: ThemePalette) -> None:
        self.palette = palette
        bg = palette.card if self._card_surface else palette.bg
        self.configure(style="Card.TFrame" if self._card_surface else "App.TFrame")
        self.content.configure(style="Card.TFrame" if self._card_surface else "App.TFrame")
        self.header_frame.configure(background=bg)
        self.chevron_lbl.configure(background=bg, foreground=palette.text_muted)
        self.title_lbl.configure(background=bg, foreground=palette.text)


class PdfPreviewPane(tk.Frame):
    """Elevated left-column card displaying live sheet-music page previews and page picking."""

    PREVIEW_W = 240
    PREVIEW_H = 310
    CACHE_LIMIT = 16

    def __init__(
        self,
        master: tk.Misc,
        on_browse: Callable[[], None],
        on_toggle_page: Callable[[int], None],
        page_renderer: Callable[[Path, int, tuple[int, int]], Image.Image] = render_pdf_page_image,
        palette: Optional[ThemePalette] = None,
    ) -> None:
        self.palette = palette or THEMES["light"]
        super().__init__(
            master,
            background=self.palette.card,
            highlightbackground=self.palette.border,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        self.on_browse = on_browse
        self.on_toggle_page = on_toggle_page
        self.page_renderer = page_renderer

        self.pdf: Optional[Path] = None
        self.total_pages: int = 0
        self.current_page: int = 1
        self._busy = False
        self._selected_pages: Optional[set[int]] = None
        self._cache: "OrderedDict[tuple[Path, int], Image.Image]" = OrderedDict()
        self._photo_ref: Optional[ImageTk.PhotoImage] = None
        self._emblem_ref: Optional[ImageTk.PhotoImage] = None

        self._build()
        self.apply_theme(self.palette)

    def _build(self) -> None:
        self.header_lbl = ttk.Label(self, text="SHEET MUSIC PREVIEW", style="SectionHeader.TLabel")
        self.header_lbl.pack(anchor="w", pady=(0, 8))

        self.stage = tk.Frame(
            self,
            width=self.PREVIEW_W,
            height=self.PREVIEW_H,
            highlightthickness=1,
            cursor="hand2",
        )
        self.stage.pack_propagate(False)
        self.stage.pack()

        # Empty dropzone frame
        self.empty_frame = tk.Frame(self.stage, cursor="hand2")
        self.empty_icon_lbl = tk.Label(self.empty_frame, text="♪", font=("Segoe UI", 26, "bold"), cursor="hand2")
        try:
            png = get_assets_dir() / "icon.png"
            if png.is_file():
                emblem = Image.open(png).convert("RGBA")
                emblem.thumbnail((52, 52), Image.Resampling.LANCZOS)
                self._emblem_ref = ImageTk.PhotoImage(emblem, master=self)
                self.empty_icon_lbl.configure(image=self._emblem_ref, text="")
        except Exception as exc:  # pragma: no cover - defensive icon fallback
            log.debug("Preview emblem fallback: %s", exc)
        self.empty_icon_lbl.pack(pady=(72, 10))

        self.empty_title_lbl = tk.Label(
            self.empty_frame,
            text="Select or Drop Sheet Music",
            font=("Segoe UI", 9, "bold"),
            justify="center",
            wraplength=self.PREVIEW_W - 24,
            cursor="hand2",
        )
        self.empty_title_lbl.pack()

        self.empty_sub_lbl = tk.Label(
            self.empty_frame,
            text="Drop a PDF or photo here,\nclick, or press Ctrl+O",
            font=("Segoe UI", 8),
            justify="center",
            wraplength=self.PREVIEW_W - 20,
            cursor="hand2",
        )
        self.empty_sub_lbl.pack(pady=(6, 0))

        for w in (self.stage, self.empty_frame, self.empty_icon_lbl, self.empty_title_lbl, self.empty_sub_lbl):
            w.bind("<Button-1>", self._handle_stage_click, add="+")

        # Image / fallback label for loaded PDFs
        self.image_lbl = tk.Label(self.stage, cursor="arrow", justify="center", wraplength=self.PREVIEW_W - 24)

        # Bottom navigation bar
        nav = ttk.Frame(self, style="Card.TFrame")
        nav.pack(fill="x", pady=(10, 4))
        self.prev_btn = ttk.Button(nav, text="◀", width=3, style="Secondary.TButton", command=self.go_prev)
        self.prev_btn.pack(side="left")
        self.page_lbl = ttk.Label(nav, text="No PDF loaded", style="Muted.TLabel", anchor="center")
        self.page_lbl.pack(side="left", fill="x", expand=True, padx=4)
        self.next_btn = ttk.Button(nav, text="▶", width=3, style="Secondary.TButton", command=self.go_next)
        self.next_btn.pack(side="right")

        self.pick_btn = ttk.Button(
            self,
            text="Include page in range",
            style="Secondary.TButton",
            command=lambda: self.on_toggle_page(self.current_page),
        )
        self.pick_btn.pack(fill="x", pady=(4, 0))

        self._show_empty()
        self._sync_controls()

    def _handle_stage_click(self, _event: Optional[tk.Event] = None) -> None:
        if not self._busy and self.pdf is None:
            self.on_browse()

    def _show_empty(self) -> None:
        self.image_lbl.pack_forget()
        self.empty_frame.pack(fill="both", expand=True)
        self.stage.configure(cursor="hand2")

    def _show_loaded(self) -> None:
        self.empty_frame.pack_forget()
        self.image_lbl.pack(fill="both", expand=True, padx=4, pady=4)
        self.stage.configure(cursor="arrow")

    def set_pdf(self, pdf: Optional[Path], total_pages: int = 0) -> None:
        self.pdf = pdf
        self.total_pages = max(0, int(total_pages))
        self.current_page = 1
        self._cache.clear()
        self._photo_ref = None
        if pdf is None or self.total_pages <= 0:
            self._show_empty()
            self.page_lbl.configure(text="No PDF loaded")
        else:
            self._show_loaded()
            self._render_current_page()
        self._sync_controls()

    def _render_current_page(self) -> None:
        if self.pdf is None or self.total_pages <= 0:
            return
        self.page_lbl.configure(text=f"Page {self.current_page} of {self.total_pages}")
        key = (self.pdf, self.current_page)
        pil_img = self._cache.get(key)
        if pil_img is None:
            try:
                pil_img = self.page_renderer(
                    self.pdf,
                    self.current_page,
                    (self.PREVIEW_W - 14, self.PREVIEW_H - 14),
                )
                self._cache[key] = pil_img
                while len(self._cache) > self.CACHE_LIMIT:
                    self._cache.popitem(last=False)
            except Exception as exc:
                log.debug("Preview render fallback for %s page %s: %s", self.pdf, self.current_page, exc)
                self._photo_ref = None
                self.image_lbl.configure(
                    image="",
                    text=f"{self.pdf.name}\n\nPage {self.current_page} of {self.total_pages}",
                    font=("Segoe UI", 9, "bold"),
                )
                return
        else:
            self._cache.move_to_end(key)

        photo = ImageTk.PhotoImage(pil_img, master=self)
        self._photo_ref = photo
        self.image_lbl.configure(image=photo, text="")

    def goto_page(self, page: int) -> None:
        if self.pdf is None or self.total_pages <= 0:
            return
        clamped = max(1, min(self.total_pages, int(page)))
        if clamped != self.current_page:
            self.current_page = clamped
            self._render_current_page()
            self._sync_controls()

    def go_prev(self) -> None:
        if self.pdf is not None and self.current_page > 1:
            self.goto_page(self.current_page - 1)

    def go_next(self) -> None:
        if self.pdf is not None and self.current_page < self.total_pages:
            self.goto_page(self.current_page + 1)

    def update_selection_state(self, selected_pages: Optional[set[int]]) -> None:
        self._selected_pages = set(selected_pages) if selected_pages is not None else None
        self._sync_controls()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._sync_controls()

    def _sync_controls(self) -> None:
        has_pdf = self.pdf is not None and self.total_pages > 0
        nav_prev = "normal" if (has_pdf and not self._busy and self.current_page > 1) else "disabled"
        nav_next = "normal" if (has_pdf and not self._busy and self.current_page < self.total_pages) else "disabled"
        self.prev_btn.configure(state=nav_prev)
        self.next_btn.configure(state=nav_next)

        if not has_pdf:
            self.pick_btn.configure(text="Include page in range", style="Secondary.TButton", state="disabled")
            return

        pick_state = "disabled" if self._busy else "normal"
        if self._selected_pages is not None and self.current_page in self._selected_pages:
            self.pick_btn.configure(
                text=f"✓ p.{self.current_page} in custom range",
                style="PresetActive.TButton",
                state=pick_state,
            )
        else:
            self.pick_btn.configure(
                text=f"+ Pick p.{self.current_page} only" if self._selected_pages is None else f"+ Include p.{self.current_page}",
                style="Secondary.TButton",
                state=pick_state,
            )

    def apply_theme(self, palette: ThemePalette) -> None:
        self.palette = palette
        self.configure(background=palette.card, highlightbackground=palette.border)
        self.stage.configure(background=palette.preview_bg, highlightbackground=palette.border)
        self.empty_frame.configure(background=palette.preview_bg)
        self.empty_icon_lbl.configure(background=palette.preview_bg, foreground=palette.accent)
        self.empty_title_lbl.configure(background=palette.preview_bg, foreground=palette.text)
        self.empty_sub_lbl.configure(background=palette.preview_bg, foreground=palette.text_muted)
        self.image_lbl.configure(background=palette.preview_bg, foreground=palette.text_muted)

    def destroy(self) -> None:
        self._photo_ref = None
        self._emblem_ref = None
        self._cache.clear()
        super().destroy()
