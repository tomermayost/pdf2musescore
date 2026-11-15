"""Conversion parameters (Section 2: Engine/Preprocessing & Section 3: Score Post-Processing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PREPROCESS_MODES = ("auto", "faded", "binarize", "none")
CLEF_CHOICES = ("auto", "treble", "bass", "alto", "tenor")
INSTRUMENT_PRESETS = ("auto", "Piano", "Trombone", "Violin", "Flute", "Trumpet", "Cello", "Clarinet", "Guitar")


@dataclass(frozen=True)
class ParameterDoc:
    key: str
    section: str
    label: str
    default: str
    examples: str
    description: str


PARAMETER_DOCS: tuple[ParameterDoc, ...] = (
    # Section 2: PDF Rendering, Image Preprocessing & AI Engine
    ParameterDoc(
        key="render_dpi",
        section="2. PDF Rendering & AI Engine",
        label="Render DPI",
        default="300",
        examples="200 (fast), 300 (standard), 400 (small/dense staves)",
        description=(
            "Resolution (dots per inch, 150-600) used to rasterize each PDF page before AI recognition. "
            "Higher DPI (400) helps scores with tiny staves or grace notes; 200 DPI is faster."
        ),
    ),
    ParameterDoc(
        key="preprocess_mode",
        section="2. PDF Rendering & AI Engine",
        label="Image Preprocessing",
        default="auto",
        examples="auto, faded, binarize, none",
        description=(
            "Enhances page contrast before the AI reads the score:\n"
            "  - auto: Standard CLAHE local contrast (best for clean printed PDFs).\n"
            "  - faded: Boosts contrast, darkens faint ink, and sharpens staff lines (best for old/faded scans).\n"
            "  - binarize: Adaptive black-and-white thresholding (best for phone photos with uneven lighting).\n"
            "  - none: Raw page image without pre-enhancement."
        ),
    ),
    ParameterDoc(
        key="deskew",
        section="2. PDF Rendering & AI Engine",
        label="Auto-Deskew",
        default="False",
        examples="False (off), True (straighten tilted scans)",
        description=(
            "Automatically measures staff-line tilt on scanned pages (up to +/-10 degrees) and rotates "
            "the page so staff lines are horizontal before recognition."
        ),
    ),
    ParameterDoc(
        key="segnet_step_size",
        section="2. PDF Rendering & AI Engine",
        label="SegNet Tile Step (px)",
        default="320",
        examples="160 (max overlap/accuracy), 240 (balanced), 320 (fast default)",
        description=(
            "Sliding-window step size (80-320 pixels) for the 320x320 neural segmentation network. "
            "Smaller step sizes (e.g. 240 or 160) overlap adjacent tiles for smoother symbol edges at the cost of speed."
        ),
    ),
    ParameterDoc(
        key="min_notehead_size",
        section="2. PDF Rendering & AI Engine",
        label="Min Notehead Size (px)",
        default="4",
        examples="3 (catch tiny/faint notes), 4 (default), 6 (ignore paper noise/specks)",
        description=(
            "Minimum pixel width/height (2-12) for a detected blob to be accepted as a notehead. "
            "Lower values (3) catch very small or faint grace notes; higher values (6) filter out dust specks."
        ),
    ),
    # Section 3: Score Post-Processing (MusicXML / MuseScore)
    ParameterDoc(
        key="transpose_semitones",
        section="3. Score Post-Processing",
        label="Transpose (Semitones)",
        default="0",
        examples="0 (none), +2 (Concert -> Bb Trumpet/Tenor Sax), -12 (down 1 octave), +12 (up 1 octave)",
        description=(
            "Transposes all notes and key signatures in the output score by N semitones (-24 to +24). "
            "Use +2 for Bb instruments (Trumpet, Clarinet, Tenor Sax), +9 or -3 for Eb instruments, or +/-12 for octaves."
        ),
    ),
    ParameterDoc(
        key="clef_override",
        section="3. Score Post-Processing",
        label="Clef Override",
        default="auto",
        examples="auto, treble, bass, alto, tenor",
        description=(
            "Keeps the AI-detected clef (auto) or forces the primary staff clef across the score: "
            "treble (G2), bass (F4, e.g. Trombone/Cello), alto (C3, Viola), or tenor (C4, Trombone/Cello/Bassoon)."
        ),
    ),
    ParameterDoc(
        key="instrument_name",
        section="3. Score Post-Processing",
        label="Instrument",
        default="auto",
        examples="auto, Piano, Trombone, Violin, Flute, Trumpet, Cello",
        description=(
            "Sets the <part-name> and <instrument-name> in the generated MusicXML/MuseScore file so MuseScore "
            "labels the staff and selects the matching playback instrument."
        ),
    ),
    ParameterDoc(
        key="title_override",
        section="3. Score Post-Processing",
        label="Title Override",
        default="",
        examples="Telemann - Sonata in F Minor (leave blank for auto)",
        description=(
            "Custom score title written to the top of the output score. Leave blank to automatically use "
            "the PDF filename / OCR title."
        ),
    ),
)


class OptionsError(ValueError):
    """Raised when a conversion option is outside its valid range."""


@dataclass(frozen=True)
class ConversionOptions:
    # Section 2: PDF Rendering, Image Preprocessing & AI Engine
    render_dpi: int = 300
    preprocess_mode: str = "auto"
    deskew: bool = False
    segnet_step_size: int = 320
    min_notehead_size: int = 4
    # Section 3: Score Post-Processing
    transpose_semitones: int = 0
    clef_override: str = "auto"
    instrument_name: str = "auto"
    title_override: Optional[str] = None

    def __post_init__(self) -> None:
        if not (100 <= int(self.render_dpi) <= 600):
            raise OptionsError(f"Render DPI must be between 100 and 600 (got {self.render_dpi})")
        mode = (self.preprocess_mode or "auto").strip().lower()
        if mode not in PREPROCESS_MODES:
            raise OptionsError(f"preprocess_mode must be one of {PREPROCESS_MODES} (got {self.preprocess_mode!r})")
        object.__setattr__(self, "preprocess_mode", mode)

        if not (80 <= int(self.segnet_step_size) <= 320):
            raise OptionsError(f"SegNet Tile Step must be between 80 and 320 px (got {self.segnet_step_size})")
        if not (2 <= int(self.min_notehead_size) <= 12):
            raise OptionsError(f"Min Notehead Size must be between 2 and 12 px (got {self.min_notehead_size})")
        if not (-24 <= int(self.transpose_semitones) <= 24):
            raise OptionsError(
                f"Transpose semitones must be between -24 and +24 (got {self.transpose_semitones})"
            )

        clef = (self.clef_override or "auto").strip().lower()
        if clef not in CLEF_CHOICES:
            raise OptionsError(f"Clef override must be one of {CLEF_CHOICES} (got {self.clef_override!r})")
        object.__setattr__(self, "clef_override", clef)

        inst = (self.instrument_name or "auto").strip()
        object.__setattr__(self, "instrument_name", inst if inst else "auto")

        title = (self.title_override or "").strip()
        object.__setattr__(self, "title_override", title if title else None)


def format_parameter_guide() -> str:
    """Return a human-readable guide covering all parameters, defaults, examples, and explanations."""
    lines: list[str] = []
    current_section = ""
    for doc in PARAMETER_DOCS:
        if doc.section != current_section:
            if lines:
                lines.append("")
            lines.append(f"=== {doc.section} ===")
            current_section = doc.section
        default_disp = doc.default if doc.default != "" else "(blank / auto)"
        lines.append(f"\n• {doc.label}  [Default: {default_disp}]")
        lines.append(f"  Examples: {doc.examples}")
        lines.append(f"  {doc.description}")
    return "\n".join(lines)
