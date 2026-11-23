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
        label="Scan Quality (DPI)",
        default="300",
        examples="200 (faster), 300 (normal), 400 (tiny or crowded notes)",
        description=(
            "Controls how closely the app zooms in on each PDF page (100-600). "
            "Keep at 300 for normal sheet music. Try 400 if the notes are very small or crowded together, "
            "or 200 if you want a faster conversion."
        ),
    ),
    ParameterDoc(
        key="preprocess_mode",
        section="2. PDF Rendering & AI Engine",
        label="Page Cleanup",
        default="auto",
        examples="auto, faded, binarize, none",
        description=(
            "Helps the app read difficult scans:\n"
            "  - auto: Best for normal, clean PDFs.\n"
            "  - faded: Darkens light or washed-out ink (great for old photocopies or faded scans).\n"
            "  - binarize: Turns the page pure black & white (great for phone camera photos with shadows).\n"
            "  - none: Uses the page picture as-is without any cleanup."
        ),
    ),
    ParameterDoc(
        key="deskew",
        section="2. PDF Rendering & AI Engine",
        label="Straighten Tilted Page",
        default="False",
        examples="Off (normal), On (straighten crooked scans)",
        description=(
            "Check this box if the sheet music was scanned or photographed slightly crooked. "
            "The app will automatically straighten the staff lines before reading the notes."
        ),
    ),
    ParameterDoc(
        key="segnet_step_size",
        section="2. PDF Rendering & AI Engine",
        label="Reading Thoroughness (px)",
        default="320",
        examples="320 (normal speed), 240 (more careful), 160 (most careful, slower)",
        description=(
            "The app reads the page in overlapping squares (80-320). Lower numbers (240 or 160) make the "
            "squares overlap more so notes near the edges aren't missed - helpful on crowded pages, though "
            "it takes a little longer."
        ),
    ),
    ParameterDoc(
        key="min_notehead_size",
        section="2. PDF Rendering & AI Engine",
        label="Smallest Note Size (px)",
        default="4",
        examples="3 (keep tiny grace notes), 4 (normal), 6 (ignore paper dust/specks)",
        description=(
            "Controls the smallest black dot (2-12) the app treats as a musical note. "
            "Lower to 3 if small grace notes are being missed; raise to 6 if dusty spots on the paper "
            "are turning into extra fake notes."
        ),
    ),
    # Section 3: Score Post-Processing (MusicXML / MuseScore)
    ParameterDoc(
        key="transpose_semitones",
        section="3. Score Post-Processing",
        label="Transpose (Half-Steps)",
        default="0",
        examples="0 (keep original), +2 (for Bb Trumpet/Clarinet/Tenor Sax), -12 (down 1 octave), +12 (up 1 octave)",
        description=(
            "Moves every note and the key signature up or down by the number of half-steps (semitones, -24 to +24) "
            "you enter."
        ),
    ),
    ParameterDoc(
        key="clef_override",
        section="3. Score Post-Processing",
        label="Clef",
        default="auto",
        examples="auto, treble, bass, alto, tenor",
        description=(
            "Keep at auto to use whatever clef is printed on the page, or pick a clef (bass or tenor for "
            "Trombone/Cello, treble for Violin/Flute, alto for Viola) if the scan misread the clef sign."
        ),
    ),
    ParameterDoc(
        key="instrument_name",
        section="3. Score Post-Processing",
        label="Instrument",
        default="auto",
        examples="auto, Piano, Trombone, Violin, Flute, Trumpet, Cello",
        description=(
            "Sets the instrument name and playback sound in MuseScore. Pick one from the list or type any "
            "instrument name you like."
        ),
    ),
    ParameterDoc(
        key="title_override",
        section="3. Score Post-Processing",
        label="Custom Title",
        default="",
        examples="Telemann - Sonata in F Minor (leave blank for automatic)",
        description=(
            "Type a title here if you want to set the heading at the top of the sheet music. "
            "Leave blank to use the PDF's title automatically."
        ),
    ),
    ParameterDoc(
        key="keep_line_breaks",
        section="3. Score Post-Processing",
        label="Keep Original Line Breaks",
        default="False",
        examples="Off (let MuseScore arrange lines automatically), On (lock line/page breaks to the PDF)",
        description=(
            "When turned off (recommended), removes forced system and page breaks so MuseScore flows "
            "measures neatly across lines and pages. Turn on only if you want every line and page to break "
            "in the exact same places as the original PDF."
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
    keep_line_breaks: bool = False

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


def get_parameter_tooltip(key: str) -> str:
    """Return a formatted tooltip string (title, default, examples, description) for ``key``."""
    for doc in PARAMETER_DOCS:
        if doc.key == key:
            default_disp = doc.default if doc.default != "" else "(blank / auto)"
            return (
                f"{doc.label}  (Default: {default_disp})\n"
                f"Examples: {doc.examples}\n\n"
                f"{doc.description}"
            )
    raise KeyError(f"Unknown parameter key: {key!r}")


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
