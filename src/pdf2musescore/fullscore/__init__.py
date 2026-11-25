"""Full Score Mode package: isolated multi-staff OMR, instrument OCR, and alignment."""

from __future__ import annotations

from .homr_patch import (
    DEFAULT_TARGET_WIDTH,
    MAX_TARGET_HEIGHT,
    calc_fullscore_image_size,
    fullscore_homr_context,
)
from .instruments import (
    InstrumentLabel,
    abbreviate_instrument_name,
    clean_instrument_label,
    consolidate_page_labels,
    expand_abbreviated_name,
    extract_instrument_labels,
    is_percussion_instrument,
    looks_abbreviated_label,
    looks_low_quality_label,
)
from .staves import StaffLayoutResult, flatten_filter_and_extend_staves

__all__ = [
    "DEFAULT_TARGET_WIDTH",
    "MAX_TARGET_HEIGHT",
    "InstrumentLabel",
    "StaffLayoutResult",
    "abbreviate_instrument_name",
    "calc_fullscore_image_size",
    "clean_instrument_label",
    "consolidate_page_labels",
    "expand_abbreviated_name",
    "extract_instrument_labels",
    "flatten_filter_and_extend_staves",
    "fullscore_homr_context",
    "is_percussion_instrument",
    "looks_abbreviated_label",
    "looks_low_quality_label",
]
