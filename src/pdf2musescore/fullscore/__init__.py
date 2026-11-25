"""Full Score Mode package: isolated multi-staff OMR, instrument OCR, and alignment."""

from __future__ import annotations

from .align import (
    PageSidecar,
    StaffMatchSpec,
    align_page_score,
    apply_conductor_metadata,
    compute_conductor_scaling,
    dedup_boundary_attributes,
    match_staves_monotonic_dp,
    merge_fullscore_files,
    merge_fullscore_pages,
    read_page_sidecar,
    reconcile_pages_to_master,
    sidecar_path_for,
    verify_fullscore_invariants,
    write_page_sidecar,
)
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
from .runner import run_fullscore_homr
from .staves import StaffLayoutResult, flatten_filter_and_extend_staves

__all__ = [
    "DEFAULT_TARGET_WIDTH",
    "MAX_TARGET_HEIGHT",
    "InstrumentLabel",
    "PageSidecar",
    "StaffLayoutResult",
    "StaffMatchSpec",
    "abbreviate_instrument_name",
    "align_page_score",
    "apply_conductor_metadata",
    "calc_fullscore_image_size",
    "clean_instrument_label",
    "compute_conductor_scaling",
    "consolidate_page_labels",
    "dedup_boundary_attributes",
    "expand_abbreviated_name",
    "extract_instrument_labels",
    "flatten_filter_and_extend_staves",
    "fullscore_homr_context",
    "is_percussion_instrument",
    "looks_abbreviated_label",
    "looks_low_quality_label",
    "match_staves_monotonic_dp",
    "merge_fullscore_files",
    "merge_fullscore_pages",
    "read_page_sidecar",
    "reconcile_pages_to_master",
    "run_fullscore_homr",
    "sidecar_path_for",
    "verify_fullscore_invariants",
    "write_page_sidecar",
]
