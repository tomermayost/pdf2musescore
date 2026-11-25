from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from pdf2musescore.fullscore.staves import flatten_filter_and_extend_staves


@dataclass
class FakeStaff:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    average_unit_size: float = 12.0
    is_grandstaff: bool = False
    symbols: list[str] = field(default_factory=list)
    split_parts: list["FakeStaff"] | None = None

    def split(self):
        return self.split_parts

    def extend_to_x_range(self, min_x: int, max_x: int) -> "FakeStaff":
        # Mimics homr.model.Staff.extend_to_x_range: returns a new Staff WITHOUT symbols!
        return FakeStaff(
            min_x=min(min_x, self.min_x),
            max_x=max(max_x, self.max_x),
            min_y=self.min_y,
            max_y=self.max_y,
            average_unit_size=self.average_unit_size,
            is_grandstaff=False,
            symbols=[],
        )


def test_flatten_empty_and_all_narrow():
    empty_res = flatten_filter_and_extend_staves([])
    assert empty_res.kept_staffs == []
    assert empty_res.staff_cys == []

    all_zero = flatten_filter_and_extend_staves(
        [SimpleNamespace(staffs=[FakeStaff(0, 0, 10, 50)])],
        min_width_ratio=2.0,  # impossible ratio when median is 0 -> min_allowed=0 so 0 >= 0; use negative width
    )
    assert len(all_zero.kept_staffs) == 1

    neg_w = flatten_filter_and_extend_staves([SimpleNamespace(staffs=[FakeStaff(100, 50, 10, 50)])])
    assert neg_w.kept_staffs == []


def test_flatten_splits_grandstaff_filters_dedupes_and_preserves_symbols(caplog):
    gs_child_1 = FakeStaff(460, 3540, 200, 260, symbols=["s1"])
    gs_child_2 = FakeStaff(460, 3540, 400, 460, symbols=["s2"])
    grand = FakeStaff(460, 3540, 200, 460, is_grandstaff=True, split_parts=[gs_child_1, gs_child_2])
    grand_none = FakeStaff(460, 3540, 600, 660, is_grandstaff=True, split_parts=None, symbols=["gn"])

    # Narrow fragment (should be dropped)
    narrow_frag = FakeStaff(1000, 1400, 500, 550)
    # Short solo staff (should be extended and keep its symbols)
    short_solo = FakeStaff(1280, 3540, 100, 150, symbols=["solo_note_1", "solo_note_2"])
    # Overlapping duplicate at y~400 (narrower than gs_child_2, so dropped; and a wider one at y~800)
    dup_narrow = FakeStaff(500, 3400, 405, 465)
    base_800 = FakeStaff(500, 3200, 800, 860, symbols=["old"])
    dup_wider = FakeStaff(460, 3540, 804, 864, symbols=["wider"])
    # Single large gap at bottom to trigger multi-system warning
    far_bottom = FakeStaff(460, 3540, 2200, 2260)

    multi = [
        SimpleNamespace(staffs=[short_solo, grand, narrow_frag]),
        SimpleNamespace(staffs=[dup_narrow, grand_none, base_800, dup_wider, far_bottom]),
    ]

    with caplog.at_level("WARNING"):
        res = flatten_filter_and_extend_staves(
            multi,
            to_input_image=lambda pt: (pt[0] * 1.1, pt[1] * 1.1),
        )

    assert res.dropped_narrow_count == 1
    assert res.deduped_overlap_count == 2
    assert res.extended_count == 1
    assert len(res.kept_staffs) == 6
    assert res.kept_staffs[0].min_x == 460
    assert res.kept_staffs[0].max_x == 3540
    assert res.kept_staffs[0].symbols == ["solo_note_1", "solo_note_2"]
    assert all(not s.is_grandstaff for s in res.kept_staffs)
    # Wider duplicate won at y~800
    assert any(s.symbols == ["wider"] for s in res.kept_staffs)
    assert res.staff_y_norms[0] == 0.0 and res.staff_y_norms[-1] == 1.0
    assert res.crop_x >= 100
    assert "Large vertical gap detected" in caplog.text


def test_single_staff_y_norm_and_default_gap():
    res = flatten_filter_and_extend_staves([SimpleNamespace(staffs=[FakeStaff(100, 2000, 300, 360)])])
    assert res.staff_y_norms == [0.5]
    assert res.median_gap == 120.0
