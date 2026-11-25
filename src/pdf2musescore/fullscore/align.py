"""Full Score alignment, variable-staff DP reconciliation, merging, metadata, and invariant verification."""

from __future__ import annotations

import copy
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

from ..musicxml import MergeError, read_score, strip_layout_breaks, write_musicxml
from ..options import ConversionOptions
from .instruments import (
    InstrumentLabel,
    consolidate_page_labels,
    is_percussion_instrument,
    looks_low_quality_label,
)

log = logging.getLogger(__name__)

_PITCHED_MIDI_CHANNELS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16)


@dataclass(frozen=True)
class PageSidecar:
    """Per-page full-score geometry and OCR labels persisted beside each page's MusicXML."""

    page_number: int
    staff_count: int
    staff_cys: list[float]
    staff_y_norms: list[float]
    labels: list[InstrumentLabel]
    sys_min_x: int = 0
    sys_max_x: int = 0
    crop_x: int = 0


def sidecar_path_for(xml_path: Path) -> Path:
    """Return the ``.fullscore.json`` sidecar path paired with ``xml_path``."""
    return Path(xml_path).with_suffix(".fullscore.json")


def write_page_sidecar(sidecar: PageSidecar, xml_path: Path) -> Path:
    """Persist ``sidecar`` JSON alongside ``xml_path``."""
    path = sidecar_path_for(xml_path)
    payload = {
        "page_number": sidecar.page_number,
        "staff_count": sidecar.staff_count,
        "staff_cys": list(sidecar.staff_cys),
        "staff_y_norms": list(sidecar.staff_y_norms),
        "labels": [asdict(lbl) for lbl in sidecar.labels],
        "sys_min_x": sidecar.sys_min_x,
        "sys_max_x": sidecar.sys_max_x,
        "crop_x": sidecar.crop_x,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.debug("Wrote full-score sidecar %s (%d staves)", path.name, sidecar.staff_count)
    return path


def read_page_sidecar(xml_path: Path) -> Optional[PageSidecar]:
    """Read sibling ``.fullscore.json`` sidecar if present and valid; otherwise return None."""
    path = sidecar_path_for(xml_path)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        labels = [
            InstrumentLabel(
                full_name=str(item["full_name"]),
                abbreviation=str(item["abbreviation"]),
                raw_ocr=str(item.get("raw_ocr", "")),
                is_fallback=bool(item.get("is_fallback", False)),
                is_abbreviated_source=bool(item.get("is_abbreviated_source", False)),
            )
            for item in raw.get("labels", [])
        ]
        return PageSidecar(
            page_number=int(raw["page_number"]),
            staff_count=int(raw["staff_count"]),
            staff_cys=[float(v) for v in raw.get("staff_cys", [])],
            staff_y_norms=[float(v) for v in raw.get("staff_y_norms", [])],
            labels=labels,
            sys_min_x=int(raw.get("sys_min_x", 0)),
            sys_max_x=int(raw.get("sys_max_x", 0)),
            crop_x=int(raw.get("crop_x", 0)),
        )
    except Exception as exc:
        log.warning("Could not read full-score sidecar %s (%s: %s) — falling back", path.name, type(exc).__name__, exc)
        return None


def _is_rest_only_measure(measure: ET.Element) -> bool:
    notes = measure.findall("note")
    if not notes:
        return True
    return all(n.find("rest") is not None for n in notes)


def _extract_last_time(measure: ET.Element) -> Optional[tuple[str, str]]:
    last_ts: Optional[tuple[str, str]] = None
    for attrs in measure.findall("attributes"):
        for t in attrs.findall("time"):
            b = (t.findtext("beats") or "").strip()
            bt = (t.findtext("beat-type") or "").strip()
            if b and bt:
                last_ts = (b, bt)
    return last_ts


def _remove_all_time_elements(measure: ET.Element) -> None:
    for attrs in list(measure.findall("attributes")):
        for t in list(attrs.findall("time")):
            attrs.remove(t)
        if len(attrs) == 0:
            measure.remove(attrs)


def _set_single_time(measure: ET.Element, beats: str, beat_type: str) -> None:
    _remove_all_time_elements(measure)
    attrs = measure.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        measure.insert(0, attrs)
    t_el = ET.SubElement(attrs, "time")
    ET.SubElement(t_el, "beats").text = str(beats)
    ET.SubElement(t_el, "beat-type").text = str(beat_type)


def _replace_measure_notes_with_rest(measure: ET.Element, expected_dur: int) -> None:
    for child in list(measure):
        if child.tag in ("note", "backup", "forward", "direction"):
            measure.remove(child)
    note = ET.Element("note")
    ET.SubElement(note, "rest", measure="yes")
    ET.SubElement(note, "duration").text = str(max(1, expected_dur))
    ET.SubElement(note, "voice").text = "1"
    ET.SubElement(note, "staff").text = "1"
    bar = measure.find("barline")
    if bar is not None:
        idx = list(measure).index(bar)
        measure.insert(idx, note)
    else:
        measure.append(note)


def _time_sig_quarters(ts: tuple[str, str]) -> Optional[Fraction]:
    try:
        b = int(ts[0])
        bt = int(ts[1])
    except (TypeError, ValueError):
        return None
    if b <= 0 or bt <= 0:
        return None
    return Fraction(b * 4, bt)


def _fraction_to_time_sig(q: Fraction) -> Optional[tuple[str, str]]:
    if q.denominator == 1 and 2 <= q.numerator <= 7:
        return (str(q.numerator), "4")
    if q.denominator == 2 and q.numerator in (3, 5, 7, 9, 11):
        return (str(q.numerator), "8")
    return None


def _measure_quarter_duration(measure: ET.Element, div: int) -> Optional[Fraction]:
    if _is_rest_only_measure(measure):
        return None
    voice_totals: dict[str, int] = {}
    for note in measure.findall("note"):
        if note.find("chord") is not None or note.find("grace") is not None:
            continue
        dur_txt = (note.findtext("duration") or "").strip()
        if not dur_txt.isdigit():
            continue
        dur = int(dur_txt)
        voice = (note.findtext("voice") or "1").strip() or "1"
        voice_totals[voice] = voice_totals.get(voice, 0) + dur
    if not voice_totals:
        return None
    chosen = voice_totals["1"] if "1" in voice_totals else max(voice_totals.values())
    if chosen <= 0:
        return None
    return Fraction(chosen, max(1, div))


def align_page_score(
    root: ET.Element,
    page_num: int,
    *,
    inherited_time_sig: Optional[tuple[str, str]] = None,
) -> tuple[ET.Element, tuple[str, str]]:
    """Harmonize measure counts and time signatures across all parts of a single page."""
    parts = root.findall("part")
    fallback_ts = inherited_time_sig or ("4", "4")
    if not parts:
        return root, fallback_ts

    part_measures_list = [p.findall("measure") for p in parts]
    counts = [len(ms) for ms in part_measures_list]
    count_freq = Counter(counts)
    pitched_by_count: Counter[int] = Counter()
    for ms, cnt in zip(part_measures_list, counts):
        if any(not _is_rest_only_measure(m) for m in ms):
            pitched_by_count[cnt] += 1

    target_measures = max(
        count_freq.keys(),
        key=lambda c: (count_freq[c], pitched_by_count[c], c),
    )
    log.info("Aligning page %d: part_measures_dist=%s, target_measures=%d", page_num, dict(sorted(count_freq.items())), target_measures)
    if target_measures == 0:
        return root, fallback_ts

    part_divs: list[list[int]] = []
    for ms in part_measures_list:
        divs_for_part: list[int] = []
        cur_div = 4
        for m in ms:
            for a in m.findall("attributes"):
                d_txt = a.findtext("divisions")
                if d_txt and d_txt.strip().isdigit():
                    cur_div = max(1, int(d_txt.strip()))
            divs_for_part.append(cur_div)
        part_divs.append(divs_for_part)

    global_ts: list[tuple[str, str]] = []
    active_global_ts: tuple[str, str] = fallback_ts
    min_playing = 2 if len(parts) <= 4 else 3
    for mi in range(target_measures):
        pitched_votes: Counter[tuple[str, str]] = Counter()
        all_votes: Counter[tuple[str, str]] = Counter()
        pitched_q_votes: Counter[Fraction] = Counter()
        for ms, divs in zip(part_measures_list, part_divs):
            if mi >= len(ms):
                continue
            m = ms[mi]
            q_dur = _measure_quarter_duration(m, divs[mi])
            if q_dur is not None:
                pitched_q_votes[q_dur] += 1
            ts = _extract_last_time(m)
            if ts is None:
                continue
            if mi == 0 or ts != active_global_ts:
                all_votes[ts] += 1
                if not _is_rest_only_measure(m):
                    pitched_votes[ts] += 1

        if mi == 0:
            votes = pitched_votes if pitched_votes else all_votes
            if votes:
                active_global_ts = max(
                    votes.keys(),
                    key=lambda ts: (
                        votes[ts],
                        pitched_q_votes.get(_time_sig_quarters(ts), 0)
                        if _time_sig_quarters(ts) is not None
                        else 0,
                    ),
                )
            else:
                active_global_ts = fallback_ts
        else:
            explicit_accepted = False
            if all_votes:
                top_ts, top_all = all_votes.most_common(1)[0]
                pitched_total_ts = sum(pitched_votes.values())
                top_pitched = pitched_votes.get(top_ts, 0)
                quorum = top_all >= max(2, int(0.25 * len(parts))) or (
                    pitched_total_ts >= 3 and top_pitched >= 0.4 * pitched_total_ts
                )
                if quorum:
                    active_global_ts = top_ts
                    explicit_accepted = True
                    log.info(
                        "Page %d measure %d: accepted global time-sig change -> %s/%s (all=%s, pitched=%s)",
                        page_num,
                        mi + 1,
                        top_ts[0],
                        top_ts[1],
                        dict(all_votes),
                        dict(pitched_votes),
                    )
                else:
                    log.debug(
                        "Page %d measure %d: ignored minority time-sig noise %s (kept %s/%s)",
                        page_num,
                        mi + 1,
                        dict(all_votes),
                        active_global_ts[0],
                        active_global_ts[1],
                    )
            if not explicit_accepted and pitched_q_votes:
                pitched_total_q = sum(pitched_q_votes.values())
                top_q, top_q_cnt = pitched_q_votes.most_common(1)[0]
                active_q = _time_sig_quarters(active_global_ts)
                cand_ts = _fraction_to_time_sig(top_q)
                if (
                    pitched_total_q >= min_playing
                    and cand_ts is not None
                    and active_q is not None
                    and top_q != active_q
                    and top_q_cnt >= min_playing
                    and top_q_cnt >= 0.65 * pitched_total_q
                    and pitched_q_votes.get(active_q, 0) <= 0.25 * pitched_total_q
                ):
                    active_global_ts = cand_ts
                    log.info(
                        "Page %d measure %d: inferred global time-sig change -> %s/%s from playing-part duration consensus (%d/%d parts = %s quarters)",
                        page_num,
                        mi + 1,
                        cand_ts[0],
                        cand_ts[1],
                        top_q_cnt,
                        pitched_total_q,
                        top_q,
                    )
        global_ts.append(active_global_ts)

    log.info("Page %d global_ts timeline: %s", page_num, [f"{b}/{bt}" for b, bt in global_ts])

    updated_rest_measures = 0
    padded_count = 0
    trimmed_count = 0

    for p in parts:
        ms = p.findall("measure")
        if len(ms) > target_measures:
            extras = ms[target_measures:]
            for em in extras:
                p.remove(em)
            trimmed_count += len(extras)
            ms = p.findall("measure")

        div = 4
        for mi, m in enumerate(ms):
            for a in m.findall("attributes"):
                d_txt = a.findtext("divisions")
                if d_txt and d_txt.strip().isdigit():
                    div = max(1, int(d_txt.strip()))

            b, bt = global_ts[mi]
            need_time = (mi == 0) or (global_ts[mi] != global_ts[mi - 1])
            if need_time:
                _set_single_time(m, b, bt)
            else:
                _remove_all_time_elements(m)

            if _is_rest_only_measure(m):
                expected_dur = int(round(div * int(b) * 4 / max(1, int(bt))))
                _replace_measure_notes_with_rest(m, expected_dur)
                updated_rest_measures += 1

        while len(p.findall("measure")) < target_measures:
            mi = len(p.findall("measure"))
            b, bt = global_ts[mi]
            expected_dur = int(round(div * int(b) * 4 / max(1, int(bt))))
            new_m = ET.Element("measure", number=str(mi + 1))
            if mi == 0 or global_ts[mi] != global_ts[mi - 1]:
                _set_single_time(new_m, b, bt)
            _replace_measure_notes_with_rest(new_m, expected_dur)
            p.append(new_m)
            padded_count += 1

    log.info(
        "Page %d alignment complete: updated_rest_measures=%d, padded=%d, trimmed=%d",
        page_num,
        updated_rest_measures,
        padded_count,
        trimmed_count,
    )
    return root, global_ts[-1]


@dataclass(frozen=True)
class StaffMatchSpec:
    """Signature of one part/staff used for cross-page monotonic alignment."""

    label: InstrumentLabel
    clef: Optional[tuple[str, str]]
    fifths: Optional[str]
    y_norm: float
    pitched_notes: int = 0


def _extract_part_signature(part: ET.Element) -> tuple[Optional[tuple[str, str]], Optional[str], int]:
    clef: Optional[tuple[str, str]] = None
    fifths: Optional[str] = None
    for attr in part.findall("./measure/attributes"):
        if clef is None:
            for c_el in attr.findall("clef"):
                if c_el.get("number") in (None, "1"):
                    sign = (c_el.findtext("sign") or "").strip()
                    line = (c_el.findtext("line") or "").strip()
                    if sign and line:
                        clef = (sign, line)
                        break
        if fifths is None:
            k_el = attr.find("key/fifths")
            if k_el is not None and k_el.text is not None and k_el.text.strip():
                fifths = k_el.text.strip()
        if clef is not None and fifths is not None:
            break
    pitched = sum(1 for n in part.findall(".//note") if n.find("rest") is None)
    return clef, fifths, pitched


def _label_tokens(label: InstrumentLabel) -> set[str]:
    text = f"{label.full_name} {label.abbreviation}".lower()
    return {tok for tok in re.findall(r"[a-z0-9]+", text) if len(tok) >= 2}


def _name_similarity(a: InstrumentLabel, b: InstrumentLabel) -> float:
    if a.is_fallback or b.is_fallback:
        return 0.0
    ta = _label_tokens(a)
    tb = _label_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _pair_alignment_score(item: StaffMatchSpec, master: StaffMatchSpec) -> float:
    score = 1.0 - abs(item.y_norm - master.y_norm)
    sim = _name_similarity(item.label, master.label)
    if not item.label.is_fallback and not master.label.is_fallback:
        score += 4.0 * sim
    if item.clef is not None and master.clef is not None:
        score += 1.2 if item.clef == master.clef else -0.8
    if item.fifths is not None and master.fifths is not None and item.fifths == master.fifths:
        score += 0.6
    return score


def match_staves_monotonic_dp(
    page_items: Sequence[StaffMatchSpec],
    master_specs: Sequence[StaffMatchSpec],
) -> list[int]:
    """Map ``K <= M`` page staves to strictly increasing master indices ``0..M-1``."""
    k = len(page_items)
    m = len(master_specs)
    if k == 0:
        return []
    if k > m:
        raise ValueError(f"Cannot monotonically map {k} items into {m} master slots")
    if k == m:
        return list(range(m))

    neg_inf = float("-inf")
    dp = [[neg_inf] * m for _ in range(k)]
    parent = [[-1] * m for _ in range(k)]

    for i in range(m - k + 1):
        dp[0][i] = _pair_alignment_score(page_items[0], master_specs[i])

    for j in range(1, k):
        best_prev_val = neg_inf
        best_prev_idx = -1
        for i in range(j, m - (k - 1 - j)):
            if dp[j - 1][i - 1] > best_prev_val:
                best_prev_val = dp[j - 1][i - 1]
                best_prev_idx = i - 1
            if best_prev_idx >= 0:
                dp[j][i] = best_prev_val + _pair_alignment_score(page_items[j], master_specs[i])
                parent[j][i] = best_prev_idx

    last_idx = max(range(k - 1, m), key=lambda i: dp[k - 1][i])
    mapping = [0] * k
    cur = last_idx
    for j in range(k - 1, -1, -1):
        mapping[j] = cur
        cur = parent[j][cur]
    return mapping


def _build_specs_for_page(
    root: ET.Element,
    sidecar: Optional[PageSidecar],
) -> list[StaffMatchSpec]:
    parts = root.findall("part")
    n = len(parts)
    specs: list[StaffMatchSpec] = []
    for idx, part in enumerate(parts):
        clef, fifths, pitched = _extract_part_signature(part)
        if sidecar is not None and idx < len(sidecar.labels):
            lbl = sidecar.labels[idx]
        else:
            lbl = InstrumentLabel(full_name=f"Part {idx + 1}", abbreviation=f"P{idx + 1}", is_fallback=True)
        if sidecar is not None and idx < len(sidecar.staff_y_norms):
            y_norm = sidecar.staff_y_norms[idx]
        else:
            y_norm = 0.5 if n <= 1 else idx / float(n - 1)
        specs.append(StaffMatchSpec(label=lbl, clef=clef, fifths=fifths, y_norm=y_norm, pitched_notes=pitched))
    return specs


def _build_rest_part_for_page(
    part_id: str,
    master_spec: StaffMatchSpec,
    reference_part: ET.Element,
) -> ET.Element:
    """Create a full-measure rest <part> matching the measure count & time signatures of ``reference_part``."""
    part = ET.Element("part", id=part_id)
    ref_measures = reference_part.findall("measure")
    active_ts = ("4", "4")
    for mi, ref_m in enumerate(ref_measures):
        ts = _extract_last_time(ref_m)
        if ts is not None:
            active_ts = ts
        b, bt = active_ts
        m_el = ET.SubElement(part, "measure", number=str(mi + 1))
        if mi == 0 or ts is not None:
            attrs = ET.SubElement(m_el, "attributes")
            if mi == 0:
                ET.SubElement(attrs, "divisions").text = "1"
                if master_spec.fifths is not None:
                    key_el = ET.SubElement(attrs, "key")
                    ET.SubElement(key_el, "fifths").text = master_spec.fifths
                if master_spec.clef is not None:
                    c_el = ET.SubElement(attrs, "clef", number="1")
                    ET.SubElement(c_el, "sign").text = master_spec.clef[0]
                    ET.SubElement(c_el, "line").text = master_spec.clef[1]
            if ts is not None:
                t_el = ET.SubElement(attrs, "time")
                ET.SubElement(t_el, "beats").text = b
                ET.SubElement(t_el, "beat-type").text = bt
        expected_dur = max(1, int(round(1 * int(b) * 4 / max(1, int(bt)))))
        _replace_measure_notes_with_rest(m_el, expected_dur)
    return part


def _rebuilt_score_with_parts(
    template_root: ET.Element,
    master_score_parts: Sequence[ET.Element],
    ordered_parts: Sequence[ET.Element],
) -> ET.Element:
    out = copy.deepcopy(template_root)
    pl = out.find("part-list")
    if pl is None:
        pl = ET.Element("part-list")
        out.insert(0, pl)
    for child in list(pl):
        pl.remove(child)
    for idx, sp in enumerate(master_score_parts, start=1):
        spc = copy.deepcopy(sp)
        spc.set("id", f"P{idx}")
        si = spc.find("score-instrument")
        if si is not None:
            si.set("id", f"P{idx}-I1")
        mi = spc.find("midi-instrument")
        if mi is not None:
            mi.set("id", f"P{idx}-I1")
        pl.append(spc)

    for old_p in list(out.findall("part")):
        out.remove(old_p)
    for idx, p in enumerate(ordered_parts, start=1):
        pc = copy.deepcopy(p)
        pc.set("id", f"P{idx}")
        out.append(pc)
    return out


def reconcile_pages_to_master(
    aligned_pages: Sequence[tuple[int, ET.Element, Optional[PageSidecar]]],
) -> tuple[list[ET.Element], list[InstrumentLabel]]:
    """Reconcile pages with possibly differing staff counts onto a single canonical instrument roster."""
    if not aligned_pages:
        raise MergeError("No full-score pages to reconcile")

    counts = [len(root.findall("part")) for _pno, root, _sc in aligned_pages]
    if any(c == 0 for c in counts):
        raise MergeError("One or more full-score pages have 0 parts")

    # Drop obvious empty stray detections on outlier pages when a smaller valid roster matches cleanly
    count_freq = Counter(counts)
    consensus_count = max(
        count_freq.keys(),
        key=lambda c: (count_freq[c], -c),
    )
    consensus_pages = count_freq[consensus_count]
    cleaned_pages: list[tuple[int, ET.Element, Optional[PageSidecar]]] = list(aligned_pages)

    if len(set(counts)) > 1 and (consensus_pages >= 2 or any(c < max(counts) for c in counts)):
        # Use the most common count (or largest count <= consensus with valid labels) as reference
        ref_count = consensus_count if consensus_pages >= 2 else max(
            c for c in counts if c < max(counts)
        )
        ref_idx = counts.index(ref_count)
        ref_specs = _build_specs_for_page(cleaned_pages[ref_idx][1], cleaned_pages[ref_idx][2])
        for pidx, (pno, root, sc) in enumerate(list(cleaned_pages)):
            if len(root.findall("part")) <= ref_count:
                continue
            specs = _build_specs_for_page(root, sc)
            # Match ref_count reference staves into this larger page's staves
            kept_indices = set(match_staves_monotonic_dp(ref_specs, specs))
            extra_indices = [i for i in range(len(specs)) if i not in kept_indices]
            all_extra_stray = all(
                specs[i].pitched_notes == 0 and (specs[i].label.is_fallback or looks_low_quality_label(specs[i].label.full_name))
                for i in extra_indices
            )
            if all_extra_stray:
                new_root = copy.deepcopy(root)
                all_parts = new_root.findall("part")
                all_sps = new_root.findall("./part-list/score-part")
                pl = new_root.find("part-list")
                for ei in sorted(extra_indices, reverse=True):
                    new_root.remove(all_parts[ei])
                    if pl is not None and ei < len(all_sps):
                        pl.remove(all_sps[ei])
                new_sc = None
                if sc is not None:
                    new_sc = PageSidecar(
                        page_number=sc.page_number,
                        staff_count=ref_count,
                        staff_cys=[sc.staff_cys[i] for i in sorted(kept_indices) if i < len(sc.staff_cys)],
                        staff_y_norms=[sc.staff_y_norms[i] for i in sorted(kept_indices) if i < len(sc.staff_y_norms)],
                        labels=[sc.labels[i] for i in sorted(kept_indices) if i < len(sc.labels)],
                        sys_min_x=sc.sys_min_x,
                        sys_max_x=sc.sys_max_x,
                        crop_x=sc.crop_x,
                    )
                cleaned_pages[pidx] = (pno, new_root, new_sc)
                log.info(
                    "Page %d: dropped %d empty/unlabeled stray staff detection(s) %s to match %d-part score",
                    pno,
                    len(extra_indices),
                    extra_indices,
                    ref_count,
                )

    updated_counts = [len(root.findall("part")) for _pno, root, _sc in cleaned_pages]
    max_count = max(updated_counts)

    # Pick richest master page among those with max_count
    max_indices = [i for i, c in enumerate(updated_counts) if c == max_count]
    master_idx = max(
        max_indices,
        key=lambda i: (
            sum(
                1
                for sp in _build_specs_for_page(cleaned_pages[i][1], cleaned_pages[i][2])
                if not sp.label.is_fallback and not looks_low_quality_label(sp.label.full_name)
            ),
            -i,
        ),
    )

    label_pages = [
        _build_specs_for_page(cleaned_pages[i][1], cleaned_pages[i][2])
        for i in max_indices
    ]
    consolidated_labels = consolidate_page_labels([[s.label for s in lp] for lp in label_pages])

    master_root = cleaned_pages[master_idx][1]
    master_raw_specs = _build_specs_for_page(master_root, cleaned_pages[master_idx][2])
    master_specs = [
        StaffMatchSpec(
            label=consolidated_labels[i] if i < len(consolidated_labels) else master_raw_specs[i].label,
            clef=master_raw_specs[i].clef,
            fifths=master_raw_specs[i].fifths,
            y_norm=master_raw_specs[i].y_norm,
            pitched_notes=master_raw_specs[i].pitched_notes,
        )
        for i in range(max_count)
    ]
    master_score_parts = master_root.findall("./part-list/score-part")
    if len(master_score_parts) < max_count:
        master_score_parts = [ET.Element("score-part", id=f"P{i + 1}") for i in range(max_count)]

    reconciled: list[ET.Element] = []
    for pno, root, sc in cleaned_pages:
        parts = root.findall("part")
        if len(parts) == max_count:
            reconciled.append(_rebuilt_score_with_parts(master_root, master_score_parts, parts))
            continue

        page_specs = _build_specs_for_page(root, sc)
        mapped_slots = match_staves_monotonic_dp(page_specs, master_specs)
        slot_to_part = {slot: parts[j] for j, slot in enumerate(mapped_slots)}
        ref_part = parts[0]

        ordered_parts: list[ET.Element] = []
        missing_names: list[str] = []
        for slot in range(max_count):
            if slot in slot_to_part:
                ordered_parts.append(slot_to_part[slot])
            else:
                missing_names.append(master_specs[slot].label.full_name)
                ordered_parts.append(_build_rest_part_for_page(f"P{slot + 1}", master_specs[slot], ref_part))

        log.info(
            "Page %d reconciled %d -> %d parts via monotonic DP (rest-filled %d missing/hidden: %s)",
            pno,
            len(parts),
            max_count,
            len(missing_names),
            ", ".join(missing_names),
        )
        reconciled.append(_rebuilt_score_with_parts(master_root, master_score_parts, ordered_parts))

    return reconciled, consolidated_labels


def dedup_boundary_attributes(
    measure: ET.Element,
    is_later_page_start: bool,
    state: dict[str, object],
) -> int:
    """Remove redundant divisions/key/time/clef on page boundaries while preserving genuine changes."""
    removed = 0
    for attrs in list(measure.findall("attributes")):
        if is_later_page_start:
            div_el = attrs.find("divisions")
            if div_el is not None and div_el.text and div_el.text.strip() == state["divisions"]:
                attrs.remove(div_el)
                removed += 1

            for key_el in list(attrs.findall("key")):
                fifths = key_el.findtext("fifths")
                if fifths is not None and fifths.strip() == state["fifths"]:
                    attrs.remove(key_el)
                    removed += 1

            for time_el in list(attrs.findall("time")):
                pair = (
                    (time_el.findtext("beats") or "").strip(),
                    (time_el.findtext("beat-type") or "").strip(),
                )
                if pair == state["time"]:
                    attrs.remove(time_el)
                    removed += 1

            for clef_el in list(attrs.findall("clef")):
                if clef_el.get("number") in (None, "1"):
                    cpair = (
                        (clef_el.findtext("sign") or "").strip(),
                        (clef_el.findtext("line") or "").strip(),
                    )
                    if cpair == state["clef"]:
                        attrs.remove(clef_el)
                        removed += 1

        if len(attrs) == 0:
            measure.remove(attrs)
            continue

        div_el = attrs.find("divisions")
        if div_el is not None and div_el.text:
            state["divisions"] = div_el.text.strip()
        key_el = attrs.find("key")
        if key_el is not None and key_el.findtext("fifths") is not None:
            state["fifths"] = key_el.findtext("fifths").strip()
        time_el = attrs.find("time")
        if time_el is not None:
            b = (time_el.findtext("beats") or "").strip()
            bt = (time_el.findtext("beat-type") or "").strip()
            if b and bt:
                state["time"] = (b, bt)
        for clef_el in attrs.findall("clef"):
            if clef_el.get("number") in (None, "1"):
                s = (clef_el.findtext("sign") or "").strip()
                ln = (clef_el.findtext("line") or "").strip()
                if s and ln:
                    state["clef"] = (s, ln)
    return removed


def compute_conductor_scaling(part_count: int) -> tuple[str, str, str]:
    """Return adaptive ``(millimeters, page_height_tenths, page_width_tenths)`` for A4 full scores."""
    if part_count >= 24:
        return "3.4", "3494", "2470"
    if part_count >= 16:
        mm = 4.2
    elif part_count >= 8:
        mm = 5.2
    else:
        mm = 6.5
    return f"{mm:.1f}", str(round(297 * 40 / mm)), str(round(210 * 40 / mm))


def apply_conductor_metadata(
    root: ET.Element,
    title: str,
    labels: Sequence[InstrumentLabel],
    *,
    keep_line_breaks: bool = False,
) -> ET.Element:
    """Apply conductor score title, adaptive scaling, instrument names/abbreviations, and valid MIDI channels."""
    for xp in ("movement-title",):
        el = root.find(xp)
        if el is not None:
            root.remove(el)
    work = root.find("work")
    if work is None:
        work = ET.Element("work")
        root.insert(0, work)
    wt = work.find("work-title")
    if wt is None:
        wt = ET.SubElement(work, "work-title")
    wt.text = title

    part_count = len(root.findall("part"))
    mm, page_h, page_w = compute_conductor_scaling(part_count)
    old_def = root.find("defaults")
    if old_def is not None:
        root.remove(old_def)
    defaults = ET.Element("defaults")
    scaling = ET.SubElement(defaults, "scaling")
    ET.SubElement(scaling, "millimeters").text = mm
    ET.SubElement(scaling, "tenths").text = "40"
    pl = ET.SubElement(defaults, "page-layout")
    ET.SubElement(pl, "page-height").text = page_h
    ET.SubElement(pl, "page-width").text = page_w
    pm = ET.SubElement(pl, "page-margins", type="both")
    ET.SubElement(pm, "left-margin").text = "140"
    ET.SubElement(pm, "right-margin").text = "120"
    ET.SubElement(pm, "top-margin").text = "120"
    ET.SubElement(pm, "bottom-margin").text = "120"

    ident = root.find("identification")
    insert_pos = list(root).index(ident) + 1 if ident is not None else 1
    root.insert(insert_pos, defaults)

    if not keep_line_breaks:
        strip_layout_breaks(root)

    # Full Score Mode must NEVER emit <transpose> (would corrupt concert vs transposed playback)
    removed_transpose = 0
    for parent in root.iter():
        for tr in list(parent.findall("transpose")):
            parent.remove(tr)
            removed_transpose += 1
    if removed_transpose:
        log.info("Removed %d unexpected <transpose> element(s) from full score", removed_transpose)

    pitched_i = 0
    for idx, sp in enumerate(root.findall("./part-list/score-part")):
        part_id = sp.get("id", f"P{idx + 1}")
        if idx < len(labels):
            inst_name, inst_abbr = labels[idx].full_name, labels[idx].abbreviation
        else:
            inst_name, inst_abbr = f"Part {idx + 1}", f"P{idx + 1}"

        pn = sp.find("part-name")
        if pn is None:
            pn = ET.Element("part-name")
            sp.insert(0, pn)
        pn.text = inst_name

        for old_pa in list(sp.findall("part-abbreviation")):
            sp.remove(old_pa)
        pa = ET.Element("part-abbreviation")
        pa.text = inst_abbr
        sp.insert(list(sp).index(pn) + 1, pa)

        si = sp.find("score-instrument")
        if si is None:
            si = ET.SubElement(sp, "score-instrument", id=f"{part_id}-I1")
        iname = si.find("instrument-name")
        if iname is None:
            iname = ET.SubElement(si, "instrument-name")
        iname.text = inst_name
        isound = si.find("instrument-sound")
        if isound is None:
            isound = ET.SubElement(si, "instrument-sound")
        isound.text = "voice"

        mi_el = sp.find("midi-instrument")
        if mi_el is None:
            mi_el = ET.SubElement(sp, "midi-instrument", id=f"{part_id}-I1")
        ch = mi_el.find("midi-channel")
        if ch is None:
            ch = ET.SubElement(mi_el, "midi-channel")
        if is_percussion_instrument(inst_name):
            ch.text = "10"
        else:
            ch.text = str(_PITCHED_MIDI_CHANNELS[pitched_i % len(_PITCHED_MIDI_CHANNELS)])
            pitched_i += 1

    log.info(
        "Applied conductor metadata (%d parts, scaling=%smm, page=%sx%s)",
        part_count,
        mm,
        page_w,
        page_h,
    )
    return root


def verify_fullscore_invariants(root: ET.Element) -> dict[str, int]:
    """Verify all 6 MusicXML full-score invariants; raise MergeError on any violation."""
    parts = root.findall("part")
    score_parts = root.findall("./part-list/score-part")
    if not parts:
        raise MergeError("Full score has no <part> elements")
    if len(score_parts) != len(parts):
        raise MergeError(f"part-list count {len(score_parts)} != part count {len(parts)}")

    mcounts = {len(p.findall("measure")) for p in parts}
    if len(mcounts) != 1:
        raise MergeError(f"Inconsistent part measure counts: {sorted(mcounts)}")
    total_m = next(iter(mcounts))
    if total_m <= 0:
        raise MergeError("Full score parts contain 0 measures")

    for p in parts:
        for m in p.findall("measure"):
            times = m.findall("./attributes/time")
            if len(times) > 1:
                raise MergeError(
                    f"Measure {m.get('number')} in {p.get('id')} has {len(times)} <time> elements"
                )

    for mi in range(total_m):
        declared: list[tuple[str, str]] = []
        for p in parts:
            t_el = p.findall("measure")[mi].find("./attributes/time")
            if t_el is not None:
                declared.append(
                    ((t_el.findtext("beats") or "").strip(), (t_el.findtext("beat-type") or "").strip())
                )
        if mi == 0 and len(declared) != len(parts):
            raise MergeError(f"Measure 1: missing time signature on some parts ({len(declared)}/{len(parts)})")
        if len(declared) not in (0, len(parts)):
            raise MergeError(
                f"Measure {mi + 1}: partial time signature across parts ({len(declared)}/{len(parts)}): {Counter(declared)}"
            )
        if len(set(declared)) > 1:
            raise MergeError(f"Measure {mi + 1}: conflicting time signatures: {Counter(declared)}")

    if root.findall(".//transpose"):
        raise MergeError("Unexpected <transpose> element in full score")

    for ch_el in root.findall(".//midi-channel"):
        txt = (ch_el.text or "").strip()
        try:
            ch_val = int(txt)
        except ValueError as exc:
            raise MergeError(f"Invalid non-integer MIDI channel {txt!r}") from exc
        if not (1 <= ch_val <= 16):
            raise MergeError(f"Out-of-range MIDI channel {ch_val} (must be 1..16)")

    note_counts = [sum(1 for n in p.findall(".//note") if n.find("rest") is None) for p in parts]
    summary = {
        "parts": len(parts),
        "measures": total_m,
        "non_empty_parts": sum(1 for c in note_counts if c > 0),
        "total_notes": sum(note_counts),
    }
    log.info(
        "Verified all 6 full-score XML invariants (%d parts x %d measures, %d notes across %d non-empty parts)",
        summary["parts"],
        summary["measures"],
        summary["total_notes"],
        summary["non_empty_parts"],
    )
    return summary


def merge_fullscore_pages(
    pages: Sequence[tuple[int, ET.Element, Optional[PageSidecar]]],
    title: str,
    *,
    keep_line_breaks: bool = False,
) -> ET.Element:
    """Align per page, reconcile varying staff counts, merge sequentially, and verify invariants."""
    if not pages:
        raise MergeError("No full-score pages to merge")

    aligned_pages: list[tuple[int, ET.Element, Optional[PageSidecar]]] = []
    inherited_ts: Optional[tuple[str, str]] = None
    for pno, root, sidecar in pages:
        aligned_root, ending_ts = align_page_score(copy.deepcopy(root), pno, inherited_time_sig=inherited_ts)
        inherited_ts = ending_ts
        aligned_pages.append((pno, aligned_root, sidecar))

    reconciled_roots, labels = reconcile_pages_to_master(aligned_pages)

    merged = copy.deepcopy(reconciled_roots[0])
    base_parts = merged.findall("part")
    for p in base_parts:
        for m in list(p.findall("measure")):
            p.remove(m)

    counters = [0] * len(base_parts)
    part_states: list[dict[str, object]] = [
        {"divisions": None, "fifths": None, "time": None, "clef": None}
        for _ in base_parts
    ]
    dedup_removed = 0
    for page_idx, root in enumerate(reconciled_roots, start=1):
        parts = root.findall("part")
        for pi, part in enumerate(parts):
            for mi, m in enumerate(part.findall("measure")):
                mc = copy.deepcopy(m)
                dedup_removed += dedup_boundary_attributes(
                    mc,
                    is_later_page_start=(page_idx > 1 and mi == 0),
                    state=part_states[pi],
                )
                counters[pi] += 1
                mc.set("number", str(counters[pi]))
                base_parts[pi].append(mc)

    log.info("Deduplicated %d redundant page-boundary attributes (divisions/key/time/clef)", dedup_removed)
    apply_conductor_metadata(merged, title, labels, keep_line_breaks=keep_line_breaks)
    verify_fullscore_invariants(merged)
    return merged


def merge_fullscore_files(
    files: Sequence[Path],
    out: Path,
    title: Optional[str] = None,
    options: Optional[ConversionOptions] = None,
) -> Path:
    """Read per-page MusicXML + sidecar files, merge in Full Score Mode, and write ``out``."""
    ordered_files = sorted(Path(f) for f in files)
    if not ordered_files:
        raise MergeError("No full-score MusicXML files to merge")

    pages: list[tuple[int, ET.Element, Optional[PageSidecar]]] = []
    for idx, xml_path in enumerate(ordered_files, start=1):
        root = read_score(xml_path)
        sidecar = read_page_sidecar(xml_path)
        pno = sidecar.page_number if sidecar is not None else idx
        pages.append((pno, root, sidecar))

    keep_breaks = bool(options.keep_line_breaks) if options is not None else False
    score_title = title or Path(out).stem
    merged = merge_fullscore_pages(pages, score_title, keep_line_breaks=keep_breaks)
    return write_musicxml(merged, Path(out))
