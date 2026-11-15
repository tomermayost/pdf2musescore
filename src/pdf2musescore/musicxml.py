"""Read MusicXML (.mxl / .musicxml) and merge multiple movements into one score."""

from __future__ import annotations

import copy
import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)


class MergeError(ValueError):
    pass


def read_score(path: Path) -> ET.Element:
    """Return the <score-partwise> root of a .mxl or .musicxml/.xml file."""
    path = Path(path)
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None:
                raise MergeError(f"{path}: no rootfile in container.xml")
            root = ET.fromstring(z.read(rootfile.get("full-path")))
    else:
        root = ET.parse(path).getroot()
    if root.tag != "score-partwise":
        raise MergeError(f"{path}: unsupported root <{root.tag}> (only score-partwise)")
    return root


def _movement_title(root: ET.Element, index: int) -> str:
    for xp in ("movement-title", "work/work-title"):
        el = root.find(xp)
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    return f"Movement {index}"


def _mark_movement_start(measure: ET.Element, title: str, new_page: bool) -> None:
    pos = 0
    if new_page:
        measure.insert(0, ET.Element("print", {"new-page": "yes"}))
        pos = 1
    for i, child in enumerate(list(measure)):
        if child.tag == "attributes":
            pos = i + 1
            break
    direction = ET.Element("direction", {"placement": "above"})
    words = ET.SubElement(ET.SubElement(direction, "direction-type"), "words",
                          {"font-weight": "bold", "font-size": "14"})
    words.text = title
    measure.insert(pos, direction)


def merge_scores(roots: Sequence[ET.Element], title: str | None = None) -> ET.Element:
    """Concatenate movements part-by-part into a single score-partwise tree."""
    if not roots:
        raise MergeError("No scores to merge")
    merged = copy.deepcopy(roots[0])
    base_parts = merged.findall("part")
    if not base_parts:
        raise MergeError("First score has no parts")

    for p in base_parts:
        for m in p.findall("measure"):
            p.remove(m)

    counters = [0] * len(base_parts)
    for idx, root in enumerate(roots, start=1):
        mtitle = _movement_title(root, idx)
        parts = root.findall("part")
        if len(parts) != len(base_parts):
            log.warning("Movement %d (%s) has %d part(s), expected %d - extra parts dropped, missing left empty",
                        idx, mtitle, len(parts), len(base_parts))
        for pi, part in enumerate(parts[:len(base_parts)]):
            measures = part.findall("measure")
            for mi, m in enumerate(measures):
                m = copy.deepcopy(m)
                counters[pi] += 1
                m.set("number", str(counters[pi]))
                if mi == 0 and pi == 0:
                    _mark_movement_start(m, mtitle, new_page=idx > 1)
                base_parts[pi].append(m)
        log.info("Merged movement %d: %s (%d measures)", idx, mtitle,
                 len(parts[0].findall("measure")) if parts else 0)

    if title:
        for xp in ("movement-title",):
            el = merged.find(xp)
            if el is not None:
                merged.remove(el)
        work = merged.find("work")
        if work is None:
            work = ET.Element("work")
            merged.insert(0, work)
        wt = work.find("work-title")
        if wt is None:
            wt = ET.SubElement(work, "work-title")
        wt.text = title
    return merged


def write_musicxml(root: ET.Element, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
    log.info("Wrote MusicXML %s", path)
    return path


def merge_files(files: Sequence[Path], out: Path, title: str | None = None) -> Path:
    files = sorted(Path(f) for f in files)
    log.info("Merging %d file(s) into %s", len(files), out)
    return write_musicxml(merge_scores([read_score(f) for f in files], title), out)
