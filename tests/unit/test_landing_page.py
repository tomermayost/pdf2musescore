"""Tests for Phase 2 GitHub Pages landing site in docs/."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


class _LandingPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_chunks: list[str] = []
        self._in_title = False
        self._in_jsonld = False
        self.jsonld_chunks: list[str] = []
        self.meta_tags: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self.anchors: list[str] = []
        self.imgs: list[str] = []
        self.scripts: list[str] = []
        self.audio_tags: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        t = tag.lower()
        if t == "title":
            self._in_title = True
        elif t == "meta":
            self.meta_tags.append(attr_map)
        elif t == "link":
            self.links.append(attr_map)
        elif t == "a":
            if "href" in attr_map:
                self.anchors.append(attr_map["href"])
        elif t == "img":
            if "src" in attr_map:
                self.imgs.append(attr_map["src"])
        elif t == "audio":
            self.audio_tags.append(attr_map)
        elif t == "script":
            if attr_map.get("type") == "application/ld+json":
                self._in_jsonld = True
            if "src" in attr_map:
                self.scripts.append(attr_map["src"])

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = False
        elif t == "script":
            self._in_jsonld = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_chunks.append(data)
        if self._in_jsonld:
            self.jsonld_chunks.append(data)


def test_landing_page_files_exist() -> None:
    assert (DOCS / ".nojekyll").is_file()
    for rel in ("index.html", "styles.css", "app.js", "assets/favicon.png"):
        path = DOCS / rel
        assert path.is_file(), f"Missing landing page file: docs/{rel}"
        assert path.stat().st_size > 0, f"Empty landing page file: docs/{rel}"


def test_landing_page_html_seo_assets_and_constraints() -> None:
    html_text = (DOCS / "index.html").read_text(encoding="utf-8")
    parser = _LandingPageParser()
    parser.feed(html_text)

    # Zero audio elements or audio asset links
    assert parser.audio_tags == []
    assert not re.search(r"\.(mp3|wav|ogg)\b", html_text, re.IGNORECASE)

    # Zero links to .pdf files (GitHub Pages / GitHub renders PDF links poorly)
    for href in parser.anchors:
        assert not href.lower().endswith(".pdf"), f"Unexpected PDF link: {href}"

    # Required title, SEO meta tags, OpenGraph, and Twitter card
    title = "".join(parser.title_chunks).strip()
    assert "PDF to MuseScore" in title

    by_name = {m.get("name"): m.get("content", "") for m in parser.meta_tags if "name" in m}
    by_prop = {m.get("property"): m.get("content", "") for m in parser.meta_tags if "property" in m}

    assert len(by_name.get("description", "")) >= 50
    assert by_name.get("twitter:card") == "summary_large_image"
    assert by_prop.get("og:image", "").endswith("assets/social-preview.png")
    assert by_prop.get("og:url") == "https://tomermayost.github.io/pdf2musescore/"

    # Valid JSON-LD SoftwareApplication metadata
    ld = json.loads("".join(parser.jsonld_chunks))
    assert ld["@type"] == "SoftwareApplication"
    assert ld["offers"]["price"] == "0"
    assert ld["downloadUrl"] == (
        "https://github.com/tomermayost/pdf2musescore/releases/latest/download/pdf2musescore-setup.exe"
    )

    # Primary installer download link exists
    assert (
        "https://github.com/tomermayost/pdf2musescore/releases/latest/download/pdf2musescore-setup.exe"
        in parser.anchors
    )

    # Every relative local asset referenced in index.html exists under docs/
    local_refs: list[str] = []
    local_refs.extend(parser.imgs)
    local_refs.extend(parser.scripts)
    local_refs.extend(
        l["href"] for l in parser.links if "href" in l and not l["href"].startswith(("http://", "https://"))
    )
    assert "styles.css" in local_refs
    assert "app.js" in local_refs
    assert "assets/favicon.png" in local_refs
    assert "assets/before-after-bach.png" in local_refs

    for rel in local_refs:
        asset_path = DOCS / rel
        assert asset_path.is_file(), f"Referenced local asset missing: docs/{rel}"
        if asset_path.suffix.lower() == ".png":
            with Image.open(asset_path) as img:
                assert img.width > 0 and img.height > 0

    # Verify JS and README references match
    js_text = (DOCS / "app.js").read_text(encoding="utf-8")
    assert "assets/ui-dark.png" in js_text
    assert "assets/ui-light.png" in js_text
    assert (DOCS / "assets" / "ui-dark.png").is_file()
    assert (DOCS / "assets" / "ui-light.png").is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://tomermayost.github.io/pdf2musescore/" in readme
    assert "What are the minimum system requirements?" in html_text
    assert "AVX2" in html_text
