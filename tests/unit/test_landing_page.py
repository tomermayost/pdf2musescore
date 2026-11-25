"""Tests for Phase 2 GitHub Pages landing site in docs/."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SITE_URL = "https://tomermayost.github.io/pdf2musescore/"


class _LandingPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_chunks: list[str] = []
        self._in_title = False
        self._in_jsonld = False
        self._current_jsonld: list[str] = []
        self.jsonld_blocks: list[str] = []
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
                self._current_jsonld = []
            if "src" in attr_map:
                self.scripts.append(attr_map["src"])

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = False
        elif t == "script" and self._in_jsonld:
            self.jsonld_blocks.append("".join(self._current_jsonld))
            self._current_jsonld = []
            self._in_jsonld = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_chunks.append(data)
        if self._in_jsonld:
            self._current_jsonld.append(data)


def test_landing_page_files_exist() -> None:
    assert (DOCS / ".nojekyll").is_file()
    for rel in (
        "index.html",
        "styles.css",
        "app.js",
        "robots.txt",
        "sitemap.xml",
        "google602c6e31d4004624.html",
        "assets/favicon.png",
    ):
        path = DOCS / rel
        assert path.is_file(), f"Missing landing page file: docs/{rel}"
        assert path.stat().st_size > 0, f"Empty landing page file: docs/{rel}"
    assert (
        (DOCS / "google602c6e31d4004624.html").read_text(encoding="utf-8")
        == "google-site-verification: google602c6e31d4004624.html"
    )


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
    assert "index" in by_name.get("robots", "") and "follow" in by_name.get("robots", "")
    assert by_name.get("author") == "Tomer Mayost"
    assert by_name.get("theme-color") == "#0f172a"
    assert by_name.get("twitter:card") == "summary_large_image"
    assert len(by_name.get("twitter:image:alt", "")) >= 20
    assert by_prop.get("og:site_name") == "PDF to MuseScore"
    assert by_prop.get("og:image", "").endswith("assets/social-preview.png")
    assert len(by_prop.get("og:image:alt", "")) >= 20
    assert by_prop.get("og:url") == SITE_URL

    # Valid JSON-LD SoftwareApplication and FAQPage metadata
    jsonld_by_type = {doc.get("@type"): doc for doc in (json.loads(b) for b in parser.jsonld_blocks)}
    assert {"SoftwareApplication", "FAQPage"}.issubset(set(jsonld_by_type))

    ld = jsonld_by_type["SoftwareApplication"]
    assert "PDF2MuseScore" in ld["alternateName"]
    assert "https://github.com/tomermayost/pdf2musescore" in ld["sameAs"]
    assert "https://apps.microsoft.com/detail/9N6NCBWM3BHT" in ld["sameAs"]
    assert len(ld["screenshot"]) >= 2
    assert len(ld["featureList"]) >= 4
    assert ld["offers"]["price"] == "0"
    assert ld["installUrl"] == "https://apps.microsoft.com/detail/9N6NCBWM3BHT"
    assert ld["downloadUrl"] == "https://github.com/tomermayost/pdf2musescore/releases/latest"

    faq = jsonld_by_type["FAQPage"]
    assert len(faq["mainEntity"]) == 6
    for q in faq["mainEntity"]:
        assert q["@type"] == "Question"
        assert len(q["name"]) > 10
        assert len(q["acceptedAnswer"]["text"]) > 20

    # Primary Store install link and standalone .exe fallback link exist
    assert "https://apps.microsoft.com/detail/9N6NCBWM3BHT" in parser.anchors
    assert "https://github.com/tomermayost/pdf2musescore/releases/latest" in parser.anchors
    assert 'id="bottomDownloadBtn"' in html_text

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
    assert "browser_download_url" in js_text
    assert (DOCS / "assets" / "ui-dark.png").is_file()
    assert (DOCS / "assets" / "ui-light.png").is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://tomermayost.github.io/pdf2musescore/" in readme
    assert "What are the minimum system requirements?" in html_text
    assert "AVX2" in html_text
    assert "preparing a Microsoft-signed" not in html_text
    assert "submitted for Store review" not in html_text
    assert "drag and drop" in html_text.lower()


def test_robots_txt_and_sitemap_xml() -> None:
    robots = (DOCS / "robots.txt").read_text(encoding="utf-8")
    assert "User-agent: *" in robots
    assert "Allow: /" in robots
    assert f"Sitemap: {SITE_URL}sitemap.xml" in robots

    tree = ET.parse(DOCS / "sitemap.xml")
    root = tree.getroot()
    ns = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "image": "http://www.google.com/schemas/sitemap-image/1.1",
    }
    locs = [el.text for el in root.findall("sm:url/sm:loc", ns)]
    assert locs == [SITE_URL]

    img_locs = [el.text for el in root.findall("sm:url/image:image/image:loc", ns) if el.text]
    assert len(img_locs) >= 4
    for img_url in img_locs:
        assert img_url.startswith(SITE_URL)
        rel = img_url.removeprefix(SITE_URL)
        asset_path = DOCS / rel
        assert asset_path.is_file(), f"Sitemap image missing: docs/{rel}"
        with Image.open(asset_path) as im:
            assert im.width > 0 and im.height > 0
