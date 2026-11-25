"""Unit tests for dual-install detection and version comparison helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pdf2musescore import install_detect
from pdf2musescore.install_detect import (
    InstalledEdition,
    check_dual_install,
    compare_versions,
    detect_other_edition,
    detect_standalone_install,
    detect_store_install,
    normalize_version_display,
    parse_msix_package_version,
)


def test_normalize_and_parse_versions() -> None:
    assert normalize_version_display("") == "0.0.0"
    assert normalize_version_display("v0.4.3.0") == "0.4.3"
    assert normalize_version_display("V1.2.3.4") == "1.2.3.4"
    assert normalize_version_display(" 0.4.6 ") == "0.4.6"

    assert parse_msix_package_version("TomerMayost.PDF2MuseScore_0.4.3.0_x64__ws19rcwb5efa4") == "0.4.3"
    assert parse_msix_package_version("tomermayost.pdf2musescore_0.4.6.0_x64__abc") == "0.4.6"
    assert parse_msix_package_version("TomerMayost.PDF2MuseScore_") is None
    assert parse_msix_package_version("OtherApp_1.0.0.0_x64__abc") is None
    assert parse_msix_package_version("TomerMayost.PDF2MuseScore_badver_x64__abc") is None

    assert compare_versions("0.4.3.0", "0.4.3") == 0
    assert compare_versions("0.4.6", "0.4.3") == 1
    assert compare_versions("0.4.3", "0.4.6") == -1


def test_detect_store_and_standalone_with_injectors() -> None:
    # Picks highest version among MSIX subkeys
    store = detect_store_install(
        enum_subkeys=lambda _sub: [
            "Other.App_1.0.0.0_x64__zzz",
            "TomerMayost.PDF2MuseScore_0.4.3.0_x64__ws19rcwb5efa4",
            "TomerMayost.PDF2MuseScore_0.4.5.0_x64__ws19rcwb5efa4",
        ]
    )
    assert store == InstalledEdition(
        kind="store",
        version="0.4.5",
        package_full_name="TomerMayost.PDF2MuseScore_0.4.5.0_x64__ws19rcwb5efa4",
    )
    assert detect_store_install(enum_subkeys=lambda _sub: (_ for _ in ()).throw(RuntimeError("reg err"))) is None
    assert detect_store_install(enum_subkeys=lambda _sub: ["Other.App_1.0.0.0_x64__zzz"]) is None

    # Standalone checks HKCU before HKLM and skips empty versions / exceptions
    calls: list[str] = []

    def fake_uninstall(root: str, _sub: str):
        calls.append(root)
        if root == "HKCU":
            raise RuntimeError("locked")
        return ("0.4.6", r"C:\Program Files\PDF to MuseScore")

    standalone = detect_standalone_install(query_uninstall=fake_uninstall)
    assert calls == ["HKCU", "HKLM"]
    assert standalone == InstalledEdition(
        kind="standalone",
        version="0.4.6",
        location=r"C:\Program Files\PDF to MuseScore",
    )

    assert detect_standalone_install(query_uninstall=lambda _r, _s: ("", "")) is None
    assert detect_standalone_install(query_uninstall=lambda _r, _s: None) is None

    # detect_other_edition catches exceptions gracefully
    assert (
        detect_other_edition(
            store_mode=True,
            standalone_detector=lambda: (_ for _ in ()).throw(OSError("boom")),
        )
        is None
    )


def test_default_winreg_helpers_cover_branches_and_non_windows(monkeypatch) -> None:
    class FakeKey:
        def __init__(self, subkeys=(), values=None):
            self.subkeys = list(subkeys)
            self.values = dict(values or {})

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def fake_open_key(hive, _subkey, _res, _sam):
        if hive == 1:  # HKEY_CURRENT_USER
            return FakeKey(
                subkeys=["TomerMayost.PDF2MuseScore_0.4.3.0_x64__ws19rcwb5efa4"],
                values={"DisplayVersion": "0.4.5"},
            )
        raise OSError("missing hklm key")

    def fake_enum_key(key, idx):
        if idx >= len(key.subkeys):
            raise OSError("no more items")
        return key.subkeys[idx]

    def fake_query_value_ex(key, name):
        if name not in key.values:
            raise OSError("missing value")
        return key.values[name], 1

    fake_winreg = SimpleNamespace(
        HKEY_CURRENT_USER=1,
        HKEY_LOCAL_MACHINE=2,
        KEY_READ=0x20019,
        OpenKey=fake_open_key,
        EnumKey=fake_enum_key,
        QueryValueEx=fake_query_value_ex,
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(install_detect, "winreg", fake_winreg)

    store = detect_store_install()
    assert store is not None and store.version == "0.4.3"

    standalone = detect_standalone_install()
    assert standalone == InstalledEdition(kind="standalone", version="0.4.5", location="")

    # OpenKey errors return empty / None safely
    monkeypatch.setattr(
        fake_winreg,
        "OpenKey",
        lambda *_a: (_ for _ in ()).throw(OSError("access denied")),
    )
    assert detect_store_install() is None
    assert detect_standalone_install() is None

    # Non-Windows returns None safely without touching winreg
    monkeypatch.setattr(sys, "platform", "linux")
    assert detect_store_install() is None
    assert detect_standalone_install() is None


@pytest.mark.parametrize(
    ("store_mode", "cur_ver", "other_ver", "expected_cmp", "expected_store_btn"),
    [
        (True, "0.4.3", "0.4.5", "other_newer", True),
        (True, "0.4.6", "0.4.6", "same", False),
        (True, "0.4.6", "0.4.4", "current_newer", False),
        (False, "0.4.5", "0.4.6", "other_newer", False),
        (False, "0.4.6", "0.4.6", "same", False),
        (False, "0.4.6", "0.4.3", "current_newer", True),
    ],
)
def test_check_dual_install_matrix(
    store_mode: bool,
    cur_ver: str,
    other_ver: str,
    expected_cmp: str,
    expected_store_btn: bool,
) -> None:
    other_kind = "standalone" if store_mode else "store"
    other = InstalledEdition(kind=other_kind, version=other_ver)

    # Suppressed when not frozen (dev runs)
    assert (
        check_dual_install(
            cur_ver,
            store_mode=store_mode,
            frozen=False,
            store_detector=lambda: other,
            standalone_detector=lambda: other,
        )
        is None
    )

    # None when sibling not present
    assert (
        check_dual_install(
            cur_ver,
            store_mode=store_mode,
            frozen=True,
            store_detector=lambda: None,
            standalone_detector=lambda: None,
        )
        is None
    )

    notice = check_dual_install(
        cur_ver,
        store_mode=store_mode,
        frozen=True,
        store_detector=lambda: other,
        standalone_detector=lambda: other,
    )
    assert notice is not None
    assert notice.comparison == expected_cmp
    assert notice.show_store_update is expected_store_btn
    assert cur_ver in notice.message
    assert other_ver in notice.message
    assert notice.dismiss_key.startswith(("store:", "standalone:"))
