"""Win32 HDROP / WM_DROPFILES test helper for native drag-and-drop integration tests."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Sequence

WM_DROPFILES = 0x0233
_GMEM_MOVEABLE_ZEROINIT = 0x0042


def post_fake_drop(hwnd: int, paths: Sequence[Path | str]) -> None:
    """Allocate a Win32 Unicode ``DROPFILES`` structure and post ``WM_DROPFILES`` to ``hwnd``."""
    str_paths = [str(p) for p in paths]
    if not str_paths:
        raise ValueError("paths must not be empty")

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p

    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL

    payload = ("\0".join(str_paths) + "\0\0").encode("utf-16-le")
    header = (ctypes.c_uint32 * 5)(20, 0, 0, 0, 1)  # pFiles=20, pt=(0,0), fNC=0, fWide=1

    hglobal = kernel32.GlobalAlloc(_GMEM_MOVEABLE_ZEROINIT, 20 + len(payload))
    if not hglobal:
        raise ctypes.WinError()

    ptr = kernel32.GlobalLock(hglobal)
    if not ptr:
        kernel32.GlobalFree(hglobal)
        raise ctypes.WinError()
    try:
        ctypes.memmove(ptr, header, 20)
        ctypes.memmove(ptr + 20, payload, len(payload))
    finally:
        kernel32.GlobalUnlock(hglobal)

    ok = user32.PostMessageW(int(hwnd), WM_DROPFILES, int(hglobal), 0)
    if not ok:
        kernel32.GlobalFree(hglobal)
        raise ctypes.WinError()
