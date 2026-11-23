"""Shared Tkinter root fixture so Windows Python 3.12 doesn't tear down Tcl/Tk mid-session."""

import tkinter as tk

import pytest


@pytest.fixture(scope="session")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("No display")
    r.withdraw()
    yield r
    r.destroy()
