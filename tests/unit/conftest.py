"""Shared Tkinter root and offline ModelManager test fixtures."""

from pathlib import Path
import tkinter as tk

import pytest


class _ReadyModelManager:
    def __init__(self, models_dir, **_kwargs):
        self.models_dir = Path(models_dir)

    def status(self):
        from pdf2musescore.model_store import ModelStatus

        return ModelStatus(
            ready=True,
            missing=(),
            homr_dir=self.models_dir / "homr",
            ocr_dir=self.models_dir / "rapidocr",
        )

    @staticmethod
    def is_downloading() -> bool:
        return False

    def ensure(self, progress=None, is_cancelled=lambda: False):
        return self.status()


@pytest.fixture(autouse=True)
def _no_network_model_manager(monkeypatch):
    import pdf2musescore.gui as gui_mod
    import pdf2musescore.pipeline as pipe_mod

    monkeypatch.setattr(gui_mod, "ModelManager", _ReadyModelManager)
    monkeypatch.setattr(pipe_mod, "ModelManager", _ReadyModelManager)


@pytest.fixture(scope="session")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("No display")
    r.withdraw()
    yield r
    r.destroy()
