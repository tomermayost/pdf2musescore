# PDF to MuseScore

**By Tomer Mayost**

Minimal Windows desktop app (Windows 10 Home & upwards, Windows 11): **Load PDF → Convert & Save as** MuseScore (`.mscz`) or MusicXML (`.musicxml`).
Optical music recognition is powered by [Audiveris](https://github.com/Audiveris/audiveris), with multi-movement merging and automatic partial-sheet recovery if individual pages fail.

## End-User Installation (No Python Required)

1. Run **`tmp\dist\pdf2musescore-setup.exe`** and follow the Setup Wizard.
2. Launch **PDF to MuseScore** from the Start Menu or Desktop shortcut.
3. Click **Load PDF...**, choose your sheet music PDF, then click **Convert & Save as...**.
4. If Audiveris or MuseScore is not yet installed on the laptop, the app will ask permission and install it automatically.

## Development & Testing

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Run from source:
```powershell
.\.venv\Scripts\python -m pdf2musescore
```

Coverage report: `tmp/coverage_html/index.html` (minimum coverage gate: 85%).

## Building the Windows Installer (`pdf2musescore-setup.exe`)

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Outputs in `tmp/dist/`:
- `tmp/dist/pdf2musescore/pdf2musescore.exe` — standalone portable folder
- `tmp/dist/pdf2musescore-setup.exe` — Windows Setup Wizard installer

## Configuration

Optional `%APPDATA%\pdf2musescore\config.toml`, overridable by environment variables `PDF2MSCZ_<KEY>`:

| Key | Default | Description |
|---|---|---|
| `log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `log_dir` | `%LOCALAPPDATA%\pdf2musescore\logs` (or `tmp/logs` in dev) | Rotating log files |
| `work_dir` | `%LOCALAPPDATA%\pdf2musescore\work` (or `tmp/work` in dev) | Temporary working directory |
| `audiveris_path` | auto-detect | Path to `Audiveris.exe` |
| `musescore_path` | auto-detect | Path to `MuseScore4.exe` |
| `audiveris_timeout_sec` | `1800` | Maximum OMR time per PDF |
