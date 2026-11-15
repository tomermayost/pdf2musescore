# pdf2musescore

Minimal Windows 10 desktop app: **Load PDF → Convert → Save as** MusicXML (and `.mscz` when MuseScore is installed).
Optical music recognition is done by [Audiveris](https://github.com/Audiveris/audiveris).

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Coverage report: `tmp/coverage_html/index.html` (gate: 85%).

## Configuration

Optional `%APPDATA%\pdf2musescore\config.toml`, overridable by env vars `PDF2MSCZ_<KEY>`:

| Key | Default | Description |
|---|---|---|
| `log_level` | `INFO` | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| `log_dir` | `tmp/logs` | Rotating log files |
| `work_dir` | `tmp/work` | Intermediate files |
| `audiveris_path` | auto-detect | Path to `Audiveris.exe` |
| `musescore_path` | auto-detect | Path to `MuseScore4.exe` |
| `audiveris_timeout_sec` | `1800` | Max OMR time per PDF |
