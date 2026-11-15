# PDF to MuseScore

**By Tomer Mayost**

Minimal Windows desktop app (Windows 10 Home & upwards, Windows 11): **Load PDF → Convert & Save as** MuseScore (`.mscz`) or MusicXML (`.musicxml`).
Optical music recognition is powered by [homr](https://github.com/liebharc/homr) deep-learning AI (ONNX Runtime on CPU), with multi-page merging and automatic tuplet/tremolo sanitization for MuseScore 4.

## Download & End-User Installation (No Python or Java Required)

👉 **[Download Windows Installer (`pdf2musescore-setup.exe`, ~100 MB)](https://github.com/tomermayost/pdf2musescore/releases/latest/download/pdf2musescore-setup.exe)** *(or view all [GitHub Releases](https://github.com/tomermayost/pdf2musescore/releases))*

1. Download and run **`pdf2musescore-setup.exe`** and follow the Setup Wizard.
   - *Note on Windows SmartScreen:* Because this open-source installer is not signed with a paid Microsoft Authenticode certificate, Windows SmartScreen may show **"Windows protected your PC"**. Click **More info** → **Run anyway**.
   - **SHA-256 (`v0.2.0`):** `279C44E7B8A23C5DC3EFCB39DDDE02710D9C7D5470DC4953A43B0101ADB08A72`
2. Launch **PDF to MuseScore** from the Start Menu or Desktop shortcut.
3. Click **Load PDF...**, choose your sheet music PDF, optionally select a page range (e.g. `2-3`) or adjust **Engine & Score Parameters** (hover over any option in the app to see a tooltip with examples and defaults), then click **Convert & Save as...**.
4. On the first conversion, the app automatically downloads the AI models once (~134 MB) and caches them locally.

## Engine & Score Parameters

All parameters are available directly in the **Engine & Score Parameters** panel in the GUI (hover over any label or input box for a tooltip with default values and examples, or click **Reset Defaults**):

### PDF Rendering, Image Preprocessing & AI Engine
| Parameter | Default | Examples | Description & When to Adjust |
|---|---|---|---|
| **Render DPI** (`render_dpi`) | `300` | `200` (fast), `300` (standard), `400` (small/dense staves) | Resolution (`100`–`600` DPI) used to rasterize each PDF page before AI recognition. Increase to `400` for miniature scores or dense staves with small grace notes; lower to `200` for faster rendering on large PDFs. |
| **Preprocess** (`preprocess_mode`) | `auto` | `auto`, `faded`, `binarize`, `none` | Page contrast enhancement before the neural network reads the score:<br>• **`auto`** *(default)*: Standard CLAHE local contrast (best for clean printed PDFs).<br>• **`faded`**: Boosts contrast (`alpha=1.35`), darkens faint ink, and sharpens staff lines (ideal for old or faded scans).<br>• **`binarize`**: Adaptive Gaussian black-and-white thresholding (ideal for phone photos with uneven lighting/shadows).<br>• **`none`**: Raw page image without pre-enhancement. |
| **Auto-Deskew** (`deskew`) | `False` | `False` (off), `True` (on) | Measures staff-line tilt on scanned pages (up to `±10°`) via Hough line detection and rotates the page so staff lines are horizontal before recognition. Enable for crooked scanner/camera pages. |
| **SegNet Tile Step (px)** (`segnet_step_size`) | `320` | `160` (max overlap), `240` (balanced), `320` (fast default) | Sliding-window step size (`80`–`320` px) for the `320×320` neural segmentation network. Smaller step sizes (`240` or `160`) overlap adjacent tiles for smoother symbol edges on dense pages at the cost of extra CPU time. |
| **Min Notehead Size (px)** (`min_notehead_size`) | `4` | `3` (catch tiny/faint notes), `4` (default), `6` (ignore specks) | Minimum pixel width & height (`2`–`12` px) for a detected blob to be accepted as a notehead. Lower (`3`) catches tiny grace notes; higher (`6`) filters out paper dust/specks. |

### Score Post-Processing (MusicXML / MuseScore)
| Parameter | Default | Examples | Description & When to Adjust |
|---|---|---|---|
| **Transpose (Semitones)** (`transpose_semitones`) | `0` | `0` (none), `+2` (Concert → B♭), `-12` (down 1 octave), `+12` (up 1 octave) | Shifts all pitches and key signatures (`<fifths>`) in the output score by `N` semitones (`-24` to `+24`). Examples: `+2` for B♭ instruments (Trumpet, Clarinet, Tenor Sax), `+9` or `-3` for E♭ instruments, `±12` for octave shifts. |
| **Clef Override** (`clef_override`) | `auto` | `auto`, `treble`, `bass`, `alto`, `tenor` | Keeps the AI-detected clef (`auto`) or forces the primary staff clef across the score: `treble` (G2), `bass` (F4, e.g. Trombone/Cello), `alto` (C3, Viola), or `tenor` (C4, Trombone/Cello/Bassoon). |
| **Instrument** (`instrument_name`) | `auto` | `auto`, `Piano`, `Trombone`, `Violin`, `Flute`, `Trumpet`, `Cello` | Sets `<part-name>` and `<instrument-name>` in the generated MusicXML/MuseScore file so MuseScore labels the staff and selects the matching playback sound. You can pick a preset or type any custom instrument name. |
| **Title Override** (`title_override`) | `""` *(auto)* | `Telemann - Sonata in F Minor` | Custom score title written to `<work-title>` at the top of the score. Leave blank to automatically use the PDF filename / OCR title. |

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
| `musescore_path` | auto-detect | Path to `MuseScore4.exe` |
| `render_dpi` | `300` | PDF page rendering DPI for OMR |

## Code Signing Policy

Free code signing is provided by [SignPath.io](https://signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

- **Committers and Reviewers:** [Tomer Mayost](https://github.com/tomermayost)
- **Privacy Policy:** This program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it (on the first conversion run, the app downloads the open-source `homr` ONNX optical music recognition model weights from GitHub Releases and caches them locally).

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Tomer Mayost.
