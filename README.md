# PDF to MuseScore

**By Tomer Mayost**

Minimal Windows desktop app (Windows 10 Home & upwards, Windows 11): **Load PDF → Convert & Save as** MuseScore (`.mscz`) or MusicXML (`.musicxml`).
Optical music recognition is powered by [homr](https://github.com/liebharc/homr) 0.7+ deep-learning AI (ONNX Runtime on CPU), recognizing **notes, slurs, ties, dynamics, expression modifiers (`dolce`, `a tempo`, etc.), beams, and multi-page scores** while automatically flowing line breaks and sanitizing tuplets/tremolos for MuseScore 4.

## Download & End-User Installation (No Python or Java Required)

👉 **[Download Windows Installer (`pdf2musescore-setup.exe`, ~105 MB)](https://github.com/tomermayost/pdf2musescore/releases/latest/download/pdf2musescore-setup.exe)** *(or view all [GitHub Releases](https://github.com/tomermayost/pdf2musescore/releases))*

1. Download and run **`pdf2musescore-setup.exe`** and follow the Setup Wizard.
   - *Note on Windows SmartScreen:* Because this open-source installer is not signed with a paid Microsoft Authenticode certificate, Windows SmartScreen may show **"Windows protected your PC"**. Click **More info** → **Run anyway**.
   - **SHA-256 (`v0.3.0`):** `583BF06A978ADA75044E3139E401240188A4CBF03DD1BCD01242AEC87FC2C5E3`
2. Launch **PDF to MuseScore** from the Start Menu or Desktop shortcut.
3. Click **Load PDF...**, choose your sheet music PDF, optionally select a page range (e.g. `2-3`) or adjust **Engine & Score Parameters** (hover over any option in the app to see a tooltip with examples and defaults), then click **Convert & Save as...**.
4. On the first conversion, the app automatically downloads the AI models once (~134 MB) and caches them locally.

## Engine & Score Parameters

All parameters are available directly in the **Engine & Score Parameters** panel in the GUI (hover over any label or input box for a tooltip with default values and examples, or click **Reset Defaults**):

### PDF Rendering, Image Preprocessing & AI Engine
| Parameter | Default | Examples | Description & When to Adjust |
|---|---|---|---|
| **Scan Quality (DPI)** (`render_dpi`) | `300` | `200` (faster), `300` (normal), `400` (tiny or crowded notes) | Controls how closely the app zooms in on each PDF page (`100`–`600`). Keep at `300` for normal sheet music. Try `400` if the notes are very small or crowded together, or `200` if you want a faster conversion. |
| **Page Cleanup** (`preprocess_mode`) | `auto` | `auto`, `faded`, `binarize`, `none` | Helps the app read difficult scans:<br>• **`auto`** *(default)*: Best for normal, clean PDFs.<br>• **`faded`**: Darkens light or washed-out ink (great for old photocopies or faded scans).<br>• **`binarize`**: Turns the page pure black & white (great for phone camera photos with shadows).<br>• **`none`**: Uses the page picture as-is without any cleanup. |
| **Straighten Tilted Page** (`deskew`) | `False` | `Off` (normal), `On` (straighten crooked scans) | Check this box if the sheet music was scanned or photographed slightly crooked. The app will automatically straighten the staff lines before reading the notes. |
| **Reading Thoroughness (px)** (`segnet_step_size`) | `320` | `320` (normal speed), `240` (more careful), `160` (most careful, slower) | The app reads the page in overlapping squares (`80`–`320`). Lower numbers (`240` or `160`) make the squares overlap more so notes near the edges aren't missed — helpful on crowded pages, though it takes a little longer. |
| **Smallest Note Size (px)** (`min_notehead_size`) | `4` | `3` (keep tiny grace notes), `4` (normal), `6` (ignore paper dust/specks) | Controls the smallest black dot (`2`–`12`) the app treats as a musical note. Lower to `3` if small grace notes are being missed; raise to `6` if dusty spots on the paper are turning into extra fake notes. |

### Score Post-Processing (MusicXML / MuseScore)
| Parameter | Default | Examples | Description & When to Adjust |
|---|---|---|---|
| **Transpose (Half-Steps)** (`transpose_semitones`) | `0` | `0` (keep original), `+2` (for B♭ Trumpet/Clarinet/Tenor Sax), `-12` (down 1 octave), `+12` (up 1 octave) | Moves every note and the key signature up or down by the number of half-steps (semitones, `-24` to `+24`) you enter. |
| **Clef** (`clef_override`) | `auto` | `auto`, `treble`, `bass`, `alto`, `tenor` | Keep at `auto` to use whatever clef is printed on the page, or pick a clef (`bass` or `tenor` for Trombone/Cello, `treble` for Violin/Flute, `alto` for Viola) if the scan misread the clef sign. |
| **Instrument** (`instrument_name`) | `auto` | `auto`, `Piano`, `Trombone`, `Violin`, `Flute`, `Trumpet`, `Cello` | Sets the instrument name and playback sound in MuseScore. Pick one from the list or type any instrument name you like. |
| **Custom Title** (`title_override`) | `""` *(automatic)* | `Telemann - Sonata in F Minor` | Type a title here if you want to set the heading at the top of the sheet music. Leave blank to use the PDF's title automatically. |
| **Keep Original Line Breaks** (`keep_line_breaks`) | `False` | `Off` (auto-flow), `On` (lock breaks to PDF) | When turned off *(recommended)*, removes forced system and page breaks so MuseScore flows measures neatly across lines and pages. Turn on only if you want every line and page to break in the exact same places as the original PDF. |

## Development & Testing

Requires Python 3.11+ (3.12 recommended):

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
