$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Building standalone app with PyInstaller into tmp/dist..."
$pyinstaller = if (Test-Path ".\.venv\Scripts\pyinstaller.exe") { ".\.venv\Scripts\pyinstaller.exe" } else { "pyinstaller" }
& $pyinstaller packaging\pdf2musescore.spec `
    --distpath tmp\dist `
    --workpath tmp\build `
    --noconfirm `
    --clean

$isccCandidates = @(
    (Get-Command ISCC.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $isccCandidates) {
    Write-Host "==> Inno Setup not found. Installing JRSoftware.InnoSetup via winget..."
    winget install --id JRSoftware.InnoSetup -e --silent --accept-package-agreements --accept-source-agreements
    $isccCandidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ }
}

$iscc = $isccCandidates | Select-Object -First 1
if (-not $iscc) {
    throw "ISCC.exe (Inno Setup) could not be found even after installation."
}

Write-Host "==> Building Setup Wizard with Inno Setup ($iscc)..."
$initText = Get-Content -Path "src\pdf2musescore\__init__.py" -Raw
if ($initText -notmatch '__version__\s*=\s*"([^"]+)"') {
    throw "Could not parse __version__ from src\pdf2musescore\__init__.py"
}
$rawVersion = $Matches[1]
Get-ChildItem -Path "tmp\dist" -Filter "pdf2musescore*setup*.exe" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
& $iscc packaging\installer.iss

Write-Host "==> Done! Outputs:"
Get-ChildItem "tmp\dist\pdf2musescore-$rawVersion-setup-x64.exe", "tmp\dist\pdf2musescore\pdf2musescore.exe" | Select-Object FullName, Length
