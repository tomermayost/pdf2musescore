$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Building standalone app with PyInstaller into tmp/dist..."
.\.venv\Scripts\pyinstaller.exe packaging\pdf2musescore.spec `
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
& $iscc packaging\installer.iss

Write-Host "==> Done! Outputs:"
Get-ChildItem tmp\dist\pdf2musescore-setup.exe, tmp\dist\pdf2musescore\pdf2musescore.exe | Select-Object FullName, Length
