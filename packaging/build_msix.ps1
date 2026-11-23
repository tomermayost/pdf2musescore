$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$initFile = Join-Path (Get-Location) "src\pdf2musescore\__init__.py"
$initText = Get-Content -Path $initFile -Raw
if ($initText -notmatch '__version__\s*=\s*"([^"]+)"') {
    throw "Could not parse __version__ from $initFile"
}
$rawVersion = $Matches[1]
$parts = $rawVersion.Split(".") | ForEach-Object {
    if ($_ -match "^(\d+)") { [int]$Matches[1] } else { 0 }
}
while ($parts.Count -lt 3) {
    $parts += 0
}
$msixVersion = "{0}.{1}.{2}.0" -f $parts[0], $parts[1], $parts[2]

$distApp = Join-Path (Get-Location) "tmp\dist\pdf2musescore"
$exePath = Join-Path $distApp "pdf2musescore.exe"
if (-not (Test-Path $exePath)) {
    throw "PyInstaller output not found at $exePath. Run build.ps1 or pyinstaller first."
}

$assetsSrc = Join-Path (Get-Location) "packaging\msix_assets"
if (-not (Test-Path (Join-Path $assetsSrc "StoreLogo.png"))) {
    throw "MSIX visual assets missing in $assetsSrc. Run scripts/generate_icons.py --all first."
}

$stageDir = Join-Path (Get-Location) "tmp\msix_stage"
if (Test-Path $stageDir) {
    Remove-Item -Path $stageDir -Recurse -Force
}
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

Write-Host "==> Staging MSIX payload (version $msixVersion) into $stageDir..."
Copy-Item -Path "$distApp\*" -Destination $stageDir -Recurse -Force

$stageAssets = Join-Path $stageDir "Assets"
New-Item -ItemType Directory -Path $stageAssets -Force | Out-Null
Copy-Item -Path "$assetsSrc\*.png" -Destination $stageAssets -Force

$templatePath = Join-Path (Get-Location) "packaging\AppxManifest.xml.template"
$manifestXml = (Get-Content -Path $templatePath -Raw).Replace("__MSIX_VERSION__", $msixVersion)
$manifestOut = Join-Path $stageDir "AppxManifest.xml"
[System.IO.File]::WriteAllText($manifestOut, $manifestXml, [System.Text.UTF8Encoding]::new($false))

$makeappxCandidates = @(
    (Get-Command makeappx.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
)
$kitsRoot = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
if (Test-Path $kitsRoot) {
    $sdkHits = Get-ChildItem -Path $kitsRoot -Recurse -Filter "makeappx.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "\\x64\\makeappx\.exe$" } |
        Sort-Object FullName -Descending |
        Select-Object -ExpandProperty FullName
    $makeappxCandidates += $sdkHits
}
$makeappx = $makeappxCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $makeappx) {
    throw "makeappx.exe (Windows 10/11 SDK) was not found. Install the Windows SDK or run on GitHub Actions windows-latest."
}

$outMsix = Join-Path (Get-Location) "tmp\dist\pdf2musescore.msix"
if (Test-Path $outMsix) {
    Remove-Item -Path $outMsix -Force
}

Write-Host "==> Packing MSIX using $makeappx..."
& $makeappx pack /o /d $stageDir /p $outMsix
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outMsix)) {
    throw "makeappx failed with exit code $LASTEXITCODE"
}

$msixFile = Get-Item $outMsix
if ($msixFile.Length -le 1MB) {
    throw "Generated MSIX is unexpectedly small ($($msixFile.Length) bytes)"
}

Write-Host "==> Done! MSIX output:"
$msixFile | Select-Object FullName, Length
