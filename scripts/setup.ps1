$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = (Get-Command python -ErrorAction Stop).Source
& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 or newer is required"
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    & $Python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m videox setup --in-venv
if ($LASTEXITCODE -ne 0) {
    throw "videox setup failed"
}
