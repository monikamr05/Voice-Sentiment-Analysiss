# Run with .venv (Python 3.12 + TensorFlow)
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Run setup first: py -3.12 -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Activate:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "Python:    $py" -ForegroundColor Cyan
& $py app.py
