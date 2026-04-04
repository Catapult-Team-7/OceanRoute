param()

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python virtual environment not found at $PythonExe"
}

$env:PYTHONPATH = Join-Path $RepoRoot "apps\api"

Push-Location $RepoRoot
try {
    & $PythonExe -m app.cli seed-demo
}
finally {
    Pop-Location
}
