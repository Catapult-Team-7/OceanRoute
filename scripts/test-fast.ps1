$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
& ".\.venv\Scripts\python.exe" -m pytest apps/api/tests -q -m "not slow and not e2e"
