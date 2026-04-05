param(
    [switch]$BootstrapDb
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ActivateScript = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
$GpuActivateScript = Join-Path $RepoRoot ".venv-gpu\Scripts\Activate.ps1"
$ApiActivateScript = if (Test-Path $GpuActivateScript) { $GpuActivateScript } else { $ActivateScript }
$InferenceActivateScript = if (Test-Path $GpuActivateScript) { $GpuActivateScript } else { $ActivateScript }
$BootstrapScript = Join-Path $RepoRoot "scripts\bootstrap-db.ps1"
$PreferredNodeDir = "C:\Users\clewr\ATCS_APP_File\nvm\v20.19.0"
$PreferredNpmCmd = Join-Path $PreferredNodeDir "npm.cmd"
$FrontendRunner = if (Test-Path $PreferredNpmCmd) { "& '$PreferredNpmCmd'" } else { "npm" }
$NodePathPrefix = if (Test-Path $PreferredNodeDir) { "`$env:PATH = '$PreferredNodeDir;' + `$env:PATH" } else { "" }
$ApiCommand = @"
& '$ApiActivateScript'
Set-Location '$RepoRoot'
`$env:OCEANROUTE_INFERENCE_SERVICE_URL = 'http://127.0.0.1:8100'
python -m uvicorn app.main:app --app-dir apps\api --reload
"@
$InferenceCommand = @"
& '$InferenceActivateScript'
Set-Location '$RepoRoot'
python -m uvicorn app.inference_main:app --app-dir apps\api --port 8100 --reload
"@
$FrontendCommand = @"
Set-Location '$(Join-Path $RepoRoot "apps\web")'
$NodePathPrefix
$FrontendRunner run dev
"@

if (-not (Test-Path $ApiActivateScript)) {
    throw "Virtual environment activation script not found at $ApiActivateScript"
}

if ($BootstrapDb) {
    & powershell -ExecutionPolicy Bypass -File $BootstrapScript
}

Start-Process powershell.exe -WorkingDirectory $RepoRoot -ArgumentList @("-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $InferenceCommand) | Out-Null
Start-Process powershell.exe -WorkingDirectory $RepoRoot -ArgumentList @("-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $ApiCommand) | Out-Null
Start-Process powershell.exe -WorkingDirectory (Join-Path $RepoRoot "apps\web") -ArgumentList @("-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $FrontendCommand) | Out-Null

Write-Host "OceanRoute local stack started."
Write-Host "API docs: http://127.0.0.1:8000/docs"
Write-Host "Inference health: http://127.0.0.1:8100/health"
Write-Host "Frontend: http://127.0.0.1:3000"
