param()

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$EnvPath = Join-Path $RepoRoot ".env"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"

if (-not (Test-Path $PythonExe)) {
    throw "Python virtual environment not found at $PythonExe. Create .venv and install apps/api requirements first."
}

if (-not (Test-Path $EnvPath)) {
    if (-not (Test-Path $EnvExamplePath)) {
        throw "Missing .env.example at $EnvExamplePath"
    }
    Copy-Item -LiteralPath $EnvExamplePath -Destination $EnvPath
    Write-Host "Created .env from .env.example"
}

Push-Location $RepoRoot
try {
    docker compose up -d postgres | Out-Null

    $ready = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        docker compose exec -T postgres sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' *> $null
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $ready) {
        throw "Postgres did not become ready within 60 seconds."
    }

    & $PythonExe -m alembic upgrade head

    Write-Host "SeaSweep database is ready."
    Write-Host "Next:"
    Write-Host "  1. powershell -ExecutionPolicy Bypass -File scripts\\seed-sample-data.ps1"
    Write-Host "  2. .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir apps\\api --reload"
    Write-Host "  3. powershell -ExecutionPolicy Bypass -File infra\\windows\\run_forecast.ps1 -HorizonHours 24 -SourceMode auto"
}
finally {
    Pop-Location
}
