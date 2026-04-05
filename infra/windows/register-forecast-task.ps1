param(
    [string]$TaskName = "SeaSweep Forecast Cycle",
    [int]$IntervalMinutes = 60
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ScriptPath = Join-Path $RepoRoot "infra\windows\run_forecast.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$ScriptPath`" -HorizonHours 24 -SourceMode auto"

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)).Repetition

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Force | Out-Null
Write-Host "Registered task '$TaskName' to run every $IntervalMinutes minute(s)."
