# Launch both FDV apps in the workspace venv
param(
  [switch]$NoReport,
  [switch]$NoPoll
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$py = Join-Path $root '.venv'
$py = Join-Path $py 'Scripts'
$py = Join-Path $py 'python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
Write-Host "Using Python: $py"
& $py (Join-Path $here 'launch_fdv_apps.py')
