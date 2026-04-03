# Launch FDV EIMPRO WebApp
# Usage: .\launch_eimpro_webapp.ps1

$ErrorActionPreference = 'Stop'

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$venvPath = Join-Path -Path $rootDir -ChildPath '.venv'
$pythonExe = Join-Path -Path $venvPath -ChildPath 'Scripts' -AdditionalChildPath 'python.exe'

# Fallback to system python if venv not found
if (-not (Test-Path $pythonExe)) {
    $pythonExe = 'py'
    Write-Host "Using system Python: $pythonExe"
} else {
    Write-Host "Using workspace venv: $pythonExe"
}

# Check if Flask is installed
Write-Host "Checking dependencies..."
& $pythonExe -m pip list | Select-String -Pattern "Flask" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required packages..."
    & $pythonExe -m pip install -r (Join-Path $scriptDir 'requirements.txt')
}

# Launch the app
Write-Host "Starting FDV EIMPRO WebApp..."
Write-Host "Access at: http://localhost:5058"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

& $pythonExe (Join-Path $scriptDir 'fdv_eimpro_webapp.py')
