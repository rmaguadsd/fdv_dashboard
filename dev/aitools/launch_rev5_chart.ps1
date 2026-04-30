param([int]$Port = 5059, [switch]$StopOnly)

<#
  Launcher for FDV Chart rev5 (Performance Options Edition)
  - Features: Optional sampling (Random, Decimation), WebGL rendering option
  - Port: 5059 (configurable - use port > 1024 to avoid browser restrictions)
  - Requires: Python with fdv_chart_rev5/fdv_chart.py
#>

function Stop-ByPort([int]$Port) {
  try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  } catch {
    $conns = @()
  }
  if ($conns) {
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
      try {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Output "Killed PID $procId on port $Port"
      } catch {
        Write-Output ("Failed to kill PID {0} on port {1}: {2}" -f $procId, $Port, $_.Exception.Message)
      }
    }
  } else {
    Write-Output "No listeners on port $Port"
  }
}

Stop-ByPort -Port $Port

if ($StopOnly) {
  Write-Output "StopOnly specified; exiting after stopping listeners on port $Port."
  exit 0
}

# Prefer a repo-local venv first, else fall back to Python launcher
# PSScriptRoot = D:\FDV\git\fdv_dashboard\dev\aitools
# We need: D:\FDV\git\fdv_dashboard
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$repoVenvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$userVenvPython = "C:\\Users\\rmaguad\\OneDrive - NANDPS\\Documents\\Work_mirror\\dev\\.venv\\Scripts\\python.exe"
$python = $null

# Try repo venv first
if (Test-Path $repoVenvPython) { 
    $python = $repoVenvPython
}
# Then try user venv
elseif (Test-Path $userVenvPython) { 
    $python = $userVenvPython
}
# Finally fall back to system python3 (NOT python which might be Python 2)
else { 
    $python = "python3"
}

# Chart server script (rev5 includes performance options)
$chartDir = Join-Path $PSScriptRoot "fdv_chart_rev5"
$script = Join-Path $chartDir "fdv_chart.py"

if (-not (Test-Path $script)) {
  Write-Output "Script not found: $script"
  exit 1
}

# Set up environment and run chart server
$env:PORT = $Port
$env:FDV_CHART_REV = 'rev5'
$env:PYTHONPATH = $chartDir

Write-Output "Starting FDV Chart rev5 (Performance Options) on port $Port..."
Write-Output "Features: Optional Sampling Modes (None/Random/Decimation), Render Modes (Canvas/WebGL)"
Write-Output "  - Navigate to: http://localhost:$Port/"
Write-Output "  - Performance controls available in dropdown selectors"
Write-Output ""

# Run the chart server with full path
$pythonExe = if ($python -and (Test-Path $python)) { $python } else { "python" }

# Change to the chart rev5 directory first
Push-Location $chartDir
try {
    & $pythonExe fdv_chart.py $Port
} finally {
    Pop-Location
}
