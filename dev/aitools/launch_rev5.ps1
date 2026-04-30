param([int]$Port = 5059, [switch]$StopOnly)

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
$repoRoot = Split-Path $PSScriptRoot -Parent
$repoVenvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$userVenvPython = "C:\\Users\\rmaguad\\OneDrive - NANDPS\\Documents\\Work_mirror\\dev\\.venv\\Scripts\\python.exe"
$python = $null
if (Test-Path $repoVenvPython) { 
    $python = $repoVenvPython 
}
elseif (Test-Path $userVenvPython) { 
    $python = $userVenvPython 
}
else { 
    # Fall back to python3 (NOT python which might be Python 2)
    $python = "python3"
}

# Use the standard FDV report launcher (same app, different port and rev5 assets)
$script = Join-Path $PSScriptRoot "run_report2.py"

if (-not (Test-Path $script)) {
  Write-Output "Script not found: $script"
  exit 1
}

# Set environment variables for rev5
$env:FDV_REPORT2_PORT = $Port
$env:FDV_REPORT2_HOST = '0.0.0.0'
$env:FDV_REPORT2_DEBUG = '1'
$env:FDV_CHART_REV = 'rev5'

Write-Output "Starting FDV Report (rev5 - Performance Options) on port $Port..."
if ($python) {
  & $python $script
} else {
  & python $script
}
