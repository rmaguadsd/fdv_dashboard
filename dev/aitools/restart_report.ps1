param([int]$Port = 5057, [switch]$StopOnly)

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
if (Test-Path $repoVenvPython) { $python = $repoVenvPython }
elseif (Test-Path $userVenvPython) { $python = $userVenvPython }
else { $python = $null }

# Use a dedicated runner that imports the Flask app and starts it
$script = Join-Path $PSScriptRoot "run_report2.py"

if (-not (Test-Path $script)) {
  Write-Output "Script not found: $script"
  exit 1
}

# Ensure env for host/port
$env:FDV_REPORT2_PORT = "$Port"
$env:FDV_REPORT2_HOST = "0.0.0.0"

# Ensure we run from the aitools folder so relative imports and templates work
$workDir = $PSScriptRoot

# Prepare logs
$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "report2_server.log"
$errFile = Join-Path $logDir "report2_server.err.log"
"[restart] Launching at $(Get-Date -Format o) using $script" | Out-File -FilePath $logFile -Append -Encoding utf8

if ($python -and (Test-Path $python)) {
  Write-Output "Starting REPORT via venv: $python $script"
  Start-Process -FilePath $python -ArgumentList $script -WorkingDirectory $workDir -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Hidden
}
else {
  # Fallback to Python launcher
  Write-Output "Venv python not found. Falling back to 'py -3.12'"
  Start-Process -FilePath "py" -ArgumentList "-3.12", $script -WorkingDirectory $workDir -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Hidden
}
