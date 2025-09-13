param([int]$Port = 5057)

function Stop-ByPort([int]$Port) {
  try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  } catch {
    $conns = @()
  }
  if ($conns) {
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
      try {
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Write-Output "Killed PID $pid on port $Port"
      } catch {
        Write-Output ("Failed to kill PID {0} on port {1}: {2}" -f $pid, $Port, $_.Exception.Message)
      }
    }
  } else {
    Write-Output "No listeners on port $Port"
  }
}

Stop-ByPort -Port $Port

$python = "C:\Users\rmaguad\OneDrive - NANDPS\Documents\Work_mirror\dev\.venv\Scripts\python.exe"
$script = "C:\Users\rmaguad\OneDrive - NANDPS\Documents\Work_mirror\dev\aitools\fdv_report2_webapp_run.py"

if (-not (Test-Path $python)) {
  Write-Output "Python not found at expected venv path: $python"
  exit 1
}
if (-not (Test-Path $script)) {
  Write-Output "Script not found: $script"
  exit 1
}

Write-Output "Starting REPORT: $python $script"
Start-Process -FilePath $python -ArgumentList $script -WindowStyle Hidden
