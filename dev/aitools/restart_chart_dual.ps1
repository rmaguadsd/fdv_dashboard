param(
    [string]$Version = 'both',  # 'both', 'stable', 'dev'
    [switch]$StopOnly
)

<#
.SYNOPSIS
Restart FDV Chart servers on dual ports.

.PARAMETER Version
  'both'   - Restart both 5058 (stable) and 5059 (dev)
  'stable' - Restart only 5058 (fdv_chart)
  'dev'    - Restart only 5059 (fdv_chart_rev1)

.PARAMETER StopOnly
  Stop servers without restarting

.EXAMPLES
  .\restart_chart_dual.ps1                    # Restart both
  .\restart_chart_dual.ps1 -Version stable   # Restart only stable (5058)
  .\restart_chart_dual.ps1 -Version dev      # Restart only dev (5059)
  .\restart_chart_dual.ps1 -StopOnly         # Stop both without restarting
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
        Write-Output "✓ Killed PID $procId on port $Port"
      } catch {
        Write-Output "✗ Failed to kill PID $procId on port $Port"
      }
    }
  } else {
    Write-Output "ℹ No listeners on port $Port"
  }
}

# Determine which ports to manage
$Ports = @()
if ($Version -eq 'both' -or $Version -eq 'stable') { $Ports += 5058 }
if ($Version -eq 'both' -or $Version -eq 'dev') { $Ports += 5059 }

Write-Output "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Output "FDV Chart Dual-Port Manager"
Write-Output "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Output "Ports to manage: $($Ports -join ', ')"
Write-Output ""

# Stop listeners on specified ports
foreach ($Port in $Ports) {
  Stop-ByPort -Port $Port
}

if ($StopOnly) {
  Write-Output ""
  Write-Output "StopOnly specified; exiting after stopping listeners."
  exit 0
}

Write-Output ""
Write-Output "Starting servers..."
Write-Output ""

# Start the servers
if ($Version -eq 'both' -or $Version -eq 'stable') {
  Write-Output "→ Starting fdv_chart on port 5058..."
  Start-Process -NoNewWindow -FilePath "py" -ArgumentList `
    "-3.12", "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py", "5058"
  Write-Output "✓ fdv_chart started (5058)"
}

if ($Version -eq 'both' -or $Version -eq 'dev') {
  Write-Output "→ Starting fdv_chart_rev1 on port 5059..."
  Start-Process -NoNewWindow -FilePath "py" -ArgumentList `
    "-3.12", "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev1\fdv_chart.py", "5059"
  Write-Output "✓ fdv_chart_rev1 started (5059)"
}

Write-Output ""
Write-Output "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Output "Servers started. Waiting 2 seconds for ports to open..."
Start-Sleep -Seconds 2

# Verify ports are listening
Write-Output ""
foreach ($Port in $Ports) {
  try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
      $pid = $conns[0].OwningProcess
      Write-Output "✓ Port $Port is listening (PID $pid)"
    } else {
      Write-Output "✗ Port $Port is NOT listening"
    }
  } catch {
    Write-Output "✗ Error checking port $Port"
  }
}

Write-Output "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
