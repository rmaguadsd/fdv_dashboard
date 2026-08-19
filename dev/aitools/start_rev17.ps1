param([switch]$StopOnly)

# Stop existing process on port 5060
try {
    $conns = Get-NetTCPConnection -LocalPort 5060 -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Output "Killed PID $procId on port 5060"
        }
    }
} catch {
    Write-Output "No listeners on port 5060"
}

if ($StopOnly) {
    Write-Output "StopOnly specified; exiting."
    exit 0
}

# Setup directories and files
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$chartScript = Join-Path $PSScriptRoot "fdv_chart_rev17\fdv_chart.py"
$chartWorkDir = Join-Path $PSScriptRoot "fdv_chart_rev17"
$chartLog = Join-Path $logDir "fdv_chart_rev17_server.log"
$chartErr = Join-Path $logDir "fdv_chart_rev17_server.err.log"

"[restart] Launching fdv_chart_rev17.py at $(Get-Date -Format o)" | Out-File -FilePath $chartLog -Append -Encoding utf8

# Try to find Python
$repoRoot = Split-Path $PSScriptRoot -Parent
$repoVenvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = $null
if (Test-Path $repoVenvPython) { $python = $repoVenvPython }

if ($python -and (Test-Path $python)) {
    Write-Output "Starting fdv_chart_rev17.py via venv: $python $chartScript"
    Start-Process -FilePath $python -ArgumentList $chartScript, "5060", "D:\FDV\recipes" -WorkingDirectory $chartWorkDir -RedirectStandardOutput $chartLog -RedirectStandardError $chartErr -WindowStyle Hidden
} else {
    Write-Output "Starting fdv_chart_rev17.py via py -3.12"
    Start-Process -FilePath "py" -ArgumentList "-3.12", $chartScript, "5060", "D:\FDV\recipes" -WorkingDirectory $chartWorkDir -RedirectStandardOutput $chartLog -RedirectStandardError $chartErr -WindowStyle Hidden
}

Write-Output "fdv_chart_rev17.py server (re)started on port 5060"
