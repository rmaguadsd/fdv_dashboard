import pathlib

script = r"""param([int]$Port = 5059)

$PROD_PORT = 5058
$STORE_DIR = 'D:\FDV\recipes'
$SCRIPT    = Join-Path $PSScriptRoot 'fdv_chart\fdv_chart.py'
$WORK_DIR  = Join-Path $PSScriptRoot 'fdv_chart'
$LOG_FILE  = Join-Path $PSScriptRoot ("fdv_chart\server_dev_" + $Port + ".log")
$ERR_FILE  = Join-Path $PSScriptRoot ("fdv_chart\server_dev_" + $Port + ".err.log")

Write-Host "=== FDV Dev Server Restart ===" -ForegroundColor Cyan
Write-Host "  Dev  port : $Port"
Write-Host "  Prod port : $PROD_PORT (NOT touched)"
Write-Host "  Store dir : $STORE_DIR"

$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conns) {
    $pids2 = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids2) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Write-Host "  Killed PID $p on port $Port" -ForegroundColor Yellow
    }
} else {
    Write-Host "  No listener on port $Port"
}
Start-Sleep -Milliseconds 600

$prodAlive = Get-NetTCPConnection -LocalPort $PROD_PORT -State Listen -ErrorAction SilentlyContinue
if ($prodAlive) {
    Write-Host "  Prod server on $PROD_PORT still alive" -ForegroundColor Green
} else {
    Write-Host "  WARNING: prod server on $PROD_PORT not detected." -ForegroundColor Red
}

$repoRoot   = Split-Path $PSScriptRoot -Parent
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

Write-Host "  Starting dev server on port $Port ..." -ForegroundColor Cyan
if (Test-Path $venvPython) {
    Start-Process -FilePath $venvPython -ArgumentList $SCRIPT, $Port, $STORE_DIR -WorkingDirectory $WORK_DIR -RedirectStandardOutput $LOG_FILE -RedirectStandardError $ERR_FILE -WindowStyle Hidden
} else {
    Start-Process -FilePath 'py' -ArgumentList '-3.12', $SCRIPT, $Port, $STORE_DIR -WorkingDirectory $WORK_DIR -RedirectStandardOutput $LOG_FILE -RedirectStandardError $ERR_FILE -WindowStyle Hidden
}

Start-Sleep -Milliseconds 1400
$devAlive = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($devAlive) {
    Write-Host "  Dev  server RUNNING   -> http://localhost:$Port/" -ForegroundColor Green
    Write-Host "  Prod server UNTOUCHED -> http://localhost:$PROD_PORT/" -ForegroundColor Cyan
    Write-Host "  Log: $LOG_FILE"
} else {
    Write-Host "  Dev server FAILED on port $Port" -ForegroundColor Red
    Get-Content $LOG_FILE -Tail 20 -ErrorAction SilentlyContinue
}
"""

dest = pathlib.Path(r'D:\FDV\git\fdv_dashboard\dev\aitools\restart_dev.ps1')
dest.write_text(script, encoding='utf-8')
print(f'Written {len(script)} bytes to {dest}')
