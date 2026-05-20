#!/usr/bin/env powershell
<#
.SYNOPSIS
    Launch FDV Chart Rev3 on port 5059
.DESCRIPTION
    Starts the Rev3 chart server with the enhanced features:
    - Font resizing (axis, labels, point labels)
    - Text annotations anywhere in the chart
    - Enhanced markers with labels and chart targeting
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonFile = Join-Path $ScriptDir "fdv_chart.py"
$LogDir = Join-Path $ScriptDir "logs"

# Create logs directory if it doesn't exist
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$LogFile = Join-Path $LogDir "rev3_startup.log"
$ErrorLogFile = Join-Path $LogDir "rev3_error.log"

Write-Host "Starting FDV Chart Rev3..." -ForegroundColor Cyan
Write-Host "Port: 5059" -ForegroundColor Green
Write-Host "Python: $PythonFile" -ForegroundColor Green
Write-Host "Logs: $LogFile, $ErrorLogFile" -ForegroundColor Green
Write-Host ""

# Launch the Python server
try {
    python $PythonFile 5059 2>&1 | Tee-Object -FilePath $LogFile
}
catch {
    Write-Host "Error starting server: $_" -ForegroundColor Red
    exit 1
}
