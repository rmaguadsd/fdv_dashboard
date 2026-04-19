# Restart FDV Chart Server with fixes applied
Write-Host "Killing existing Python processes..." -ForegroundColor Cyan
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

Write-Host "Starting FDV Chart Server on port 5058..." -ForegroundColor Cyan
$process = Start-Process -FilePath "py" -ArgumentList '-3', "D:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py" -PassThru -NoNewWindow

Start-Sleep -Seconds 5

# Check if server is running
$port_check = netstat -ano 2>$null | Select-String "5058"
if ($port_check) {
    Write-Host "✓ Server is listening on port 5058" -ForegroundColor Green
} else {
    Write-Host "✗ Server may not be listening yet, checking again..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    $port_check = netstat -ano 2>$null | Select-String "5058"
    if ($port_check) {
        Write-Host "✓ Server is listening on port 5058" -ForegroundColor Green
    } else {
        Write-Host "✗ Server not listening - may need manual restart" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Access FDV Chart at: http://localhost:5058/" -ForegroundColor Cyan
Write-Host "Server PID: $($process.Id)" -ForegroundColor Cyan
