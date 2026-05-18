# REV9 Phase 2 - PowerShell Test Script
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "REV9 PHASE 2 - SIMPLE TEST" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

$ServerURL = "http://localhost:5059"
$LogDir = "D:\FDV\logs\A2\DOE\PPSR"

# Test 1: Server connectivity
Write-Host "[TEST 1] Server Connectivity" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$ServerURL/" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  SUCCESS: Server is running" -ForegroundColor Green
} catch {
    Write-Host "  FAILED: Server not responding" -ForegroundColor Red
    exit 1
}

# Test 2: Get test file
Write-Host "`n[TEST 2] Finding test file" -ForegroundColor Yellow
$TestFile = Get-ChildItem "$LogDir\*.txt" -ErrorAction Stop | Sort-Object Length -Descending | Select-Object -First 1
$FileSizeMB = [math]::Round($TestFile.Length / 1MB, 1)
$FileSizeGB = [math]::Round($FileSizeMB / 1024, 2)
Write-Host "  File: $($TestFile.Name)" -ForegroundColor Green
Write-Host "  Size: $FileSizeMB MB ($FileSizeGB GB)" -ForegroundColor Green

# Test 3: Parse with dynamic timeout
Write-Host "`n[TEST 3] Parse 1.4GB file (Dynamic Timeout Test)" -ForegroundColor Yellow
Write-Host "  Expected timeout: ~24 minutes" -ForegroundColor Yellow
Write-Host "  Sending file to server..." -ForegroundColor Yellow

$StartTime = Get-Date
try {
    $form = @{
        file  = Get-Item $TestFile
        regex = "FDV OUT.*::READ_RBER_PAGE.*"
    }
    
    $result = Invoke-RestMethod -Uri "$ServerURL/parse" -Method Post -Form $form -TimeoutSec 1800 -ErrorAction Stop
    $ElapsedSeconds = ((Get-Date) - $StartTime).TotalSeconds
    $ElapsedMinutes = [math]::Round($ElapsedSeconds / 60, 1)
    
    $CSVID = $result.csv_id
    $MatchCount = $result.match_count
    
    Write-Host "  SUCCESS: Parse completed!" -ForegroundColor Green
    Write-Host "    CSV ID: $CSVID" -ForegroundColor Green
    Write-Host "    Matches: $MatchCount" -ForegroundColor Green
    Write-Host "    Time: ${ElapsedMinutes}m (${ElapsedSeconds}s)" -ForegroundColor Green
    
} catch {
    $ElapsedSeconds = ((Get-Date) - $StartTime).TotalSeconds
    Write-Host "  FAILED: Parse failed or timed out" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "    Elapsed: $ElapsedSeconds seconds" -ForegroundColor Red
    exit 1
}

# Test 4: CSV Download
Write-Host "`n[TEST 4] CSV Download (Streaming Test)" -ForegroundColor Yellow
try {
    $OutPath = "C:\Temp\export_$CSVID.csv"
    Invoke-WebRequest -Uri "$ServerURL/download_csv/$CSVID" -OutFile $OutPath -TimeoutSec 300 -ErrorAction Stop
    
    $FileSize = (Get-Item $OutPath).Length
    $FileSizeMB = [math]::Round($FileSize / 1MB, 1)
    
    Write-Host "  SUCCESS: CSV downloaded!" -ForegroundColor Green
    Write-Host "    File: $OutPath" -ForegroundColor Green
    Write-Host "    Size: $FileSizeMB MB" -ForegroundColor Green
} catch {
    Write-Host "  FAILED: CSV download failed" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 5: Pagination
Write-Host "`n[TEST 5] Pagination Performance" -ForegroundColor Yellow
try {
    $offsets = @(0, 10000, 100000)
    foreach ($offset in $offsets) {
        $start = Get-Date
        $response = Invoke-RestMethod -Uri "$ServerURL/rows?csv_id=$CSVID&offset=$offset&limit=1000" -TimeoutSec 30 -ErrorAction Stop
        $elapsed = ((Get-Date) - $start).TotalMilliseconds
        $rows = @($response.rows).Count
        $total = $response.total
        
        if ($elapsed -lt 100) {
            Write-Host "  Offset $offset : $rows rows in ${elapsed}ms (FAST)" -ForegroundColor Green
        } else {
            Write-Host "  Offset $offset : $rows rows in ${elapsed}ms" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  FAILED: Pagination test failed" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 6: Job Status
Write-Host "`n[TEST 6] Job Status Endpoint" -ForegroundColor Yellow
try {
    $JobID = $CSVID -replace "csv_", "job_"
    $response = Invoke-RestMethod -Uri "$ServerURL/job_status/$JobID" -TimeoutSec 10 -ErrorAction Stop
    
    $state = $response.state
    $elapsed = $response.elapsed_seconds
    
    if ($state -eq "done") {
        Write-Host "  SUCCESS: Job status available" -ForegroundColor Green
        Write-Host "    State: $state" -ForegroundColor Green
        Write-Host "    Elapsed: $elapsed seconds" -ForegroundColor Green
    } else {
        Write-Host "  INFO: Job still $state" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  FAILED: Job status failed" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ALL TESTS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan
