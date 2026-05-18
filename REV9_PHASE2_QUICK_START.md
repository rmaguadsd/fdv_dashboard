# REV9 PHASE 2: QUICK START - RUN TESTS NOW

**Last Updated**: May 18, 2026  
**Status**: ✅ All systems ready  
**Estimated Time**: 2-3 hours (mostly waiting for parse to complete)

---

## 60-Second Setup

### Terminal 1: Start Server
```powershell
cd d:\FDV\git\fdv_dashboard
python3 dev/aitools/fdv_chart_rev9/fdv_chart.py 5059
```

**Expected Output**:
```
PYTHON_START
Starting FDV Chart Parser...
Port      : 5059
FDV Chart Parser is running at http://0.0.0.0:5059 (all interfaces)
Press Ctrl+C to stop
```

✓ Server is ready (leave this terminal running)

---

## Test 1: Dynamic Timeout (20 minutes)

**Objective**: Verify 1.4GB file parses without timing out

### Terminal 2: Run Parse
```powershell
# Get test file
$testFile = Get-Item "D:\FDV\logs\A2\DOE\PPSR\*.txt" | Sort-Object Length -Descending | Select-Object -First 1
Write-Host "Using: $($testFile.Name) ($([math]::Round($testFile.Length/1GB,2)) GB)"

# Build request
$form = @{
    file  = $testFile
    regex = "FDV OUT.*::READ_RBER_PAGE.*"
}

# Send to server (this will take 15-25 minutes)
Write-Host "Starting parse... this may take 15-25 minutes"
Write-Host "Watch the server terminal for progress"
$result = Invoke-RestMethod -Uri "http://localhost:5059/parse" -Method Post -Form $form -TimeoutSec 1800

# Save result
$result | ConvertTo-Json | Out-File "test_result_1.json"
Write-Host "Test 1 Complete!" -ForegroundColor Green
$result
```

**Expected Result**:
```json
{
  "success": true,
  "csv_id": "csv_XXXXX",
  "match_count": 12345
}
```

**Success Criteria**: ✓ Completes in 15-25 min (not killed at 10 min)

---

## Test 2: CSV Download (1 minute)

**Objective**: Verify CSV export works

### Terminal 2 (Continued):
```powershell
# Get CSV ID from test 1 result (or read from file)
$csvId = $result.csv_id
Write-Host "Downloading CSV: $csvId"

# Download CSV
$csvPath = "C:\Temp\export_$csvId.csv"
Invoke-WebRequest -Uri "http://localhost:5059/download_csv/$csvId" -OutFile $csvPath -TimeoutSec 300

# Check file
$file = Get-Item $csvPath
Write-Host "Downloaded: $($file.Name)" -ForegroundColor Green
Write-Host "Size: $([math]::Round($file.Length/1MB,1)) MB" -ForegroundColor Green

# Verify it's a CSV
$content = Get-Content $csvPath -TotalCount 2
Write-Host "First line: $($content[0].Substring(0,50))..." -ForegroundColor Green
```

**Expected Result**: CSV file of reasonable size, valid CSV format

**Success Criteria**: ✓ File downloads and opens in Excel

---

## Test 3: Pagination (30 seconds)

**Objective**: Verify pagination is fast

### Terminal 2 (Continued):
```powershell
# Test pagination at different offsets
$csvId = $result.csv_id
$baseUrl = "http://localhost:5059/rows"

Write-Host "Testing pagination..."

# Offset 0
$t1 = Measure-Command {
    $r1 = Invoke-RestMethod -Uri "$baseUrl?csv_id=$csvId&offset=0&limit=1000"
}
Write-Host "Offset 0:    $($r1.rows.Count) rows in $($t1.TotalMilliseconds)ms" -ForegroundColor Green

# Offset 10000
$t2 = Measure-Command {
    $r2 = Invoke-RestMethod -Uri "$baseUrl?csv_id=$csvId&offset=10000&limit=1000"
}
Write-Host "Offset 10000: $($r2.rows.Count) rows in $($t2.TotalMilliseconds)ms" -ForegroundColor Green

# Offset large
$offset = $result.match_count - 1000
$t3 = Measure-Command {
    $r3 = Invoke-RestMethod -Uri "$baseUrl?csv_id=$csvId&offset=$offset&limit=1000"
}
Write-Host "Offset $offset: $($r3.rows.Count) rows in $($t3.TotalMilliseconds)ms" -ForegroundColor Green
```

**Expected Result**: All responses < 100ms

**Success Criteria**: ✓ All queries return in < 100ms

---

## Test 4: Job Status (10 seconds)

**Objective**: Verify job status tracking

### Terminal 2 (Continued):
```powershell
# Check job status
$jobId = $result.csv_id -replace "csv_", "job_"
$status = Invoke-RestMethod -Uri "http://localhost:5059/job_status/$jobId"

Write-Host "Job Status:" -ForegroundColor Green
Write-Host "  State: $($status.state)" -ForegroundColor Green
Write-Host "  Elapsed: $($status.elapsed_seconds)s" -ForegroundColor Green

# Show full result
$status | ConvertTo-Json
```

**Expected Result**:
```json
{
  "success": true,
  "job_id": "job_XXXXX",
  "state": "done",
  "elapsed_seconds": 1200,
  "result": {...}
}
```

**Success Criteria**: ✓ state="done", elapsed_seconds > 0

---

## Test 5: Memory Check (During all tests)

**Objective**: Verify memory stays bounded

### Optional: Open Task Manager
1. Press `Ctrl+Shift+Esc` to open Task Manager
2. Go to Processes tab
3. Find `python.exe` (REV9 server)
4. Watch Memory column during parse

**Expected**: Peak memory < 200MB (was 50GB before)

**Success Criteria**: ✓ Memory < 200MB throughout

---

## Summary Results

After all tests, you should see:
```
TEST RESULTS:
  Test 1 - Dynamic Timeout:   PASS (completed in 15-25 min)
  Test 2 - CSV Download:      PASS (downloaded successfully)
  Test 3 - Pagination:        PASS (all queries < 100ms)
  Test 4 - Job Status:        PASS (state shows "done")
  Test 5 - Memory:            PASS (peak < 200MB)

Overall: ALL TESTS PASSED ✓
```

---

## If Tests FAIL

### Test 1 Times Out
- Problem: Parse killed at 10 minutes
- Solution: Check if code update deployed correctly
- Verify: `grep "file_size_gb" dev/aitools/fdv_chart_rev9/fdv_chart.py`

### Test 2 Fails
- Problem: CSV download not working
- Solution: Server may need restart
- Verify: `grep "download_csv" dev/aitools/fdv_chart_rev9/fdv_chart.py`

### Test 3 Slow
- Problem: Pagination > 100ms
- Solution: Check SQLite queries
- Verify: `grep "LIMIT" dev/aitools/fdv_chart_rev9/fdv_chart.py`

### Test 4 Errors
- Problem: Job status not available
- Solution: Check endpoint implementation
- Verify: `grep "job_status" dev/aitools/fdv_chart_rev9/fdv_chart.py`

### Memory Too High
- Problem: Memory > 500MB
- Solution: Possible memory leak
- Action: Restart server and retry

---

## Quick Reference Commands

### Setup
```powershell
# Terminal 1
python3 dev/aitools/fdv_chart_rev9/fdv_chart.py 5059

# Terminal 2
cd d:\FDV\git\fdv_dashboard
```

### Run All Tests
```powershell
# Get test file
$testFile = Get-Item "D:\FDV\logs\A2\DOE\PPSR\*.txt" | Sort Length -Desc | Select -First 1

# Run parse
$r = Invoke-RestMethod -Uri "http://localhost:5059/parse" -Method Post -Form @{file=$testFile;regex="FDV OUT.*::READ_RBER_PAGE.*"} -TimeoutSec 1800

# Test CSV download
Invoke-WebRequest -Uri "http://localhost:5059/download_csv/$($r.csv_id)" -OutFile "export.csv"

# Test pagination
Invoke-RestMethod -Uri "http://localhost:5059/rows?csv_id=$($r.csv_id)&offset=0&limit=1000"

# Test job status
Invoke-RestMethod -Uri "http://localhost:5059/job_status/$($r.csv_id -replace 'csv_','job_')"
```

---

## Timeline

| Time | Activity | Duration |
|------|----------|----------|
| 0:00 | Start server | 2 min |
| 0:02 | Prepare test file | 1 min |
| 0:03 | Start parse (Test 1) | 15-25 min ⏳ |
| 0:28 | Download CSV (Test 2) | 1 min |
| 0:29 | Test pagination (Test 3) | 1 min |
| 0:30 | Test job status (Test 4) | 1 min |
| 0:31 | Review results | 5-10 min |

**Total**: ~45 minutes (mostly waiting)

---

## Documentation

**For More Details, See**:
- `REV9_PHASE2_TESTING_GUIDE.md` - Detailed test procedures
- `REV9_PHASE2_EXECUTIVE_SUMMARY.md` - Overview
- `REV9_PHASE2_TEST_SUMMARY.md` - Expected outcomes

---

## Go/No-Go Decision

### After Tests PASS ✓
**Option A**: Deploy immediately
```
All tests passed → Production ready
Timeline: Deploy today
```

**Option B**: Phase 3 first (UI improvements)
```
Add progress bar, cancel button, ETA
Timeline: 1-2 weeks
```

**Option C**: Phase 4 first (benchmarking)
```
Performance profiling, stress tests
Timeline: 1 week
```

### After Tests FAIL ✗
**Steps**:
1. Review error message
2. Check server logs
3. Verify code update
4. Run diagnostic test
5. Fix issue and retry

---

## Contact & Support

**Questions?** See documentation:
- Technical: `REV9_PHASE2_IMPLEMENTATION_COMPLETE.md`
- Testing: `REV9_PHASE2_TESTING_GUIDE.md`
- Architecture: `REV9_5GB_SUPPORT_IMPLEMENTATION_PLAN.md`

---

**Ready to Test?** 👉 Start with Terminal 1 command above!

