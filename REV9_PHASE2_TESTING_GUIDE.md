# REV9 PHASE 2 TESTING GUIDE - COMPLETE

**Date**: May 18, 2026  
**Status**: Ready for comprehensive testing  
**Duration**: ~2-3 hours including parse time

---

## Quick Start (Immediate)

### 1. Server Startup
```powershell
cd d:\FDV\git\fdv_dashboard
python3 dev/aitools/fdv_chart_rev9/fdv_chart.py 5059
```

Expected output:
```
PYTHON_START
Starting FDV Chart Parser...
Port      : 5059
Store dir : D:\FDV\recipes
FDV Chart Parser is running at http://0.0.0.0:5059 (all interfaces)
Press Ctrl+C to stop
```

### 2. Test Files Available
Located in: `D:\FDV\logs\A2\DOE\PPSR`

| File Size | Time | Purpose |
|-----------|------|---------|
| 1.4 GB | 20 min | Dynamic Timeout Test |
| 1.4 GB | 20 min | CSV Download Test |
| Any | <1s | Pagination Test |

---

## TEST 1: Dynamic Timeout (Phase 2a)

**Objective**: Verify timeout scales with file size (1.4GB should get ~20 min timeout)

### Steps:
1. Keep server running in one terminal
2. In another terminal, upload 1.4GB file:

```powershell
# Prepare the request
$file = "D:\FDV\logs\A2\DOE\PPSR\Output_site111_5_15_2026_14_02_15_FDVLOG_4_tb_set_utility_PROGRAM_SUSPEND_HOTE_REL005.txt"
$url = "http://localhost:5059/parse"
$regex = "FDV OUT.*::READ_RBER_PAGE.*"

# Upload (this will take 15-20 minutes)
$form = @{
    file  = Get-Item $file
    regex = $regex
}

$result = Invoke-RestMethod -Uri $url -Method Post -Form $form -TimeoutSec 1800
$result | ConvertTo-Json
```

### Expected Results:
- ✓ Parse completes without timeout (OLD: timed out at 10 min)
- ✓ CSV ID returned (e.g., "csv_abc12345")
- ✓ Match count > 0
- ✓ Time: 15-25 minutes (not killed at 10 min)
- ✓ Memory stays ~150MB (not 50GB)

### Success Criteria:
```
{
  "success": true,
  "csv_id": "csv_XXXXX",
  "match_count": 12345,
  "message": "Parse completed"
}
```

---

## TEST 2: CSV Download (Phase 2b)

**Objective**: Verify CSV download endpoint works without memory spikes

### Prerequisites:
- Complete TEST 1 first (need CSV ID from parse result)
- Note the `csv_id` value

### Steps:
1. In new terminal, download CSV:

```powershell
# Replace csv_abc12345 with actual CSV ID from TEST 1
$csv_id = "csv_abc12345"
$url = "http://localhost:5059/download_csv/$csv_id"
$output = "C:\Temp\export_$csv_id.csv"

# Download (should complete in 30-60 seconds)
Invoke-WebRequest -Uri $url -OutFile $output -TimeoutSec 300

# Verify file
Get-Item $output | Select-Object Name, Length
```

### Expected Results:
- ✓ File downloads successfully
- ✓ File is valid CSV (has headers, comma-separated columns)
- ✓ File size = match_count * ~100 bytes (varies)
- ✓ Download time < 60 seconds
- ✓ Memory spike < 50MB (streaming, not buffering all rows)

### Success Criteria:
```
Download completes in < 60 seconds
File opens in Excel without errors
File has N rows = match_count from TEST 1
```

---

## TEST 3: Pagination Performance (Phase 2c)

**Objective**: Verify pagination is fast even with large result sets

### Prerequisites:
- Complete TEST 1 first (need CSV ID)
- Note the `csv_id` and `match_count` values

### Steps:
1. Test various offsets:

```powershell
$csv_id = "csv_abc12345"
$base_url = "http://localhost:5059/rows"

# Test 1: First page
$response = Invoke-RestMethod -Uri "$base_url?csv_id=$csv_id&offset=0&limit=1000"
$response | Add-Member -NotePropertyName "test" -NotePropertyValue "offset=0" -PassThru

# Test 2: Middle page
$response = Invoke-RestMethod -Uri "$base_url?csv_id=$csv_id&offset=10000&limit=1000"
$response | Add-Member -NotePropertyName "test" -NotePropertyValue "offset=10000" -PassThru

# Test 3: Large offset (if match_count > 100K)
$response = Invoke-RestMethod -Uri "$base_url?csv_id=$csv_id&offset=100000&limit=1000"
$response | Add-Member -NotePropertyName "test" -NotePropertyValue "offset=100000" -PassThru
```

### Expected Results:
- ✓ All responses return in < 100ms (OLD: 2-10 seconds)
- ✓ Correct number of rows (1000 except maybe last page)
- ✓ `total` field matches match_count from TEST 1
- ✓ `has_more` field indicates if more pages exist
- ✓ Memory stays constant (no loading all rows)

### Success Criteria:
```
Response time: < 100ms for all offsets
Rows per page: 1000 (or less on last page)
Total matches: Matches from TEST 1
Memory: <5MB per request
```

---

## TEST 4: Job Status (Phase 2d)

**Objective**: Verify job status endpoint tracks parse progress

### Prerequisites:
- Complete TEST 1 first
- Note the `csv_id` value

### Steps:
1. Check job status:

```powershell
$csv_id = "csv_abc12345"
$job_id = $csv_id -replace "csv_", "job_"
$url = "http://localhost:5059/job_status/$job_id"

$response = Invoke-RestMethod -Uri $url
$response | ConvertTo-Json
```

### Expected Results (after parsing completes):
- ✓ `state` = "done"
- ✓ `elapsed_seconds` = actual parse time (15-25 minutes)
- ✓ `result` contains parsing details
- ✓ No error field (or error = null)

### Success Criteria:
```
{
  "success": true,
  "job_id": "job_XXXXX",
  "state": "done",
  "elapsed_seconds": 1200,
  "result": {
    "csv_id": "csv_XXXXX",
    "match_count": 12345
  }
}
```

---

## TEST 5: Memory Efficiency (Phase 1+2 Combined)

**Objective**: Verify Phase 1+2 combined achieves 99%+ memory reduction

### Prerequisites:
- Complete TEST 1 (parse running/completed)
- Monitor system while parsing

### Steps:
1. Open Task Manager → Processes → python3 (fdv_chart)
2. Watch Memory column during parse
3. Note peak memory usage

### Expected Results:
- ✓ Peak memory < 200MB (OLD: 50GB)
- ✓ Memory stable throughout parse
- ✓ No memory leaks after multiple parses
- ✓ Disk usage < 1GB (SQLite cache)

### Success Criteria:
```
Peak Memory: < 200MB (was 50GB)
Reduction: 99.7%
Sustained Memory: <150MB
```

---

## Summary Table: All 5 Tests

| Test | Feature | Old Behavior | New Behavior | Status |
|------|---------|--------------|--------------|--------|
| **Test 1** | Dynamic Timeout | Timeout at 10 min | Scale with file size | [Test] |
| **Test 2** | CSV Download | N/A (not supported) | Streaming download | [Test] |
| **Test 3** | Pagination | 2-10 sec, memory spikes | <100ms, constant memory | [Test] |
| **Test 4** | Job Status | N/A (not available) | Real-time tracking | [Test] |
| **Test 5** | Memory | 50GB peak | <200MB peak | [Test] |

---

## Troubleshooting

### Issue: Server Won't Start
```powershell
# Check if port is in use
netstat -ano | findstr :5059

# Kill existing process if needed
Stop-Process -Id XXXXX -Force
```

### Issue: Parse Times Out
```
If parse times out after 10 minutes, check:
1. Code not updated: Restart server
2. Wrong file path: Check file exists
3. Regex error: Test regex locally
```

### Issue: Low Match Count
```
Check regex is correct:
- Regex: FDV OUT.*::READ_RBER_PAGE.*
- File should have millions of lines
- Search is case-sensitive
```

### Issue: Pagination Slow
```
If pagination > 100ms:
1. Server not using Phase 2 code
2. SQLite database corrupt
3. System disk too slow
```

---

## Performance Benchmarks (Expected)

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Parse 1.4GB file | 15-25 min | <200MB | Depends on file I/O speed |
| Download CSV (12K rows) | <60s | <50MB | Streaming in 10K batches |
| Paginate 1M rows (offset=0) | <100ms | <5MB | Direct SQLite query |
| Paginate 1M rows (offset=500K) | <100ms | <5MB | Same performance at any offset |
| Job status check | <10ms | <1MB | Non-blocking status query |

---

## Validation Checklist

After all 5 tests complete:

- [ ] Test 1: Parse completes without timeout
- [ ] Test 2: CSV exports successfully
- [ ] Test 3: Pagination < 100ms
- [ ] Test 4: Job status shows complete
- [ ] Test 5: Memory < 200MB peak
- [ ] No server crashes during tests
- [ ] All endpoints return valid JSON
- [ ] No error messages in logs
- [ ] File system not full (check temp folder)
- [ ] Process cleans up after exit

---

## Next Steps After Testing

### If All Tests PASS ✓
1. **Phase 3** (Optional): Add frontend progress UI
   - Progress bar showing % complete
   - Cancel/pause buttons
   - ETA calculation
   
2. **Phase 4**: Production deployment
   - Deploy to production servers
   - Monitor real usage
   - Collect performance metrics

### If Any Test FAILS ✗
1. Review error message
2. Check server logs
3. Verify code deployed correctly
4. Review Phase 2 implementation details
5. Run diagnostic tests

---

## Additional Notes

### About Timeouts

**Phase 2a Dynamic Timeout Calculation**:
```
Base timeout: 600 seconds (10 minutes)
Add: 600 seconds per GB of file
Cap: 3600 seconds (60 minutes)

Examples:
- 0.5 GB: 600s (baseline)
- 1.0 GB: 1200s (20 min)
- 1.4 GB: 1440s (24 min)
- 2.0 GB: 1800s (30 min)
- 5.0 GB: 3600s (60 min - capped)
```

### About Memory Efficiency

**Phase 1 Batching**:
- Rows written to SQLite in 50K batches
- Memory freed after each batch
- Total memory = buffer size (~150MB), not file size

**Phase 2 Pagination**:
- Queries SQLite with LIMIT/OFFSET
- Only requested rows loaded to memory
- Response time constant regardless of offset

### About CSV Downloads

**Streaming Strategy**:
- Rows fetched from SQLite in 10K batches
- Written to CSV file directly
- Memory = 10K rows * ~100 bytes = ~1MB
- Works for unlimited row counts

---

## Test Commands Quick Reference

```powershell
# Start server
python3 dev/aitools/fdv_chart_rev9/fdv_chart.py 5059

# Upload and parse file
$result = Invoke-RestMethod -Uri "http://localhost:5059/parse" -Method Post -Form @{
    file  = Get-Item "D:\FDV\logs\A2\DOE\PPSR\*.txt" -ErrorAction Stop | Sort-Object Length | Select -Last 1
    regex = "FDV OUT.*::READ_RBER_PAGE.*"
} -TimeoutSec 1800

# Download CSV
Invoke-WebRequest -Uri "http://localhost:5059/download_csv/$($result.csv_id)" -OutFile "export.csv"

# Check pagination
Invoke-RestMethod -Uri "http://localhost:5059/rows?csv_id=$($result.csv_id)&offset=0&limit=1000"

# Check job status
Invoke-RestMethod -Uri "http://localhost:5059/job_status/$($result.csv_id -replace 'csv_','job_')"
```

---

**READY TO TEST** - All Phase 2 features implemented and syntax-checked ✓

