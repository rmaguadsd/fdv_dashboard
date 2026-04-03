# FDV EIMPRO WebApp - Testing Guide

This guide helps you verify that the FDV EIMPRO WebApp is working correctly.

## Pre-Launch Checklist

### 1. Verify Python Installation
```powershell
py -3 --version
# Expected: Python 3.7 or higher
```

### 2. Verify Dependencies are Installed
```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp"
py -3 -m pip list | Select-String -Pattern "Flask|pandas|numpy|matplotlib|seaborn"
```

Expected output should show all 5 packages.

If not installed:
```powershell
py -3 -m pip install -r requirements.txt
```

### 3. Check Temp Directory
```powershell
# Create if needed
New-Item -ItemType Directory -Path "D:\fdv_eimpro_tmp" -Force -ErrorAction SilentlyContinue
Get-Item "D:\fdv_eimpro_tmp" | Format-Table
```

## Launch Tests

### Test 1: Start Application

```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp"
py -3 fdv_eimpro_webapp.py
```

**Expected Output:**
```
Starting FDV EIMPRO WebApp...
Visit http://localhost:5058
 * Running on http://0.0.0.0:5058
 * Debug mode: off
```

### Test 2: Verify Port is Listening
In another PowerShell window:
```powershell
Test-NetConnection localhost -Port 5058 -InformationLevel Quiet
```

**Expected Output:** `True`

### Test 3: Test HTTP Connection
```powershell
$response = Invoke-WebRequest -Uri "http://localhost:5058" -ErrorAction SilentlyContinue
$response.StatusCode
```

**Expected Output:** `200`

## Functional Tests

### Test 4: Access Web Interface

1. Open browser to: `http://localhost:5058`
2. Verify you see:
   - Header: "FDV EIMPRO Log Parser & Visualizer"
   - 6 navigation tabs
   - Purple gradient styling
   - Upload section with file input

### Test 5: Create Sample Test Data

Create a sample log file `test_sample.log`:

```
Test Start Date (34_tb_set_utility_READ_OPERATIONS): 2025_08_20 Test Start Time: 2:43:29

FDV OUTPUT [D:\NAND\150S\FDV\STAGING\RMAGUAD/BASIC_ERASE_PROGRAM_PAGE_READ_EIMPRO/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV::READ_ECC_BLK_778_PG_44_PGTYPE_LP_WL_2_SB_8_BL_3,SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.725000,SPECOFFSET=0.125,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]: DUT1,PASS,18592,29,0.0016,29,0.00019,0.008,FAILCOUNT_ONLY,
FDV OUTPUT [D:\NAND\150S\FDV\STAGING\RMAGUAD/BASIC_ERASE_PROGRAM_PAGE_READ_EIMPRO/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV::READ_ECC_BLK_778_PG_45_PGTYPE_LP_WL_3_SB_8_BL_3,SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.725000,SPECOFFSET=0.125,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]: DUT1,PASS,18592,10,0.0005,10,0.00007,0.008,FAILCOUNT_ONLY,
FDV OUTPUT [D:\NAND\150S\FDV\STAGING\RMAGUAD/BASIC_ERASE_PROGRAM_PAGE_READ_EIMPRO/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV::READ_ECC_BLK_779_PG_44_PGTYPE_UP_WL_4_SB_8_BL_3,SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.725000,SPECOFFSET=0.125,TM=12,VCC=2.6,VCCQ=1.2,TEMP=30]: DUT2,PASS,18592,50,0.0027,50,0.00034,0.008,FAILCOUNT_ONLY,
FDV OUTPUT [D:\NAND\150S\FDV\STAGING\RMAGUAD/BASIC_ERASE_PROGRAM_PAGE_READ_EIMPRO/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV::READ_ECC_BLK_779_PG_45_PGTYPE_UP_WL_5_SB_8_BL_3,SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.725000,SPECOFFSET=0.125,TM=12,VCC=2.6,VCCQ=1.2,TEMP=30]: DUT2,PASS,18592,20,0.0011,20,0.00014,0.008,FAILCOUNT_ONLY,

FDV POLL [D:\NAND\150S\FDV\STAGING\RMAGUAD/char/array_char/<TR_SSLC>.FDV::<POLL_TR_C0_SP_READ_BLK_364_PG_82_PGTYPE_SSLC_WL_4_SB_10_BL_1,SSYNC=FALSE,TRC=,DUTTEMP=-999,TAC=6.300000,SPECOFFSET=0,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]: DUT1 0,33.205807,39839,[ignore]
FDV POLL [D:\NAND\150S\FDV\STAGING\RMAGUAD/char/array_char/<TR_SSLC>.FDV::<POLL_TR_C0_SP_READ_BLK_365_PG_82_PGTYPE_SSLC_WL_5_SB_10_BL_1,SSYNC=FALSE,TRC=,DUTTEMP=-999,TAC=6.300000,SPECOFFSET=0,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]: DUT2 0,45.123456,39839,[ignore]

Test End Date (34_tb_set_utility_READ_OPERATIONS): 2025_08_20 Test End Time: 2:47:50
```

Save as `test_sample.log`

### Test 6: Test File Upload and Parsing

1. In web browser, click **"📤 Upload & Parse"** tab
2. Click "Select FDV Log File"
3. Choose `test_sample.log`
4. Click **"Parse File"**
5. Wait for spinner to finish

**Expected Results:**
- ✅ "File Parsed Successfully!" message
- ✅ Row count shows: 6
- ✅ Column count shows: 15+
- ✅ Preview table shows 6 rows with data
- ✅ "⬇️ Download CSV" button is available

### Test 7: Test CSV Download

1. After successful parse, click **"⬇️ Download CSV"**
2. File should download as `parsed_*.csv`
3. Open CSV in Excel or text editor
4. Verify it contains:
   - Header row with column names
   - 6 data rows
   - Extracted values (VCC=2.5, TEMP=25, etc.)
   - DUT numbers (DUT1, DUT2)
   - WL values (2, 3, 4, 5)

### Test 8: Test Data Table Viewing

1. Click **"📋 View Data"** tab
2. Wait for data to load
3. Verify you can see:
   - All 6 records in table format
   - Sortable columns (clickable headers)
   - Pagination controls
4. Try filtering:
   - Select column filter
   - Type a value (e.g., "LP")
   - Click "Filter"
5. Verify filtered results appear

### Test 9: Test Scatter Plot Generation

1. Click **"📈 Scatter Plot"** tab
2. Select columns:
   - X-Axis: `WL`
   - Y-Axis: `RBER`
   - Color By: `pagetype`
   - Title: "WL vs RBER by Page Type"
3. Click **"📈 Generate Plot"**
4. Wait for spinner

**Expected Results:**
- ✅ Plot image appears below
- ✅ Points colored by pagetype (LP = one color, UP = another)
- ✅ X-axis labeled "WL"
- ✅ Y-axis labeled "RBER"
- ✅ Legend shows pagetype categories

### Test 10: Test CDF Plot Generation

1. Click **"📊 CDF Plot"** tab
2. Select columns:
   - Value Column: `RBER`
   - Category Column: `pagetype`
   - Split By: `dut` (optional)
   - Title: "RBER CDF by Page Type"
3. Click **"📊 Generate Plot"**

**Expected Results:**
- ✅ CDF plot appears
- ✅ Multiple curves (one per pagetype)
- ✅ Y-axis shows cumulative probability (0 to 1)
- ✅ Legend shows categories
- ✅ If split by DUT, shows subplots

### Test 11: Test Statistics Tab

1. Click **"📉 Statistics"** tab
2. Wait for stats to load
3. Verify stat cards appear for columns:
   - Numeric columns: show mean, std dev, min, max, count
   - Categorical columns: show unique count
4. Example for RBER:
   - Mean: ~0.0012
   - Count: 4

### Test 12: Test Help Tab

1. Click **"ℹ️ Help"** tab
2. Verify content appears:
   - Getting Started section
   - Supported Log Formats
   - Available Plots
   - Key Features
   - Extracted Fields
   - Technical Details

## Edge Case Tests

### Test 13: Empty File Upload
1. Create empty file `empty.log`
2. Try to upload
3. **Expected**: Error message "No FDV records found"

### Test 14: Invalid Format
1. Create text file with random content
2. Try to upload
3. **Expected**: Error message or empty result

### Test 15: Large Dataset
1. Create log file with 1000+ records
2. Upload and parse
3. Verify:
   - ✅ Parse completes successfully
   - ✅ Table loads with pagination
   - ✅ Filters work correctly
   - ✅ Plots generate without errors

## Performance Tests

### Test 16: Memory Usage
1. Open Task Manager (Ctrl+Shift+Esc)
2. Find "Python" process
3. Note memory usage before and after parse
4. **Expected**: Less than 500 MB for typical files

### Test 17: Response Time
1. Measure time for API responses:
   - Upload & parse: <5 seconds for 100 records
   - Plot generation: <3 seconds
   - Data loading: <1 second
2. **Expected**: All within acceptable range

## API Tests (Optional - Using curl or Postman)

### Test 18: Upload API
```powershell
$file = "D:\path\to\test_sample.log"
$response = Invoke-RestMethod -Uri "http://localhost:5058/api/upload" -Method Post -Form @{file = Get-Item $file}
$response.success  # Should be $true
$response.row_count  # Should be 6
```

### Test 19: Get Data API
```powershell
$csvId = $response.csv_id
$data = Invoke-RestMethod -Uri "http://localhost:5058/api/csv/$csvId/data?page=0&per_page=50"
$data.data.Count  # Should be 6
```

### Test 20: Stats API
```powershell
$stats = Invoke-RestMethod -Uri "http://localhost:5058/api/stats/$csvId"
$stats.stats.Keys  # Should contain column names
```

## Cleanup After Testing

```powershell
# Clean up test files
Remove-Item -Path "D:\fdv_eimpro_tmp\*" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "test_sample.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "empty.log" -Force -ErrorAction SilentlyContinue
```

## Success Criteria

✅ **All tests pass if:**
1. Application starts without errors
2. Web interface loads correctly
3. File uploads parse successfully
4. CSV downloads work
5. Data table displays and filters
6. Plots generate with correct data
7. Statistics calculate correctly
8. Help documentation is accessible
9. No JavaScript errors in browser console
10. Temp files are created and cleaned up

## Troubleshooting Test Failures

### If startup fails:
```powershell
py -3 -m pip install --force-reinstall -r requirements.txt
```

### If web interface doesn't load:
1. Check firewall: Port 5058 might be blocked
2. Try accessing via IP: `http://127.0.0.1:5058`
3. Check browser console (F12) for errors

### If parse fails:
1. Verify log file format matches FDV specification
2. Check file encoding (should be UTF-8)
3. Look at error message for clues

### If plots don't appear:
1. Check temp directory exists and has write permissions
2. Verify matplotlib backend is 'Agg'
3. Check browser console for fetch errors

## Performance Optimization Tips

If tests show slow performance:

1. **Increase memory**: More RAM = faster processing
2. **Disable debug mode**: Already disabled by default
3. **Use SSD**: Faster temp file I/O
4. **Reduce page size**: Use `per_page=25` for faster pagination
5. **Archive old data**: Clean up temp directory regularly

---

## Test Report Template

Use this template to document your test results:

```
Date: _______________
Tester: _______________
Version: 1.0.0

Test Results:
□ Test 1: Python Installation - PASS/FAIL
□ Test 2: Dependencies - PASS/FAIL
□ Test 3: Temp Directory - PASS/FAIL
□ Test 4: Application Startup - PASS/FAIL
□ Test 5: Port Listening - PASS/FAIL
□ Test 6: Web Interface - PASS/FAIL
□ Test 7: File Upload - PASS/FAIL
□ Test 8: CSV Generation - PASS/FAIL
□ Test 9: Data Table - PASS/FAIL
□ Test 10: Scatter Plot - PASS/FAIL
□ Test 11: CDF Plot - PASS/FAIL
□ Test 12: Statistics - PASS/FAIL

Issues Found:
_________________________________________________
_________________________________________________

Notes:
_________________________________________________
_________________________________________________

Overall Status: ✅ PASS / ⚠️ FAIL / 🔄 PARTIAL
```

---

**Testing Complete When:**
- ✅ All 12 functional tests pass
- ✅ No console errors in browser
- ✅ Performance within acceptable limits
- ✅ Edge cases handled gracefully
- ✅ Ready for production use

**Time Estimate**: 30-45 minutes for complete test suite
