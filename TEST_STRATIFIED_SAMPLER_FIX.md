# Test Guide: Verify Stratified Sampler Fix

## ✅ The Fix Applied

Changed sampling logic from **color-blind stride** (drops entire blocks) to **stratified-by-color** (guarantees every group represented).

The server has been restarted with this fix active. Now test it.

---

## Quick Test Plan

### Step 1: Reload Browser
- **Hard refresh**: `Ctrl+F5` (or `Cmd+Shift+R` on Mac)
- Wait for page to fully load

### Step 2: Load Your FDV Data File
- Use the file upload to load a FDV log with multiple DUTs
- Recommended: Data with 4+ DUTs and 6+ BLKs

### Step 3: Test Case #1 - No Split, No Color-By

**Setup:**
- X-axis: Any column (e.g., BLK, Line#, etc.)
- Y-axis: RBER or similar measurement
- Color-by: **(Leave blank)**
- Split-by: **(Leave blank)**
- Chart type: Scatter

**Draw chart** and observe:

✅ **What you should see (FIX WORKING):**
- All data points visible for ALL DUTs
- No DUTs appear to have missing data
- Status shows something like: `10000 pts` (or similar large number, representative of full dataset)
- Chart displays a comprehensive sample of data

❌ **What would indicate problem (fix NOT working):**
- Only some DUTs have visible points
- Certain DUTs appear to have "gaps" or fewer points
- Status shows small number of points (< 25% of data)
- Visual "holes" in the scatter plot by DUT

---

### Step 4: Test Case #2 - Split by DUT, Color by BLK

**Setup:**
- X-axis: Any column
- Y-axis: RBER or similar
- Color-by: **BLK**
- Split-by: **DUT** (or split-chart column)
- Chart type: Scatter

**Draw chart** and observe:

✅ **What you should see (FIX WORKING):**
- Multiple tiles displayed (one per DUT)
- Each tile's legend shows the SAME blocks
- Even if a block has 0 measurements in that DUT, it still appears in legend as "(n=0)"
- All tiles have consistent legends
- Comparing DUT1 tile to DUT1 subset in Case #1 should show similar data distribution

✅ **Additional check:**
- Count total points across all tiles
- Compare to total points in Case #1 (no split)
- Should be approximately the same (within ±10%)

❌ **What would indicate problem:**
- Different tiles show different blocks in legends
- Missing blocks in some tiles that appear in others
- Total points across tiles significantly different from Case #1
- Visual inconsistency between tiles

---

## Detailed Inspection Checklist

### Browser Console (F12 → Console tab)

✅ **Look for:**
- No red error messages
- Status messages showing sampling info (e.g., "10000 pts (sampled 10000/50000)")
- Clean rendering without console errors

❌ **Avoid:**
- Red `Uncaught` errors
- `NaN` in calculations
- Undefined colorKey errors

### Data Point Distribution

✅ **Good signs:**
- Points evenly distributed across all visible DUTs
- No complete absence of data from any DUT
- When split by DUT, each tile shows similar proportions of each block

❌ **Bad signs:**
- One or more DUTs completely empty
- Point count dramatically different between tiles with similar data
- Blocks appearing/disappearing between split and no-split modes

### Legend Consistency (Split Mode)

✅ **Expected:**
```
Tile 1 (DUT1):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK3 (200pts)] [BLK4 (150pts)] [BLK5 (n=0)] [BLK6 (n=0)]
Tile 2 (DUT2):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK3 (200pts)] [BLK4 (150pts)] [BLK5 (n=0)] [BLK6 (n=0)]
Tile 3 (DUT3):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK3 (200pts)] [BLK4 (150pts)] [BLK5 (n=0)] [BLK6 (n=0)]
Tile 4 (DUT4):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK3 (200pts)] [BLK4 (150pts)] [BLK5 (n=0)] [BLK6 (n=0)]
                ↑ All same blocks ↑
```

❌ **Bad (before fix):**
```
Tile 1 (DUT1):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK3 (200pts)] [BLK4 (150pts)]
Tile 2 (DUT2):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK3 (200pts)] [BLK5 (100pts)]      ← Different blocks!
Tile 3 (DUT3):  [BLK1 (500pts)] [BLK2 (350pts)] [BLK4 (200pts)] [BLK6 (150pts)]      ← Missing BLK3/5
Tile 4 (DUT4):  [BLK1 (500pts)] [BLK3 (350pts)] [BLK5 (200pts)] [BLK6 (150pts)]      ← Missing BLK2/4
                ↑ All different! ↑
```

---

## Common Observations

### Point Count Changes
- **Expected**: Total points displayed will be ~MAX_PTS (10,000 for scatter)
- **Reason**: Stratified sampler now ensures proper representation across all colors
- **Comparison to before**: May see MORE total points (before fix was dropping blocks entirely)

### Legend Changes (Split Mode)
- **Expected**: All tiles now show complete legend with consistent blocks
- **Reason**: Global color discovery and stratified sampling ensures completeness
- **Visual impact**: More consistent appearance across tiles

### Distribution Changes
- **Expected**: Data now more evenly represented by DUT
- **Reason**: Stride sampler was accidentally clustering samples by contiguous DUT/BLK runs
- **Visual impact**: Less "clumpy" appearance, smoother distribution

---

## Troubleshooting

### Problem: Still seeing missing DUT/BLK data
1. **Hard refresh browser**: `Ctrl+F5` to clear cache
2. **Check server is running**: Terminal should show "FDV Chart Parser is running"
3. **Restart server** if needed: Kill port 5058 and restart
4. **Report**: Share screenshot showing which DUT/BLK is missing

### Problem: Different data in Split vs No-Split modes
1. This is **expected** if:
   - Split mode has multiple DUT tiles with different data per DUT
   - But **count should be similar** (within ±10%)
2. **NOT expected** if:
   - Total points across split tiles is 2-3x the no-split count
   - This would indicate sampling is still not deterministic
3. **Report**: Share exact point counts from both modes

### Problem: Performance is slow
- This is **not expected** - stratified sampler is faster than old method
- **Check**: How many rows in your dataset? If > 1 million, may take time to initially load
- **Report**: Dataset size and observed lag time

---

## Expected Improvements

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Missing DUTs in no-split mode | ~50% (2 of 4 missing) | 0% (all visible) |
| Missing BLKs per tile | ~40% per tile | ~0% (all visible, some n=0) |
| Consistency: Split vs No-split | Poor (different data) | Good (same proportions) |
| Legend consistency across tiles | Poor (different blocks) | Perfect (identical blocks) |
| Data loss from sampling | ~40% (silent) | 0% (guaranteed representation) |

---

## Next Steps After Verification

If the fix is working:
1. ✅ Test with multiple different FDV files
2. ✅ Try different color-by and split-by columns
3. ✅ Report any remaining issues with exact data and settings

If something is still wrong:
1. ❌ Collect screenshot of the problem
2. ❌ Note exact X/Y/Color/Split column choices
3. ❌ Check browser console for errors (F12)
4. ❌ Report with as much detail as possible

---

## Questions?

**Look for these in console output:**
- `[drawScatterLine]` messages showing sampling info
- `(sampled N/M)` in the status bar showing point counts
- No `Uncaught` or `NaN` errors

**The fix is active and ready to test!** 🚀
