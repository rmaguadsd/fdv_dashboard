# Split Chart X-Regex Not Evaluating - ROOT CAUSE & FIX

## Problem
When using split charts in rev16, the X-axis regex was not being evaluated properly. Single charts worked fine, but split charts ignored the regex pattern.

Example:
- **Single Chart**: X-col=`tname`, X-regex=`DUT._(.*)` → ✅ Regex applied correctly
- **Split Chart**: Same settings → ❌ Regex ignored, defaults to DUT

## Root Cause
The split chart code had a **double-extraction bug**:

1. **First extraction** (line 11049): Called `readFilteredFromMemory(xCol, yCol, ..., xRx, yRx, ...)` which correctly:
   - Applied the X-regex to extract numeric values
   - Returned `allPoints` with `{ x: extractedValue, y: extractedValue, _ri: rowIndex }`

2. **Second extraction** (lines 11238-11330): In the bucket population loop, the code **bypassed** `allPoints` and **re-extracted** from raw rows:
   ```javascript
   var xRaw2 = row[currentHeaders.indexOf(xCol)];
   var xv2 = extractNum(xRaw2, xRx);  // Second extraction!
   ```

The problem: the second extraction was inline and handled differently, and the code that followed it also had a flaw where it wasn't actually using the properly-filtered data.

## Solution
**Use the already-extracted points from `allPoints` instead of re-extracting:**

### Before (BROKEN):
```javascript
// WRONG: Discarded allPoints, re-extracted from raw rows
var allPoints = res.points;  // <- Correctly extracted but UNUSED!

// Later...
for (var r = 0; r < filteredIndices.length; r++) {
    var xv2 = extractNum(row[xColIdx], xRx);  // Re-extract (wrong!)
    // Use xv2 (ignoring allPoints)
}
```

### After (FIXED):
```javascript
// CORRECT: Use allPoints which already has regex-extracted values
var allPoints = res.points;  // { x: extracted, y: extracted, _ri: rowIdx }

// Later...
allPoints.forEach(function(pt) {
    // pt.x and pt.y are ALREADY extracted with regex!
    // Just attach split bucket key and preserve color data
    var pt2 = { x: pt.x, y: pt.y, _ri: pt._ri };
    buckets[key].push(pt2);
});
```

## Changes Made
**File**: `fdv_chart_rev16/fdv_chart.html`

**Location**: Lines 11234-11286 (bucket population section in `_drawSplitCharts`)

**What Changed**:
- Removed redundant re-extraction loop
- Changed from iterating over `filteredIndices` to iterating over `allPoints`
- Direct use of `pt.x` and `pt.y` which are already regex-extracted
- Only extract split column and color information from raw rows

## Benefits
✅ Split charts now respect X-regex settings (matching single chart behavior)
✅ Eliminates double-extraction overhead
✅ More efficient memory usage
✅ Clearer code logic

## Testing
Test with:
- **X-col**: `tname`
- **X-regex**: `DUT._(.*)` or similar
- **Split mode**: Enabled with any split-chart dimension
- **Result**: Scatter plot should show properly extracted numeric values on X-axis

Compare with single chart mode to verify they produce the same axis values.
