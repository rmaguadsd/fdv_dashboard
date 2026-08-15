# FINAL FIX: Unified Color Map Between Main Chart and Split Charts

## Problem

Your screenshots showed the bug persists because **the main chart (`drawScatterLine()`) and split charts (`_drawSplitCharts()`) had SEPARATE color maps!**

**What Happened**:
1. Split charts discovered colors: [FALSE, TRUE] → Assigned colors: [FALSE→RED, TRUE→BLUE]
2. Main chart ran afterwards and OVERWROTE the map with only its groups: [TRUE] → [TRUE→BLUE]
3. Split chart 2 then tried to use FALSE, but it wasn't in the map → **fallback to same blue as TRUE**

## Root Cause: Two Independent Color Discovery Processes

**`_drawSplitCharts()` (lines ~11000-11070)**:
```javascript
/* Discovers ALL color values from filteredIndices */
for each row in filteredIndices:
    colorVal = extract color/split value
    allColorValues[colorVal] = true

sorted = sort(allColorValues)
for each colorVal, idx in sorted:
    colorMap[colorVal] = PALETTE[idx]
```

**`drawScatterLine()` (lines ~12050-12150)**:
```javascript
/* Discovers ONLY from plotted points in main chart */
points.forEach(function(pt):
    colorVal = extract color value
    allColorValues[colorVal] = true

groups = Object.keys(groups)
for each group, idx in groups:
    colorMap[group] = PALETTE[idx]  /* OVERWRITES split chart map! */
```

**THE PROBLEM**: 
- Different datasets discovered
- Different sorted order
- Different indices → Different colors!

## Solution: Unified Color Discovery

### Change 1: Main Chart Also Scans ALL Rows (No Sampling)

**File**: `fdv_chart_rev15/fdv_chart.html`  
**Location**: Lines ~12053-12100

**New Code**:
```javascript
/* Discover from sampled points (for legend) */
var allColorValues = {};
points.forEach(function(pt) {
    var colorVal = extract color
    allColorValues[colorVal] = true;
});

/* CRITICAL: Also scan ALL filteredIndices (NO sampling) to match split chart discovery */
var allDiscoveredColors = Object.assign({}, allColorValues);
for (var r = 0; r < filteredIndices.length; r++) {
    var row = allRows[filteredIndices[r]];
    var colorVal = extract color
    allDiscoveredColors[colorVal] = true;  /* Include ALL unique colors */
}
```

**Why**: Now main chart discovers the SAME colors as split charts (both scan all rows without sampling).

### Change 2: Build Map From ALL Discovered Colors (Sorted)

**Location**: Lines ~12330-12345

**Before**:
```javascript
var allGroupsSorted = Object.keys(groups).sort();  /* Only groups in main chart */
allGroupsSorted.forEach(function(g, idx) {
    colorMap[g] = PALETTE[idx]  /* Index from main chart groups only */
});
```

**After**:
```javascript
var allColorsSorted = Object.keys(allDiscoveredColors).sort();  /* ALL unique colors */
allColorsSorted.forEach(function(colorVal, idx) {
    if (!window._groupColorMap[colorVal]) {
        colorMap[colorVal] = PALETTE[idx]  /* Index from ALL discovered colors */
    }
});
```

**Why**: Both main chart and split charts now use the SAME sorted list → Same indices → Same colors!

### Change 3: Add Empty Datasets for ALL Colors

**Location**: Line ~12283

```javascript
/* Add empty datasets for ALL discovered colors (not just plotted ones) */
Object.keys(allDiscoveredColors).forEach(function(colorVal) {
    if (!groups[colorVal]) {
        groups[colorVal] = [];  /* Ensure even unplotted colors appear in legend */
    }
});
```

**Why**: Ensures even if a color appears in split charts but not main chart, it's still created.

## How It Works Now

### Scenario: Two Split Charts

**Dataset**:
```
ag-18: Only TRUE values
ak-172: FALSE and TRUE values
```

**Unified Discovery Process**:
1. `_drawSplitCharts()` runs:
   - Scans ALL rows → discovers [FALSE, TRUE]
   - Sorts → [FALSE, TRUE]
   - Assigns → FALSE=PALETTE[0], TRUE=PALETTE[1]
   - map = {FALSE: "#dc3545", TRUE: "#007bff"}

2. Tile rendering uses map:
   - ag-18: Uses TRUE → #007bff (BLUE) ✅
   - ak-172: Uses FALSE → #dc3545 (RED), TRUE → #007bff (BLUE) ✅

3. `drawScatterLine()` runs later:
   - Scans ALL rows → discovers [FALSE, TRUE]
   - Sorts → [FALSE, TRUE]
   - Color map already exists with both colors
   - **Does NOT overwrite** (checks `if (!window._groupColorMap[colorVal])`)
   - Uses existing: FALSE="#dc3545", TRUE="#007bff"

**Result**: ✅ **All charts use CONSISTENT colors!**

## Key Insight: Alphabetical Sorting is Deterministic

Both `_drawSplitCharts()` and `drawScatterLine()` now:
1. Discover ALL colors from ALL filtered rows (no sampling)
2. Sort alphabetically: [FALSE, TRUE]
3. Assign to PALETTE by index: [PALETTE[0], PALETTE[1]]

Since both functions use:
- Same data source (filteredIndices)
- Same sorting (alphabetical)
- Same indexing (idx % PALETTE.length)

They **ALWAYS produce identical color maps!** ✅

## Testing

**Expected Results**:
```
Chart 1 (ag-18 - TRUE only):
  ✅ TRUE points: BLUE (#007bff)

Chart 2 (ak-172 - FALSE and TRUE):
  ✅ FALSE points: RED (#dc3545) - DIFFERENT from TRUE
  ✅ TRUE points: BLUE (#007bff) - SAME as Chart 1

Console Logs:
  [_drawSplitCharts] Pre-assigned color for "FALSE": #dc3545
  [_drawSplitCharts] Pre-assigned color for "TRUE": #007bff
  [drawScatterLine] Pre-assigned color for "FALSE": #dc3545  (or skipped if map exists)
  [drawScatterLine] Pre-assigned color for "TRUE": #007bff   (or skipped if map exists)
```

## Files Modified

**Single File**: `fdv_chart_rev15/fdv_chart.html`

**Locations**:
1. Lines ~12053-12100: Add full row scan in `drawScatterLine()`
2. Lines ~12283: Update empty dataset creation to use `allDiscoveredColors`
3. Lines ~12330-12345: Build color map from `allDiscoveredColors` (sorted)

## Performance Impact

✅ **Minimal** - Both functions already scan large datasets:
- `readFilteredFromMemory()`: Already O(n)
- New scan: Also O(n)
- Only scans indexes, not raw data
- Same as split chart discovery we already did

## Architecture Summary

### Before (Broken)
```
_drawSplitCharts()          drawScatterLine()
    ↓                            ↓
Discover A                  Discover B (different!)
    ↓                            ↓
Build Map A             Overwrite with Map B
    ↓                            ↓
Split uses A, Main uses B → **INCONSISTENT COLORS**
```

### After (Fixed)
```
_drawSplitCharts()         drawScatterLine()
    ↓                            ↓
Discover ALL            Discover ALL (same!)
    ↓                            ↓
Build Map from ALL     Use/Extend Map from ALL
    ↓                            ↓
Both use same map → **CONSISTENT COLORS**
```

## Status

✅ **Problem identified**: Two independent color discoveries causing conflicts  
✅ **Root cause found**: Main chart overwrote split chart color map  
✅ **Fixed**: Unified color discovery between both functions  
✅ **Validated**: No syntax errors  
✅ **Deployed**: Server restarted with fix applied  

## Next: Verification

Your split charts should now show:
- **FALSE**: Different color (RED/distinct)
- **TRUE**: Consistent color (BLUE) across all tiles
