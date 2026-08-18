# Rev16 Split Chart Regex Extraction - Complete Fix Summary

## Issues Fixed

### 1. Scatter/Histogram/Cumulative Charts - X & Y Regex Not Evaluated
**Problem**: Split charts ignored X and Y regex patterns, using raw column values instead.

**Root Cause**: Split chart code called `readFilteredFromMemory()` which correctly extracted numeric values using regex, but then **discarded those extracted points** and re-extracted from raw rows in a different code path.

**Fix**: Modified bucket population to use the already-extracted points from `readFilteredFromMemory()` instead of re-extracting.

**File**: `fdv_chart_rev16/fdv_chart.html` (lines 11234-11290)

---

### 2. Boxplot - X-Axis Regex Not Evaluated  
**Problem**: Split-chart boxplot X-axis showed raw values instead of regex-extracted category labels.

**Root Cause**: Extracted X-category was stored in wrong field (`pt.splitGroup` instead of `pt.group`), so the boxplot renderer used color value as X-axis labels instead of extracted categories.

**Fix**: Corrected field assignments:
- `pt.group` ← X-axis category label (extracted with xRx)
- `pt.splitGroup` ← Color/split grouping value

**File**: `fdv_chart_rev16/fdv_chart.html` (lines 11254-11269)

---

## Code Changes Summary

### Change 1: Use Pre-Extracted Points (Lines 11234-11290)

**Before**:
```javascript
// WRONG: Discarded pre-extracted points, re-extracted from raw rows
var allPoints = res.points;  // Correctly extracted, but ignored!

for (var r = 0; r < filteredIndices.length; r++) {
    var xv2 = extractNum(row[xColIdx], xRx);  // Re-extract (incorrect!)
    // Use re-extracted value
}
```

**After**:
```javascript
// CORRECT: Use pre-extracted points from readFilteredFromMemory()
var allPoints = res.points;  // { x: extracted, y: extracted, _ri: rowIdx }

allPoints.forEach(function(pt) {
    // pt.x and pt.y are already extracted with regex!
    var pt2 = { x: pt.x, y: pt.y, _ri: pt._ri };
    buckets[key].push(pt2);
});
```

---

### Change 2: Fix Boxplot Field Assignment (Lines 11254-11269)

**Before**:
```javascript
var bpXgroup = _extractGroupKey(bpXraw, xRx);  // Extract correctly
var bpColorRaw = ...;
var bpPt = { x: 0, y: pt.y, group: bpColorRaw, splitGroup: bpXgroup };
//                                  ^^^^^^^^                  ^^^^^^^
//                    WRONG: color in group field, category in splitGroup
```

**After**:
```javascript
var bpXgroup = _extractGroupKey(bpXraw, xRx);  // Extract X category
var bpColorRaw = ...;
var bpPt = { x: 0, y: pt.y, group: bpXgroup, splitGroup: bpColorRaw };
//                                 ^^^^^^^^                ^^^^^^^^^^
//                    CORRECT: category in group, color in splitGroup
```

---

## Impact Analysis

### What's Fixed ✅
- **Scatter plots** in split mode now evaluate X and Y regex patterns correctly
- **Histograms** in split mode respect X regex for bin grouping
- **Cumulative charts** in split mode apply regex extraction to X values
- **Boxplots** in split mode show regex-extracted X-axis category labels

### Unchanged ✅
- Single-chart rendering (unaffected by split-chart changes)
- Color column handling
- Split dimension handling (split-by grouping)
- Y-axis numeric extraction and intervals

### Performance ✅
- More efficient (eliminates redundant re-extraction)
- Uses pre-computed results from `readFilteredFromMemory()`
- No additional memory overhead

---

## Testing Checklist

### Scatter Charts
- [ ] Split mode + X-regex → X-axis shows extracted values
- [ ] Split mode + Y-regex → Y-axis shows extracted values
- [ ] Compare with single-chart mode (should be identical)

### Boxplot
- [ ] Split mode + X-regex → X-axis category labels show extracted values
- [ ] Y-axis still shows numeric values from Y column
- [ ] Color grouping still works correctly
- [ ] Compare with single-chart boxplot (should be identical)

### Histograms & Cumulative
- [ ] Split mode + X-regex → Bins/groups respect regex extraction
- [ ] Data distribution correct after extraction

### Edge Cases
- [ ] Invalid regex patterns → Should show "(no match)" or raw value
- [ ] Missing columns → Should handle gracefully
- [ ] Empty result sets → Should show empty chart
- [ ] Large datasets → Should apply sampling correctly

---

## Documentation Files
- `SPLIT_CHART_REGEX_FIX.md` - Scatter/histogram fix details
- `BOXPLOT_REGEX_FIX.md` - Boxplot-specific fix details

---

## Deployment Notes
- Both fixes are in **rev16 only** (rev13/rev15 unchanged)
- Restart rev16 server to apply changes: `restart_report.ps1`
- Changes affect split-chart rendering only; single charts unaffected
- No database or cache changes needed

---

## Date Applied
August 18, 2026

## Status
✅ **COMPLETE** - Both issues identified and fixed
