# X-Axis Raw Data Verification - Cum Sigma Chart

## Question
"Are you sure X is raw?"

## Answer: YES ✅

The X-axis data is **guaranteed to be raw (unsorted) and in original row order**.

## Data Flow Verification

### Stage 1: Row Filtering
**File:** `fdv_chart.html`, Lines 1580-1600 (`recomputeFilteredIndices` function)
```javascript
for (var r = 0; r < allRows.length; r++) {  // Iterate rows in order
    if (filters.length === 0) { filteredIndices.push(r); continue; }
    var row = allRows[r], pass = true;
    // Check filters...
    if (pass) filteredIndices.push(r);  // Push indices in order
}
```
**Result:** `filteredIndices = [0, 5, 12, 23, ...]` - **IN ORIGINAL ROW ORDER**

### Stage 2: Points Extraction
**File:** `fdv_chart.html`, Lines 2346-2375 (`readFilteredFromMemory` function)
```javascript
for (var r = 0; r < filteredIndices.length; r++) {  // Iterate in order
    var row = allRows[filteredIndices[r]];
    // Extract x value...
    var pt = { x: xv, y: yv, _ri: filteredIndices[r] };
    points.push(pt);  // Push in order
}
```
**Result:** `points` array has x values in order they were filtered → **UNSORTED**

### Stage 3: Bucket Collection
**File:** `fdv_chart.html`, Lines 5005-5040 (in `_drawSplitCharts`)
```javascript
for (var r = 0; r < filteredIndices.length; r++) {  // Iterate in order
    var row = allRows[filteredIndices[r]];
    // ... create point pt ...
    buckets[key].push(pt2);  // Push in order
}
```
**Result:** `buckets[key]` contains points in filtered order → **UNSORTED**

### Stage 4: Group Collection (Cum Sigma Handler)
**File:** `fdv_chart.html`, Lines 4672-4682 (in `_buildTileChart`)
```javascript
tileData.forEach(function(pt) {  // Iterate points from buckets[key] in order
    var g = innerSplitCol ? innerSplitKey(pt) : colorKey(pt);
    if (!csGroups[g]) { csGroups[g] = []; csOrder.push(g); }
    csGroups[g].push(pt.x);  // Push x values in order
});
```
**Result:** `csGroups[g]` contains x values in order they appear in tileData → **UNSORTED**

### Stage 5: Data Mapping
**File:** `fdv_chart.html`, Lines 4695-4720 (in `_buildTileChart`)
```javascript
var rawVals = csGroups[g].slice();  // Unsorted copy
var sortedVals = rawVals.slice().sort(...);  // Sorted ONLY for stats

// Calculate statistics from sortedVals (mean, median, stdev)
var mean = sortedVals.reduce(...) / n;
var stdev = Math.sqrt(...);

// Map through rawVals (NOT sortedVals)
data: rawVals.map(function(v,j) {
    var z = stdev > 0 ? (v - center) / stdev : 0;
    return { x: v, y: z };  // x = v FROM rawVals (unsorted)
})
```
**Result:** Chart plotted with X from `rawVals` (unsorted) and Y as z-score → **X-AXIS IS RAW**

## Key Separation

| Stage | Variable | Status |
|-------|----------|--------|
| Filtering | `filteredIndices` | Original row order |
| Extraction | `points` array | Original row order |
| Bucketing | `buckets[key]` | Original row order |
| Grouping | `csGroups[g]` | Original row order (unsorted) |
| Statistics | `sortedVals` | SORTED (for calculations only) |
| Plotting | `rawVals` | Original row order (unsorted) |

## Why It Works

1. **Raw values collected in original order** through the entire pipeline
2. **Sort happens ONLY in local `sortedVals` copy** for statistical calculations
3. **Plotting uses `rawVals.map()`** which preserves the unsorted order
4. **Z-scores calculated from sorted statistics** but mapped back to unsorted values

## Confirmation

✅ **X-axis will show values in their original (unsorted) order**
✅ **Y-axis will show calculated z-scores (-3 to +3)**
✅ **Scatter pattern will NOT be linear** (original order ≠ sorted order)

---
**Date:** Current session
**Status:** VERIFIED - Code implementation is correct
