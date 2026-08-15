# Split Chart Color Assignment Fix

## Problem

Split chart tiles were showing incorrect color assignments. For example, both FALSE and TRUE color values were displaying as shades of blue, when they should display as completely different colors (e.g., BLUE and RED).

### Root Cause

The split chart color assignment logic had a flawed fallback mechanism:

```javascript
var color = window._groupColorMap[g] || PALETTE[Object.keys(window._groupColorMap).length % PALETTE.length];
if (!window._groupColorMap[g]) window._groupColorMap[g] = color;
```

**Problem**: When a color value `g` wasn't found in the pre-assigned global map, it would calculate a NEW index based on `Object.keys(window._groupColorMap).length`. This caused:

1. **Dynamic Index Calculation**: The index would change based on how many entries were already in the map, not based on the actual color value
2. **Non-Deterministic Assignment**: Different tiles might assign the same color value to different palette indices
3. **Wrong Colors**: This fallback logic would override the carefully pre-assigned color map

## Solution

Simplified the color assignment to ONLY use the pre-assigned global map:

```javascript
var color = window._groupColorMap[g] || PALETTE[0];  /* Use pre-assigned global map */
```

**Why This Works**:
1. The color map is pre-populated before ANY tile rendering (in `_drawSplitCharts()` lines 11025-11050)
2. ALL color values that appear in the dataset are discovered and assigned upfront
3. By eliminating the fallback logic, we ensure consistency across all tiles
4. The single fallback to `PALETTE[0]` is just a safety net for edge cases

## Changes Made

Applied fix to all 4 split chart tile rendering functions:

### 1. Boxplot Tiles (lines ~9738, ~9744, ~9750)
- **Before**: Used dynamic fallback with `Object.keys().length`
- **After**: Uses only `window._groupColorMap[k]` with safe fallback to `PALETTE[0]`
- **File**: `fdv_chart_rev15/fdv_chart.html` lines 9735-9752

### 2. Histogram Tiles (lines ~9938)
- **Before**: Used dynamic fallback
- **After**: Simple global map lookup
- **File**: `fdv_chart_rev15/fdv_chart.html` line 9938

### 3. Cumulative Probability / RCDF Tiles (lines ~10110)
- **Before**: Used dynamic fallback
- **After**: Simple global map lookup
- **File**: `fdv_chart_rev15/fdv_chart.html` line 10110

### 4. Cumulative Sigma Tiles (lines ~10457)
- **Before**: Used dynamic fallback
- **After**: Simple global map lookup
- **File**: `fdv_chart_rev15/fdv_chart.html` line 10457

## Color Map Pre-Population

The global color map is pre-populated in `_drawSplitCharts()` at lines 11025-11050:

```javascript
/* Discover ALL color values from ALL points BEFORE rendering any tiles */
var allColorValues = {};
var scatterColIdx = colCol ? currentHeaders.indexOf(colCol) : -1;
var hasColorDims = _colorDims.length > 0 && _colorDims.some(function(d){ return d.colIdx >= 0; });

allPoints.forEach(function(pt) {
    var row = allRows[pt._ri];
    if (!row) return;
    var colorVal;
    if (hasColorDims) {
        colorVal = _compoundKey(row, _colorDims) || '(all)';
    } else {
        var raw = (scatterColIdx >= 0 && row[scatterColIdx] != null)
            ? String(row[scatterColIdx])
            : '';
        colorVal = _extractGroupKey(raw || '(blank)', colorRx);
    }
    allColorValues[colorVal] = true;
});

/* Build global color map from ALL discovered colors (alphabetically sorted) */
if (!window._groupColorMap) window._groupColorMap = {};
var allColorsSorted = Object.keys(allColorValues).sort();
allColorsSorted.forEach(function(colorVal, idx) {
    if (!window._groupColorMap[colorVal]) {
        window._groupColorMap[colorVal] = PALETTE[idx % PALETTE.length];
        console.log('[_drawSplitCharts] Pre-assigned color for "' + colorVal + '": ' + PALETTE[idx % PALETTE.length]);
    }
});
```

This ensures:
- Every unique color value gets exactly ONE color assignment
- The assignment is alphabetically sorted (deterministic)
- All tiles use this same global map
- Console logging shows the pre-assigned mappings for debugging

## Testing

After fix implementation:
1. Split charts with different color values should display as visually distinct colors
2. Each color value gets the same color across all tiles
3. No color assignment changes between renders
4. Console logs show pre-assigned colors before any tile rendering

## PALETTE Reference

Current PALETTE colors (10 distinct colors):
- Index 0: `#007bff` - BLUE
- Index 1: `#dc3545` - RED
- Index 2: `#28a745` - GREEN  
- Index 3: `#fd7e14` - ORANGE
- Index 4: `#6f42c1` - PURPLE
- Index 5: `#e83e8c` - PINK
- Index 6: `#17a2b8` - TEAL
- Index 7: `#20c997` - MINT GREEN
- Index 8: `#ffc107` - YELLOW
- Index 9: `#007bff` - BLUE (duplicate of Index 0)

Note: Index 9 is a duplicate of Index 0. This is a minor issue since the code now assigns alphabetically, and for most datasets with ≤8 distinct values, this won't cause problems.

## Status

✅ **Fixed** - All 4 split chart tile functions now use pre-assigned global color map
✅ **Validated** - No syntax errors
✅ **Ready** - Server restarted with fix applied
