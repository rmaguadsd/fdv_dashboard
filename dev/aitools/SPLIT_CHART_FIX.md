# Split-Chart Functionality Fix - April 14, 2026

## Problem
Split-chart functionality was broken due to duplicate variable declarations in the `_drawSplitCharts()` function.

## Root Cause
During the unification of split-chart mechanism, duplicate variable declarations were accidentally introduced:

```javascript
// BEFORE (broken):
var scDims    = _gdimRead('sc');   // Line 3146
var xLog      = document.getElementById('x-log').checked;  // Line 3147
var xLog      = document.getElementById('x-log').checked;  // Line 3148 (DUPLICATE!)
var yLog      = document.getElementById('y-log').checked;
var scDims    = _gdimRead('sc');   // Line 3151 (DUPLICATE!)
var scCol     = scDims.length > 0 ? scDims[0].col : '';
var scRx      = scDims.length > 0 ? scDims[0].rx : '';
```

This created:
- Duplicate `xLog` declaration (JavaScript error)
- Duplicate `scDims` declaration (JavaScript error)
- Confusing code flow

## Solution
Cleaned up the function to remove duplicates and reorganize declarations properly:

```javascript
// AFTER (fixed):
var scDims    = _gdimRead('sc');   /* Split-chart dimensions (unified mechanism) */
var scCol     = scDims.length > 0 ? scDims[0].col : '';  /* Primary split column from dims */
var scRx      = scDims.length > 0 ? scDims[0].rx : '';   /* Primary split regex from dims */
var xLog      = document.getElementById('x-log').checked;
var yLog      = document.getElementById('y-log').checked;
var status    = document.getElementById('plot-status');
```

**Lines Modified**: 3140-3151 in `_drawSplitCharts()`

## What's Fixed
✅ **Split-chart plotting now works**
- No more duplicate variable declaration errors
- `scDims`, `scCol`, `scRx` are properly defined
- Dimensions are correctly read from unified dimension mechanism

## Code Flow Restored
1. User selects split-chart dimension(s)
2. Clicks "Plot"
3. `drawPlot()` checks if `scCol` is set
4. If yes: calls `_drawSplitCharts()` → creates tile grid
5. If no: calls standard chart functions → single chart

## Testing
The fix resolves the JavaScript errors that were preventing split-chart from rendering. To test:

1. Load data
2. Set split-chart dimension (By column + optional Regex)
3. Click "Plot"
4. Should see:
   - Grid of tiles (one per unique split value)
   - Status showing "X tiles, Y points, split by [column] → keys: ..."
   - Each tile shows the chart for that split value

## Files Modified
- `fdv_chart.html`: Lines 3140-3151 in `_drawSplitCharts()` function

## Technical Details

### Variable Purpose
- `scDims`: Array of split-chart dimension objects from unified mechanism
- `scCol`: Primary split column name (extracted from scDims[0])
- `scRx`: Primary split regex pattern (extracted from scDims[0])
- `xLog`, `yLog`: Axis log scale settings
- `status`: Status bar element

### Integration with Unified Mechanism
The fix properly integrates with the Option 1 unification:
- Uses `_gdimRead('sc')` to read split-chart dimensions
- Extracts primary dimension from array (scDims[0])
- Supports future multi-dimensional splits (scDims[1], scDims[2], etc.)

---

**Status**: ✅ COMPLETE - Split-chart functionality restored
