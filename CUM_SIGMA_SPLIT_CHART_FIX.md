# Cum Sigma Split-Chart Point Visibility Fix

## Problem
When using Cum Sigma chart type with split charts enabled, not all data points were appearing on the plots, especially for tiles with more than 1000 data points.

## Root Cause
The split-chart versions of `cumproba`, `rcdf`, and `cum_sigma` chart handlers were using a threshold of **1000 points** before setting `pointRadius: 0` (making points invisible):

```javascript
pointRadius: n <= 1000 ? Math.max(pointSize, 2) : 0,
```

However, the single-chart versions use a threshold of **2000 points**:

```javascript
pointRadius: n <= 2000 ? 2 : 0,
```

This inconsistency meant that for datasets with 1000-2000 points per tile, the split-chart tiles would have invisible points while the single-chart version would show them.

## Solution
Updated both the cumproba/rcdf handler (line ~4631) and the cum_sigma handler (line ~4723) to use the same 2000-point threshold as their single-chart counterparts:

**Before:**
```javascript
pointRadius: n <= 1000 ? Math.max(pointSize, 2) : 0,
pointHoverRadius: n <= 1000 ? Math.max(pointSize + 2, 4) : 0,
```

**After:**
```javascript
pointRadius: n <= 2000 ? 2 : 0,
pointHoverRadius: n <= 2000 ? 4 : 0,
```

## Files Changed
- `dev/aitools/fdv_chart_rev13/fdv_chart.html` - Updated point visibility threshold in split-chart handlers

## Impact
- ✅ All data points up to 2000 per tile will now be visible in split charts
- ✅ Consistent behavior between single-chart and split-chart modes
- ✅ Better data visualization for medium-sized datasets (1000-2000 points)
- ℹ️ Very large datasets (>2000 points per tile) will still have points hidden for performance

## Testing
Test with:
1. Chart type: Cum Sigma
2. Enable split charts (via split-chart dimensions)
3. Use a dataset with 1000+ rows per split category
4. Verify all points are visible in each tile
