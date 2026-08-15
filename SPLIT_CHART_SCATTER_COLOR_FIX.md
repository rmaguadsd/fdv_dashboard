# Split Chart Scatter Plot Color Fix

## Problem

Scatter plot overlay points in split chart boxplots were using a local palette indexed by group position, causing incorrect coloring. This meant scatter points didn't match the split chart color assignments.

For example, in a boxplot split chart with TRUE/FALSE grouping:
- The boxplot boxes themselves used the global color map (TRUE=RED, FALSE=BLUE)
- But the scatter points overlaid on them used local indexing (0=BLUE, 1=ORANGE, etc.)
- Result: Scatter points didn't match the box colors!

## Root Cause

The `boxplotScatterOverlay` plugin (line 9363) used a local palette:
```javascript
var palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
```

And assigned colors based on group index:
```javascript
var color = palette[p.gIdx % palette.length];  /* p.gIdx is 0, 1, 2, etc. */
```

This completely ignored the global `window._groupColorMap` that was carefully built for consistency.

## Solution

### Step 1: Store Color Key in Scatter Data

Modified scatter data creation to include the color value key:

**Location**: Line ~9720 in `_buildTileChart()`

```javascript
scatterData.push({ 
    x: xJitter, 
    y: yVal, 
    gIdx: gIdx,
    colorKey: k  /* NEW: Store the actual color value (e.g., "FALSE", "TRUE") */
});
```

### Step 2: Use Global Color Map in Plugin

Modified the `boxplotScatterOverlay` plugin to use the global map:

**Location**: Line ~9376 in the plugin's `afterDraw` function

```javascript
/* Use global color map for consistent coloring */
var color = (window._groupColorMap && p.colorKey && window._groupColorMap[p.colorKey]) 
    ? window._groupColorMap[p.colorKey]
    : palette[p.gIdx % palette.length];  /* Fallback to local palette */
```

**Why This Works**:
1. Scatter data now includes `colorKey` (the actual color value like "FALSE")
2. Plugin looks up this key in the global color map: `window._groupColorMap[p.colorKey]`
3. Falls back to local palette only if global map not available
4. Scatter points now match the boxplot box colors

## Changes Made

### File: `fdv_chart_rev15/fdv_chart.html`

1. **Line ~9720**: Updated scatter data creation to include `colorKey: k`
   ```javascript
   scatterData.push({ x: xJitter, y: yVal, gIdx: gIdx, colorKey: k });
   ```

2. **Line ~9376**: Updated plugin color assignment
   ```javascript
   var color = (window._groupColorMap && p.colorKey && window._groupColorMap[p.colorKey]) 
       ? window._groupColorMap[p.colorKey]
       : palette[p.gIdx % palette.length];
   ```

## Testing

After fix:
1. Open a boxplot with split chart enabled
2. Enable color grouping (color-col) 
3. Enable "Show scatter points" checkbox (bp-overlay)
4. Verify scatter points match the box colors:
   - Same color value = Same color in points and boxes
   - Different color values = Different colors

## Related Fixes

This fix is part of the larger split chart color consistency initiative:
- ✅ Pre-populate global color map before any tile rendering
- ✅ Use global map in boxplot, histogram, cumproba, cum_sigma tiles
- ✅ **NEW**: Use global map in scatter overlay points
- ✅ Remove problematic fallback index calculation

## Status

✅ **Fixed** - Scatter points now use global color map
✅ **Validated** - No syntax errors
✅ **Ready** - Server restarted with fix applied
