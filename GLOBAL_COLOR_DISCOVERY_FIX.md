# Fix: Global Color Discovery for Split Charts

## Problem
When using split-by-DUT with color-by-BLK, each DUT tile only showed the blocks that had measurements in that specific DUT, not all blocks globally. This created inconsistent legends across tiles.

**Example of the problem:**
- Global dataset has BLKs: 1, 2, 3, 4, 5, 6
- DUT1 tile shows: BLK 1, 2, 3, 4
- DUT2 tile shows: BLK 1, 2, 3, 5
- DUT3 tile shows: BLK 1, 2, 4, 6
- DUT4 tile shows: BLK 1, 3, 5, 6

Each tile shows a different subset of blocks, making it hard to compare.

## Root Cause Analysis

In `_drawSplitCharts()`:
1. We discover ALL color values globally and store in `allColorValues` ✓
2. We track per-bucket colors in `colorsByBucket[k]` ✓
3. But when calling `_buildTileChart(canvas, ..., colorsByBucket[key])`, we only pass the colors for that specific bucket
4. Each tile independently discovers colors from its own data
5. Tiles with different data have different color discovery

## Solution

**Three-part fix:**

### Part 1: Update Function Signature
Modified `_buildTileChart()` function signature to accept `allColorValues` parameter:
```javascript
function _buildTileChart(canvas, ct, tileData, xCol, yCol, xRx, yRx, colCol, colorRx,
                          xLog, yLog, innerSplitCol, innerSplitRx, chartId, bucketColors, allColorValues)
```

### Part 2: Pass Global Colors to Tiles
Updated the call to `_buildTileChart()` to pass `allColorValues`:
```javascript
var inst = _buildTileChart(canvas, ct, buckets[key] || [],
            xCol, yCol, xRx, yRx, colCol, colorRx, xLog, yLog, splitCol, splitRx, ki+1,
            colorsByBucket[key] || {}, allColorValues);  // <- added allColorValues
```

### Part 3: Use Global Colors in Color Discovery
In `_buildTileChart()`, added logic to ensure ALL global color values appear in the legend:
```javascript
/* CRITICAL FIX: For split charts, also ensure ALL GLOBAL COLOR VALUES are represented
   in the legend of this tile, even if they don't have measurements in this specific tile.
   This ensures consistent legends across all tiles when split-by-DUT or similar. */
if (allColorValues && Object.keys(allColorValues).length > 0) {
    Object.keys(allColorValues).forEach(function(colorVal) {
        if (!allColorGroups[colorVal]) {
            allColorGroups[colorVal] = true;
        }
    });
}
```

## Data Flow After Fix

```
Row with color value but no valid X/Y (e.g., BLK=3, DUT=DUT1, but no measurements)
    ↓
Added to tileData as placeholder point
    ↓
_drawSplitCharts discovers: allColorValues = {1,2,3,4,5,6}
    ↓
For each DUT tile:
  - Discover colors from tile's own data
  - ALSO add all global colors from allColorValues
  ↓
All tiles show same legend: {1,2,3,4,5,6}, with n=0 where applicable
```

## Expected Outcome

When split by DUT and color by BLK:
- All 4 DUT tiles show the same blocks in their legends
- Missing blocks show as "(n=0)"
- Consistent visualization across all tiles
- Makes cross-DUT comparison clearer

## Files Changed
- `fdv_chart.html` (3 modifications)
  - Function signature (line ~3495)
  - Color discovery logic (lines ~3693-3725)
  - Function call (line ~4071)

## Testing Steps

1. **Clear browser cache**: Ctrl+F5 on the application
2. **Reload data file**: Use the FDV UI to load a data file
3. **Test split-by-DUT with color-by-BLK**:
   - Select a data file
   - X = any numeric column (e.g., BLK or sequence)
   - Y = any numeric column (e.g., RBER)
   - Color = BLK
   - Split by DUT
   - Expected: All 4 DUT tiles show the same blocks in legend
4. **Compare with single chart**:
   - Remove split
   - Color = BLK (single chart mode)
   - Verify same blocks appear as in split mode

## Technical Notes

- The fix is backward compatible (all new parameters have defaults)
- No changes to rendering logic or data points
- Empty datasets for missing blocks are created (shows n=0 in legend)
- Placeholder points remain hidden during rendering (filtered before display)
- Works with both single-column and multi-dimensional color grouping

## Debugging

If issues persist, check browser console for:
- `[_drawSplitCharts] All COLOR values found (from all filtered rows):` - shows global colors discovered
- `[_buildTileChart] allColorGroups:` - shows colors in each tile's legend
- Count of placeholder vs. real points per tile

