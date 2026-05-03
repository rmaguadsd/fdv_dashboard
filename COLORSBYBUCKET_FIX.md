# Fix: colorsByBucket Not Tracking Colors from Invalid X/Y Rows

## Problem
Even with placeholder points being added for invalid X/Y rows, blocks were still not appearing consistently across all split-by-DUT tiles. The issue was that while placeholder points were being added to buckets, the `colorsByBucket[k]` tracking dictionary was NOT being updated for those placeholder rows.

## Root Cause
In `_drawSplitCharts()` (lines 3805-4050):
- When a row has valid X/Y: `colorsByBucket[k][colorVal]` is updated ✓
- When a row has invalid X/Y: `colorsByBucket[k][colorVal]` was NOT being updated ✗

When `_buildTileChart()` is called for each split tile, it receives `colorsByBucket[key]` as a parameter. This dictionary is used to discover ALL colors that should appear in that tile's legend, including colors from rows with no valid measurements.

Since `colorsByBucket[k]` was incomplete, some colors were silently missing from the legend discovery process in `_buildTileChart()`.

## Solution
Modified the placeholder point addition logic (lines 3950-3970) to also update `colorsByBucket[k]` when adding placeholder points:

```javascript
keys_r.forEach(function(key) { 
    buckets[key].push(ptPlaceholder);
    /* ALSO track the color value for this bucket (even though X/Y invalid) */
    if (gi2 >= 0 && row[gi2] != null) {
        var colorVal = String(row[gi2]);
        if (!colorsByBucket[key]) colorsByBucket[key] = {};
        colorsByBucket[key][colorVal] = true;
    }
});
```

## Data Flow After Fix
1. Row with invalid X/Y but has color value
2. Placeholder point added to bucket: `buckets[k].push(ptPlaceholder)`
3. Color value tracked: `colorsByBucket[k][colorVal] = true`
4. When `_buildTileChart(canvas, ..., colorsByBucket[k])` called
5. Color discovery in `_buildTileChart` now includes this color
6. Color appears in legend even with n=0

## Testing
1. Reload browser (Ctrl+F5)
2. Load FDV data file
3. **Case #1**: Color by BLK (no split) → Should see all blocks in legend
4. **Case #2**: Split by DUT, Color by BLK → Should see same blocks in ALL DUT tiles

## Expected Outcome
All blocks now consistently appear in legends across all split tiles, even if they have zero valid measurements in that tile.

## Technical Notes
- The fix is minimal (6 lines added)
- No changes to rendering logic
- Placeholder points remain filtered out during rendering (line 3689)
- Only affects color discovery in legend generation
- Backward compatible with existing code
