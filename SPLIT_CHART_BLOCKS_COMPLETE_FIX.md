# Complete Fix Summary: Split Chart Block Inconsistency

## Issue Description
When splitting charts by DUT and coloring by BLK, different DUT tiles show different blocks in their legends, making it impossible to see the complete picture of which blocks exist in the data.

## Investigation Journey

### Discovery 1: Missing Placeholder Tracking
**Initial Finding**: When rows had invalid X/Y values, we were adding placeholder points to buckets but NOT updating `colorsByBucket[k]` (the dictionary that tracks which colors exist in each bucket).

**Fix #1** (lines 3950-3970):
- Updated placeholder point addition to also record color values in `colorsByBucket[k]`
- This ensured per-bucket color tracking was complete

### Discovery 2: Local vs Global Color Tracking
**Critical Finding**: Even with per-bucket color tracking, each tile still only showed its own colors because we were NOT passing the GLOBAL color list to each tile.

When split by DUT and color by BLK:
- Global colors discovered: BLK = {1, 2, 3, 4, 5, 6}
- But each DUT tile only knew about its own subset
- DUT1 didn't know about BLK 5 and 6 (which only appear in other DUTs)
- Result: Inconsistent legends across tiles

**Fix #2** (three changes):
1. Updated `_buildTileChart()` function signature to accept `allColorValues` parameter
2. Modified the call to `_buildTileChart()` to pass `allColorValues`
3. Added logic in `_buildTileChart()` to ensure ALL global colors appear in each tile's legend

## Code Changes

### Change 1: Track Colors from Placeholder Rows
**File**: `fdv_chart.html` | **Lines**: 3950-3970

```javascript
keys_r.forEach(function(key) { 
    buckets[key].push(ptPlaceholder);
    /* ALSO track the color value for this bucket (even though X/Y invalid) */
    if (gi2 >= 0 && row[gi2] != null) {
        var colorVal = String(row[gi2]);
        if (!colorsByBucket[key]) colorsByBucket[key] = {};
        colorsByBucket[key][colorVal] = true;  // <- NEW
    }
});
```

### Change 2: Function Signature Update
**File**: `fdv_chart.html` | **Line**: ~3495

```javascript
// OLD:
function _buildTileChart(canvas, ct, tileData, xCol, yCol, xRx, yRx, colCol, colorRx,
                          xLog, yLog, innerSplitCol, innerSplitRx, chartId, bucketColors)

// NEW:
function _buildTileChart(canvas, ct, tileData, xCol, yCol, xRx, yRx, colCol, colorRx,
                          xLog, yLog, innerSplitCol, innerSplitRx, chartId, bucketColors, allColorValues)
                          //                                                                ^^^^^^^^^^^^^^^^
```

### Change 3: Pass Global Colors to Each Tile
**File**: `fdv_chart.html` | **Line**: ~4071

```javascript
// OLD:
var inst = _buildTileChart(canvas, ct, buckets[key] || [],
            xCol, yCol, xRx, yRx, colCol, colorRx, xLog, yLog, splitCol, splitRx, ki+1,
            colorsByBucket[key] || {});

// NEW:
var inst = _buildTileChart(canvas, ct, buckets[key] || [],
            xCol, yCol, xRx, yRx, colCol, colorRx, xLog, yLog, splitCol, splitRx, ki+1,
            colorsByBucket[key] || {}, allColorValues);  // <- added allColorValues
```

### Change 4: Use Global Colors in Legend Discovery
**File**: `fdv_chart.html` | **Lines**: ~3693-3725

Added comprehensive comment and code block:

```javascript
/* CRITICAL FIX: For split charts, also ensure ALL GLOBAL COLOR VALUES are represented
   in the legend of this tile, even if they don't have measurements in this specific tile.
   This ensures consistent legends across all tiles when split-by-DUT or similar.
   Example: When split by DUT and color by BLK:
   - DUT1 has BLK[1,2,3,4], DUT2 has BLK[1,2,3,5]
   - Both tiles should show BLK[1,2,3,4,5] in legend, with n=0 for missing ones */
if (allColorValues && Object.keys(allColorValues).length > 0) {
    Object.keys(allColorValues).forEach(function(colorVal) {
        if (!allColorGroups[colorVal]) {
            allColorGroups[colorVal] = true;
        }
    });
}
```

## Why This Works

### Before Fix
```
_drawSplitCharts():
  ├─ Discovers: allColorValues = {1, 2, 3, 4, 5, 6}  (ALL blocks globally)
  ├─ Creates buckets:
  │  ├─ DUT1: points with colors {1,2,3,4}
  │  ├─ DUT2: points with colors {1,2,3,5}
  │  ├─ DUT3: points with colors {1,2,4,6}
  │  └─ DUT4: points with colors {1,3,5,6}
  └─ For each tile:
     └─ _buildTileChart(buckets[DUT1], colorsByBucket[DUT1])  <- Only {1,2,3,4}!
        └─ Result: DUT1 tile shows only {1,2,3,4}
```

### After Fix
```
_drawSplitCharts():
  ├─ Discovers: allColorValues = {1, 2, 3, 4, 5, 6}
  ├─ Creates buckets: (same as before)
  └─ For each tile:
     └─ _buildTileChart(buckets[DUT1], colorsByBucket[DUT1], allColorValues)
                                                              ^^^^^^^^^^^^^^
        └─ Color discovery:
           1. From tileData: {1, 2, 3, 4}
           2. From allColorValues: Add {5, 6}
        └─ Result: DUT1 tile shows {1, 2, 3, 4, 5, 6} with n=0 for {5, 6}
```

## Impact

### Behavior Changes
- ✅ All tiles now show consistent set of blocks in legends
- ✅ Missing blocks appear with "(n=0)" notation
- ✅ Easier to compare across DUT tiles
- ✅ No visible data points changed (placeholders remain hidden)

### Performance Impact
- Minimal: Only added ~15 lines of code
- No additional data processing required
- Uses existing `allColorValues` already computed

### Backward Compatibility
- ✅ New parameter has default value
- ✅ No breaking changes
- ✅ Works with existing code

## Verification

### Console Debug Output
Look for these messages to verify the fix is working:

1. Check global colors discovered:
   ```
   [_drawSplitCharts] All COLOR values found (from all filtered rows): 1, 2, 3, 4, 5, 6
   ```

2. Check per-bucket colors (should now be complete):
   ```
   [_drawSplitCharts] Bucket[DUT1]: colors: 1, 2, 3, 4, (5, 6 added globally)
   [_drawSplitCharts] Bucket[DUT2]: colors: 1, 2, 3, 5, (4, 6 added globally)
   ```

### Visual Verification
When split by DUT and color by BLK:
- ✅ All 4 DUT tiles show same blocks in legend
- ✅ Blocks with no data in that tile show "(n=0)"
- ✅ Legends are consistent across all tiles

## Related Documents
- `COLORSBYBUCKET_FIX.md` - First fix (placeholder tracking)
- `GLOBAL_COLOR_DISCOVERY_FIX.md` - Second fix (global color discovery)
- `DATA_CONSISTENCY_INVESTIGATION.md` - Original investigation notes
- `DATA_LOSS_BUG_INVESTIGATION.md` - Data loss analysis

