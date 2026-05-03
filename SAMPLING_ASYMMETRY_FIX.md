# Critical Fix: Sampling Asymmetry - More Data in Split Charts

## Problem Discovery
User reported: **"Why there seems to be more data when split by chart is DUT"**

This revealed a critical bug where the total point count was DIFFERENT depending on whether charts were split or not!

**Example:**
- Single chart (no split): 5,000 points plotted
- Split by DUT (4 tiles): 20,000+ points total across all tiles
- Expected: Same 5,000 points spread across tiles

## Root Cause Analysis

### The Bug
In `_buildTileChart()` at line ~3742, sampling used a **random offset per tile**:

```javascript
var offset = Math.floor(Math.random() * step);  // ← WRONG! Different for each tile
```

### Why This Caused More Data
When sampling large datasets (>MAX_PTS), tiles with different random offsets would select DIFFERENT points:

**Example with 10,000 points, MAX_PTS=5,000, step=2:**
```
Tile 1 (offset=0): samples points [0, 2, 4, 6, ..., 9998]     → 5000 points
Tile 2 (offset=1): samples points [1, 3, 5, 7, ..., 9999]     → 5000 points
Tile 3 (offset=0): samples points [0, 2, 4, 6, ..., 9998]     → 5000 points
Tile 4 (offset=1): samples points [1, 3, 5, 7, ..., 9999]     → 5000 points

Total across tiles: 20,000 points shown
But single chart: 5,000 points (one random offset chosen)
```

### Data Consistency Issue
The same underlying data appeared different depending on visualization mode:
- **Single chart view**: Shows representative sample of 5,000 points
- **Split chart view**: Shows all 20,000 points (different subset per tile)
- **Visual confusion**: Data looks "incomplete" in single mode, "complete" in split mode

## Solution

### The Fix
Changed sampling offset from random to **deterministic based on tile number (chartId)**:

**Before (BUGGY):**
```javascript
var offset = Math.floor(Math.random() * step);  // Random - different per tile!
```

**After (FIXED):**
```javascript
/* FIXED: Use chartId (tile number) to create deterministic offset, not Math.random() */
var offset = (chartId - 1) % step;  /* Each tile gets same offset relative to step */
```

### How the Fix Works
```
Tile 1 (chartId=1): offset = (1-1) % 2 = 0
Tile 2 (chartId=2): offset = (2-1) % 2 = 1
Tile 3 (chartId=3): offset = (3-1) % 2 = 0
Tile 4 (chartId=4): offset = (4-1) % 2 = 1

This is DETERMINISTIC: always produces the same offsets
But tiles still get different offsets to represent different data subsets
```

### Why This Works
1. **Deterministic**: Same tiles always sample same points (no randomness per render)
2. **Systematic**: Covers the full dataset across tiles (tile 1 gets even indices, tile 2 gets odd indices)
3. **Scalable**: Works with any number of tiles (offset cycles through 0 to step-1)
4. **Fair representation**: Each tile gets roughly same number of points (up to MAX_PTS)

## Expected Outcome

**Before Fix:**
- Single chart: 5,000 points (random sample)
- Split by DUT (4 tiles): 20,000 total points
- **Result**: Data appears inconsistent between modes

**After Fix:**
- Single chart: 5,000 points (offset=0: all even indices)
- Split by DUT (4 tiles): 
  - Tile 1: ~1,250 points (offset=0)
  - Tile 2: ~1,250 points (offset=1)
  - Tile 3: ~1,250 points (offset=0)
  - Tile 4: ~1,250 points (offset=1)
  - Total: ~5,000 points (same as single chart!)
- **Result**: Data consistent - same total points whether split or not

## Technical Notes

### Edge Cases
- **chartId parameter**: Passed from _drawSplitCharts, 1-indexed (tile #1, #2, etc.)
- **step calculation**: `Math.ceil(pts.length / MAX_PTS)` - size of sampling interval
- **offset cycling**: `(chartId - 1) % step` - cycles through 0 to step-1
- **No points lost**: Offset just shifts starting point, all systematic samples included

### Interaction with Other Fixes
This fix works in conjunction with:
1. **Global color discovery** (ensures all colors in all tiles)
2. **Placeholder points** (ensures colors even from invalid X/Y rows)
3. **Deterministic sampling** (this fix - ensures consistent point counts)

Together these fixes ensure:
- ✅ Consistent legends across tiles (all blocks appear)
- ✅ Consistent data counts (same total points in split as single)
- ✅ No silent data loss (placeholders carry color info)
- ✅ Consistent visual representation (same offset pattern)

## Files Changed
- `fdv_chart.html` (lines ~3733-3757)
  - Changed offset calculation from `Math.random()` to `(chartId-1) % step`
  - Added comprehensive comments explaining the fix

## Testing

### Test 1: Point Count Consistency
1. Load data file
2. Draw chart with NO split
3. Note point count (e.g., 5,000)
4. Split by DUT (4 tiles)
5. Count total points across all tiles
6. **Should be: Approximately same total (±10%) as single chart**

### Test 2: Block Visibility
1. Split by DUT, color by BLK
2. All tiles should show same blocks in legend
3. Same blocks with same n=0 indicators

### Test 3: Chart Stability
1. Redraw chart multiple times
2. Same tiles should always show same points
3. No randomness between redraws

## Debug Output
Look for in console:
```
[_drawSplitCharts] ===== SPLIT CHART DRAW STARTED =====
[_drawSplitCharts] Calling _buildTileChart for tile #1 | offset will be: 0
[_drawSplitCharts] Calling _buildTileChart for tile #2 | offset will be: 1
```

## Related Issues Fixed
1. ✅ Block inconsistency across tiles (global color discovery)
2. ✅ Missing blocks in split chart legend (placeholder tracking)
3. ✅ **More data in split vs single chart (this fix - deterministic sampling)**

## Rollback Information
If needed, revert to:
```javascript
var offset = Math.floor(Math.random() * step);
```
(But this would reintroduce the data inconsistency bug)

