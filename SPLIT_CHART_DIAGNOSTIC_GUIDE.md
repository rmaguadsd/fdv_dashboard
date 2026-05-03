# Diagnostic Guide: Split Chart Block Inconsistency

## Steps to Test and Diagnose

### 1. Open Browser Developer Console
- Press **F12** to open Developer Tools
- Click on the **Console** tab
- Clear any existing messages: Click the trash icon or type `clear()`

### 2. Reload the Application
- Press **Ctrl+F5** (or **Cmd+Shift+R** on Mac) to clear cache
- Wait for the application to fully load

### 3. Load a Data File
- Use the FDV UI to load a data file with multiple DUTs and BLKs

### 4. Set Up Split-by-tname with Color-by-BLK
In the chart configuration panel:
- **X-axis column**: Select any numeric column (e.g., Line#, STEP, or RBER)
- **Y-axis column**: Select RBER or similar
- **Color by**: BLK
- **Split-chart column**: tname (or DUT if available)
- **Chart type**: scatter

### 5. Observe Console Output

When the chart renders, you should see debug messages in this order:

#### Message 1: Global Color Discovery
```
[_drawSplitCharts] All COLOR values found (from all filtered rows): 1, 2, 3, 4, 5, 6
```
This shows ALL blocks discovered globally from the entire dataset.

#### Message 2: Per-Tile Bucket Info (one per tile)
```
[_drawSplitCharts] Calling _buildTileChart for tile #1 key=tname_READ_1_BLK | passing allColorValues: 1, 2, 3, 4, 5, 6
[_buildSplitChart] Bucket[tname_READ_1_BLK]: 150 points | colors in this bucket: 1, 2, 3, 4
```

Notice:
- `passing allColorValues` shows ALL global colors (1-6)
- `colors in this bucket` shows only colors that have measurements in this bucket (1-4)

#### Message 3: Color Group Discovery in Tile (one per tile)
```
[_buildTileChart] chartId=1 allColorGroups after global merge: 1, 2, 3, 4, 5, 6
[_buildTileChart] chartId=1 tileData.length=150 validPts.length=150
[_buildTileChart] chartId=1 allColorValues (global): 1, 2, 3, 4, 5, 6
```

This shows:
- Final `allColorGroups` includes ALL colors (1-6) after adding global values
- The tile has 150 valid points
- Global colors received: 1-6

This pattern should repeat for each tile.

### 6. Verify the Fix is Working

✅ **Fix is working if:**
- All tiles show the SAME blocks in their legends
- Blocks with n=0 appear in all tiles consistently
- Console shows `allColorGroups after global merge` includes ALL global colors

❌ **Fix is NOT working if:**
- Different tiles show different sets of blocks
- Some blocks missing from some tiles
- Console shows `allColorGroups` with only local colors (missing global ones)

## Expected Console Pattern (4-tile split example)

```
[_drawSplitCharts] All COLOR values found (from all filtered rows): 1, 2, 3, 4, 5, 6
[_drawSplitCharts] All DUT values found (from all filtered rows): DUT1, DUT2, DUT3, DUT4
[_drawSplitCharts] Calling _buildTileChart for tile #1 key=tname_READ_1_BLK | passing allColorValues: 1, 2, 3, 4, 5, 6
[_buildTileChart] chartId=1 allColorGroups after global merge: 1, 2, 3, 4, 5, 6
[_drawSplitCharts] Calling _buildTileChart for tile #2 key=tname_READ_2_BLK | passing allColorValues: 1, 2, 3, 4, 5, 6
[_buildTileChart] chartId=2 allColorGroups after global merge: 1, 2, 3, 4, 5, 6
[_drawSplitCharts] Calling _buildTileChart for tile #3 key=tname_READ_3_BLK | passing allColorValues: 1, 2, 3, 4, 5, 6
[_buildTileChart] chartId=3 allColorGroups after global merge: 1, 2, 3, 4, 5, 6
[_drawSplitCharts] Calling _buildTileChart for tile #4 key=tname_READ_4_BLK | passing allColorValues: 1, 2, 3, 4, 5, 6
[_buildTileChart] chartId=4 allColorGroups after global merge: 1, 2, 3, 4, 5, 6
```

Notice how `allColorValues: 1, 2, 3, 4, 5, 6` and `allColorGroups after global merge: 1, 2, 3, 4, 5, 6` are the same for ALL tiles.

## Troubleshooting

### Problem: allColorValues is empty or undefined

**Possible causes:**
1. Color-by column not selected
2. No data rows being processed
3. Bug in color discovery loop

**How to verify:**
- Check that `[_drawSplitCharts] All COLOR values found` shows your blocks
- If empty, check that color-by is set and data is loaded

### Problem: allColorValues not received by tiles

**Possible causes:**
1. Function call not passing parameter (code issue)
2. allColorValues being set to undefined

**How to verify:**
- Check `passing allColorValues:` message
- Should show same values as `All COLOR values found`
- If empty or missing, code wasn't updated properly

### Problem: allColorGroups doesn't include global colors

**Possible causes:**
1. Global merge logic not executing
2. allColorValues parameter not being used
3. JavaScript error in loop

**How to verify:**
- Check `allColorGroups after global merge:` message
- Should match `allColorValues (global):` message
- If different, the merge isn't working

## Console Filter Tips

To see only split chart messages, type in console:
```javascript
// Filter for split chart messages (Chrome/Firefox)
console.log = (msg) => { if (msg.includes('_draw') || msg.includes('_build')) { 
    orig_log.call(console, msg); } };
```

Or use the browser's built-in search in console with keywords:
- `_drawSplit` - Shows main split chart messages
- `_buildTile` - Shows per-tile messages
- `allColor` - Shows color discovery messages

## Next Steps

1. **Collect console output** after applying the fix
2. **Share the output** so we can verify the fix is working
3. **If working**: Test with different data and configurations
4. **If not working**: We'll dig deeper based on the console messages

