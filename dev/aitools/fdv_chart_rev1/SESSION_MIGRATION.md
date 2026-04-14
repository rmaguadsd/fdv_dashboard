# Session Migration: Old to New Format

## What Changed

The new FDV Chart Rev1 now supports saving and restoring **color-by** and **split-chart** dimension settings in sessions.

### Old Session Format (Pre-Migration)

Old sessions stored dimension settings in individual inputs:
- `color-col`: The selected color column
- `color-regex`: The regex for color extraction
- `split-chart-col`: The split-chart column
- `split-chart-rx`: The split-chart regex

### New Session Format (Rev1)

New sessions use standardized dimension arrays:
- `__colorDims`: JSON array `[{col, colIdx, rx}, ...]`
- `__scDims`: JSON array `[{col, colIdx, rx}, ...]`
- `__splitDims`: JSON array `[{col, colIdx, rx}, ...]`

## Automatic Migration

**Good news: You don't need to do anything!**

When you load an old session, the application automatically:

1. **Detects** if the session is in old format (missing `__colorDims`, `__scDims`)
2. **Reads** the legacy `color-col`, `split-chart-col`, etc. from the session
3. **Converts** them to the new dimension array format
4. **Restores** the dimensions using the new restoration logic

### How It Works

The migration happens in the `_migrateSessionSnapshot()` function:

```javascript
/* Old session with color-col="RESULT" and color-regex="(\\d+)" */
→ Converted to __colorDims: "[{col:"RESULT", colIdx:2, rx:"(\\d+)"}]"

/* Old session with split-chart-col="PAGETYPE" */
→ Converted to __scDims: "[{col:"PAGETYPE", colIdx:5, rx:""}]"
```

## Testing Migration

To verify migration is working:

1. **Load an old session** from before this feature was added
2. **Open browser console** (F12 → Console)
3. **Look for messages** starting with `[_migrateSessionSnapshot]`:
   - `"Old session detected, migrating..."`
   - `"Migrated color-by: [...]"`
   - `"Migrated split-chart: [...]"`
4. **Check if dimensions appear** in the UI after loading

## New Session Saving

When you **save a new session**, it automatically uses the new format with dimension arrays. The next time you load it, it will load directly without needing migration.

## Backwards Compatibility

This approach is **backwards compatible**:
- Old sessions still load (with automatic migration)
- New sessions save in the new format (more reliable)
- The system works seamlessly either way

## Benefits

✅ Old sessions are automatically converted
✅ No manual intervention needed
✅ Dimension settings now properly restored
✅ More robust handling of multiple dimensions
✅ Easier to extend with more dimension types in the future

