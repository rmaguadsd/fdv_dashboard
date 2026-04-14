# Dimension Restoration - Complete Solution

## Problem Statement

The FDV Chart Rev1 was not properly restoring **color-by** and **split-chart** dimension settings when loading saved sessions, even though the settings were being saved to the snapshot.

## Root Cause Analysis

### Issue 1: Complex DOM-Based Restoration
The original approach tried to restore dimensions by:
1. Reading saved JSON arrays
2. Setting HTML select element values
3. Hoping the DOM would update properly

This was fragile because:
- Select options might not be populated at the right time
- Setting a value that doesn't exist as an option silently fails
- The order of operations was critical and error-prone

### Issue 2: Old Sessions Without Dimension Keys
Sessions saved before this feature was added don't have `__colorDims` or `__scDims` keys because they were never captured. They only have legacy `color-col` and `split-chart-col` inputs.

## Solution Implemented

### 1. Direct Restoration Function (`_gdimRestoreDirect()`)

Instead of trying to update existing DOM elements, the new function:
1. **Clears** all existing dimension rows
2. **Creates fresh** dimension rows from scratch
3. **Populates options** from `currentHeaders` while creating rows
4. **Sets values** (they exist as options)
5. **Updates** global dimension arrays

**Code location**: `fdv_chart.html`, function `_gdimRestoreDirect()`

**Key advantage**: No ambiguity about whether options exist - we build them fresh!

### 2. Restoration Entry Points

Dimensions are restored at TWO points:

**Point A: Immediate Restoration** (`_recipeApply()` function)
- Used when headers are already loaded in the UI
- Happens before loading rows
- Calls `_gdimRebuildSelects()` to populate options
- Calls `_gdimRestoreDirect()` to restore values

**Point B: Deferred Restoration** (`populatePlotSelectors()` function)
- Used when headers load from server (new parse)
- Happens after `populatePlotSelectors()` is called
- Calls `_gdimRestoreDirect()` on the pending snapshot

### 3. Automatic Session Migration (`_migrateSessionSnapshot()`)

This function runs when loading any session and:
1. **Detects** if the session is in old format (pre-dimension-arrays)
2. **Extracts** legacy `color-col`, `split-chart-col` values
3. **Converts** them to new `__colorDims`, `__scDims` format
4. **Patches** the snapshot before restoration

**Enables**: Loading old sessions that have no dimension arrays!

## Complete Restoration Flow

```
User clicks "Load Session"
    ↓
Server returns snap = {color-col: "RESULT", ...}
    ↓
_migrateSessionSnapshot() runs
    - Detects old format
    - Converts to __colorDims: "[{col:"RESULT",...}]"
    ↓
_recipeApply(snap) runs
    - Checks if headers loaded
    - If yes: immediate restoration
    - If no: defers to populatePlotSelectors()
    ↓
_gdimRestoreDirect('color', [...]) runs
    - Clears old dimension rows
    - Creates fresh rows with options
    - Sets values from snapshot
    - Updates global _colorDims array
    ↓
Dimensions appear in UI!
```

## Testing the Solution

### Test 1: New Session (Current Version)
1. Parse CSV
2. Add color-by dimension (select a column)
3. Add split-chart dimension (select a column)
4. Save session
5. Clear dimensions (click × buttons)
6. Load session
7. ✅ Dimensions should appear

### Test 2: Old Session (Pre-Dimension Feature)
1. Load an old saved session (created before this feature)
2. Open browser console (F12)
3. Look for `[_migrateSessionSnapshot]` messages
4. ✅ Dimensions should appear (migrated from legacy format)

## Code Changes Summary

| Function | Change | Purpose |
|----------|--------|---------|
| `_gdimRestoreDirect()` | New | Direct restoration without DOM ambiguity |
| `_migrateSessionSnapshot()` | New | Converts old sessions to new format |
| `_recipeApply()` | Modified | Uses `_gdimRestoreDirect()` instead of `_gdimSetDims()` |
| `populatePlotSelectors()` | Modified | Uses `_gdimRestoreDirect()` for deferred case |
| `_sessionLoad()` | Modified | Calls `_migrateSessionSnapshot()` before `_recipeApply()` |

## Files Modified

- `fdv_chart.html` (5893 lines)
  - Added `_gdimRestoreDirect()` function (~45 lines)
  - Added `_migrateSessionSnapshot()` function (~60 lines)
  - Updated restoration calls in `_recipeApply()` (~8 changes)
  - Updated restoration calls in `populatePlotSelectors()` (~4 changes)
  - Updated `_sessionLoad()` to call migration (~1 change)

## Key Features

✅ **Backwards Compatible**: Old sessions load with automatic migration
✅ **Robust**: No ambiguity about whether select options exist
✅ **Simple**: Direct creation instead of complex DOM manipulation
✅ **Extensible**: Easy to add more dimension types
✅ **Logged**: Console shows exactly what's happening for debugging

## Browser Console Logging

When loading a session, you'll see messages like:

```
[_migrateSessionSnapshot] Old session detected, migrating...
[_migrateSessionSnapshot] Migrated color-by: "[{"col":"RESULT",...}]"
[_recipeApply] Restoring dimensions immediately (headers loaded)
[_recipeApply] Rebuilding dimension selects with headers
[_recipeApply] Found __colorDims: "[...]"
[_recipeApply] Calling _gdimRestoreDirect for color with 1 dims
[_gdimRestoreDirect] Restoring color with 1 dims
[_gdimRestoreDirect] Adding row 0 for col=RESULT
[_gdimRestoreDirect] Done. Global array for color: [...]
```

These logs help verify the restoration is working correctly!

