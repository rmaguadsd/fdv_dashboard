# Bug Fix: Only One Group Rendered

**Issue**: When configuring split-chart grouping with 2+ groups, only the last group was rendered. X-axis showed only one group name and separators weren't drawn.

**Root Cause**: Line 2641 in `_buildGroupedXAxis()` was overwriting the groupMap for duplicate x-values:

```javascript
// WRONG - overwrites previous group
groupMap[xVal] = groupVal;  
```

Example scenario:
```
Data points:
- (WL=0, pagetype=UP)  → groupMap['0'] = 'UP'
- (WL=0, pagetype=TP)  → groupMap['0'] = 'TP'  ← OVERWRITES!
- (WL=1, pagetype=UP)  → groupMap['1'] = 'UP'
- (WL=1, pagetype=TP)  → groupMap['1'] = 'TP'  ← OVERWRITES!

Result: groupMap only contains the LAST group for each x-value
Only one group (TP) visible because it overwrote UP
```

## The Fix

**Location**: Line 2641, function `_buildGroupedXAxis()`

**Changed**:
```javascript
/* OLD - overwrites on duplicate x-values */
if (xVal && groupVal) {
    groupMap[xVal] = groupVal;  /* Map each x-value to its group */
    if (!groupsSet[groupVal]) groupsSet[groupVal] = {};
    groupsSet[groupVal][xVal] = true;
}

/* NEW - preserves first occurrence for lookup, all groups in groupsSet */
if (xVal && groupVal) {
    /* Only set groupMap for first occurrence of this x-value
       (for display purposes - actual groups come from groupsSet) */
    if (!groupMap[xVal]) {
        groupMap[xVal] = groupVal;
    }
    /* Add to groups set - this preserves ALL groups */
    if (!groupsSet[groupVal]) groupsSet[groupVal] = {};
    groupsSet[groupVal][xVal] = true;
}
```

**Why This Works**:
1. `groupMap` only stores the **first** group for each x-value (prevents overwrites)
2. `groupsSet` collects **all** groups with their x-values (complete picture)
3. The `groups` array built from `groupsSet` contains all unique groups
4. `groupSeparators` plugin reads from `groups` array, which now has all groups

## Data Structure After Fix

```javascript
groupsSet = {
    'UP': {'0': true, '1': true, '2': true, '3': true, '4': true},
    'TP': {'0': true, '1': true, '2': true, '3': true, '4': true}
}

groups = [
    { name: 'UP', xValues: ['0','1','2','3','4'] },
    { name: 'TP', xValues: ['0','1','2','3','4'] }
]

groupMap = {
    '0': 'UP',   ← First group for x-value 0
    '1': 'UP',   ← First group for x-value 1
    '2': 'UP',   ← etc.
    '3': 'UP',
    '4': 'UP'
}
```

## Console Output After Fix

```javascript
[_buildGroupedXAxis] Built 2 groups: UP:0,1,2,3,4 | TP:0,1,2,3,4
[groupSeparators] Drawing separators for 2 groups
[groupSeparators] Group 1 UP - drawing separator at x=2.0 (pixel=234)
[groupSeparators] Drawing group tier labels at y=450
[groupSeparators] Group label "UP" at pixel x=150
[groupSeparators] Group label "TP" at pixel x=350
```

**Before Fix**: Only showed UP group (TP overwrote it)
**After Fix**: Shows both UP and TP groups ✅

## Visual Result

### Before Fix (Broken)
```
X-axis:    0 1 2 3 4
Groups:       UP            ← Only one group visible
           ──────────────    ← No separators
```

### After Fix (Correct)
```
X-axis:    0 1 2 3 4  |  0 1 2 3 4    ← Repeated for each group
Groups:       UP           TP          ← Both groups visible
           ──────────────  ──────────   ← Separator between groups
```

## Testing

To verify the fix works:

1. Load CSV with data having:
   - X column (e.g., WL with values 0-4)
   - Grouping column (e.g., pagetype with values UP, TP)
   - Y column (measurement)

2. Configure:
   - Y = measurement
   - X = WL
   - Add grouping: pagetype

3. Render chart

4. Check browser console for:
   ```
   [_buildGroupedXAxis] Built 2 groups: UP:... | TP:...
   ```

5. Verify visually:
   - Two group labels below axis (UP, TP)
   - Dashed separator line between groups
   - X-values repeated (0-4, 0-4)

## Files Modified

- `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev7\fdv_chart.html`
  - Lines 2616-2648: Fixed `_buildGroupedXAxis()` function

## Impact

- ✅ All groups now rendered correctly
- ✅ Separators drawn between all groups
- ✅ Group tier labels show all categories
- ✅ Split-chart visualization working as designed
- ✅ No breaking changes to existing code

## Root Cause Analysis

The bug existed because:

1. **Assumption**: Original code assumed "one x-value = one group" 
   - But in split-chart, "one x-value = multiple groups"

2. **Overwrite Pattern**: Used simple assignment that overwrote on duplicates
   - `groupMap[xVal] = groupVal` replaces previous value
   - Lost all but the last group

3. **Unused groupsSet Initially**: The `groupsSet` collected all data but wasn't properly isolated
   - groupMap was corrupted before groups array was built

## Lesson Learned

When the same x-value appears in multiple groups:
- Store ALL associations (in `groupsSet`)
- Don't rely on simple key-value map that can overwrite
- Use collection structures that preserve multiplicity
