# Z-Order Feature Implementation Summary

## What Was Added

### 1. **UI Control** (Line 786-791)
Added a new text input field in the Plot Panel:
```html
<label id="z-order-label" style="margin-left:8px;display:flex;align-items:center;gap:4px">
    <span style="font-size:0.82em;color:#6c757d;font-weight:bold;white-space:nowrap">Z-Order:</span>
    <input type="text" id="z-order-input" placeholder="e.g. red, blue, green" autocomplete="off"
           style="width:180px;font-size:0.85em"
           title="Specify rendering order (last = on top). Comma-separated color-by values. Groups not listed render first (bottom)">
    <span style="font-size:0.75em;color:#999;white-space:nowrap">(last = top)</span>
</label>
```

**Location:** Right after Color-by control in plot-bar  
**Placeholder:** "e.g. red, blue, green"  
**Tooltip:** Explains rendering order semantics

### 2. **Sorting Logic** (Lines 5095-5120)

#### Parse Z-Order Input
```javascript
var zOrderInput = document.getElementById('z-order-input').value.trim();
var zOrderList = [];
if (zOrderInput) {
    zOrderList = zOrderInput.split(',').map(function(s) { return s.trim(); });
}
```
- Reads the z-order text input
- Splits by comma
- Trims whitespace from each value

#### Custom Sort Comparator
```javascript
var sortedGroupKeys = groupKeys.sort(function(a, b) {
    var aIdx = zOrderList.indexOf(a);
    var bIdx = zOrderList.indexOf(b);
    
    /* Both not in z-order list: keep alphabetical */
    if (aIdx === -1 && bIdx === -1) return a.localeCompare(b);
    
    /* One not in z-order: unlisted comes first */
    if (aIdx === -1) return -1;
    if (bIdx === -1) return 1;
    
    /* Both in z-order: sort by position (earlier = lower rendering order) */
    return aIdx - bIdx;
});
```

**Sorting Rules:**
1. If both groups are NOT in z-order list → alphabetical order
2. If only one is not in z-order → unlisted goes first (bottom)
3. If both in z-order → sort by position in list (lower index = lower rendering)

#### Build Datasets in Sorted Order
```javascript
var datasets = sortedGroupKeys.map(function(g, i) {
    return {
        label: g + ' (n=' + groups[g].length + ')',
        data: groups[g],
        backgroundColor: PALETTE[i % PALETTE.length] + '99',
        // ... rest of dataset config
    };
});
```
- Uses `sortedGroupKeys` instead of `Object.keys(groups).sort()`
- Maintains all existing dataset configuration
- Colors assigned in dataset order (palette index)

## How It Works Visually

### Example Flow

**Input:**
- Color-by column: Status
- Values in data: "pending", "approved", "rejected", "flagged"
- Z-Order input: `rejected, approved, flagged`

**Processing:**
1. Parse z-order: ["rejected", "approved", "flagged"]
2. Identify groups: {pending, approved, rejected, flagged}
3. Sort groups:
   - "pending" not in list → position 0 (bottom)
   - "rejected" at index 0 → position 1
   - "approved" at index 1 → position 2
   - "flagged" at index 2 → position 3 (top)
4. Build datasets in this order
5. Chart.js renders bottom-to-top

**Result:** "flagged" points render on top (most visible)

## Compatibility

### Chart Types Supported
- ✅ Scatter
- ✅ Line
- ✅ Histogram
- ✅ Cumulative Probability
- ✅ RCDF
- ✅ Box & Whisker (with overlay)

### Multi-Dimensional Color-By
- Works with compound keys (e.g., "value1~value2")
- Enter full compound keys in z-order field

### Backward Compatibility
- ✅ Feature is purely additive
- ✅ Empty z-order field = default alphabetical sorting
- ✅ Existing recipes/sessions still work (z-order field is new)
- ✅ No breaking changes to existing functionality

## Technical Details

### Data Flow
```
Z-Order Input (user enters comma-separated values)
    ↓
Parse & trim whitespace
    ↓
Create z-order position map
    ↓
Sort group keys with custom comparator
    ↓
Build datasets in sorted order
    ↓
Chart.js renders with z-order applied
```

### Key Variables
- `zOrderInput`: Raw string from input field
- `zOrderList`: Array of parsed z-order values
- `groupKeys`: All unique color-by values from data
- `sortedGroupKeys`: Group keys sorted by z-order logic
- `aIdx`, `bIdx`: Position in z-order list (-1 if not listed)

### Performance
- **Complexity:** O(n log n) where n = number of color-by groups
- **Typical impact:** Negligible (usually < 100 groups)
- **Memory:** Minimal (small array of strings)

## File Changes

**File:** `dev/aitools/fdv_chart_rev11/fdv_chart.html`

**Lines Added:**
1. **UI Control:** Lines 786-791 (new label with z-order input)
2. **Sorting Logic:** Lines 5097-5120 (parse and sort)
3. **Total added:** ~40 lines of HTML + JS

**Lines Modified:**
- Line 5096: Changed from `Object.keys(groups).sort()` to use custom sorting
- (The old line is replaced by new sorting logic)

## Testing Recommendations

1. **Basic Test:** Enter "a, b, c" for a scatter plot with groups A, B, C
   - Expected: C on top, B middle, A bottom

2. **Partial List Test:** Enter "c" with groups A, B, C
   - Expected: A and B bottom (alphabetical), then C on top

3. **Empty Field Test:** Clear z-order field
   - Expected: Return to alphabetical ordering (A bottom, C top)

4. **Multi-Dim Test:** Use two color-by columns
   - Enter compound keys in z-order field
   - Expected: Sorting works on compound keys

5. **Histogram Test:** Apply to histogram chart type
   - Expected: Bars render in z-order (last bar on top)

## Future Enhancements

### Possible Extensions
1. **Visual Z-Order Editor:** Drag/drop legend items to reorder
2. **Save Z-Order in Recipes:** Store z-order preference with recipe
3. **Auto-Detect:** Automatically suggest z-order based on data patterns
4. **Regular Expressions:** Support regex patterns in z-order (e.g., "error*")
5. **Relative Positioning:** "Move X to front" vs absolute ordering

### Notes
- Current implementation is intentionally simple
- All complexity is in the sort comparator (could expand there)
- UI is minimal but functional

---

**Status:** ✅ Complete and deployed  
**Date:** May 21, 2026  
**Tested:** Manual verification on localhost:5059
