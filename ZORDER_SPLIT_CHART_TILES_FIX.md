# Z-Order for Split-Chart Tiles - Fixed!

## 🎯 The Problem You Found

You were using **split-chart mode** (with SourceFile column creating multiple tiles), but z-order **wasn't working for the scatter/line tiles**. 

The logs you showed indicated:
```
[drawPlot] Split-chart mode detected
```

This meant the code was using `_buildTileChart()` to render each tile, but that function **was missing z-order support for scatter/line tiles**!

---

## ✅ What I Fixed

### Issue Location
- **Function:** `_buildTileChart()` (used for rendering each split-chart tile)
- **Chart types affected:** Scatter and Line charts in split-chart mode
- **Line:** 4527 (original code)

### The Bug
```javascript
// BEFORE (no z-order support):
var datasets = Object.keys(groups).sort().map(function(g, i) {
    return {
        label: g + ' (n=' + groups[g].length + ')',
        backgroundColor: PALETTE[i % PALETTE.length] + '99',  // Color by array index!
        ...
    };
});
```

**Problems:**
1. Using `.sort()` on group keys (alphabetical only)
2. Assigning colors by array index `[i]` (changes when array reorders)
3. No z-order support at all!

### The Solution
```javascript
// AFTER (full z-order support with persistent colors):
/* Create persistent group-to-color mapping */
if (!window._tileGroupColorMap) window._tileGroupColorMap = {};
var allGroupsSortedTile = Object.keys(groups).sort();
allGroupsSortedTile.forEach(function(g, idx) {
    if (!window._tileGroupColorMap[g]) {
        window._tileGroupColorMap[g] = PALETTE[idx % PALETTE.length];
    }
});

/* Parse z-order and convert indices to group names */
var zOrderElem = document.getElementById('z-order-input');
var zOrderInput = (zOrderElem && zOrderElem.value) ? zOrderElem.value.trim() : '';
// ... convert indices to names ...

/* Sort by z-order (or alphabetical if empty) */
if (convertedZOrderList.length > 0) {
    // Use z-order list as primary ordering
    // ... [sorting logic]
} else {
    sortedGroupKeys = groupKeys.sort();
}

/* Assign colors from persistent map (not by array index) */
var datasets = sortedGroupKeys.map(function(g) {
    var color = window._tileGroupColorMap[g] || PALETTE[0];
    return {
        label: g + ' (n=' + groups[g].length + ')',
        backgroundColor: color + '99',  // Uses persistent map!
        ...
    };
});
```

**Improvements:**
1. ✅ Added full z-order support for scatter/line tiles
2. ✅ Index-to-name conversion (1, 2, 3 → A, B, C)
3. ✅ Persistent color mapping (colors don't change when reordering)
4. ✅ Same logic as main chart for consistency

---

## 🧪 How to Test It Now

### Step 1: Hard Refresh
```
Press Ctrl+F5 to clear cache
```

### Step 2: Restore Your Session
```
If you had a session with split-chart and SourceFile column
Click "Restore Session" or re-open your chart
```

### Step 3: Plot Split-Chart with Z-Order
```
You should see multiple tiles (one per SourceFile value)
```

### Step 4: Apply Z-Order
```
Find the Z-Order field
Enter: 2, 1  (or 1, 2 to reverse)
Click Plot
```

### Step 5: Verify
```
✅ Tiles should redraw with NEW rendering order
✅ Colors should stay the SAME (not change)
✅ If z-order says "2, 1", group 1 should appear on top
```

---

## 📊 What Changed

**File:** `dev/aitools/fdv_chart_rev11/fdv_chart.html`
**Size:** 363,599 → 366,204 bytes (+2,605 bytes)
**Function:** `_buildTileChart()` for scatter/line tiles

**Additions:**
1. Persistent color mapping (`window._tileGroupColorMap`)
2. Z-order parsing (with index-to-name conversion)
3. Z-order sorting logic (matching main chart)
4. Color assignment from persistent map

---

## 🔍 Z-Order Features in Tiles

### Supported Chart Types
- ✅ **Scatter tiles** (split-chart mode)
- ✅ **Line tiles** (split-chart mode)
- ✅ **Boxplot tiles** (already had z-order)
- ✅ **Histogram tiles** (already had z-order)
- ✅ **Cumproba tiles** (already had z-order)
- ✅ **RCDF tiles** (already had z-order)

### Input Format
```
Z-Order: 1, 2, 3         # Use indices (1-based)
Z-Order: A, B, C         # Use group names
Z-Order: 3, B, 1         # Mixed (both work)
```

### Behavior
```
Legend (alphabetical):  A, B, C
No z-order (default):   C on top

Z-Order: 3, 2, 1 (or C, B, A):
Result: A on top, C at bottom

Z-Order: 1, 2, 3 (or A, B, C):
Result: C on top (same as default)
```

---

## ⚠️ Important Notes

### Color Persistence
```
Colors stay ASSIGNED to groups regardless of z-order:
- Group A = always blue (if alphabetically first)
- Group B = always green
- Group C = always red

Even if you reorder to "C, B, A", colors don't change!
```

### Split-Chart Specific
```
Z-order is applied per tile independently
Each tile has its own set of groups
Z-order input affects ALL tiles the same way
```

### Index Mapping
```
Indices are based on alphabetically sorted groups:

If you have groups: East, North, South, West
Alphabetical:       1     2      3      4

Z-Order: 4, 3, 2, 1 puts West on top of each tile
```

---

## 🚀 How It Works

### Before (without z-order for tiles)
```
1. Read tile data → groups: {A: [...], B: [...], C: [...]}
2. Sort alphabetically → [A, B, C]
3. Assign colors by index → A=blue(0), B=green(1), C=red(2)
4. Render → C on top (last in list)

Result: No control! Always alphabetical!
```

### After (with z-order for tiles)
```
1. Read tile data → groups: {A: [...], B: [...], C: [...]}
2. Read z-order input → "3, 2, 1"
3. Convert indices → ["C", "B", "A"]
4. Create persistent map → A=blue, B=green, C=red
5. Sort by z-order → [C, B, A]
6. Assign colors from map → C=red, B=green, A=blue
7. Render → A on top (last in reordered list)

Result: Full control with persistent colors!
```

---

## ✅ Summary

**What was fixed:**
- ✅ Added z-order support to scatter/line tile rendering
- ✅ Fixed color persistence issue (colors no longer change)
- ✅ Added index-to-name conversion (same as main chart)
- ✅ Implemented consistent sorting logic across all tiles

**What works now:**
- ✅ Z-order in split-chart mode (scatter/line tiles)
- ✅ Persistent colors (don't change when reordering)
- ✅ Index-based input (1, 2, 3 or A, B, C)
- ✅ All 6 chart types now have z-order support

**Where to use it:**
- ✅ Single scatter/line charts: YES
- ✅ Split-chart scatter/line tiles: YES (NEW!)
- ✅ Boxplot (main and tiles): YES
- ✅ Histogram tiles: YES
- ✅ Cumproba/RCDF tiles: YES

**Server status:**
✅ Running (366,204 bytes)
✅ All fixes deployed
✅ Ready for testing

Try it now with your split-chart data! 🎉
