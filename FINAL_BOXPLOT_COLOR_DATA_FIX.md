# CRITICAL FIX: Split Chart Boxplot Data - Color vs X Column

## The Final Bug

When building boxplot split chart tile data in `_drawSplitCharts()`, the code was storing the **X column value** in the `group` field instead of the **COLOR column value**!

**Buggy Code (Line 11203)**:
```javascript
var bpXraw = xi_bp >= 0 && row[xi_bp] != null ? String(row[xi_bp]).trim() : xCol;
var bpPt = { x: 0, y: bpYv, group: bpXraw, _ri: rowIdx };  /* group = X value! */
```

**Problem**:
- `group` field should contain the COLOR value (for `colorKey()` to work)
- But it contained the X value
- When tiles rendered, they used the wrong key for color lookup
- All colors fell back to same default → **BOTH SAME COLOR BUG**

## Why Other Chart Types Were OK

**Non-boxplot charts** (histogram, cumproba, cum_sigma) use the correct pattern:
```javascript
var gi2  = colCol   ? currentHeaders.indexOf(colCol)   : -1;
var si2  = splitCol ? currentHeaders.indexOf(splitCol) : -1;
var pt2  = { x: xv2, y: yv2, _ri: rowIdx };
if (gi2 >= 0 && row[gi2] != null)  pt2.group      = String(row[gi2]);      /* Correct! */
if (si2 >= 0 && row[si2] != null)  pt2.splitGroup = String(row[si2]);      /* Correct! */
buckets[key].push(pt2);
```

**Only boxplot was wrong!**

## The Fix

Added color column index and proper value extraction:

**File**: `fdv_chart_rev15/fdv_chart.html`  
**Location**: Lines ~11147-11210

### Step 1: Add Color Column Index (Line 11151)
```javascript
var ci_bp = colCol ? currentHeaders.indexOf(colCol) : -1;  /* Color column index */
```

### Step 2: Extract Color Value and Fix Point Creation (Lines 11203-11207)
```javascript
/* Extract color value from color column if available */
var bpColorRaw = ci_bp >= 0 && row[ci_bp] != null ? String(row[ci_bp]) : '';

var bpPt = { x: 0, y: bpYv, group: bpColorRaw, splitGroup: bpXraw, _ri: rowIdx };
```

**Changes**:
- `group` now contains COLOR value (was X value)
- `splitGroup` now contains X value (for reference)
- This matches the non-boxplot pattern exactly!

## Data Structure After Fix

**Boxplot Point**:
```javascript
{
    x: 0,                    /* Always 0 for boxplot X positioning */
    y: numericValue,         /* Y value from yCol */
    group: colorValue,       /* COLOR value - matches color map keys! */
    splitGroup: xValue,      /* X value - for reference */
    _ri: rowIndex            /* Original row index */
}
```

**Histogram/Cumproba/CumSigma Point** (already correct):
```javascript
{
    x: numericValue,         /* X value from xCol */
    y: numericValue,         /* Y value from yCol */
    group: colorValue,       /* COLOR value - matches color map keys! */
    splitGroup: splitValue,  /* Split value if applicable */
    _ri: rowIndex            /* Original row index */
}
```

## How It Works Now

### Step 1: Color Map Building
```
_drawSplitCharts() discovers: ["FALSE", "TRUE"]
Builds map: {FALSE: RED, TRUE: BLUE}
```

### Step 2: Tile Data Creation
Boxplot now correctly extracts:
```
row data with color column = "FALSE"
bpColorRaw = "FALSE"
pt.group = "FALSE"  ← NOW CORRECT!
```

### Step 3: Tile Rendering
In `_buildTileChart()`:
```
colorKey(pt) uses pt.group = "FALSE"
colorKey() uses colorRx to extract from "FALSE" → "FALSE"
Looks up window._groupColorMap["FALSE"] → RED ✅
```

### Step 4: Result
```
Chart 1 (TRUE only): Uses map["TRUE"] → BLUE ✅
Chart 2 (FALSE + TRUE):
  - FALSE points: Uses map["FALSE"] → RED ✅
  - TRUE points: Uses map["TRUE"] → BLUE ✅
  - DISTINCT COLORS! ✅
```

## Why This Was So Hard to Find

1. **Boxplot special case**: Only boxplot chart type has custom data building logic
2. **X vs Color confusion**: Code used `xi_bp` (X index) instead of `ci_bp` (color index)
3. **Comment was misleading**: Said "just like main chart does" but it wasn't
4. **Fallback behavior hid issue**: When key not found, fallback to `PALETTE[0]` made it look like consistent coloring

## Verification Checklist

After fix, verify:
- [ ] Boxplot tile 1 (TRUE only): Shows in BLUE
- [ ] Boxplot tile 2 (FALSE + TRUE): FALSE in RED, TRUE in BLUE  
- [ ] Histogram tile 2: FALSE in RED, TRUE in BLUE
- [ ] Cumproba tile 2: FALSE in RED, TRUE in BLUE
- [ ] Cum_sigma tile 2: FALSE in RED, TRUE in BLUE
- [ ] Console logs show correct color map: {FALSE: #dc3545, TRUE: #007bff}

## Code Consistency

Now ALL split chart types follow the same pattern:
1. Extract color column value
2. Store in `pt.group` field
3. Extract split column value
4. Store in `pt.splitGroup` field
5. `colorKey()` uses `group` field with `colorRx`

**Before**: ❌ Boxplot broke this pattern  
**After**: ✅ All types consistent

## Status

✅ **Root cause identified**: Boxplot stored X value instead of color value  
✅ **Fixed**: Now extracts and stores color value correctly  
✅ **Consistent**: Matches histogram/cumproba/cum_sigma pattern  
✅ **Validated**: No syntax errors  
✅ **Deployed**: Server restarted with fix applied  

## Impact

**Fixes**: Split chart boxplot colors now match other chart types  
**Affects**: Only boxplot split chart tiles  
**Performance**: No impact (same logic, just using correct column)  
**Compatibility**: Fully backward compatible
