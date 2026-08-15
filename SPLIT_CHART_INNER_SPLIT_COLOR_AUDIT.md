# Split Chart Inner Split Color Audit & Fix

## Problem Statement

When using split charts with an **inner split dimension** (split-by), if there are multiple plots/groups within a single split chart tile, all groups were using the same color or colors from the wrong palette.

**Example Scenario**:
- Outer split: By DUT (creates separate tiles for DUT=A, DUT=B)
- Inner split: By Block (within each DUT tile, show Block=1, Block=2, Block=3)
- **Bug**: All blocks in each tile would be colored from a limited palette, not getting distinct colors

## Root Cause Analysis

### Color Map vs Group Keys Mismatch

The issue was an **architectural mismatch** between how group keys are discovered vs. how they're used:

**Color Map Building (lines ~11000-11050)**:
```javascript
/* Discovers from COLOR dimension only */
if (hasColorDims) {
    colorVal = _compoundKey(row, _colorDims) || '(all)';
} else {
    colorVal = _extractGroupKey(raw || '(blank)', colorRx);  /* Uses color-col */
}
allColorValues[colorVal] = true;
```

**Group Key Usage in Rendering (lines ~9827, ~10044, ~10237)**:
```javascript
/* Uses SPLIT dimension if available, otherwise color dimension */
var g = innerSplitCol ? innerSplitKey(pt) : colorKey(pt);
```

**THE PROBLEM**:
1. Color map is built from COLOR dimension values (e.g., "Block1", "Block2")
2. Split rendering uses SPLIT dimension values (e.g., "Wafer1-Block1", "Wafer2-Block2") when `splitCol` is set
3. These don't match! → **All groups fall back to `PALETTE[0]` (same color)**

### Timeline

1. Lines 11000-11050: Build color map from **color dimension ONLY**
2. Line 11075: Read **split dimension** (TOO LATE!)
3. Line 11378: Pass both color and split params to `_buildTileChart()`
4. Lines 9827+: Rendering uses **split keys** that aren't in the color map!

## Solution Implemented

### Early Discovery of Split Dimensions

**New Approach**: Move split dimension discovery BEFORE color map building so we know what grouping will actually be used.

**File**: `fdv_chart_rev15/fdv_chart.html`  
**Location**: Lines ~11000-11050 (in `_drawSplitCharts()`)

### Key Changes

#### 1. Discover Split Dimensions Early (Lines ~11004-11014)

```javascript
/* ALSO need to discover split dimension values since they'll be used for grouping if splitCol is set */
var splitDims = _gdimRead('split');  /* Discover early to check if inner split will be used */
var splitCol  = splitDims.length > 0 ? splitDims[0].col : '';
var splitRx   = splitDims.length > 0 ? splitDims[0].rx : '';
var splitIdx  = splitCol ? currentHeaders.indexOf(splitCol) : -1;
var hasSplitDims = _splitDims.length > 0 && _splitDims.some(function(d){ return d.colIdx >= 0; });
```

**Why**: We need to know if an inner split will be used BEFORE discovering color values.

#### 2. Conditional Color/Split Discovery (Lines ~11017-11047)

```javascript
allPoints.forEach(function(pt) {
    var row = allRows[pt._ri];
    if (!row) return;

    var colorVal;

    /* If there's an inner split dimension, use it for grouping (takes precedence) */
    if (splitCol) {
        /* Extract from split dimension like _buildTileChart does */
        if (hasSplitDims) {
            colorVal = _compoundKey(row, _splitDims) || '(all)';
        } else {
            var raw = (splitIdx >= 0 && row[splitIdx] != null)
                ? String(row[splitIdx])
                : '';
            colorVal = _extractGroupKey(raw || '(blank)', splitRx);
        }
    } else {
        /* No inner split - use color dimension like normal */
        if (hasColorDims) {
            colorVal = _compoundKey(row, _colorDims) || '(all)';
        } else {
            var raw = (scatterColIdx >= 0 && row[scatterColIdx] != null)
                ? String(row[scatterColIdx])
                : '';
            colorVal = _extractGroupKey(raw || '(blank)', colorRx);
        }
    }

    allColorValues[colorVal] = true;
});
```

**Why**: Now we discover the ACTUAL keys that will be used for rendering, whether they come from the color dimension OR the split dimension.

#### 3. Removed Duplicate Split Dimension Read (Line ~11083)

**Before**:
```javascript
var splitDims = _gdimRead('split');  /* Duplicate read */
var splitCol  = splitDims.length > 0 ? splitDims[0].col : '';
var splitRx   = splitDims.length > 0 ? splitDims[0].rx : '';
```

**After**:
```javascript
/* splitDims/splitCol/splitRx already read above for color discovery */
```

**Why**: Avoid redundant code and ensure same values are used everywhere.

## Scenarios Covered

### Scenario 1: No Split (Normal Color Grouping)
```
Color By: BLK (Block)
Split By: (none)
```
- Color map discovers: [Block1, Block2, Block3]
- Rendering uses: color dimension
- Result: ✅ Each block gets distinct color

### Scenario 2: Inner Split Only
```
Color By: (none)
Split By: BLK (Block)
```
- Color map discovers: [Block1, Block2, Block3] (from split dimension)
- Rendering uses: split dimension
- Result: ✅ Each block gets distinct color

### Scenario 3: Both Split and Color (FIXED!)
```
Color By: DUT
Split By: BLK (Block)
```
- **Before Fix**: Color map discovers DUT values, but rendering uses BLK values → mismatch
- **After Fix**: Color map discovers BLK values (split takes precedence) → ✅ matches rendering

### Scenario 4: Outer Split + Inner Split (FIXED!)
```
Split-Chart By: WAFER
Split By: BLK
```
- Color map discovers: [Block1, Block2, Block3]
- Each tile (Wafer1, Wafer2) shows blocks with same colors
- Result: ✅ Consistent colors across all tiles

## Technical Details

### How `_buildTileChart()` Uses Group Keys

The rendering code (lines 9827, 10044, 10237):
```javascript
var g = innerSplitCol ? innerSplitKey(pt) : colorKey(pt);
```

This means:
- **Priority 1**: If inner split column is set → use `innerSplitKey()` 
- **Priority 2**: Otherwise → use `colorKey()` (color dimension)

The fix ensures the color map is built with the same priority logic!

### Color Map Building Logic

1. Read all control values: `colCol`, `colorRx`, `splitCol`, `splitRx`
2. For each point, determine what key will be used:
   - If `splitCol` exists → extract from split dimension
   - Otherwise → extract from color dimension
3. Build color map with discovered keys (sorted alphabetically)
4. All rendering uses this same map with same priority logic

## Verification

To verify the fix works:

1. **Scenario: Split by Color with Multiple Values**
   - Set up split chart
   - Use inner split (split-by) dimension
   - Render chart
   - **Expected**: Each inner split value gets a distinct color (not all same color)

2. **Scenario: Multiple Color Values with Inner Split**
   - Set color-by dimension
   - Also set split-by dimension  
   - Render chart
   - **Expected**: Inner split values get colors (split takes precedence), not color values

3. **Consistency Across Tiles**
   - Create split chart (split-chart by outer dimension)
   - Set inner split dimension
   - View multiple tiles
   - **Expected**: Same inner split value has same color across all tiles

## Files Modified

**Single File**: `fdv_chart_rev15/fdv_chart.html`

**Locations**:
- Lines ~11000-11050: Early split dimension discovery
- Lines ~11017-11047: Conditional color/split key discovery
- Lines ~11083-11088: Removed duplicate split dimension read

## Impact Assessment

**✅ Positive**:
- Split charts with inner split now show proper distinct colors
- Color consistency across all split chart tiles
- Precedence is clear: split dimension > color dimension for grouping

**⚠️ Considerations**:
- Only affects charts with inner split dimension set
- Normal color-only charts unaffected
- Performance: Minimal (early reads same data, slightly earlier in function)

## Status

✅ **Audited** - Identified root cause of color mismatch  
✅ **Fixed** - Early split dimension discovery implemented  
✅ **Validated** - No syntax errors  
✅ **Ready** - Server restarted with fix applied  

## Related Fixes

This fix complements the earlier split chart color consistency fixes:
1. ✅ Pre-populate global color map before tile rendering
2. ✅ Use global map in all tile types (boxplot, histogram, cumproba, cum_sigma)
3. ✅ Use global map for scatter overlay points
4. ✅ **NEW**: Discover BOTH color AND split dimensions for accurate key matching
