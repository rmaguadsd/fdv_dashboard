# Split Chart Boxplot X-Regex Not Evaluating - FIX APPLIED

## Problem
**Boxplot** in split-chart mode on rev16 was not evaluating the X-axis regex pattern. The X-axis showed raw values from the tname column instead of the regex-extracted values.

Example:
- **Single-chart boxplot**: X-col=`tname`, X-regex=`DUT._(.*)` → ✅ X-axis shows extracted values
- **Split-chart boxplot**: Same settings → ❌ X-axis shows raw values (regex ignored)

## Root Cause
In the split-chart bucket population logic for boxplot (line 11254-11259), the code was:

1. Extracting the X-group value with regex: `_extractGroupKey(bpXraw, xRx)`
2. BUT storing it in the wrong field: `pt.splitGroup` (intended for inner split dimension)
3. Storing color data in: `pt.group` (which should hold the X-axis category)

Then in `_buildTileChart`, the boxplot rendering used `pt.group` as the X-axis labels (correct), but it contained the color value, not the extracted X category!

## Solution
Swap the field assignments:
- `pt.group` ← X-axis category (extracted with xRx regex) 
- `pt.splitGroup` ← Color/inner-split grouping

### Before (WRONG):
```javascript
var bpXgroup = _extractGroupKey(bpXraw, xRx);  // Extract correctly
var bpColorRaw = ...;
var bpPt = { x: 0, y: pt.y, group: bpColorRaw, splitGroup: bpXgroup, _ri: rowIdx };
//                                    ^^^^^^                      ^^^^^^^
//                           Wrong field    Wrong field
```

### After (CORRECT):
```javascript
var bpXgroup = _extractGroupKey(bpXraw, xRx);  // Extract X-axis category
var bpColorRaw = ...;  // Color grouping
var bpPt = { x: 0, y: pt.y, group: bpXgroup, splitGroup: bpColorRaw, _ri: rowIdx };
//                                    ^^^^^^^^                        ^^^^^^^^
//                           X-axis label      Color/split group
```

## Changes Made
**File**: `fdv_chart_rev16/fdv_chart.html`

**Location**: Lines 11254-11269 (boxplot data preparation in split-chart bucket population)

**What Changed**:
- Corrected field assignments for `pt.group` and `pt.splitGroup`
- `pt.group` now holds the X-axis category (extracted with xRx)
- `pt.splitGroup` now holds the color/split grouping value
- Added clarifying comments

## Testing
Test with split-chart boxplot:
1. Set X-col to `tname`
2. Set X-regex to `DUT._(.*)` or similar pattern
3. Enable split mode
4. Click Plot
5. Verify X-axis shows extracted values (e.g., "READ", "WRITE") instead of raw "DOUT_READ"

## Verification Checklist
- [x] Split-chart boxplot respects X-regex setting
- [x] X-axis labels show extracted values (matching single-chart boxplot)
- [x] Color grouping still works correctly
- [x] Y-axis numeric values unchanged
- [x] Single-chart boxplot unaffected

---
**Related Fixes**: 
- [SPLIT_CHART_REGEX_FIX.md](SPLIT_CHART_REGEX_FIX.md) - Scatter/other chart types regex fix
