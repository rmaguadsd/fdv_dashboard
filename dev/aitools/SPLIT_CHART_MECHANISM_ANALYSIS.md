# Split-Chart vs Color-By Mechanism & Feel Analysis

## Current State

### Color-By Structure
**Location in HTML:** Lines 671-675
```html
<span id="color-col-label" style="margin-left:8px;display:inline-flex;align-items:flex-start;gap:4px">
    <span style="font-size:0.82em;color:#495057;white-space:nowrap;padding-top:3px">Color&nbsp;by:</span>
    <span id="color-dims-wrap" class="gdim-wrap"><!-- rows injected by _gdimInit --></span>
    <button class="gdim-add" onclick="_gdimAdd('color')" title="Add another color-by dimension">+</button>
</span>
```

**Mechanism:**
- Uses a "dims-wrap" container (`color-dims-wrap`)
- Initially shows ONE row via `_gdimInit()` → calls `_gdimAdd('color')`
- Each row = a "dimension" with:
  - `<select>` dropdown for column selection
  - `<input>` text field for regex pattern
  - `<button>` to delete that dimension
- **Multiple dimensions supported**: User clicks `+` button to add more
- **Key function**: `_compoundKey(row, dims)` joins multiple dimensions with ` │ ` separator
- **Arrays**: `_colorDims` holds the active color dimensions at runtime
- **CSS Class**: `.gdim-row` for each dimension row, `.gdim-wrap` for container

### Split-Chart Current Structure
**Location in HTML:** Lines 676-696
```html
<span style="margin-left:8px;display:inline-flex;align-items:center;gap:5px;border-left:1px dashed #ced4da;padding-left:8px">
    <span style="font-size:0.82em;color:#17a2b8;font-weight:bold;white-space:nowrap">&#9707;&nbsp;Split-chart:</span>
    <label style="font-size:0.82em;color:#17a2b8">By:
        <select id="split-chart-col" onchange="_onSplitColChange()" style="border-color:#17a2b8">
            <option value="">-- none --</option>
        </select></label>
    <label style="font-size:0.82em;color:#17a2b8">Regex:
        <input type="text" id="split-chart-rx" placeholder="e.g. DUT(\d+)" autocomplete="off"
               style="width:100px;border-color:#17a2b8" onchange="_onSplitColChange()"
               title="..."></label>
    <span id="sc-dims-wrap" class="gdim-wrap" style="display:inline-flex;flex-direction:row;flex-wrap:wrap;gap:3px;align-items:center"></span>
    <button class="gdim-add" onclick="_gdimAdd('sc')" title="Add an extra split-chart dimension (combined with the By column above)">+</button>
    <!-- Layout controls: Cols, Tile H, Grid H -->
</span>
```

**Current Issues:**
1. **Two separate sections**: 
   - Main "By" and "Regex" are hardcoded `<select>` and `<input>`
   - Extra dimensions go in `sc-dims-wrap` (split-chart dimensions)
2. **Inconsistent feel**: 
   - Color-by: Everything is a "dimension" row from the start
   - Split-chart: "By" column is special, extras are bolt-on
3. **No unified mechanism**: 
   - Color-by uses `_gdimRead('color')` to read all dimensions
   - Split-chart has primary (`split-chart-col`, `split-chart-rx`) + extras (`_scDims`)
   - Must check BOTH in plotting code

## Key Differences

| Aspect | Color-By | Split-Chart |
|--------|----------|-------------|
| **Initial state** | One dimension row in dims-wrap | One hardcoded select + input + extra dims-wrap |
| **Adding more** | Click + → adds another gdim-row | Click + → adds to sc-dims-wrap but "By" stays separate |
| **Reading values** | `_gdimRead('color')` gets ALL dims | `split-chart-col` + `split-chart-rx` + `_scDims` (must combine) |
| **Visual consistency** | All rows same layout | By/Regex different from extra dims |
| **Compound key** | `_compoundKey(row, _colorDims)` | Must concatenate split-chart-col + _scDims manually |
| **Feel** | Symmetrical, extensible | Asymmetrical, primary + extras |

## How to Make Split-Chart Like Color-By

### Option 1: Full Unification (Most Elegant)

**Goal**: Make split-chart use the same `gdim-row` mechanism, with all dimensions treated equally.

**Steps**:

1. **Replace hardcoded split-chart By/Regex with dims-wrap approach**
   - Remove lines with hardcoded `split-chart-col` select and `split-chart-rx` input
   - Create a single `sc-dims-wrap` that serves as the ONLY place for split-chart dimensions
   - Initialize with `_gdimAdd('sc')` on startup (like color-by does)
   - All extra dims added via `+` button will be siblings to the first one

2. **Update the layout flow**
   ```
   OLD:
   Split-chart: [By select] [Regex input] [+ button] [extra dims in sc-dims-wrap] [Cols/H controls]
   
   NEW:
   Split-chart: [dims rows in sc-dims-wrap (each with select + regex + delete)] [+ button] [Cols/H controls]
   ```

3. **Change the plotting code**
   - Replace all `split-chart-col` + `split-chart-rx` references with `_gdimRead('sc')`
   - Use `_compoundKey()` if multiple dimensions are set, or handle single dimension case
   - Remove checks like `if (scCol) ...` and replace with `if (_scDims.length && _scDims[0].colIdx >= 0)`

4. **Update the recipe/session system**
   - Remove hardcoded `split-chart-col` and `split-chart-rx` from recipe IDs
   - Let `_gdimSetDims('sc', dims)` handle restoration of split-chart dimensions from saved sessions

### Option 2: Hybrid Approach (Less Change, Better UX)

**Goal**: Keep the "By" and "Regex" fields for quick/simple use, but make extras work like color-by, and make the whole section VISUALLY consistent.

**Steps**:

1. **Restructure HTML layout**:
   - Move hardcoded `split-chart-col` and `split-chart-rx` INSIDE the `sc-dims-wrap`
   - Wrap them in a `gdim-row` div just like color-by rows
   - Make the first row special (don't show delete button) OR show delete but make it optional

2. **Update JavaScript**:
   - When reading split-chart config: first check if `split-chart-col/rx` have values, then read `_scDims`
   - When drawing: combine primary "By" column + any extra dimensions into a single compound key like color-by does
   - Maintain backward compatibility with existing recipes

3. **Visual improvements**:
   - Use `.gdim-row` CSS class for consistency
   - Show layout controls (Cols, H, Grid H) BELOW the dimensions, not inline
   - Make the section feel more "expandable" like color-by

### Option 3: Complete Refactor (Breaking Change)

**Goal**: Identical mechanism to color-by — every split-chart dimension is a `gdim-row`.

**Steps**:

1. Remove ALL hardcoded `split-chart-col` and `split-chart-rx` from HTML
2. Replace them with just:
   ```html
   <span id="sc-dims-wrap" class="gdim-wrap"></span>
   <button class="gdim-add" onclick="_gdimAdd('sc')" title="...">+</button>
   ```
3. Update `_gdimInit()` to call `_gdimAdd('sc')` once to create first split-chart dimension
4. Change plotting code to ONLY use `_gdimRead('sc')`
5. Update recipe system to save/restore `_scDims` the same way as `_colorDims`
6. **Breaking**: Old recipes with `split-chart-col` and `split-chart-rx` will lose those values

## CSS Classes Already in Use

```css
.gdim-wrap {
    display: flex;
    flex-direction: column;  /* for color-by: vertical stacking */
    gap: 2px;
}

.gdim-row {
    display: flex;
    gap: 4px;
    align-items: center;
}

.gdim-add {
    /* existing styles for + button */
}

.gdim-del {
    /* existing styles for delete button */
}

.gdim-rx {
    /* class for regex input fields */
}
```

## Current JavaScript Functions & Arrays

| Function | Purpose |
|----------|---------|
| `_gdimInit()` | Initialize all dimension builders (color, split, sc) |
| `_gdimAdd(type)` | Add one dimension row to a builder |
| `_gdimDel(btn, type)` | Remove a dimension row |
| `_gdimRead(type)` | Read all active dimensions from the wrap into array |
| `_gdimChanged(type)` | Sync arrays + rebuild |
| `_gdimSetDims(type, dims)` | Restore from saved dimensions array |
| `_gdimRebuildSelects(headers)` | Refresh column options when headers change |
| `_compoundKey(row, dims)` | Create compound key from multiple dimensions |

| Array | Holds | Size |
|-------|-------|------|
| `_colorDims` | Active color-by dimensions | Usually 0-2 |
| `_splitDims` | Active split-by (cumproba) dimensions | Usually 0-1 |
| `_scDims` | Extra split-chart dimensions | Usually 0-1 |

## Recommendation

**Option 1 (Full Unification)** is best because:
- ✅ Consistent mechanism across both features
- ✅ Users understand "dimensions" work the same way everywhere
- ✅ Code duplication removed
- ✅ Future maintenance easier
- ✅ Compound multi-dimensional split-chart is naturally supported

**Trade-off**: Must update plotting code in ~6-8 locations to use `_gdimRead('sc')` instead of direct element access.

**Option 2 (Hybrid)** if you want:
- ✅ Keep simple use case (just select By column)
- ✅ Backward compatible with existing recipes
- ✅ Visual consistency improvements without huge refactor

---

## Implementation Checklist for Option 1

- [ ] Remove hardcoded `split-chart-col` select from HTML
- [ ] Remove hardcoded `split-chart-rx` input from HTML  
- [ ] Remove `id="split-chart-col"` and `id="split-chart-rx"` from hidden legacy elements
- [ ] Ensure `sc-dims-wrap` is the ONLY place split-chart dimensions are built
- [ ] Update `_gdimInit()` to call `_gdimAdd('sc')` for initial row
- [ ] Update line ~3138: Use `_gdimRead('sc')` instead of direct element access
- [ ] Update line ~4240: Use `_gdimRead('sc')` instead of direct element access
- [ ] Update line ~4440: Use `_gdimRead('sc')` instead of direct element access
- [ ] Update line ~4886: Use `_gdimRead('sc')` instead of direct element access
- [ ] Update recipe system: remove `split-chart-col` and `split-chart-rx` from `_recipeIds()`
- [ ] Test: Add multiple split-chart dimensions
- [ ] Test: Save/restore session with split-chart dimensions
- [ ] Test: All chart types work correctly with split-chart

