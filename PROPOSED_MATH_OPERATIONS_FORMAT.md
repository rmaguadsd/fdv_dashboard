# Proposed Format for Math/Logical Operations on Regex Split-Chart & Color-By

## Overview

Currently, the system supports:
- **Regex extraction** on split-chart and color-by dimensions
- **Axis intervals** (y-min, y-max, x-min, x-max) for numeric filtering

This proposal adds **mathematical transformations** and **logical grouping** to regex splits, enabling calculated columns, transformations, and conditional grouping without modifying the source data.

---

## Current Architecture

### Color-By Dimensions Structure
```javascript
_colorDims = [
    {
        col: "ColumnName",    // Column to group by
        colIdx: 3,            // Index in currentHeaders
        rx: "regex(pattern)"  // Optional regex to extract key
    }
]
```

### Split-Chart Dimensions Structure
```javascript
_scDims = [
    {
        col: "ColumnName",    // Column for split grouping
        colIdx: 2,            // Index in currentHeaders
        rx: "regex(pattern)"  // Optional regex to extract key
    }
]
```

### Usage
- Extract group keys: `_extractGroupKey(rawValue, rxStr)`
- Build compound keys: `_compoundKey(row, dims)`

---

## Proposed Extension: Math Operations Format

### 1. **Formula Syntax (Optional `formula` field)**

Add an optional `formula` field to dimension objects:

```javascript
_colorDims = [
    {
        col: "Value",           // Source column
        colIdx: 4,
        rx: "(\\d+)",           // Extract numeric part
        formula: "x > 100 ? 'HIGH' : 'LOW'"  // NEW: Transform extracted value
    }
]
```

### 2. **Simple Cases (Without Regex)**

**Numeric bucket grouping:**
```javascript
{
    col: "Temperature",
    formula: "x < 0 ? 'Frozen' : x < 10 ? 'Cold' : x < 20 ? 'Cool' : 'Warm'"
}
```

**String transformation:**
```javascript
{
    col: "Product",
    formula: "x.substring(0, 3).toUpperCase()"  // Group by first 3 chars, uppercase
}
```

**Math transformation:**
```javascript
{
    col: "Diameter",
    formula: "Math.round(x / 10) * 10"  // Round to nearest 10
}
```

### 3. **Complex Cases (With Regex + Formula)**

**Extract number, then bucket:**
```javascript
{
    col: "Model",
    rx: "\\d+",                                  // Extract: "MODEL123" → "123"
    formula: "parseInt(x) >= 100 ? 'Series100+' : 'Series<100'"
}
```

**Extract and transform:**
```javascript
{
    col: "Filename",
    rx: "test_(\\w+)_\\d+",                      // Extract: "test_ABC_001" → "ABC"
    formula: "x.toLowerCase() + '_group'"
}
```

### 4. **Multi-Dimensional Formulas (With Compound Keys)**

For split-charts with **multiple dimensions**, formulas can reference all source values:

```javascript
_scDims = [
    {
        col: "Temperature",
        formula: "x < 10 ? 'Cold' : 'Warm'"
    },
    {
        col: "Pressure",
        formula: "x > 50 ? 'High' : 'Low'"
    }
]
// Result: "Cold, High" or "Warm, Low" (compound key)
```

---

## Syntax Details

### Variable Naming
- `x` = extracted value (after regex, if provided; otherwise raw value)
- `row` = entire row object (for future cross-column operations, if needed)
- `_` = Lodash/utility functions available? (to be determined)

### Formula Types

| Type | Example | Input | Output |
|------|---------|-------|--------|
| Ternary | `x > 100 ? 'HIGH' : 'LOW'` | `150` | `'HIGH'` |
| String method | `x.toUpperCase()` | `'abc'` | `'ABC'` |
| Math function | `Math.round(x / 10) * 10` | `157` | `160` |
| Logical AND/OR | `x > 10 && x < 50 ? 'MID' : 'OTHER'` | `25` | `'MID'` |
| Substring | `x.substring(0, 2)` | `'APPLE'` | `'AP'` |
| Null coalesce | `x || '(unknown)'` | `null` | `'(unknown)'` |

### Escaping
- Use single quotes inside string literals (avoid HTML quote conflicts)
- Use `\"` for double-quote literals if needed
- Backslashes in regex already require `\\` (no change)

---

## Implementation Locations

### 1. **UI Changes**

#### Color-By Dimension Row (Existing Pattern)
```html
<div class="gdim-row">
    <select id="gdim-color-col-ABC123"><!-- columns --></select>
    <input type="text" class="gdim-rx" placeholder="regex…">
    <!-- NEW: Formula input -->
    <input type="text" class="gdim-formula" placeholder="formula (optional)…" 
           title="JavaScript expression: 'x > 100 ? HIGH : LOW' or 'x.toUpperCase()'">
    <button class="gdim-del" onclick="_gdimDel(this,'color')">×</button>
</div>
```

#### Layout
- Optional: Add collapsible section "Advanced" after regex field
- Or: Single-row inline layout with 3 input fields (column, regex, formula)

### 2. **Data Storage**

Extend dimension object structure:
```javascript
{
    col: "ColumnName",
    colIdx: 3,
    rx: "regex",           // Existing
    formula: "x > 100 ? 'HIGH' : 'LOW'"  // NEW
}
```

Session save/restore:
```javascript
// Saved in `__colorDims` as JSON string
snap['__colorDims'] = JSON.stringify([
    { col: 'Value', rx: '(\\d+)', formula: 'x > 100 ? "HIGH" : "LOW"' }
]);
```

### 3. **Extraction Logic**

Current:
```javascript
function _extractGroupKey(raw, rxStr) {
    if (!rxStr) return raw || '(blank)';
    var m = new RegExp(rxStr).exec(raw);
    if (!m) return '(no match)';
    return groups.length > 0 ? groups.join(', ') : m[0];
}
```

Enhanced:
```javascript
function _extractGroupKey(raw, rxStr, formula) {
    var value = raw;
    
    // Step 1: Apply regex if provided
    if (rxStr) {
        var m = new RegExp(rxStr).exec(raw);
        if (!m) return '(no match)';
        var groups = [];
        for (var g = 1; g < m.length; g++) {
            if (m[g] !== undefined) groups.push(m[g]);
        }
        value = groups.length > 0 ? groups.join(', ') : m[0];
    }
    
    // Step 2: Apply formula if provided
    if (formula) {
        try {
            var x = value;  // Formula variable
            var result = eval('(' + formula + ')');
            return String(result);
        } catch(e) {
            console.warn('[formula] Error evaluating:', formula, 'on value:', value, e.message);
            return value;  // Fallback to unformatted value
        }
    }
    
    return value;
}
```

### 4. **Compound Key Builder**

Current:
```javascript
function _compoundKey(row, dims) {
    return dims.map(function(d) {
        var raw = (d.colIdx >= 0 && row[d.colIdx] != null) 
            ? String(row[d.colIdx]) : '';
        return _extractGroupKey(raw, d.rx);
    }).join(', ');
}
```

Enhanced:
```javascript
function _compoundKey(row, dims) {
    return dims.map(function(d) {
        var raw = (d.colIdx >= 0 && row[d.colIdx] != null) 
            ? String(row[d.colIdx]) : '';
        return _extractGroupKey(raw, d.rx, d.formula);  // Pass formula
    }).join(', ');
}
```

---

## Example Use Cases

### 1. Temperature Bucketingfor Boxplot Split-Chart

**Scenario**: Split boxplots by temperature range without creating new columns

**Configuration**:
```
Split-Chart Column: Temperature
Regex: (empty)
Formula: x < 0 ? 'Frozen' : x < 10 ? 'Cold' : x < 20 ? 'Mild' : 'Warm'
```

**Result**: Each temperature value grouped into 4 categories, boxplot rendered per category

---

### 2. Product Code Shortening for Color-By

**Scenario**: Color points by first 3 characters of product code, ignore variant suffix

**Configuration**:
```
Color-By Column: ProductCode
Regex: ^([A-Z]+)\d+  ← Extract letters only
Formula: x.substring(0, 3).toUpperCase()
```

**Data**: `ABC123`, `ABCDEF456`, `ABCxyz` → Color groups: `ABC`, `ABC`, `ABC`

---

### 3. Log-Scale Binning for Histogram

**Scenario**: Histogram with logarithmic buckets (1, 10, 100, 1000)

**Configuration**:
```
Histogram X-Axis Column: Value
Formula: Math.floor(Math.log10(x)).toString()
```

**Data**: `50` → `log10=1.7` → `floor=1` → bucket `"1"` (10^1)

---

### 4. Quality Score Thresholding for Split-Chart

**Scenario**: Split charts into Pass/Fail based on quality threshold

**Configuration**:
```
Split-Chart Column: QualityScore
Regex: ^([0-9.]+)  ← Extract numeric score
Formula: parseFloat(x) > 0.95 ? 'PASS' : 'FAIL'
```

---

### 5. Multi-Dimensional Split (Cartesian Product)

**Scenario**: Split scatter into 4 quadrants by temperature AND pressure

**Configuration**:
```
Split-Chart Dimension 1:
  Column: Temperature
  Formula: x > 15 ? 'WARM' : 'COLD'

Split-Chart Dimension 2:
  Column: Pressure
  Formula: x > 50 ? 'HIGH' : 'LOW'
```

**Result**: Tiles labeled "COLD, LOW", "COLD, HIGH", "WARM, LOW", "WARM, HIGH"

---

## Validation & Error Handling

### Input Validation (UI Level)
1. **Syntax check**: Try to validate formula in browser before submission (optional nice-to-have)
2. **Common errors**: Warn if formula references undefined variable
3. **Feedback**: Show warning badge if formula has syntax error

### Runtime Handling
1. **Parse error**: Log to console, fallback to unformatted value
2. **Runtime error** (e.g., `NaN` result): Convert to string, use as group key
3. **Infinite loops**: Timeout/sandbox formula execution? (advanced future feature)

### Edge Cases
| Situation | Behavior |
|-----------|----------|
| Formula returns `null` | Stringify to `"null"` |
| Formula returns `undefined` | Use original extracted value |
| Formula throws error | Log error, use original value |
| Formula returns object | Call `.toString()` on result |

---

## Backward Compatibility

### Existing Sessions
- Old sessions have NO `formula` field → defaults to `undefined`
- When reading: `formula` is optional, safely ignored if not present
- No breaking changes to existing dimension objects

### Existing Recipes
- Legacy `color-col`, `color-regex`, `split-chart-col`, `split-chart-rx` inputs don't change
- Migration logic already in place: `_migrateSessionSnapshot()`
- No changes needed if user doesn't use formulas

---

## Optional Extensions (Future)

### 1. **Multiple Regex Capture Groups + Formula**
```javascript
{
    col: "Filename",
    rx: "(\\w+)_([0-9]+)_([A-Z]+)",  // Extract: (word, number, letters)
    formula: "g1 + '_' + g2"  // g1, g2, g3 = capture groups
}
```

### 2. **Cross-Column Operations**
```javascript
{
    col: "Value1",
    formula: "row[4] > row[5] ? 'A>B' : 'A<=B'"  // Compare two columns
}
```

### 3. **Chained Transformations**
```javascript
{
    col: "RawValue",
    transforms: [
        { rx: "(\\d+)" },          // Extract number
        { formula: "x * 0.453" },  // Pounds → kg
        { formula: "Math.round(x)" }  // Round
    ]
}
```

### 4. **Function Library**
```javascript
{
    col: "Value",
    formula: "bucket(x, [0, 10, 50, 100])"  // Predefined bucket function
}
```

---

## Proposed Implementation Priority

1. **Phase 1 (MVP)**: Basic formula support
   - Add `formula` field to dimension objects
   - Extend `_extractGroupKey()` to apply formula
   - Update UI with formula input field
   - Update session save/restore

2. **Phase 2 (Polish)**: Error handling & UX
   - Formula validation in browser
   - Better error messages
   - Optional formula help panel

3. **Phase 3 (Advanced)**: Cross-column & functions
   - Support `row` variable for multi-column references
   - Predefined function library
   - Chained transformations

---

## Questions for Feedback

1. **Variable scope**: Should formulas have access to full `row` object, or just extracted value `x`?
2. **Security**: Is `eval()` acceptable for formulas, or use sandboxed approach?
3. **UI complexity**: Single line formula input, or collapsible "Advanced" section?
4. **Error messages**: How verbose should formula errors be in console vs. UI?
5. **Naming**: Better name than "formula"? (transform, expression, operation, mapping?)

---

## Summary

This proposal enables **mathematical and logical transformations** on grouping dimensions without modifying source data. The implementation is:

- ✅ **Backward compatible** (optional field, no breaking changes)
- ✅ **Simple** (single `formula` field per dimension)
- ✅ **Extensible** (can add `row` access, chaining, function library later)
- ✅ **Familiar** (JavaScript syntax, matches `eval()` usage already in codebase)

The format works for:
- **Split-chart dimensions** (bucketingfor tile generation)
- **Color-by dimensions** (grouping point colors)
- **Split-by dimensions** (inner grouping within tiles)

Ready to proceed with Phase 1 implementation once feedback is received.
