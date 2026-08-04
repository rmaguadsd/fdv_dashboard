# Regex and Formula Processing in Rev14 - X-Regex and Y-Regex

## Quick Answer
**No, the regex formula (with `|>` syntax) does NOT currently apply to x-regex and y-regex.** 

The x-regex and y-regex fields only use **simple regex extraction**, not the advanced formula syntax.

## Detailed Breakdown

### What EXISTS: Simple Regex Extraction

**X-Regex and Y-Regex:** Use the `extractNum()` function (line 5370)

```javascript
function extractNum(val, rxStr) {
    if (rxStr) {
        try {
            var m = new RegExp(rxStr).exec(String(val));
            if (m) val = (m[1] !== undefined ? m[1] : m[0]);  // Extract capture group or full match
        } catch(e) {}
    }
    var n = parseFloat(String(val).replace(/[^0-9.eE+\-]/g,''));  // Remove non-numeric chars
    return isNaN(n) ? null : n;
}
```

**Flow for x-regex:**
1. Input: Raw cell value from CSV
2. Apply regex pattern to extract substring (capture group or full match)
3. Remove all non-numeric characters
4. Parse as float
5. Return numeric value

**Example:**
```
Raw value: "2.5mA"
X-regex: "([0-9.]+)"
After regex: "2.5"
After cleanup: "2.5"
Result: 2.5 (numeric)
```

### What EXISTS: Advanced Formula (Color-By and Group-By ONLY)

**Color-Regex and Group-Regex:** Use the `_extractGroupKey()` function (line 9032)

The formula syntax with `|>` **ONLY** works for:
- **color-regex** (color dimension grouping)
- **group-regex** (group dimension grouping) 
- **x-hier-regex** (hierarchical axis grouping)

**NOT for:**
- ❌ x-regex
- ❌ y-regex

### The Formula System: `regex |> formula`

**Available for color-by, group-by:**

```
Pattern: REGEX |> FORMULA

Example: (\d+) |> x * 2 + 10

Breakdown:
  1. REGEX: (\d+) - Extract digits
  2. |> - Pipe operator
  3. FORMULA: x * 2 + 10 - Transform extracted value
  
Result: If raw="5", extract 5, compute 5*2+10 = 20
```

**Formula Variables:**
- `x` - The extracted value (or full string if no capture group)
- `g1`, `g2`, `g3` - Individual capture groups from regex
- Math functions: `Math.log()`, `Math.sqrt()`, etc.

**Code Location:** `_extractGroupKey()` at line 9032-9163
```javascript
function _extractGroupKey(raw, rxStr) {
    if (!rxStr) return raw || '(blank)';
    
    try {
        var parts = rxStr.split(/\s*\|>\s*/);  // Split on |>
        var regexPart = parts[0] ? parts[0].trim() : '';
        var formulaPart = parts[1] ? parts[1].trim() : '';  // Formula part
        
        // ... execute formula with eval() ...
        var result = eval('(function() { var x = ' + JSON.stringify(x) + '; return (' + formulaPart + '); })()');
        
        return String(result) || '(blank)';
    } catch(e) { ... }
}
```

## Why the Difference?

### X-Regex / Y-Regex: Purpose
- **Goal:** Extract numeric values from text
- **Use Case:** CSV cell contains "2.5mA", extract "2.5"
- **Need:** Only regex extraction, no math transformations
- **Type:** Grouping not needed - direct numeric conversion

### Color-Regex / Group-Regex: Purpose
- **Goal:** Transform text into categorical grouping keys
- **Use Case:** "Test_01" → "Test_1", "Batch_002" → "Batch_2"
- **Need:** Both extraction AND transformation
- **Type:** Creates categorical groups - needs flexible transformations

## Current Architecture

### Data Flow with X-Regex / Y-Regex (Numeric)
```
CSV Row Value
    ↓
readFilteredFromMemory()
    ↓
extractNum(xRaw, xRx)  ← Simple regex extraction
    ↓
Numeric value: 2.5
    ↓
plotted as: (2.5, yValue) on chart
```

### Data Flow with Color-Regex (Categorical)
```
CSV Row Value
    ↓
drawScatterLine()
    ↓
colorKey(pt) → _extractGroupKey(raw, colorRx)  ← Advanced formula
    ↓
String result: "Test_1" (after formula transformation)
    ↓
Used as legend/color group: "Test_1"
```

## Feature Request: Would You Like Formula Support for X-Regex / Y-Regex?

If you want to enable `|>` formula syntax for x-regex and y-regex, here's what would need to change:

### Option 1: Extend `extractNum()` to support formulas

```javascript
function extractNum(val, rxStr) {
    var extractedVal = val;
    
    if (rxStr) {
        var parts = rxStr.split(/\s*\|>\s*/);
        var regexPart = parts[0] ? parts[0].trim() : '';
        var formulaPart = parts[1] ? parts[1].trim() : '';
        
        // Apply regex extraction
        if (regexPart) {
            try {
                var m = new RegExp(regexPart).exec(String(val));
                if (m) extractedVal = (m[1] !== undefined ? m[1] : m[0]);
            } catch(e) {}
        }
        
        // Apply formula transformation (NEW)
        if (formulaPart) {
            try {
                var x = parseFloat(extractedVal);
                if (!isNaN(x)) {
                    extractedVal = eval('(function() { var x = ' + x + '; return (' + formulaPart + '); })()');
                }
            } catch(e) {}
        }
    }
    
    var n = parseFloat(String(extractedVal).replace(/[^0-9.eE+\-]/g,''));
    return isNaN(n) ? null : n;
}
```

### Use Cases for Formula in X/Y-Regex:

```
Raw X Value: "2.5mA"
X-Regex: ([0-9.]+) |> x * 1000
Result: 2500 (convert mA to µA)

Raw Y Value: "log(100)"
Y-Regex: \(([0-9]+)\) |> Math.log10(x)
Result: 2 (convert explicit log notation to numeric)

Raw Value: "2.5dB"
Regex: ([0-9.]+) |> x / 10
Result: 0.25 (normalize dB scale)
```

## Summary Table

| Feature | X-Regex | Y-Regex | Color-Regex | Group-Regex |
|---------|---------|---------|-------------|-------------|
| Simple Regex Extraction | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Formula `\|>` Syntax | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Purpose | Numeric extract | Numeric extract | Categorical group | Categorical group |
| Current Function | `extractNum()` | `extractNum()` | `_extractGroupKey()` | `_extractGroupKey()` |

## Files

- **Server:** `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev14\fdv_chart.py`
- **Client:** `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev14\fdv_chart.html`
  - `extractNum()`: Line 5370
  - `_extractGroupKey()`: Line 9032 (with formula support)
  - `readFilteredFromMemory()`: Line 5015 (uses extractNum)

## Recommendation

If you need formula support for x/y-regex, I can implement it. Would you like me to:
1. Enable `|>` formula syntax for x-regex and y-regex?
2. Update the documentation?
3. Test with example use cases?
