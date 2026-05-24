# REV12 Formula Feature: Quick Reference Card

**Separator:** `=>` (space-equals-greater-than)

---

## Three Syntax Modes

```
MODE 1: Regex only (backward compatible)
(\d+)

MODE 2: Formula only (new)
=> x > 100 ? 'HIGH' : 'LOW'

MODE 3: Regex + Formula (new)
(\d+) => parseInt(x) > 100 ? 'HIGH' : 'LOW'
```

---

## Variables in Formulas

```
x       Extracted value (after regex, or raw if no regex)
g1      First capture group
g2      Second capture group
g3      Third, etc...
```

---

## Operators & Functions

### Comparison
```
>  <  ==  !=  >=  <=
```

### Logical
```
&&  ||  !
```

### Ternary
```
condition ? true_value : false_value
```

### Arithmetic
```
+  -  *  /  %  **
```

### String Methods
```
.toUpperCase()      .toLowerCase()      .substring(start, end)
.replace(a, b)      .trim()             .slice(start, end)
.charAt(index)      .split(sep)         .includes(str)
.startsWith(str)    .endsWith(str)      .length
.replaceAll(a, b)   .padStart(len, ch)  .padEnd(len, ch)
```

### Math Functions
```
Math.floor(x)       Math.ceil(x)        Math.round(x)
Math.abs(x)         Math.sqrt(x)        Math.cbrt(x)
Math.pow(x, y)      Math.log(x)         Math.log10(x)
Math.log2(x)        Math.min(a, b)      Math.max(a, b)
```

### Type Conversion
```
parseInt(x)         parseFloat(x)       Number(x)
String(x)           isNaN(x)            isFinite(x)
```

---

## Usage Examples

### Example 1: Simple Bucketing
```
Input:   "250"
Formula: => parseInt(x) > 100 ? 'HIGH' : 'LOW'
Output:  "HIGH"
```

### Example 2: Regex + Bucketing
```
Input:   "DUT250"
Regex:   (\d+)
Formula: => parseInt(x) > 100 ? 'HIGH' : 'LOW'
Output:  "HIGH"
```

### Example 3: Multi-Group Join
```
Input:   "AB_123"
Regex:   ^(..)_(\d+)$
Formula: => g1 + '-' + g2
Output:  "AB-123"
```

### Example 4: String Transformation
```
Input:   "hello"
Formula: => x.toUpperCase()
Output:  "HELLO"
```

### Example 5: Complex Math
```
Input:   "47"
Formula: => Math.round(parseInt(x) / 10) * 10
Output:  "50"
```

---

## Error Indicators

| Indicator | Meaning |
|-----------|---------|
| `(blank)` | Input was empty/null |
| `(no match)` | Regex didn't match |
| `(formula error)` | Formula syntax/runtime error |
| `(undefined)` | Formula returned undefined |

---

## Where to Use

✅ **Color dimensions** — Group/color by transformed values  
✅ **Split-chart dimensions** — Organize sub-charts  
✅ **Split dimensions** — Partition main chart  

---

## How to Test

1. Add dimension → Select column
2. Enter formula in field
3. Press Enter or click elsewhere
4. Watch chart update
5. Check console (F12 → Console) for errors

---

## Common Formulas

```javascript
// Bucketing by value
x > 100 ? 'LARGE' : 'SMALL'

// Case conversion
x.toUpperCase()

// Rounding
Math.round(parseInt(x) / 10) * 10

// Multi-level bucketing
parseInt(x) < 50 ? 'LOW' : parseInt(x) < 100 ? 'MID' : 'HIGH'

// String operations
x.substring(0, 3).toUpperCase()

// Multi-group join
g1 + '-' + g2

// Type checking
isNaN(x) ? '(invalid)' : parseInt(x)

// Null coalescing
g1 || 'DEFAULT'

// Complex expression
(parseInt(g1) * 100 + parseInt(g2)) / 10
```

---

## Debugging

**If formula not working:**
1. Open console: F12 → Console tab
2. Look for `[Formula Error]` message
3. Fix the syntax error
4. Try simpler version first

**Example Error:**
```
[Formula Error] x > 100 ? 'HIGH' : 'LOW' | Error: SyntaxError: Unexpected token
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `REV12_QUICK_TEST_GUIDE.md` | 10 quick tests |
| `REV12_FORMULA_IMPLEMENTATION.md` | Complete reference |
| `REV12_IMPLEMENTATION_INDEX.md` | Navigation |
| `REV12_EXACT_CODE_CHANGES.md` | Technical details |

---

## Features

✅ Regex extraction (backward compatible)  
✅ Formula transformation (new)  
✅ Variable binding (x, g1, g2...)  
✅ Error handling with console logging  
✅ Safe operations (limited scope)  
✅ Fast execution  

---

## Not Allowed

❌ Array methods (map, filter, reduce)  
❌ Object creation/access  
❌ Function definitions  
❌ Variable assignments  
❌ DOM access  

---

## File Size

- Code: 379,688 bytes (+2,157 bytes from base)
- Backward Compatible: 100% ✅

---

**Separator:** `=>`  
**Status:** Ready for Testing ✅  
**Next:** Read `REV12_QUICK_TEST_GUIDE.md`
