# Complete List of Supported Formulas

This document lists all JavaScript formulas and operations supported when using the combined regex + formula field (with `|>` separator).

---

## Variables Available in Formulas

### Primary Variables
- **`x`** = extracted value (after regex extraction, or raw value if no regex)
- **`g1`, `g2`, `g3`, ...** = capture groups from regex (if capture groups present)
- **`row`** = entire row object (planned for future, not in MVP)

### Examples
```javascript
|> x + 10                    // x = "5" → result: 15
|> g1 + '_' + g2             // g1="ABC", g2="25" → result: "ABC_25"
```

---

## Supported Operations

### 1. Comparison Operators

| Operator | Syntax | Example | Result |
|----------|--------|---------|--------|
| Equal | `x == value` | `x == 100` | `true` or `false` |
| Not equal | `x != value` | `x != 100` | `true` or `false` |
| Greater than | `x > value` | `x > 100` | `true` or `false` |
| Less than | `x < value` | `x < 100` | `true` or `false` |
| Greater or equal | `x >= value` | `x >= 100` | `true` or `false` |
| Less or equal | `x <= value` | `x <= 100` | `true` or `false` |

### Usage Examples
```javascript
|> x > 100 ? 'HIGH' : 'LOW'
|> x == 'PASS' ? 'SUCCESS' : 'FAIL'
|> x >= 0.95 ? 'EXCELLENT' : 'GOOD'
```

---

### 2. Logical Operators

| Operator | Syntax | Example | Result |
|----------|--------|---------|--------|
| AND | `a && b` | `x > 10 && x < 50` | `true` or `false` |
| OR | `a \|\| b` | `x < 0 \|\| x > 100` | `true` or `false` |
| NOT | `!a` | `!isNaN(x)` | `true` or `false` |

### Usage Examples
```javascript
|> x > 10 && x < 50 ? 'MID' : 'EXTREME'
|> x < 0 || x > 100 ? 'OUT_OF_RANGE' : 'NORMAL'
|> !isNaN(x) ? 'NUMBER' : 'NOT_NUMBER'
```

---

### 3. Ternary Conditional (if-then-else)

| Pattern | Example | Usage |
|---------|---------|-------|
| Single condition | `condition ? 'yes' : 'no'` | `x > 100 ? 'HIGH' : 'LOW'` |
| Nested conditions | `cond1 ? val1 : cond2 ? val2 : val3` | `x < 0 ? 'NEG' : x == 0 ? 'ZERO' : 'POS'` |
| Multiple levels | Chain multiple `?:` | `x < -10 ? 'VERY_NEG' : x < 0 ? 'NEG' : x < 10 ? 'POS' : 'VERY_POS'` |

### Usage Examples
```javascript
|> x > 100 ? 'HIGH' : 'LOW'

|> x < 0 ? 'Frozen' : x < 10 ? 'Cold' : x < 20 ? 'Mild' : 'Warm'

|> x == 'PASS' ? 'SUCCESS' : x == 'FAIL' ? 'FAILURE' : 'UNKNOWN'
```

---

### 4. Arithmetic Operations

| Operation | Syntax | Example | Input | Result |
|-----------|--------|---------|-------|--------|
| Addition | `a + b` | `x + 10` | `5` | `15` |
| Subtraction | `a - b` | `x - 10` | `25` | `15` |
| Multiplication | `a * b` | `x * 2` | `5` | `10` |
| Division | `a / b` | `x / 2` | `10` | `5` |
| Modulo (remainder) | `a % b` | `x % 10` | `25` | `5` |
| Power (exponent) | `a ** b` | `x ** 2` | `5` | `25` |

### Usage Examples
```javascript
|> x * 0.453                    // Convert pounds to kg
|> (x + 32) * 5/9              // Complex formula
|> x % 10                       // Last digit
|> x ** 2                       // Square value
```

---

### 5. String Methods

#### Case Conversion
```javascript
x.toUpperCase()                    // "apple" → "APPLE"
x.toLowerCase()                    // "APPLE" → "apple"
```

#### Substring Operations
```javascript
x.substring(0, 3)                  // "APPLE" → "APP" (chars 0-2)
x.substring(2)                     // "APPLE" → "PLE" (from char 2 onward)
x.slice(0, 3)                      // "APPLE" → "APP" (alternative)
x.charAt(0)                        // "APPLE" → "A" (first character)
x[0]                               // "APPLE" → "A" (bracket notation)
```

#### Length
```javascript
x.length                           // "APPLE" → 5
```

#### Trim
```javascript
x.trim()                           // "  hello  " → "hello"
x.trimStart()                      // "  hello  " → "hello  "
x.trimEnd()                        // "  hello  " → "  hello"
```

#### Replace
```javascript
x.replace('A', 'X')               // "APPLE" → "XPPLE" (first only)
x.replaceAll('P', 'X')            // "APPLE" → "AXXLE" (all occurrences)
```

#### Split (returns array)
```javascript
x.split(',')[0]                    // "a,b,c" → "a" (get first part)
x.split('_').join('-')             // "a_b_c" → "a-b-c" (join with different separator)
```

#### Includes (search)
```javascript
x.includes('test') ? 'FOUND' : 'NOT_FOUND'  // Search for substring
x.startsWith('PRE') ? 'PREFIX' : 'OTHER'     // Check if starts with
x.endsWith('.log') ? 'LOGFILE' : 'OTHER'     // Check if ends with
```

### Usage Examples
```javascript
|> x.substring(0, 3).toUpperCase()         // First 3 chars, uppercase
|> x.includes('ERROR') ? 'ERROR' : 'OK'    // Check if contains text
|> x.replace('_', '-')                      // Replace character
|> x.toLowerCase() + '_group'               // Transform and append
```

---

### 6. Math Object Methods

#### Rounding
```javascript
Math.round(x)                      // 5.6 → 6 (nearest integer)
Math.floor(x)                      // 5.9 → 5 (round down)
Math.ceil(x)                       // 5.1 → 6 (round up)
Math.trunc(x)                      // 5.9 → 5 (truncate decimal)
```

#### Absolute Value
```javascript
Math.abs(x)                        // -15 → 15
```

#### Power & Root
```javascript
Math.pow(x, 2)                     // x² (same as x ** 2)
Math.sqrt(x)                       // √x
Math.cbrt(x)                       // ∛x (cube root)
```

#### Logarithm
```javascript
Math.log(x)                        // Natural logarithm (ln)
Math.log10(x)                      // Base-10 logarithm
Math.log2(x)                       // Base-2 logarithm
```

#### Min/Max
```javascript
Math.min(x, 100)                   // Minimum of x and 100
Math.max(x, 0)                     // Maximum of x and 0
```

#### Trigonometry (if needed)
```javascript
Math.sin(x), Math.cos(x), Math.tan(x)  // Trig functions
```

### Usage Examples
```javascript
|> Math.round(x / 10) * 10              // Round to nearest 10
|> Math.floor(Math.log10(x)).toString() // Log-scale bucketing (10^1, 10^2, etc)
|> Math.abs(x)                          // Convert negative to positive
|> x > 0 ? Math.log10(x) : 0            // Safe log (avoid log of negative)
```

---

### 7. Type Conversion

#### parseInt & parseFloat
```javascript
parseInt(x)                        // "123abc" → 123
parseInt(x, 10)                    // Force base-10
parseFloat(x)                      // "123.45abc" → 123.45
Number(x)                          // "123" → 123 (number type)
String(x)                          // 123 → "123" (always converted)
```

#### isNaN / isFinite
```javascript
isNaN(x) ? 'NaN' : 'Valid'        // Check if value is NaN
isFinite(x) ? 'Finite' : 'Inf'    // Check if value is finite
```

### Usage Examples
```javascript
|> parseInt(x) > 100 ? 'HIGH' : 'LOW'
|> parseFloat(x) * 2.5
|> isNaN(parseFloat(x)) ? 'INVALID' : 'VALID'
```

---

### 8. Null Coalescing & Default Values

| Pattern | Example | Behavior |
|---------|---------|----------|
| OR (fallback) | `x \|\| 'default'` | Use x if truthy, else 'default' |
| Nullish coalesce | `x ?? 'default'` | Use x if not null/undefined, else 'default' |

### Usage Examples
```javascript
|> x || '(unknown)'                  // Use x or "(unknown)" if falsy
|> g1 ?? 'NO_MATCH'                  // Use g1 or "NO_MATCH" if undefined
|> x || parseFloat(x) || 0           // Try x, then parse, then 0
```

---

### 9. Multi-Group Capture Combinations

When regex has multiple capture groups, combine them:

```javascript
^(\w+)_(\d+)$ |> g1 + '_' + g2                    // Concat groups
^([A-Z]+)_(\d+)$ |> g1 + '-' + (parseInt(g2)*2)  // Group 1 + calculated group 2
^(\w+)_(\w+)_(\w+)$ |> g1 + g2 + g3              // Combine 3 groups
^(.+)_([0-9.]+)$ |> g1 + ':' + parseFloat(g2)    // Extract and convert
```

---

### 10. Complex Expressions

Chain multiple operations:

```javascript
|> x.substring(0, 2).toUpperCase() + x.substring(2).toLowerCase()  // "aBcDe" → "ABcde"
|> parseFloat(x) > 50 ? (x * 1.1).toFixed(2) : x                   // Conditional math
|> x.replace(/[^0-9]/g, '') || '0'                                  // Extract digits, default to 0
|> g1 ? g1.toUpperCase() + '_' + g2 : 'UNKNOWN'                    // Conditional group join
```

---

## Not Supported (Intentionally Restricted)

For security and simplicity, these are **NOT** available in formulas:

❌ **Variables / Functions to Avoid**:
- `document`, `window`, `eval()`, `Function()`
- `fetch()`, `XMLHttpRequest`
- File I/O operations
- `require()`, `import`
- `process`, `child_process`

(Note: Current implementation uses `eval()`, so strict sandboxing would require future enhancement)

---

## Error Handling

When a formula fails:

| Scenario | Behavior | Example |
|----------|----------|---------|
| Syntax error | Logged to console, value unchanged | `x +` (incomplete) |
| Type error | Fallback to extracted value | `x.toUpperCase()` when x is number |
| Undefined variable | Use original value | `x + y` when y doesn't exist |
| Returns `null` | Converted to string "null" | `null` → `"null"` |
| Returns `undefined` | Uses original extracted value | Fallback mechanism |
| Divide by zero | Infinity or NaN (JavaScript default) | `x / 0` |

**Console Logging**: All errors logged to browser console with tag `[formula]`

---

## Complete Examples Reference

### Example 1: Temperature Range Bucketing
```javascript
|> x < 0 ? 'Frozen' : x < 10 ? 'Cold' : x < 20 ? 'Mild' : 'Warm'
```
Input: `-5`, `5`, `15`, `30`  
Output: `Frozen`, `Cold`, `Mild`, `Warm`

---

### Example 2: Extract Digits, Then Bucket
```javascript
(\d+) |> parseInt(x) > 100 ? 'HIGH' : parseInt(x) > 50 ? 'MID' : 'LOW'
```
Input: `"TEMP_150_F"` → regex extracts `"150"` → formula → `"HIGH"`

---

### Example 3: Multi-Group Extraction & Combination
```javascript
^([A-Z]+)_(\d+)_([a-z]+)$ |> g1 + '-' + parseInt(g2) * 2 + '-' + g3.toUpperCase()
```
Input: `"ABC_25_test"`  
g1="ABC", g2="25", g3="test"  
Output: `"ABC-50-TEST"`

---

### Example 4: String Normalization
```javascript
|> x.trim().toLowerCase().substring(0, 3).toUpperCase()
```
Input: `"  HELLO WORLD  "`  
Output: `"HEL"`

---

### Example 5: Conditional Type Conversion
```javascript
|> isNaN(parseFloat(x)) ? 'TEXT' : parseFloat(x) > 100 ? 'HIGH_NUM' : 'LOW_NUM'
```
Input: `"abc"` → `"TEXT"`  
Input: `"150"` → `"HIGH_NUM"`  
Input: `"50"` → `"LOW_NUM"`

---

### Example 6: Log Scale Bucketing (for histogram)
```javascript
|> x > 0 ? Math.floor(Math.log10(x)).toString() : 'NON_POSITIVE'
```
Input: `5` → log₁₀(5)=0.7 → floor=0 → `"0"` (bucket 10⁰)  
Input: `150` → log₁₀(150)=2.18 → floor=2 → `"2"` (bucket 10²)

---

### Example 7: Extract & Scale
```javascript
([0-9.]+) |> (parseFloat(x) * 0.453).toFixed(2)
```
Input: `"Weight: 10 lbs"` → extracts `"10"` → converts → `"4.53"` kg

---

### Example 8: Regex with Fallback
```javascript
^TEST_(\d+) |> g1 || 'NOTFOUND'
```
Input: `"TEST_123"` → g1="123" → `"123"`  
Input: `"PROD_123"` → no match → g1=undefined → `"NOTFOUND"`

---

### Example 9: Multi-Condition with Math
```javascript
|> x < 10 ? 'SINGLE' : x < 100 ? Math.round(x/10) + '0S' : 'BULK'
```
Input: `5` → `"SINGLE"`  
Input: `45` → `"40S"` (rounded to nearest 10s)  
Input: `500` → `"BULK"`

---

### Example 10: Safe Number Parsing
```javascript
|> parseFloat(x) || 0
```
Input: `"123.45"` → `123.45`  
Input: `"abc"` → `0` (fallback)

---

## Performance Notes

- **Regex execution**: O(n) for string length
- **Formula evaluation**: Generally O(1) for simple expressions, scales with complexity
- **No loops/recursion**: Avoid infinite loops; timeout not currently implemented
- **String concatenation**: Efficient for < 10KB strings

---

## Future Extensions (Not Yet Implemented)

### Planned Features
- **Row access**: `row[0]`, `row['columnName']` for cross-column operations
- **Function library**: `bucket(x, [0,10,50,100])` for predefined operations
- **Array methods**: `.map()`, `.filter()`, `.reduce()` on parsed values
- **Chained transforms**: Multiple `|>` operations in sequence
- **Named capture groups**: `(?<name>pattern)` and `name` variable access
- **Custom functions**: User-defined functions for reuse

---

## Quick Reference Cheatsheet

```javascript
// Comparisons
x > 100
x == 'PASS'
x >= 0 && x <= 100

// Ternary (if-then-else)
x > 50 ? 'HIGH' : 'LOW'

// String operations
x.toUpperCase()
x.substring(0, 3)
x.includes('test') ? 'YES' : 'NO'

// Math operations
Math.round(x)
Math.floor(Math.log10(x))
parseInt(x) * 2

// Defaults
x || 'default'
g1 ?? 'NO_MATCH'

// Multi-group
g1 + '_' + g2

// Complex
x > 0 ? Math.log10(x).toFixed(2) : '(invalid)'
```

---

## Summary

✅ **Fully Supported**:
- Comparison operators (`>`, `<`, `==`, `!=`, `>=`, `<=`)
- Logical operators (`&&`, `||`, `!`)
- Ternary conditional (`? :`)
- Arithmetic (`+`, `-`, `*`, `/`, `%`, `**`)
- String methods (toUpperCase, substring, slice, includes, etc.)
- Math functions (round, floor, ceil, log, sqrt, abs, etc.)
- Type conversion (parseInt, parseFloat, isNaN, Number, String)
- Null coalescing (`||`, `??`)
- Capture group combinations (`g1 + g2`)

⚠️ **Use Caution**:
- `eval()` (currently used, not sandboxed)
- Complex nested expressions (readability)
- Performance-heavy operations (large data)

❌ **Not Supported**:
- DOM access
- Network requests
- File I/O
- Async/await
- External libraries

