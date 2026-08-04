# Quick Reference: X/Y-Regex Formula Feature

## Syntax
```
REGEX_PATTERN |> FORMULA_EXPRESSION
```

## Examples

### Multiply by Constant
```
Input:     "2.5mA"
X-Regex:   ([0-9.]+) |> x * 1000
Output:    2500
```

### Log Transform
```
Input:     "100"
Y-Regex:   ([0-9]+) |> Math.log10(x)
Output:    2.0
```

### Exponential (dB to Linear)
```
Input:     "20dB"
X-Regex:   ([0-9]+) |> Math.pow(10, x/20)
Output:    10.0
```

### Normalize to 0-1
```
Input:     "75"
X-Regex:   ([0-9]+) |> x / 100
Output:    0.75
```

### Absolute Value
```
Input:     "-3.5"
X-Regex:   (-?[0-9.]+) |> Math.abs(x)
Output:    3.5
```

### Square Root
```
Input:     "16"
Y-Regex:   ([0-9]+) |> Math.sqrt(x)
Output:    4.0
```

## Common Math Functions

| Function | Example |
|----------|---------|
| `x * 2` | Multiply |
| `x / 10` | Divide |
| `x + 5` | Add |
| `x - 3` | Subtract |
| `Math.sqrt(x)` | Square root |
| `Math.pow(x, 2)` | Power (x²) |
| `Math.abs(x)` | Absolute value |
| `Math.log(x)` | Natural log |
| `Math.log10(x)` | Log base 10 |
| `Math.exp(x)` | e to the power x |
| `Math.sin(x)` | Sine |
| `Math.cos(x)` | Cosine |
| `Math.floor(x)` | Round down |
| `Math.ceil(x)` | Round up |
| `Math.round(x)` | Round nearest |

## Tips

1. **Optional Formula:** If no `|>` present, just regex extraction
2. **Error Handling:** Invalid formulas gracefully fall back to extracted value
3. **Works Everywhere:** Scatter, boxplot, histogram, cum_sigma, etc.
4. **All Charts:** Any chart type using x-regex or y-regex

## Troubleshooting

**Problem:** Chart shows wrong values
**Solution:** Check browser console for formula errors

**Problem:** No data points appear
**Solution:** Formula might be returning NaN - verify with test data

**Problem:** "undefined is not a function"
**Solution:** Typo in Math function name (e.g., "Math.Sqrt" should be "Math.sqrt")

## Status
✅ Implemented and tested on port 5059
