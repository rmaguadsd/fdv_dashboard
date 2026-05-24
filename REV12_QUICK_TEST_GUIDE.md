# REV12 Formula Feature: Quick Test Guide

## Quick Reference

### Separator: `=>`
Use space-equals-greater-than to separate regex from formula.

### Three Syntax Options

```javascript
// Option 1: Regex only (traditional)
(\d+)

// Option 2: Formula only (new)
=> x > 100 ? 'HIGH' : 'LOW'

// Option 3: Regex + Formula (new)
(\d+) => parseInt(x) > 100 ? 'HIGH' : 'LOW'
```

### Available Variables

```javascript
x   // The extracted value (or raw if no regex)
g1  // First capture group
g2  // Second capture group
g3  // Third, etc...
```

---

## 10 Quick Tests to Try

### Test 1: Simple Bucketing (Formula Only)
**Input Field:** `=> x > 50 ? 'LARGE' : 'SMALL'`  
**Sample Values:**  
- "75" → "LARGE"
- "30" → "SMALL"

### Test 2: Regex with Numbers (Regex Only)
**Input Field:** `(\d+)`  
**Sample Values:**  
- "DUT123" → "123"
- "TEST_456_DATA" → "456"

### Test 3: Extract and Bucket (Regex + Formula)
**Input Field:** `(\d+) => parseInt(x) > 100 ? 'HIGH' : 'LOW'`  
**Sample Values:**  
- "DUT250" → "HIGH"
- "DUT45" → "LOW"

### Test 4: Multiple Capture Groups (Regex)
**Input Field:** `^(..)_(\d+)$`  
**Sample Values:**  
- "AB_123" → "AB, 123"
- "XY_999" → "XY, 999"

### Test 5: Groups with Formula
**Input Field:** `^(..)_(\d+)$ => g1.toUpperCase() + '-' + g2`  
**Sample Values:**  
- "ab_123" → "AB-123"
- "xy_456" → "XY-456"

### Test 6: String Methods
**Input Field:** `=> x.substring(0, 3).toUpperCase()`  
**Sample Values:**  
- "hello" → "HEL"
- "world" → "WOR"

### Test 7: Math Operations
**Input Field:** `(\d+) => Math.round(parseInt(x) / 10) * 10`  
**Sample Values:**  
- "TEST_47_DATA" → "50"
- "TEST_123_DATA" → "120"

### Test 8: Complex Conditional
**Input Field:** `(\d+) => parseInt(x) < 50 ? 'LOW' : parseInt(x) < 100 ? 'MID' : 'HIGH'`  
**Sample Values:**  
- "VALUE_30" → "LOW"
- "VALUE_75" → "MID"
- "VALUE_150" → "HIGH"

### Test 9: Error Handling (Invalid Formula)
**Input Field:** `=> x >! invalid`  
**Expected:** Returns `(formula error)` + logs to console (F12 → Console)

### Test 10: Error Handling (Invalid Regex)
**Input Field:** `([) => x`  
**Expected:** Returns raw value + logs to console

---

## How to Test

1. **Open the Chart:**
   - Navigate to your FDV dashboard
   - Load a dataset with numeric/text columns

2. **Add a Color Dimension:**
   - Click "Add Dimension" under Color By
   - Select a column
   - Enter formula in the "regex or regex => formula" field
   - Watch the colors update

3. **Check for Errors:**
   - Open browser console: `F12` key
   - Look for `[Formula Error]` messages
   - Check for any chart rendering issues

4. **Verify Results:**
   - Check color groupings changed as expected
   - Spot-check a few values manually
   - Look for `(no match)` or `(formula error)` in legend

---

## Expected Outcomes

✅ **Should Work:**
- Old regex-only dimensions continue working
- New formula-only dimensions work
- Combined regex + formula work together
- Errors show `(formula error)` text and console logs
- Colors group by transformed values

❌ **Should NOT Work (Intentional):**
- Array methods: `x.map(...)` — not allowed
- Object creation: `{a: 1}` — not allowed
- Function definitions: `function() {}` — not allowed
- Variable assignment: `var y = 5` — not allowed

---

## Error Examples

### If You See This... | This Means...
```javascript
(no match)          // Regex didn't match the input value
(formula error)     // Formula had a syntax or runtime error
(blank)             // Input was empty or null
(undefined)         // Formula returned undefined
```

**Action:** Check console for details. Press `F12` → Console tab.

---

## Console Error Messages Format

```
[Formula Error] x > 100 ? 'HIGH' : 'LOW' | Error: SyntaxError: Unexpected token '?'
[extractGroupKey Error] Invalid regular expression | Input: ([)
```

---

## Performance Notes

- ✅ Simple formulas: Instant
- ✅ Complex math: Fast
- ✅ String operations: Very fast
- ⚠️ Many dimensions: May slow rendering
- ⚠️ Nested ternary: Test with real data first

---

## Rollback (If Needed)

If anything breaks:
```bash
cd d:\FDV\git\fdv_dashboard
git checkout dev/aitools/fdv_chart_rev12/fdv_chart.html
```

---

## Feedback

After testing, note:
- [ ] What formulas worked well?
- [ ] What errors did you encounter?
- [ ] Performance issues?
- [ ] UI suggestions?
- [ ] Missing operators?

---

**File Size:** 379,688 bytes  
**Separator:** `=>`  
**Status:** Ready for Testing
