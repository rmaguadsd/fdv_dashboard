# Proposal: Combined Regex + Formula Field (Single Input)

## Overview

Instead of adding a separate `formula` field, consolidate both regex extraction **and** formula transformation into a **single unified input field** with a simple separator syntax. This reduces UI clutter while maintaining all functionality.

---

## Current State

### UI Layout (Current)
```html
<select>Column</select> + <input class="gdim-rx" placeholder="regex…"> + <button>×</button>
```

### Data Structure (Current)
```javascript
{
    col: "ColumnName",
    colIdx: 3,
    rx: "regex_pattern"    // Only regex, no formula
}
```

### Processing (Current)
```javascript
function _extractGroupKey(raw, rxStr) {
    if (!rxStr) return raw;
    var m = new RegExp(rxStr).exec(raw);
    return m ? m[0] : '(no match)';
}
```

---

## Proposed Solution: Separator Syntax

### Format Specification

**Single field syntax with optional separator `|>`:**

```
[regex] [|> formula]
```

Where:
- **Left side** (before `|>`): Standard regex pattern (optional)
- **Separator**: `|>` (pipe-greater-than, visual "flow" indicator)
- **Right side** (after `|>`): JavaScript formula operating on extracted value (optional)

### Examples

| Input | Meaning | Behavior |
|-------|---------|----------|
| `(\d+)` | Regex only | Extract digits; no transformation |
| `(\d+) \|> x > 100 ? 'HIGH' : 'LOW'` | Regex + formula | Extract digits, then bucket |
| ` \|> x.toUpperCase()` | Formula only (no regex) | Transform raw value to uppercase |
| ` \|> x < 0 ? 'NEG' : 'POS'` | Formula only | Bucket raw value as number |
| `^(\w+)_(\d+) \|> g1 + '_' + g2` | Regex + formula | Extract groups, join with underscore |

---

## Parsing Logic

### Extended _extractGroupKey()

```javascript
function _extractGroupKey(raw, rxAndFormula) {
    // Step 1: Parse the combined field
    var parts = rxAndFormula.split(/\s*\|>\s*/);  // Split on |> with optional spaces
    var rxStr = parts[0].trim();
    var formula = parts.length > 1 ? parts[1].trim() : '';
    
    // Step 2: Apply regex extraction
    var value = raw;
    if (rxStr) {
        try {
            var m = new RegExp(rxStr).exec(raw);
            if (!m) return '(no match)';
            
            // Capture groups: g1, g2, g3, etc.
            var groups = {};
            for (var i = 1; i < m.length; i++) {
                groups['g' + i] = m[i];
            }
            
            // If capture groups exist, use them; otherwise use full match
            if (Object.keys(groups).length > 0 && groups.g1) {
                value = groups.g1;  // Primary: first capture group
                // Store full groups for formula access
                value._groups = groups;
            } else {
                value = m[0];  // Fallback: full match
            }
        } catch(e) {
            console.warn('[regex] Invalid regex:', rxStr, e.message);
            return '(regex error)';
        }
    }
    
    // Step 3: Apply formula transformation
    if (formula) {
        try {
            var x = value;
            // Also expose capture groups if they exist
            var g1 = (value._groups && value._groups.g1) || value;
            var g2 = (value._groups && value._groups.g2) || '';
            var g3 = (value._groups && value._groups.g3) || '';
            var result = eval('(' + formula + ')');
            return String(result);
        } catch(e) {
            console.warn('[formula] Error evaluating:', formula, 'on value:', value, e.message);
            return String(value);  // Fallback to unformatted value
        }
    }
    
    // Step 4: Return final group key
    return String(value);
}
```

### Updated _gdimRead()

```javascript
function _gdimRead(type) {
    var wrap = document.getElementById(type + '-dims-wrap');
    if (!wrap) return [];
    var dims = [];
    wrap.querySelectorAll('.gdim-row').forEach(function(row) {
        var sel = row.querySelector('select');
        var inp = row.querySelector('.gdim-rx');  // Still same class name
        var col = sel ? sel.value : '';
        var rxFormula = inp ? inp.value.trim() : '';  // Now contains regex|>formula
        dims.push({ col: col, colIdx: col ? (currentHeaders.indexOf(col)) : -1, rx: rxFormula });
    });
    return dims;
}
```

---

## Separator Design Rationale

### Why `|>` (pipe-greater-than)?

| Symbol | Pros | Cons |
|--------|------|------|
| `\|>` | Visual flow: extract then transform | Needs escape in regex |
| `->` | Common in functional languages | Conflict: regex range `a-z` |
| `::` | Clear scope separation | Conflict: regex anchors `::` |
| `;` | Semicolon separator | Common in code, but confusing with `eval` |
| `=>` | Arrow function syntax | Visual similarity helps learning |
| `|=>` | Clearer (pipe to arrow) | More typing |
| `⇒` | Unicode arrow | Not keyboard-friendly |
| `~>` | Tilde-arrow | Less common, harder to find on keyboard |

**Recommendation: `|>` or `->`**
- `|>` wins for visual "flow" metaphor
- Requires escaping in regex if user wants literal `|>` in pattern (rare edge case, same as any special char)
- Consistent with "piping" concept (Unix, Elixir, modern JS proposals)

---

## Usage Examples with Combined Field

### Example 1: Temperature Bucketing
**Input field:**
```
|> x < 0 ? 'Frozen' : x < 10 ? 'Cold' : x < 20 ? 'Mild' : 'Warm'
```
- No regex (left side empty)
- Formula transforms raw number into bucket

**Flow:**
```
raw value: "5" → parse as number → formula → "Cold"
```

---

### Example 2: Extract + Bucket
**Input field:**
```
(\d+) |> parseInt(x) > 100 ? 'HIGH' : 'LOW'
```
- Regex extracts digits from string
- Formula buckets the number

**Flow:**
```
raw: "TEMP_150_F" → regex extracts "150" → formula → "HIGH"
```

---

### Example 3: Multi-Group Join
**Input field:**
```
^(\w+)_(\d+)$ |> g1 + '_' + (parseInt(g2) * 2)
```
- Regex captures two groups
- Formula combines and transforms them

**Flow:**
```
raw: "ABC_25" → g1="ABC", g2="25" → formula → "ABC_50"
```

---

### Example 4: String Transformation Only
**Input field:**
```
|> x.substring(0, 3).toUpperCase()
```
- No regex
- Formula transforms string directly

**Flow:**
```
raw: "apple" → formula → "APP"
```

---

### Example 5: Case-Insensitive Group Extraction
**Input field:**
```
(?i)^(result_\w+) |> x.toUpperCase()
```
- Regex with case-insensitive flag
- Formula normalizes to uppercase

**Flow:**
```
raw: "Result_Pass" → regex → "Result_Pass" → formula → "RESULT_PASS"
```

---

## UI Layout (Unchanged)

The placeholder and layout remain the same:

```html
<div class="gdim-row">
    <select id="gdim-color-col-ABC123">
        <option>— col —</option>
        <option>Temperature</option>
        <option>Product</option>
        ...
    </select>
    <input type="text" 
           class="gdim-rx" 
           placeholder="regex… or regex |> formula"
           title="Regex pattern to extract value, optionally followed by |> and JavaScript formula">
    <button class="gdim-del" onclick="_gdimDel(this,'color')">×</button>
</div>
```

**Key change:** Just update the placeholder text.

---

## Data Structure in Sessions

**Stored as before** (no change to session format):

```javascript
{
    col: "Temperature",
    colIdx: 5,
    rx: "(\d+) |> parseInt(x) > 100 ? 'HIGH' : 'LOW'"  // Combined format
}
```

Session JSON:
```json
{
    "__colorDims": "[{\"col\":\"Temperature\",\"rx\":\"(\\d+) |> parseInt(x) > 100 ? 'HIGH' : 'LOW'\"}]"
}
```

No migration needed—existing sessions with only regex (no `|>`) work as before.

---

## Backward Compatibility

✅ **100% Compatible**

- Old sessions with plain regex (no `|>`) work unchanged
- Split logic is simple: only parse `|>` if present
- Empty left side = formula only (no regex)
- Empty right side = regex only (no formula)
- Both empty = use raw value (same as current behavior)

---

## Parsing Edge Cases

### Edge Case 1: Regex contains `|>` literally

**Input:**
```
(result_\|>_\w+)
```

**Handling:** 
- User escapes `|>` as `\|>` in regex
- Split logic looks for ` |> ` (spaces around)
- If they don't add spaces, it's treated as formula start (user error, but rare)

**Better approach:** Use spaces as delimiter:
```
Pattern: [regex] [whitespace] |> [whitespace] [formula]
Code: split(/\s+\|>\s+/)  // Requires spaces
```

This avoids most edge cases.

---

### Edge Case 2: Formula contains `|>`

**Input:**
```
x.match(/a\|>b/) |> x[0]
```

**Handling:**
- Only split on FIRST occurrence of ` |> ` (with spaces)
- Formula can safely use `|>` without spaces in regex/strings

**Implementation:**
```javascript
var idx = rxAndFormula.indexOf(' |> ');
if (idx === -1) {
    rxStr = rxAndFormula.trim();
    formula = '';
} else {
    rxStr = rxAndFormula.substring(0, idx).trim();
    formula = rxAndFormula.substring(idx + 4).trim();
}
```

---

### Edge Case 3: User forgets spaces around `|>`

**Input:**
```
(\d+)|>parseInt(x) > 50
```

**Handling:** Won't match ` |> ` pattern (requires spaces)
- Treated as regex only
- User gets unexpected result → debugging tip in error message

**Solution:** Accept both ` |> ` and `|>` (with/without spaces):
```javascript
var parts = rxAndFormula.split(/\s*\|>\s*/);  // Flexible spacing
```

---

## Formula Variables Available

### In Formula Context

```javascript
x      = extracted value (after regex, if provided; otherwise raw value)
g1     = first regex capture group (if regex present)
g2     = second regex capture group (if regex present)
g3     = third capture group (if regex present)
row    = full row object (future enhancement, not in MVP)
```

### Example with Multiple Groups

**Input:**
```
^([A-Z]+)_(\d+)_([a-z]+)$ |> g1 + '-' + parseInt(g2) * 2 + '-' + g3.toUpperCase()
```

**Execution:**
```
raw: "ABC_25_test"
  → g1="ABC", g2="25", g3="test"
  → formula: "ABC" + "-" + 50 + "-" + "TEST"
  → result: "ABC-50-TEST"
```

---

## Implementation Checklist

### Phase 1 (MVP): Single Regex Input with Optional Formula

- [ ] Update `_extractGroupKey()` to parse `|>` separator
- [ ] Add regex + formula variables (`x`, `g1`, `g2`, etc.)
- [ ] Update placeholder text in UI
- [ ] Update session save/restore (no change needed, same `rx` field)
- [ ] Add error handling for invalid regex/formula
- [ ] Console logging for debugging

### Phase 2 (Polish): UX Improvements

- [ ] Optional help tooltip with syntax examples
- [ ] Formula syntax validation in browser (optional, nice-to-have)
- [ ] Better error messages when formula fails
- [ ] Visual indicator if field contains formula (e.g., ` |> ` highlight)

### Phase 3 (Advanced): Extended Features

- [ ] Support `row` variable for cross-column operations
- [ ] Predefined function library (`bucket()`, `normalize()`, etc.)
- [ ] Named capture groups support (`?<name>` syntax)

---

## Comparison: Separate vs. Combined Field

### Option A: Separate Fields (Original Proposal)

| Aspect | Pros | Cons |
|--------|------|------|
| UI | Clear distinction | More screen space |
| Code | Easy parsing (two inputs) | More fields to read/save |
| UX | User sees both options | Cognitive load: "which do I use?" |

### Option B: Combined Field (This Proposal) ✅

| Aspect | Pros | Cons |
|--------|------|------|
| UI | Compact, single input | Less obvious options |
| Code | Single field to parse | Slightly complex parsing logic |
| UX | Natural "flow" (`|>`) | Users must understand separator |

**Recommendation: Option B**
- Cleaner UI (no extra field)
- Familiar to developers (`|>` is like function composition)
- Backward compatible
- Easier to teach ("put formula after `|>`")

---

## Example Help Text / Tooltip

```
"Regex pattern to extract value from raw data, optionally followed by ' |> ' and a JavaScript formula to transform it.

Examples:
  • (\d+)                          Extract digits only
  • (\d+) |> parseInt(x) > 100 ? 'HIGH' : 'LOW'    Extract & bucket
  • |> x.toUpperCase()              Transform to uppercase (no extraction)
  • ^(\w+)_(\d+)$ |> g1 + '_' + g2  Extract two groups, join them

Variables in formula:  x (extracted value), g1 (capture group 1), g2 (capture group 2), etc.
"
```

---

## Error Messages

| Scenario | Error Message | Solution |
|----------|---------------|----------|
| Invalid regex | `(regex error) — invalid pattern` | User checks regex syntax |
| Formula throws | `(formula error) — invalid expression` | User checks formula syntax |
| No match (regex) | `(no match)` | Filter removes rows with no match |
| Formula returns null | `null` | Result converted to string "null" |
| Formula return undefined | (uses original value) | Fallback to unformatted value |

---

## Backward Compatibility Check

### Old Session with Plain Regex
```json
{"__colorDims": "[{\"col\":\"Value\",\"rx\":\"(\\d+)\"}]"}
```

**Processing:**
```javascript
var rxAndFormula = "(\\d+)";  // No |> separator
var parts = rxAndFormula.split(/\s*\|>\s*/);
// parts = ["(\\d+)"]
var rxStr = "(\\d+)";
var formula = "";  // Empty
// Regex applied, no formula → works as before ✓
```

---

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Fields per dimension** | 1 (col) + regex input | 1 (col) + combined input |
| **Regex support** | Yes | Yes ✓ |
| **Formula support** | No | Yes ✓ |
| **Separator** | N/A | `|>` |
| **Session format** | No change | No change ✓ |
| **Backward compatible** | N/A | 100% ✓ |
| **UI complexity** | Simple | Simple ✓ |
| **Parsing complexity** | Simple | Moderate |

---

## Advantages of Combined Field Approach

1. **Single Input**: Users see one field, not two → clearer mental model
2. **Backward Compatible**: Existing regex-only values work unchanged
3. **Compact**: No extra UI clutter, better use of space
4. **Composable**: Mirrors functional programming pattern (`|>` pipes)
5. **Flexible**: Supports regex-only, formula-only, or both
6. **No Migration**: Sessions automatically work with combined format
7. **Intuitive**: Clear visual "flow" from extraction to transformation

---

## Disadvantages of Combined Field Approach

1. **Parsing Complexity**: Must handle separator logic, edge cases
2. **User Documentation**: Requires explaining `|>` syntax
3. **Error Messages**: Parsing errors harder to pinpoint (regex vs. formula)
4. **Copy-Paste Issues**: Users might copy regex from elsewhere, forget formula part
5. **Accessibility**: Single field harder to navigate with keyboard only
6. **Visual Clarity**: Less obvious that two operations are available

---

## Recommendation

**Go with Option B (Combined Field)** because:
- ✅ Matches user mental model: "extract then transform"
- ✅ Cleaner UI layout
- ✅ Single field to document and learn
- ✅ Backward compatible with existing sessions
- ✅ `|>` syntax is becoming standard in modern languages (Elixir, future JS)

---

## Next Steps

1. **Feedback**: Confirm separator choice (`|>` vs. `->`), variable names (`x`, `g1`, `g2`)
2. **Implementation**: Update `_extractGroupKey()` and parsing logic
3. **Testing**: Verify backward compatibility with existing sessions
4. **Documentation**: Update UI placeholder and help text
5. **Launch**: Deploy as Rev12 feature

---

## Alternative Separators (If `|>` Not Preferred)

| Symbol | Syntax | Pros | Cons |
|--------|--------|------|------|
| `\|>` | `(\d+) \|> formula` | Explicit escape | Ugly, not intuitive |
| `=>` | `(\d+) => formula` | Arrow familiar to JS devs | May conflict with some regex patterns |
| `~>` | `(\d+) ~> formula` | Uncommon, less conflicts | Harder to find on keyboard |
| `;` | `(\d+); formula` | Simple | Confusing with code statements |
| `→` | `(\d+) → formula` | Visual arrow | Unicode, not keyboard-friendly |
| `:` | `(\d+): formula` | Label-like syntax | Conflict with regex patterns |
| `>>` | `(\d+) >> formula` | Bitshift visual | Less intuitive |

**Ranking:**
1. `|>` (best for "piping" mental model)
2. `=>` (familiar to JS developers)
3. `->` (if conflicts with regex become issue)

