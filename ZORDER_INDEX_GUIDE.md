# Z-Order with Index Numbers - Quick Guide

## ✨ New Feature: Use Index Numbers!

You can now use **either group names OR legend indices** in the z-order field!

---

## 🎯 How It Works

### Index Mapping

Legend indices correspond to **alphabetically sorted groups**:

```
Alphabetical order: A, B, C
Index mapping:      1, 2, 3
```

### Example

**Legend (alphabetical):**
```
1 = A (blue)
2 = B (green)
3 = C (red)
```

**To arrange as C, B, A (reverse order):**
```
Z-Order: 3, 2, 1
```

That's it! ✅

---

## 📋 Real-World Examples

### Example 1: Region Data

**Your CSV has Region column:**
```
North, South, East, West
```

**Alphabetical order (legend):**
```
1 = East
2 = North
3 = South
4 = West
```

**Goal:** Put West on top, then South, then North, then East

**Command:**
```
Z-Order: 4, 3, 2, 1
```

**Result:**
- Index 4 (West) renders first (bottom)
- Index 3 (South) renders next
- Index 2 (North) renders next
- Index 1 (East) renders last (on top) ✅

---

### Example 2: Status Codes

**Status column values:**
```
error, warning, info, success
```

**Alphabetical order (legend):**
```
1 = error
2 = info
3 = success
4 = warning
```

**Goal:** Make 'error' most visible (on top)

**Command:**
```
Z-Order: 4, 3, 2, 1
```

Or just put error last:
```
Z-Order: 2, 3, 4, 1
```

**Result:** error (index 1) now on top! ✅

---

### Example 3: Product Types

**Product column:**
```
TypeA, TypeB, TypeC, TypeD
```

**Alphabetical order:**
```
1 = TypeA
2 = TypeB
3 = TypeC
4 = TypeD
```

**Goal:** Show TypeB and TypeD on top

**Command:**
```
Z-Order: 1, 3, 2, 4
```

**Result:**
- TypeA (1) renders first (bottom)
- TypeC (3) renders next
- TypeB (2) renders next
- TypeD (4) renders last (on top) ✅

---

## 🔄 You Can Mix Indices and Names!

Both work together:

**Example: Groups are A, B, C (alphabetically)**

**Command 1: All indices**
```
Z-Order: 3, 2, 1
```

**Command 2: All names**
```
Z-Order: C, B, A
```

**Command 3: Mix both!**
```
Z-Order: 3, B, 1
```
- Index 3 = C (renders first)
- Name B = B (renders next)
- Index 1 = A (renders last, on top)

All three commands produce the **same result**! ✅

---

## 🧪 Quick Test

### Step 1: Create Chart
```
X: any column
Y: any column
Color-by: any categorical column
Type: Scatter
```

### Step 2: Click Plot
```
Note the legend indices:
1 = [first group alphabetically]
2 = [second group]
3 = [third group]
etc.
```

### Step 3: Try Index Z-Order
```
Z-Order: 3, 2, 1
```

### Step 4: Click Plot
```
✅ Last group (1) should now appear ON TOP!
```

---

## 📊 Understanding the Index System

### Why Alphabetical?

Groups are always sorted alphabetically FIRST, then assigned indices:

```
Raw data groups:  B, A, C, A, B
Unique groups:    B, A, C
Sorted:           A, B, C
Indices:          1, 2, 3

So:
Index 1 = A
Index 2 = B
Index 3 = C
```

### Console Output Shows Mapping

Open browser console (F12) and look for:
```
[drawScatterLine] alphabetically sorted groups: ['A', 'B', 'C']
[drawScatterLine] index 3 → group "C"
[drawScatterLine] index 2 → group "B"
[drawScatterLine] index 1 → group "A"
```

This confirms your index-to-name mapping! ✅

---

## 💡 Pro Tips

### Tip 1: Find Index of a Specific Group

**Question:** "What index is 'Premium'?"

**Answer:** Check console (F12) output:
```
alphabetically sorted groups: ['Basic', 'Premium', 'Standard']
Premium is at index 2
```

### Tip 2: Quick Reversal

To put everything in reverse order:
```
Count your groups:  4 groups (A, B, C, D)
Indices:             1, 2, 3, 4
Reverse:             4, 3, 2, 1

Z-Order: 4, 3, 2, 1
```

### Tip 3: Partial List

You don't need ALL indices!

**Groups: A, B, C, D (indices 1, 2, 3, 4)**

**Command:**
```
Z-Order: 4, 2
```

**Result:**
- Index 4 (D) renders first (bottom)
- Index 2 (B) renders next
- Index 1 (A) renders next
- Index 3 (C) renders last (on top)

Indices not listed (1, 3) appear at the end, alphabetically! ✅

---

## 🚀 Comparison: Indices vs Names

### Using Names
```
Z-Order: West, North, East, South
```
**Pros:** Human-readable, easy to understand
**Cons:** Must type exact names, case-sensitive

### Using Indices
```
Z-Order: 4, 3, 2, 1
```
**Pros:** Quick to type, no spelling errors, no case sensitivity
**Cons:** Must know the index mapping

### Mixed
```
Z-Order: 4, North, 2, South
```
**Pros:** Best of both! Flexibility when needed
**Cons:** Slightly more complex

---

## ⚠️ Important Notes

### 1-Based Indexing

Indices start at **1**, not 0:
```
Index 1 = first group
Index 2 = second group
Index 3 = third group
```

NOT:
```
Index 0 = first group  ❌ (wrong!)
```

### Out-of-Range Indices Are Ignored

**If you have 3 groups but enter:**
```
Z-Order: 5, 2, 1
```

**Result:**
- Index 5: doesn't exist, skipped
- Index 2: used (renders next)
- Index 1: used (renders last)

Console shows: `index 5 out of range (1-3), skipping` ✅

### Case-Insensitive for Indices

```
Z-Order: 3, 2, 1    ← Works
Z-Order: 03, 02, 01 ← Works (leading zeros ignored)
Z-Order: 3.0, 2.0   ← Works (decimals truncated)
```

---

## 🆘 Troubleshooting

### Problem: "Indices don't match what I expected"

**Solution:** Check console (F12):
```
Look for: [drawScatterLine] alphabetically sorted groups: [...]
```

This shows the ACTUAL index mapping! Use that as reference.

---

### Problem: "Mix of indices and names isn't working"

**Solution:** Verify:
- [ ] Indices are valid (1 to N)
- [ ] Names match legend exactly (case-sensitive!)
- [ ] You used commas to separate

**Example that works:**
```
Z-Order: 3, North, 1, South
where:
  3 = third group alphabetically
  North = exact group name
  1 = first group alphabetically
  South = exact group name
```

---

### Problem: "I don't know which index is which"

**Solution:** Use the console!

1. Open F12 (Developer Tools)
2. Go to Console tab
3. Plot your chart
4. Look for:
   ```
   [drawScatterLine] alphabetically sorted groups: ['A', 'B', 'C']
   ```
5. Count: A=1, B=2, C=3

---

## 📝 Quick Reference

| Feature | Example | Result |
|---------|---------|--------|
| Index order | `3, 2, 1` | Reverses rendering (last renders on top) |
| Partial indices | `3, 1` | Groups 3 and 1 in that order, 2 at end |
| Named groups | `C, B, A` | Same as `3, 2, 1` if C, B, A are alphabetically sorted |
| Mixed | `3, B, 1` | Indices and names work together |
| Empty | (empty) | Uses alphabetical by default |

---

## ✅ Summary

**Z-Order now supports:**
- ✅ Group names: `Z-Order: C, B, A`
- ✅ Index numbers: `Z-Order: 3, 2, 1`
- ✅ Mixed: `Z-Order: 3, B, 1`

**Indices are:**
- ✅ 1-based (1, 2, 3, ...)
- ✅ Based on alphabetical order
- ✅ Easier than remembering group names!

**Try it now!** 🚀
