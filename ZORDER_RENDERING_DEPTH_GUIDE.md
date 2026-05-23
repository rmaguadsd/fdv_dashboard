# Z-Order Rendering Depth Control - Exact Specification

## 🎯 How Z-Order Works Now

You have **perfect control** over which data points appear on top!

### The Rule

**Z-Order defines rendering depth: FIRST item renders at BOTTOM, LAST item renders at TOP**

### Visual Example

**Default chart (no z-order):**
```
Legend (alphabetical): A, B, C
Rendering order: A (bottom) → B (middle) → C (top)

Chart view:
┌─────────────────────┐
│         C ●●●       │  ← C on top (you see it first)
│      ●●B●●●●●       │  ← B behind C
│   A●●●●●●●●●●●      │  ← A behind B
└─────────────────────┘
```

**With Z-Order "C, B, A":**
```
Z-Order input: C, B, A
Rendering order: C (first/bottom) → B (middle) → A (last/top)

Chart view:
┌─────────────────────┐
│   A●●●●●●●●●●●      │  ← A on top NOW! (you see it first)
│      ●●B●●●●●       │  ← B behind A
│         C ●●●       │  ← C at bottom
└─────────────────────┘
```

---

## 📋 Examples

### Example 1: Three Data Groups (A, B, C)

**Legend:** A (blue), B (green), C (red)

**Default behavior:**
- A renders first (bottom layer)
- B renders on top of A
- C renders on top of B (most visible)

**Command:** `Z-Order: C, B, A`
- C renders first (bottom layer)
- B renders on top of C
- A renders last (most visible, on top)

**Result:** **A points are now on top!** ✅

---

### Example 2: Region Data (North, South, East, West)

**Legend:** East, North, South, West (alphabetical default)

**Default:**
- East renders first (you see it least)
- West renders last (you see it most)

**Command:** `Z-Order: West, North, East, South`
- West renders first (now you see it LEAST)
- South renders last (now you see it MOST)

**Result:** South data dominates the view! ✅

---

### Example 3: Status Codes (error, warning, info, success)

**Legend:** error, info, success, warning

**You want:** Success visible (on top)

**Command:** `Z-Order: error, warning, info, success`
- Error renders first (bottom)
- Success renders last (top) ✅

---

## 🔧 How to Use It

### Step-by-Step

**1. Create your chart and note the legend**
```
Legend shows: A, B, C (in some order)
```

**2. Decide what you want on top**
```
I want: A on top, B in middle, C on bottom
```

**3. Enter in Z-Order field**
```
Z-Order: C, B, A
         ↑  ↑  ↑
      bottom → top
```

**4. Click Plot**
```
Chart updates with new rendering depth!
```

---

## 📊 Rendering Order Explained

### In Chart.js (How It Works)

Datasets are drawn in the order they appear in the array:

```javascript
datasets = [
    { label: 'A', data: [...] },    // Drawn FIRST (renders at bottom)
    { label: 'B', data: [...] },    // Drawn SECOND
    { label: 'C', data: [...] }     // Drawn LAST (renders on top)
]
```

**Canvas rendering principle:** Last thing drawn appears on top (like stacking paper)

### Your Z-Order Maps To This

```
You enter: "C, B, A"
     ↓
System creates datasets array:
[
    { label: 'C', data: [...] },    // Drawn first → bottom
    { label: 'B', data: [...] },    // Drawn second
    { label: 'A', data: [...] }     // Drawn last → top
]
```

---

## ✅ Verification Checklist

### Does Z-Order Work?

Test with this example:

**1. Create scatter plot:** X=any, Y=any, Color-by=any_column
**2. Plot with NO z-order**
```
Record: Which group appears on top by default?
Example: If legend is "A, B, C", then C is on top (alphabetical)
```

**3. Enter z-order:** `Z-Order: A, B, C`
**4. Click Plot**
```
Observation: C should STILL be on top (same as #2)
Why? Because we reversed the order, so C is now last
```

**5. Try z-order:** `Z-Order: C, B, A`
**6. Click Plot**
```
Observation: A should NOW be on top (different from #2!)
Why? Because A is now last in the z-order list
```

**If A moved to top: ✅ Z-Order is working!**

---

## 🎨 Color Behavior

### Colors Stay Consistent

**Important:** Colors don't change when you change z-order!

**Example:**
```
Default: A=blue, B=green, C=red
         C on top (red points visible)

Z-Order "C, B, A": A=blue, B=green, C=red
                   A on top (blue points visible)
```

**The colors STAY ASSIGNED to the same groups** ✅

---

## 🚀 Advanced Usage

### Partial Z-Order

You don't have to list ALL groups!

**Example:** Groups are A, B, C, D, E
```
Z-Order: C, A
```

**Behavior:**
- C renders first (bottom)
- A renders second
- B, D, E render next (alphabetically, unlisted groups)

**Effect:** A is on top of C, but B/D/E are on top of A

---

### Empty Z-Order

**Z-Order field empty:**
```
System uses alphabetical order by default
```

---

## 🐛 Troubleshooting

### "I entered z-order but nothing changed"

**Problem:** Spelling doesn't match legend exactly

**Legend shows:** `North`
**You entered:** `north` ← lowercase! ❌

**Fix:** `North` ← exact case ✅

---

### "Wrong group is on top"

**Problem:** You reversed the order you needed

**You wanted:** B on top
**You entered:** `A, B` ← B is last = B on top ✅

**But you wanted:** A on top?
**Enter:** `B, A` ← A is last = A on top ✅

---

### "Points are overlapping - can't see one group"

**This means:** That group IS below the others!
**Solution:** Move it to the end of z-order list

**Currently see:** Group1 only
**Enter:** `Z-Order: Group1, Group2, Group3` 
**Result:** Group3 will now be visible on top

---

## 📝 Real-World Scenario

### Sales Data with Status

**CSV:**
```
Date,Amount,Status
2024-01-01,1000,Pending
2024-01-02,1500,Completed
2024-01-03,800,Cancelled
```

**Chart setup:**
- X: Date
- Y: Amount  
- Color-by: Status
- Type: Scatter

**Default chart:**
```
Legend: Cancelled, Completed, Pending (alphabetical)
On top: Pending (you see it first, might hide others)
```

**Goal:** See Completed orders on top

**Solution:**
```
Z-Order: Pending, Cancelled, Completed
```

**Result:**
- Pending renders first (hidden below)
- Cancelled renders next
- Completed renders last (visible on top!) ✅

---

## 💡 Key Insights

| Concept | Meaning |
|---------|---------|
| **First in Z-Order** | Renders at BOTTOM (hard to see) |
| **Last in Z-Order** | Renders on TOP (clearly visible) |
| **Omitted groups** | Render in middle (alphabetically) |
| **Color assignment** | STAYS SAME regardless of z-order |
| **Empty z-order field** | Uses alphabetical by default |

---

## 🎬 Quick Test (30 seconds)

1. Go to http://localhost:5059
2. Upload any CSV
3. Create scatter plot: X=col1, Y=col2, Color-by=col3
4. Click Plot
5. **Remember which color appears on top**
6. Enter Z-Order field: `(reverse the group order)`
7. Click Plot again
8. **Check: Different color should now be on top!**

✅ If yes: Z-Order is working perfectly!

---

**Status:** ✅ Z-Order rendering depth control DEPLOYED

The logic is now: **First item in z-order = bottom layer, Last item in z-order = top layer**
