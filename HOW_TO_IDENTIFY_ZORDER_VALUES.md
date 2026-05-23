# How to Identify Z-Order Values (value1, value2, value3)

## 🎯 The Short Answer

Your **color-by values** ARE your z-order values. They come directly from your data!

---

## 📊 Step-by-Step: Where to Find Them

### Step 1: Look at Your CSV File

Open your CSV file and find the **"Color by" column**:

**Example CSV:**
```
Date,Sales,Region
Jan,1000,North
Feb,1200,North
Mar,800,South
Apr,900,East
May,1100,West
Jun,950,South
```

In this example, the **Region** column has these values:
- `North`
- `South`
- `East`
- `West`

These are your **z-order values!**

---

### Step 2: Set Color-by in the Chart

1. Open http://localhost:5059
2. Parse your CSV
3. Select **Color-by: Region**

---

### Step 3: Check the Chart Legend

After you click **Plot**, look at the bottom of your chart for the **Legend**:

```
┌─────────────────────────┐
│     Your Chart Here     │
├─────────────────────────┤
│ □ North                 │  ← These are your values!
│ □ South                 │
│ □ East                  │
│ □ West                  │
└─────────────────────────┘
```

**These legend labels = your z-order values!**

---

## 🔍 Practical Examples

### Example 1: Sales by Region

**Your CSV has:** Region column with values: North, South, East, West

**Chart Legend shows:**
```
□ East
□ North
□ South
□ West
```

**Your z-order values are:** `East, North, South, West`

**If you want West on top, enter:**
```
Z-Order: East, North, South, West
```

---

### Example 2: Product Categories

**Your CSV has:** Product column with values: TypeA, TypeB, TypeC

**Chart Legend shows:**
```
□ TypeA
□ TypeB
□ TypeC
```

**Your z-order values are:** `TypeA, TypeB, TypeC`

**If you want TypeB on top, enter:**
```
Z-Order: TypeA, TypeC, TypeB
```

---

### Example 3: Status Codes

**Your CSV has:** Status column with values: error, warning, info, success

**Chart Legend shows:**
```
□ error
□ info
□ success
□ warning
```

**Your z-order values are:** `error, info, success, warning`

**If you want errors visible on top, enter:**
```
Z-Order: warning, info, success, error
```

---

## ✅ How to Copy Values Correctly

### Method 1: Copy from Legend (RECOMMENDED)

1. Look at your chart **legend**
2. **Carefully copy each value** exactly as shown (case matters!)
3. Separate with commas

**Example - Looking at legend:**
```
Legend shows:
□ North
□ South  
□ East
□ West

Copy to Z-Order field: North, South, East, West
                       ↑     ↑      ↑     ↑
                   Copy exactly as shown!
```

---

### Method 2: Look at Your Data

If you don't have a legend yet (chart not plotted):

1. Open your CSV file
2. Find the Color-by column
3. Look at the unique values in that column
4. Copy them to Z-Order field

---

## 🚨 Common Mistakes to Avoid

### ❌ Mistake 1: Wrong Case

**Your data shows:** `North` (with capital N)

**Wrong:**
```
Z-Order: north, south  ← lowercase won't work!
```

**Correct:**
```
Z-Order: North, South  ← matches exactly
```

---

### ❌ Mistake 2: Typos

**Your legend shows:** `Region_1`, `Region_2`

**Wrong:**
```
Z-Order: Region1, Region2  ← missing underscore!
```

**Correct:**
```
Z-Order: Region_1, Region_2  ← matches exactly
```

---

### ❌ Mistake 3: Extra Spaces

**Your legend shows:** `North` (no extra spaces)

**Wrong:**
```
Z-Order: North , South , East  ← spaces inside values
```

**Correct:**
```
Z-Order: North, South, East  ← or spaces after comma is OK: North, South, East
```

---

### ❌ Mistake 4: Incomplete List

**Your legend shows:** `A, B, C, D, E`

**Wrong:**
```
Z-Order: E, A  ← missing B, C, D
```

**This still works, but B, C, D will render alphabetically BEFORE A and E**

**If you want full control:**
```
Z-Order: B, C, D, A, E  ← all values listed
```

---

## 📝 Quick Checklist

Before entering z-order values, verify:

- [ ] I can see the **Chart Legend** (bottom of chart)
- [ ] I copied the **exact text** from the legend
- [ ] The **case matches** (North not north)
- [ ] I used **commas** to separate values
- [ ] I **didn't add extra spaces** inside value names
- [ ] I **copied values exactly** as they appear

---

## 🎯 Real Workflow

### Your Complete Workflow:

**1. Prepare Data**
```
CSV file with:
- Column: Region
- Values: North, South, East, West
```

**2. Open Chart**
- Go to http://localhost:5059

**3. Parse File**
- Click Parse button

**4. Set Up Chart**
```
X: [Date]
Y: [Sales]
Color-by: [Region]  ← This is KEY
Type: [Scatter]
```

**5. Click Plot**
- Chart renders
- Legend appears at bottom

**6. Read Legend**
```
Legend shows:
□ East
□ North
□ South
□ West
```

**7. Copy Values**
- These ARE your z-order values
- Copy them to Z-Order field

**8. Enter Z-Order**
```
Z-Order: East, North, South, West
         (or in any order you want)
```

**9. Click Plot Again**
- New z-order is applied

---

## 🔧 Multi-Dimensional Color-By

If you're using **multiple color-by dimensions** (click the + button):

**Your values will be COMPOUND keys:**

```
Dimension 1: Department (Sales, Marketing, IT)
Dimension 2: Level (Junior, Senior)

Your z-order values become:
Sales~Junior
Sales~Senior
Marketing~Junior
Marketing~Senior
IT~Junior
IT~Senior
```

**In Z-Order field:**
```
Z-Order: Sales~Junior, IT~Senior, Marketing~Junior
```

**See the ~ symbol?** That's how compound keys are joined!

---

## 📊 Visual Guide

### Your Data
```
Date      | Sales | Region
----------|-------|--------
Jan 1     | 1000  | North
Jan 2     | 1200  | North
Jan 3     | 800   | South
Jan 4     | 900   | East
Jan 5     | 1100  | West
```

### Set Color-by to "Region"

### Chart Legend Shows
```
□ East    ← value1
□ North   ← value2  
□ South   ← value3
□ West    ← value4
```

### Z-Order Examples

**Put West on top:**
```
Z-Order: East, North, South, West
```

**Put North on top:**
```
Z-Order: East, South, West, North
```

**Put South first, West last:**
```
Z-Order: South, East, North, West
```

---

## ✨ Key Point

> **Your z-order values = the unique values in your Color-by column**

That's it! Just:
1. Choose a Color-by column
2. Plot the chart
3. Read the legend
4. Copy those values to the Z-Order field

---

## 🆘 Still Can't Find Them?

### Check 1: Did you select a Color-by column?

If you didn't select a Color-by column:
- You won't have different colors
- Z-Order won't have anything to sort
- First, select a Color-by column!

### Check 2: Did you click Plot?

The legend only appears AFTER clicking Plot.
- Plot first
- Then read the legend
- Then enter z-order values

### Check 3: Check Your Data

Open your CSV file and verify:
- The column you selected has different values
- Values aren't all the same (e.g., not all "A")
- There's actual variety in the data

---

## 📚 Examples by Data Type

### Text Values (Most Common)
```
Color-by values: Apple, Banana, Cherry

Z-Order: Apple, Cherry, Banana
```

### Numeric Values
```
Color-by values: 1, 2, 3, 4, 5

Z-Order: 1, 5, 3, 2, 4
```

### Mixed Values
```
Color-by values: Red_Small, Red_Large, Blue_Small, Blue_Large

Z-Order: Red_Small, Blue_Small, Red_Large, Blue_Large
```

### With Spaces
```
Color-by values: High Priority, Medium Priority, Low Priority

Z-Order: Low Priority, Medium Priority, High Priority
```

---

## 🎓 Summary

**To find your z-order values:**

1. Choose a **Color-by column**
2. **Click Plot**
3. Look at the **Chart Legend** at the bottom
4. **Copy those values** (exactly as shown)
5. **Paste into Z-Order field**
6. Separate with **commas**
7. **Click Plot again**

That's all! The values you see in the legend = your z-order values. 🚀

---

**Still confused?** The safest approach:
- Plot your chart first
- Look at the legend
- Copy/paste the legend values into Z-Order field
- Can't go wrong! ✅
