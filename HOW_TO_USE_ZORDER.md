# How to Use Z-Order - Step-by-Step Guide

## 🎯 What Does Z-Order Do?

Z-Order controls **which color-by groups render on top** when data points overlap in your chart.

**Simple Example:**
- Without z-order: Groups render alphabetically (A bottom, B middle, C on top by default)
- With z-order: You specify the order (maybe C bottom, A middle, B on top)

---

## 📍 Where Is the Z-Order Control?

In the **Plot Panel** at the top of the page, look for:

```
X: [column ▼]          Y: [column ▼]

Color by: [column ▼]  +
Z-Order:  [text field]  (last = top)
                ↑
           THIS FIELD
```

The Z-Order field appears **right after the "Color by:" selector**.

---

## 🚀 How to Use It (4 Steps)

### Step 1: Set Up Your Chart

1. **Parse a CSV file** with data
2. **Select X and Y columns** (what to plot)
3. **Select a Color-by column** (what to split by color)
4. **Choose chart type** (Scatter, Line, Histogram, Boxplot, etc.)

**Example:** 
- X: Date
- Y: Sales  
- Color-by: Region (North, South, East, West)

### Step 2: Find the Z-Order Field

Look for the text input field labeled **"Z-Order:"**

It says `(last = top)` which means **the last value you enter will render on top**.

### Step 3: Enter Your Preferred Order

Type the color-by values **in the order you want them rendered**:

```
Z-Order: North, South, East, West
```

**What this does:**
- North renders 1st (bottom)
- South renders 2nd
- East renders 3rd
- West renders 4th (TOP - fully visible)

### Step 4: Click Plot

Click the **"Plot"** button to apply the new z-order.

---

## 📋 Real-World Examples

### Example 1: Show Sales Performance (Put "Best" on Top)

**Scenario:** You have regions (Good, Average, Poor) and want to see "Good" clearly

**Z-Order field:** `Average, Poor, Good`

**Result:** Good region renders on top (fully visible, not hidden behind others)

---

### Example 2: Hide Background Data

**Scenario:** You have treatment data (Control, Baseline, Treatment) and want Treatment visible

**Z-Order field:** `Control, Baseline, Treatment`

**Result:** Treatment renders last (on top)

---

### Example 3: Emphasize One Category

**Scenario:** Categories are (Normal, Important), want Important on top

**Z-Order field:** `Normal, Important`

**Result:** Important renders on top

---

## ✅ Quick Reference

| What You Want | How to Do It |
|---------------|------------|
| Default alphabetical order | Leave Z-Order field empty |
| Put X on top | Enter: `..., X` (X last) |
| Put X on bottom | Enter: `X, ...` (X first) |
| Custom order | Enter all values in desired order |
| Reset to default | Clear the field and click Plot |

---

## 🎨 Important Rules

### ✓ DO

- ✅ **Match exact spellings** - Copy from your legend (case-sensitive!)
- ✅ **Use commas** to separate values: `red, blue, green`
- ✅ **Spaces around commas are OK** - `red , blue , green` works
- ✅ **Use the exact case** - "Red" ≠ "red"
- ✅ **List values in render order** - Last = top

### ✗ DON'T

- ❌ **Don't invent values** - Only use values that appear in your legend
- ❌ **Don't use semicolons** - Use commas: `a, b` NOT `a; b`
- ❌ **Don't mix cases** - Check your legend for exact capitalization
- ❌ **Don't forget commas** - Each value must be separated by comma
- ❌ **Don't leave trailing commas** - `a, b,` won't work; use `a, b`

---

## 🔍 How to Verify It's Working

### Method 1: Visual Check
1. Enter z-order values
2. Click Plot
3. Look at your chart legend (usually at bottom)
4. The **last group you entered should be on top** (most visible when overlapping)

### Method 2: Browser Console
1. Open browser dev tools: **Press F12**
2. Click **"Console"** tab
3. Look for lines like:
   ```
   [drawScatterLine] z-order input: red, blue, green
   [drawScatterLine] z-order list: ["red", "blue", "green"]
   [drawScatterLine] sorted group keys: ["red", "blue", "green"]
   ```
4. If you see these, z-order is working!

---

## 📊 Supported Chart Types

Z-Order works with **ALL chart types**:

- ✅ **Scatter** - Points rendered in z-order
- ✅ **Line** - Lines rendered in z-order  
- ✅ **Histogram** - Bars rendered in z-order
- ✅ **Box & Whisker** - Groups rendered in z-order
- ✅ **Cumulative Probability** - Lines rendered in z-order
- ✅ **RCDF** - Lines rendered in z-order
- ✅ **Split-Chart tiles** - Each tile respects z-order

---

## 🎯 Common Scenarios

### Scenario: Comparative Analysis

You're comparing a new treatment against control and want treatment clearly visible:

```
Z-Order: Control, Standard, NewTreatment
```

→ NewTreatment renders on top

---

### Scenario: Outlier Detection

You want to see rare/outlier data points clearly:

```
Z-Order: Normal, Normal_High, Normal_Low, Outlier
```

→ Outlier always visible on top

---

### Scenario: Time-Based Visibility

You want to see recent data most clearly:

```
Z-Order: Week1, Week2, Week3, Week4
```

→ Week4 renders on top (newest data most visible)

---

## 🆘 Troubleshooting

### Problem: Z-Order Not Working

**Checklist:**
1. ✓ Did you click **Plot** button after entering z-order?
2. ✓ Are the values spelled **exactly** as shown in your legend?
3. ✓ Is the **case correct**? (Red vs red matters!)
4. ✓ Did you use **commas** to separate values?
5. ✓ Are you using the **right chart type**? (All types support it)

**Solution:** Check browser console (F12) for error messages

---

### Problem: Some Groups Still in Alphabetical Order

**This is normal!** Groups you don't list in z-order still get rendered (alphabetically sorted, at the bottom).

**Example:**
- Your groups: A, B, C, D, E
- Z-Order: `E, A`
- Result: B, C, D (alphabetical) then A, then E on top

**To control all:** List all groups: `A, B, C, D, E`

---

### Problem: Still Not Working?

**Check these:**
1. Open F12 → Console
2. Look for z-order log messages
3. If missing: z-order field not found
4. If messages show wrong values: check your spelling
5. If messages correct but chart wrong: there may be a rendering issue

---

## 💾 Saving Your Z-Order Preference

### Option 1: Save as Recipe
- Click **"Save"** under "Recipe" section
- Your z-order + all plot settings save together
- Load anytime with **"Load Recipe"**

### Option 2: Save as Session  
- Click **"Save"** under "Session" section
- Your z-order + all data + settings save together
- Load anytime with **"Load Session"**

### Option 3: Browser Memory
- Your z-order persists while the tab is open
- Lost on refresh unless saved above

---

## 📝 Example Walkthrough

### Starting State
```
File: sales_data.csv
Columns: Month, Revenue, Region
Data has: North (500 points), South (400 points), East (300 points)
```

### Steps:
1. **Parse file** → Click Parse button
2. **Set chart:**
   - X: Month
   - Y: Revenue
   - Color-by: Region
   - Type: Scatter
3. **Enter z-order:** `South, North, East`
4. **Click Plot**

### Result:
- South renders first (bottom)
- North renders second
- East renders last **(on top - fully visible)**

---

## 🎓 Learning Tips

### Beginner
- Start with just 2 values: `B, A`
- See if they change order
- Then try 3+ values

### Intermediate  
- Use with complex data (many groups)
- Combine with filtering (intervals)
- Experiment with different orders

### Advanced
- Use for multi-dimensional color-by
- Combine with split-chart mode
- Stack specific patterns for analysis

---

## ✨ Key Takeaway

**Z-Order = Rendering Order**

Last value in your list = renders last = appears on top (most visible)

```
Z-Order: bottom, middle, top
         ↓      ↓      ↓
      1st    2nd    3rd (ON TOP)
```

---

## 🚀 Ready to Try?

1. Open http://localhost:5059
2. Upload a CSV with categorical data
3. Select chart type + columns
4. **Enter z-order values** in the Z-Order field
5. **Click Plot**
6. Watch your preferred group render on top! 🎉

---

**Questions?** Check the console (F12) or try a simple example first!
