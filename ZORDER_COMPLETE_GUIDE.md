# Z-Order Feature - Complete Usage Guide

## Overview

The **Z-Order** feature lets you control which color-by groups render **on top** in overlapping visualizations. This is useful when you want to ensure important data points or groups are always visible above others.

---

## 🎯 Core Concept

In chart rendering, layers are drawn sequentially:
1. **First dataset** → Renders at bottom (background)
2. **Middle datasets** → Render in middle
3. **Last dataset** → Renders on top (fully visible)

The **Z-Order control** lets you specify which groups become the "last dataset" (on top).

---

## 📋 Step-by-Step Usage

### Setup Phase

**1. Parse Your Data**
```
File: sales_data.csv
Columns: Region, Quarter, Revenue, Product
Sample: NORTH,Q1,15000,TypeA
        SOUTH,Q1,12000,TypeB
```

**2. Load File and Set Axes**
- Click Parse
- X-axis: Quarter
- Y-axis: Revenue
- Color-by: Product

**3. See Initial Chart**
- Should show groups: TypeA, TypeB, TypeC, etc.
- Default order: alphabetical (TypeA bottom, TypeC top if 3 types)

### Z-Order Phase

**4. Enter Z-Order Preference**

Click the **Z-Order field** and enter your preferred rendering order:

```
Z-Order: [input field]
Enter:   TypeB, TypeA, TypeC
```

**5. Click Plot**
- Chart refreshes with new rendering order
- TypeB renders first (bottom)
- TypeA renders second
- TypeC renders last (on top)

**6. Verify Results**
- Look at overlapping points
- Most important group should be fully visible
- Check legend to confirm order in sidebar

---

## 🔍 Detailed Examples

### Example 1: Hide a Baseline for Focus

**Scenario:**
You have 4 treatment groups: Control, Treatment1, Treatment2, New_Drug

By default, they render alphabetically:
- Control (bottom)
- New_Drug
- Treatment1
- Treatment2 (top - but you want New_Drug on top)

**Solution:**
```
Z-Order: Control, Treatment1, Treatment2, New_Drug
```

**Result:**
- Control renders at bottom
- Treatment groups in middle
- New_Drug renders last (fully visible, on top)

**Why:** New_Drug is the focus of your study

---

### Example 2: Reveal Hidden Points

**Scenario:**
- You have dense data (many points) for "common" group
- You have sparse data (few points) for "rare" group
- Rare points are hidden under common points

**Data:**
- Group "common": 1000 points
- Group "rare": 5 points

**Initial order (alphabetical):**
```
common (bottom) - many points
rare (top) - 5 points mostly visible
```

**But if you want:**
```
Z-Order: common, rare
```

**Result:**
- common renders first (under rare)
- rare renders last (always visible)
- All 5 rare points now clearly visible

---

### Example 3: Reverse Alphabetical Order

**Scenario:**
You want the exact opposite of alphabetical ordering.

**Default (alphabetical):**
```
A (bottom), B, C, D (top)
```

**Desired order:**
```
D (bottom), C, B, A (top)
```

**Solution:**
```
Z-Order: D, C, B, A
```

**Result:** A renders last (on top)

---

### Example 4: Partial Z-Order

**Scenario:**
You have 5 groups: A, B, C, D, E

You only care about putting E on top. The rest can stay alphabetical.

**Solution:**
```
Z-Order: E
```

**Processing:**
- A, B, C, D render first (alphabetically: A, B, C, D)
- E renders last (on top)

**Result:** A bottom, ..., E on top ✓

---

### Example 5: Multi-Dimensional Color-By

**Scenario:**
You're using 2 color-by dimensions:
- Dimension 1: Department (Sales, Marketing, IT)
- Dimension 2: Level (Junior, Senior)

Compound keys: `Sales~Junior`, `Sales~Senior`, `Marketing~Junior`, etc.

**Default order (alphabetical):**
```
IT~Junior, IT~Senior, Marketing~Junior, Marketing~Senior, Sales~Junior, Sales~Senior
```

**You want to highlight Sales~Senior:**
```
Z-Order: IT~Junior, IT~Senior, Marketing~Junior, Marketing~Senior, Sales~Junior, Sales~Senior
```

Or just put on top:
```
Z-Order: Sales~Senior
```

**Result:** Sales~Senior renders on top

---

## 🛠️ Advanced Techniques

### Technique 1: Layering for Visual Clarity

Goal: Create a visual hierarchy with clear layering

**Strategy:**
```
Input groups: background1, background2, midlayer, highlight
Z-Order:      background1, background2, midlayer, highlight
```

**Visual effect:**
```
Layer 1 (bottom): background1 - barely visible, just sets stage
Layer 2: background2 - contextual reference
Layer 3: midlayer - main comparison
Layer 4 (top): highlight - focal point
```

### Technique 2: Comparative Analysis

Goal: Compare two specific groups by layering them

**Setup:**
```
Groups: baseline, treatment, control, other1, other2
Z-Order: other1, other2, baseline, control, treatment
```

**Result:**
- Others at bottom (context)
- baseline, control in middle (reference)
- treatment on top (focus)

### Technique 3: Outlier Detection

Goal: Always show outliers on top of main data

**Setup:**
```
Groups: inlier_Q1, inlier_Q2, inlier_Q3, outlier
Z-Order: inlier_Q1, inlier_Q2, inlier_Q3, outlier
```

**Result:**
- Outliers always visible (not hidden behind inliers)
- Easy to spot patterns

---

## 📊 Real-World Example

### Scenario: Sales Performance Analysis

**Data Structure:**
```
Date: Jan, Feb, Mar, Apr, May
Sales by Region: North, South, East, West
Color-by: Region
Chart type: Line chart (showing sales trends)
```

**Lines overlapping issue:**
- North (high sales) line often covers West (low sales) line
- Can't see West region trends

**Solution:**

1. **Initial setup:**
   - X: Date (Jan-May)
   - Y: Sales
   - Color-by: Region
   - Type: Line
   - Plot to see issue

2. **Add Z-Order:**
   - Want to see West clearly (it's underperforming - needs focus)
   - Enter: `North, South, East, West`
   - Click Plot

3. **Result:**
   - North line renders first (bottom, faded behind)
   - South, East in middle
   - West line renders last (on top, crisp and clear)
   - Now you can see West's downward trend clearly

**Alternative:** If you want to see North clearly:
```
Z-Order: West, East, South, North
```
Then North renders on top instead.

---

## ✅ Best Practices

### DO
- ✅ **Start simple**: Enter just the group you want on top
- ✅ **Verify exact spelling**: Values are case-sensitive
- ✅ **Check legend**: Legend shows which group is which
- ✅ **Test with 2-3 groups first**: Before complex orderings
- ✅ **Save with recipe**: Preserve your z-order preference

### DON'T
- ❌ **Don't invent values**: Only use values from your legend
- ❌ **Don't forget commas**: `red blue` won't work; use `red, blue`
- ❌ **Don't mix cases**: `Red` ≠ `red`
- ❌ **Don't use special characters** in z-order field itself
- ❌ **Don't assume it persists**: Save settings if you want to keep them

---

## 🔧 Troubleshooting

### Issue: Z-Order has no effect

**Checklist:**
```
1. □ Did I click Plot after entering z-order?
2. □ Are the values spelled exactly as shown in legend?
3. □ Is the case correct? (Legend shows "Red" but I entered "red"?)
4. □ Are values comma-separated?
5. □ Is there only 1 color-by column active?
```

**Solution:**
- Copy the exact text from the chart legend
- Paste into z-order field
- Add comma if multiple values
- Click Plot

### Issue: Some groups still in alphabetical order

**Explanation:**
This is expected! Groups NOT in your z-order list render first (alphabetically).

**Example:**
```
Groups: A, B, C, D, E, F
Z-Order: F
Result: A, B, C, D, E (alphabetical), then F (on top)
```

**To control all groups:**
List all groups in z-order field:
```
Z-Order: A, B, C, D, E, F
```

### Issue: Chart looks wrong after z-order

**Solution:**
```
1. Clear the z-order field (delete all text)
2. Click Plot
3. Chart returns to default (alphabetical) ordering
4. Try again with correct value names
```

---

## 💾 Persistence

### Saving Your Work

**Option 1: Save as Recipe** (includes z-order)
- Saves: parse regex + all plot settings including z-order
- Click: "Save Recipe" button
- Restore: Click "Load Recipe"

**Option 2: Save as Session** (includes z-order)
- Saves: all rows + settings including z-order
- Click: "Save Session" button
- Restore: Click "Load Session"

**Option 3: Browser Storage**
- Z-order persists while browser tab open
- Lost on refresh unless saved in recipe/session

---

## 🎓 Learning Path

### Beginner
1. Load a simple CSV (3-4 rows, 2-3 color-by values)
2. Plot with default (alphabetical) ordering
3. Enter one value in z-order field (the one you want on top)
4. Click Plot → see it move to top

### Intermediate
1. Work with 5+ color-by values
2. Practice entering 2-3 values in specific order
3. Observe how rendering order changes
4. Understand: unlisted groups render first (alphabetical)

### Advanced
1. Use with multi-dimensional color-by
2. Combine with filtering/intervals to isolate focus groups
3. Save multiple recipes with different z-orders
4. Use for comparative analysis (side-by-side recipes)

---

## 📞 Quick Reference

| Task | Action |
|------|--------|
| Put one group on top | `Z-Order: groupname` |
| Control order of 3 groups | `Z-Order: first, second, third` |
| Return to alphabetical | Clear z-order field, click Plot |
| Check current values | Look at chart legend |
| Multi-dim color | Use: `dim1~value1, dim2~value2` |
| Save preference | Click "Save Recipe" or "Save Session" |

---

**Version:** 1.0  
**Feature Status:** ✅ Production Ready  
**Last Updated:** May 21, 2026
