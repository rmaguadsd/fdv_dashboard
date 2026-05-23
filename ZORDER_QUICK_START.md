# Z-Order Feature - Quick Start Guide

## 🎯 What Does Z-Order Do?

Controls which color-by groups appear **on top** when data points overlap.

## 📍 Where Is It?

Look for the **Z-Order** field in the Plot Panel:

```
┌─ Plot Panel ─────────────────────────────────────────┐
│ X: [col]    Color by: [col]  +                       │
│             Z-Order: [text input]  (last = top)      │
│ Y: [col]                                             │
│             Type: [scatter ▼]                        │
│                                                       │
│ [Plot Button]                                        │
└──────────────────────────────────────────────────────┘
```

## 🚀 How to Use

### Step 1: Set Up Chart
1. Select X and Y columns
2. Select a Color-by column
3. Choose chart type (Scatter, Line, Histogram, etc.)

### Step 2: Enter Z-Order
1. Click in the **Z-Order** field
2. Type color values **comma-separated**:
   ```
   red, blue, green
   ```
3. Click **Plot**

### Step 3: See Results
- **red** renders first (bottom layer)
- **blue** renders second
- **green** renders last (on top, fully visible)

---

## 📊 Common Examples

### Example 1: Emphasize Positive Results
```
Your groups:  negative, neutral, positive
Z-Order:      negative, neutral, positive
Result:       positive points on top ✓
```

### Example 2: Hide One Group Behind Another
```
Your groups:  background, foreground
Z-Order:      foreground, background
Result:       background renders first (behind), then foreground on top
```

### Example 3: Focus on Outliers
```
Your groups:  normal_A, normal_B, outlier
Z-Order:      normal_A, normal_B, outlier
Result:       outliers always visible on top ✓
```

---

## 💡 Important Notes

✓ **Case-sensitive** → "Red" ≠ "red"  
✓ **Spaces are trimmed** → "red , blue" works fine  
✓ **Partial lists OK** → Groups not listed render first (alphabetically)  
✓ **Empty field** → Returns to alphabetical order  
✓ **Multi-dim color-by** → Use compound keys like "value1~value2"

---

## 🎨 Visual Explanation

### Without Z-Order (Alphabetical)
```
┌─────────────────────────┐
│  Blue points on top ▲   │  (B = 3)
│  Green points ▲ ▲       │  (G = 2)
│  Red points ▲ ▲ ▲       │  (R = 1)
└─────────────────────────┘
Order: Red (bottom) → Green → Blue (top)
```

### With Z-Order: red, blue, green
```
┌─────────────────────────┐
│  Green points on top ▲  │  (G = 3)
│  Blue points ▲ ▲        │  (B = 2)
│  Red points ▲ ▲ ▲       │  (R = 1)
└─────────────────────────┘
Order: Red (bottom) → Blue → Green (top)
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Z-Order not working | Check spelling/case in chart legend |
| Groups missing | They might be filtered by intervals |
| Wrong group on top | Last group in list goes on top |
| Returning to alphabetical | Clear z-order field and replot |

---

## 📝 Syntax Reference

### Format
```
value1, value2, value3, ...
```

### Rules
- **Separate by comma**
- **Whitespace optional** (will be trimmed)
- **Order matters** → last value renders on top
- **Case matters** → must match legend exactly

### Examples
```
✓ Good:     red, blue, green
✓ Good:     red , blue , green    (extra spaces OK)
✓ Good:     my_var_1, my_var_2    (underscores OK)
✗ Bad:      red; blue; green      (semicolon won't work)
✗ Bad:      Red, blue, green      (case mismatch if data is lowercase)
```

---

## 🔄 Saving Your Preference

Your z-order setting is stored in **browser memory**:
- **Save with Recipe:** Save the recipe (includes z-order)
- **Save with Session:** Save the session (includes z-order)
- **Browser refresh:** Lost unless saved in recipe/session

---

## 🆘 Need Help?

1. **Check chart legend** → Verify exact color-by values shown
2. **Try simpler order** → Start with 2-3 values
3. **Clear and replot** → Delete z-order and plot to reset
4. **Check console** → Browser dev tools (F12) for errors

---

**Ready to try it? Open the chart and experiment! 🚀**
