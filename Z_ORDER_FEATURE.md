# Z-Order Control for Color-By Groups

## Overview

The **Z-Order control** allows you to specify which color-by groups render **on top** of others when data points overlap in scatter, line, histogram, and other chart types.

## Where to Find It

In the Plot Panel, look for the **Z-Order** input field (labeled "Z-Order: (last = top)") right after the Color-by selector.

```
Color by: [column dropdown]  +
Z-Order:  [text input]  (last = top)
```

## How It Works

**Z-Order determines rendering sequence:**
- Groups **NOT listed** in Z-Order render **first** (bottom layer)
- Groups **listed in Z-Order** render in **that sequence**
- **Last group in list** renders **on top** (fully visible)

### Example

If you have color-by values: `red`, `blue`, `green`, `yellow`

**Z-Order Input:** `blue, yellow, red`

**Rendering Order (bottom to top):**
1. `green` (not listed → renders first/bottom)
2. `blue` (first in z-order)
3. `yellow` (second in z-order)
4. `red` (last in z-order → on top)

## Usage Guide

### Basic Usage

1. **Parse and plot your data** with a color-by column
2. **Enter color values in Z-Order field**, comma-separated:
   ```
   Value1, Value2, Value3
   ```
3. **Click Plot** to apply the new rendering order

### Tips

- **Case-sensitive**: Values must match exactly as they appear in the chart legend
- **Whitespace handling**: Spaces around values are trimmed automatically
- **Partial lists**: Only list the groups you want to control; others render first in alphabetical order
- **Empty field**: If Z-Order is blank, all groups render in alphabetical order (default behavior)

### Examples

**Example 1: Ensure "important" group is always visible**
```
Z-Order: other, baseline, important
```
Result: `important` renders on top

**Example 2: Highlight a specific treatment**
```
Z-Order: control, control_2, treatment_a, treatment_b, highlight
```
Result: `highlight` appears on top of all other groups

**Example 3: Reverse alphabetical rendering**
```
Your groups: A, B, C, D (alphabetically first=A bottom, D=top by default)
Z-Order: D, C, B, A
```
Result: A renders on top (opposite of default)

## Technical Details

### Sorting Algorithm

The z-order sorting works as follows:

1. **Parse** the z-order input (comma-separated values)
2. **For each color group**, determine its position:
   - If **NOT in z-order list**: render position = early (sorted alphabetically among unlisted)
   - If **in z-order list**: render position = index in list (0 = second to bottom, higher = more on top)
3. **Sort datasets** by this calculated position
4. **Chart.js renders** in dataset order (first dataset = bottom, last dataset = top)

### Chart Types Affected

Z-Order works on all chart types:
- ✅ **Scatter** - Points on top of other points
- ✅ **Line** - Lines layering (with showLine option)
- ✅ **Histogram** - Bars on top of other bars
- ✅ **Cumulative Probability** - Lines layering
- ✅ **RCDF** - Lines layering
- ✅ **Box & Whisker** - When overlay is enabled

### Multi-Dimensional Color-By

If you're using multiple color-by dimensions (Click `+` to add):
- Z-Order applies to the **compound key** (combination of all dimensions)
- Example with 2 dimensions:
  ```
  Value1~Value2, Value3~Value4, Value1~Value3
  ```

## Resetting Z-Order

To return to **default alphabetical ordering**:
1. **Clear the Z-Order field** (delete all text)
2. **Click Plot**

## Common Use Cases

### 1. Highlight Outliers
```
Z-Order: normal, normal_high, normal_low, outlier
```
Outliers render on top, making them clearly visible

### 2. Compare Two Treatments
```
Z-Order: control, treatment1, treatment2
```
treatment2 on top for easy visual comparison

### 3. Focus on Sparse Group
```
Z-Order: dense_group1, dense_group2, sparse_group
```
Sparse group on top so individual points are visible

### 4. Emphasize Positive vs Negative
```
Z-Order: negative, zero, positive
```
Positive results on top (most visible)

## Troubleshooting

**Problem:** Z-Order not taking effect  
**Solution:** Verify the values exactly match the chart legend (case-sensitive, whitespace matters)

**Problem:** Some groups missing from chart  
**Solution:** Only listed values should appear; if a group is hidden, check:
- Is it filtered out by intervals?
- Does the regex exclude it?
- Is it in a different split-chart?

**Problem:** Wrong group on top  
**Solution:** Check the order in your Z-Order input; the **last value** renders on top

## Technical Implementation

**File:** `dev/aitools/fdv_chart_rev11/fdv_chart.html`  
**Function:** `drawScatterLine()` (lines 5095-5120)  
**Key Code:** Z-Order parsing and sorting algorithm before dataset creation

The implementation:
1. Reads z-order input field (`#z-order-input`)
2. Parses comma-separated values
3. Sorts group keys with custom comparator
4. Builds datasets in sorted order
5. Chart.js renders in dataset sequence

---

**Last Updated:** May 21, 2026  
**Feature Status:** ✅ Production Ready
