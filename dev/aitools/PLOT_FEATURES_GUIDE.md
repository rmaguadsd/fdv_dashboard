# Interactive Plot Features Guide

## Overview

The XY Plot viewer now includes powerful interactive features for data visualization and exploration. All plots are generated using **Plotly**, an interactive plotting library with rich interactivity.

---

## 1. Resizable Plot

### How to Adjust Plot Size

After generating a plot, you'll see the **"Plot Size Controls"** section:

```
Width: [1000] px    Height: [600] px    [Apply Size]
```

**Steps to resize:**

1. Modify the **Width** value (400-2000 px)
2. Modify the **Height** value (300-1500 px)
3. Click the **"Apply Size"** button
4. The plot will automatically resize and reflow data points to fit the new dimensions

### Visual Resize

You can also directly resize the plot container by:
- **Dragging the resize handle** (bottom-right corner of the plot area)
- The data points will **auto-adjust** to the new dimensions
- Plot elements scale proportionally

### Why This is Useful

- **Better fit for different screens**: Adjust to your monitor size
- **Focus on data**: Make the plot larger to see fine details
- **Space constraints**: Make it smaller to fit more content on page
- **Presentation**: Optimize size for screenshots or sharing

---

## 2. Legend Features

### What is a Legend?

A legend is a visual reference that explains what each color/pattern in the plot represents. In your plots, legends show:

- **Selection names** (e.g., "THES_QLC", "THES_SSLC")
- **Color-coding dimensions** (e.g., "vcc=1.2", "temp=25C")
- **Data groupings** when using the "Color data points by" option

### Enabling/Disabling Legend

The **"Show Legend"** checkbox controls legend visibility:

```
☑ Show Legend    ☑ Legend Clickable
```

- **Checked (default)**: Legend is displayed
- **Unchecked**: Legend is hidden to maximize plot area

### Interactive Legend

When **"Legend Clickable"** is enabled, you can:

1. **Click a legend entry** to toggle visibility of that data series
2. **Double-click** to isolate that series (all others hidden)
3. **Right-click** to show only that series
4. Makes it easy to focus on specific data subsets

### Example Scenarios

**Scenario 1: Too many colors on plot**
- Solution: Click legend entries to hide less important groups
- Reduces visual clutter while keeping data available

**Scenario 2: Comparing two specific selections**
- Solution: Double-click each selection name in legend to isolate them
- All other data hidden temporarily

**Scenario 3: Understanding what each color means**
- Solution: Hover over legend entry to highlight that data series on plot
- All points from that series highlight together

---

## 3. Color-Coding Data Points

### How to Color Data

Use the **"Color data points by (optional)"** dropdown:

```
Color data points by: [-- No color coding --]
                      - FDV
                      - VCC
                      - TEMP
                      - Status
                      - Plane
                      - Pagetype
                      - PR
                      - WL
                      - BLK
                      - TM
                      - Value
```

**Steps:**

1. Select a dimension from the dropdown (e.g., "VCC")
2. Click "Generate Plot"
3. Data points will be colored by unique values of that dimension
4. A legend appears showing color → value mapping

### When to Use Color-Coding

| Dimension | Use Case |
|-----------|----------|
| **FDV** | Distinguish between different test types |
| **VCC** | Show voltage variations across data |
| **TEMP** | Highlight temperature dependencies |
| **Status** | Group by pass/fail or device status |
| **Plane** | Separate memory planes |
| **Value** | Visualize numeric range (blue→red) |

---

## 4. Splitting Plots by Selection

### What is "Split plots by selection"?

When enabled, creates **separate subplots** (one per selected item) instead of combining everything on one plot.

### How to Use

Check the **"Split plots by selection"** checkbox:

```
☑ Split plots by selection
```

Then click "Generate Plot"

### Result

Instead of:
```
Single plot with all THES_QLC, THES_SSLC, and THES_MLC data mixed
```

You get:
```
Subplot 1: THES_QLC data only
Subplot 2: THES_SSLC data only  
Subplot 3: THES_MLC data only
```

### Benefits

- **Clearer patterns**: Each subset's trends visible without overlap
- **Easier comparison**: Subplots have consistent scale
- **Reduced clutter**: Less visual complexity per subplot

---

## 5. Interactive Toolbar

The plot toolbar appears in the **top-right corner** with these tools:

| Tool | Function | Usage |
|------|----------|-------|
| **📷** | Download Plot | Save current plot as PNG image |
| **🔍+** | Zoom In | Zoom to a box you draw |
| **🔍-** | Zoom Out | Zoom out to full view |
| **🔄** | Pan | Click and drag to move around |
| **↺** | Reset | Return to original zoom/pan |
| **⚙️** | Settings | Adjust hover, animation, etc. |

### Zoom & Pan Example

1. Click **Zoom** button (magnifying glass)
2. Draw a box around data region of interest
3. Plot zooms into that region
4. Click **Reset** to go back to full view

---

## 6. Hover Tooltips

### What Happens When You Hover

Position your mouse over any data point to see a tooltip showing:

```
Selection: THES_QLC
Color Field: vcc=1.2
X Axis: value=2500
Y Axis: fail_count=45
```

### Information Displayed

Depends on plot configuration:
- **Always shows**: X and Y field values
- **If color-coded**: The color dimension value
- **If split**: Selection/subplot name

### Why This is Useful

- **Exact values**: No need to estimate from grid
- **Identify outliers**: Quickly see which point has extreme value
- **Verify clustering**: Confirm data grouping makes sense

---

## 7. Complete Workflow Example

### Scenario: Analyzing Temperature Effects on Test Results

**Steps:**

1. **Generate initial plot**
   - X Axis: `value` (test measurement)
   - Y Axis: `fail_count` (failures observed)
   - Color by: `TEMP` (to see temperature effects)
   - Click "Generate Plot"

2. **Adjust size for better viewing**
   - Width: 1200 px
   - Height: 800 px
   - Click "Apply Size"

3. **Explore the data**
   - Hover over high-fail points to see temperature
   - Notice pattern: higher temps = more failures

4. **Focus on specific temperature**
   - Click "Legend Clickable" checkbox (if not already enabled)
   - Double-click a temperature color in legend to isolate it

5. **Compare across selections**
   - Uncheck "Split plots by selection" (all together)
   - Notice which FDV has better performance at each temp

6. **Export findings**
   - Click download button (📷) to save plot
   - Use in report or presentation

---

## 8. Legend Customization Tips

### Hide Legend to See Full Plot
```
☐ Show Legend    (legend disappears)
```

### Make Legend Clickable for Filtering
```
☑ Legend Clickable    (click entries to toggle/isolate)
```

### Combine With Color-Coding for Maximum Info
```
Color by: VCC
☑ Show Legend
☑ Legend Clickable
```
Result: Click legend to show only specific voltage values

---

## 9. Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Plot too small to see details** | Increase Width/Height in size controls |
| **Too many colors, can't distinguish** | Hide non-essential legend entries by clicking them |
| **Can't remember what each color means** | Hover over legend entry to highlight that data |
| **Plot won't resize** | Click "Apply Size" button after changing values |
| **Legend taking up too much space** | Uncheck "Show Legend" to hide it |

---

## 10. Keyboard Shortcuts (Plotly Built-in)

| Shortcut | Action |
|----------|--------|
| **Scroll wheel** | Zoom in/out on plot |
| **Click + Drag** | Pan around the plot |
| **Double-click** | Reset zoom to full view |
| **Shift + Drag** | Create zoom box |

---

## Summary

| Feature | Keyboard/Mouse | Effect |
|---------|----------------|--------|
| **Resize Plot** | Width/Height inputs + Apply button | Plot scales data points |
| **Show/Hide Legend** | Checkbox | Toggle legend visibility |
| **Toggle Legend Clickable** | Checkbox | Enable interactive filtering |
| **Color Data Points** | Dropdown selection | Colorize by dimension |
| **Split by Selection** | Checkbox | Create subplots |
| **Hover** | Mouse over point | See exact values |
| **Zoom** | Toolbar / Scroll wheel | Magnify region |
| **Pan** | Toolbar / Click-drag | Move around plot |
| **Download** | Toolbar 📷 button | Save as PNG |

---

## Need Help?

- **Hover over the info icon** (ℹ️) near controls for quick tips
- **Check the success message** after generating plot for feature tips
- **Try interactive legend** - click entries to explore data subsets
- **Resize plot** to make patterns more visible

Enjoy exploring your data! 📊
