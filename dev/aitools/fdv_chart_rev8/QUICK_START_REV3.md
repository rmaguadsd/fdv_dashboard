# Rev3 Quick Reference Guide

## 🖥️ FONT CONTROLS
Located in the plot control panel, Font row:

```
Font: Axis [12] px  |  Labels [10] px  |  Points [8] px
```

**What it does:**
- **Axis**: Changes the size of X and Y axis titles
- **Labels**: Changes the size of axis tick numbers/labels  
- **Points**: Changes the size of text labels on data points

**Range**: Axis (8-32), Labels (6-28), Points (4-20)

---

## 📝 TEXT ANNOTATIONS
Located in the plot control panel, Text row:

### Basic Usage:
```
x=100:y=0.5:Peak
```

### With Color:
```
x=100:y=0.5:Peak:red
x=100:y=0.5:Peak:#FF0000
```

### Multiple Annotations (comma-separated):
```
x=100:y=0.5:Peak, x=200:y=0.3:Valley, x=150:y=0.8:Threshold
```

**What to know:**
- X and Y values are in data coordinates (same scale as your plot)
- Text is optional; you can add just text at a location
- Colors can be CSS names (red, blue) or hex codes (#FF0000)
- Press Enter or click a marker tag to remove individual annotations
- Click "Clear" to remove all annotations at once

---

## —— MARKERS (ENHANCED)
Located in the plot control panel, Markers row:

### Basic Format:
```
AXIS=VALUE:LABEL:CHART_TARGET
```

Where:
- **AXIS**: `x` or `y`
- **VALUE**: numeric value on that axis
- **LABEL**: optional descriptive label (Threshold, UCL, etc.)
- **CHART_TARGET**: optional chart name (if specified, marker only appears on that chart)

### Examples:

| Input | Result |
|-------|--------|
| `x=100` | Vertical line at x=100 |
| `x=100:Threshold` | Vertical line with label "Threshold" |
| `x=100:Threshold:Chart1` | Vertical line on Chart1 only |
| `y=0.5` | Horizontal line at y=0.5 |
| `y=0.5:Upper Limit:all` | Horizontal line on all charts |

### Multiple Markers (comma-separated):
```
x=100:Lower, y=0.5:Threshold, x=200:Upper:Chart2
```

### Features:
- Labels automatically appear next to the marker line
- Colors auto-cycle through a palette
- Press Enter to add
- Click the ✕ in a marker tag to remove it individually
- Click "Clear markers" to remove all at once

---

## SNAPSHOT & PERSISTENCE

All three features (fonts, markers, text) are saved when you:
1. Change chart view
2. Load a recipe
3. Export/save current configuration

They will automatically restore when:
1. You come back to the same data
2. You load the same recipe
3. You refresh the page with the same data loaded

---

## QUICK WORKFLOW EXAMPLE

1. **Load data**: Upload your FDV log file
2. **Resize fonts**: Adjust Font controls if text is too small/large
3. **Add markers**: Use Markers row to highlight key values
   - `x=27000:Test Limit`
   - `y=0.99:99% Threshold`
4. **Annotate**: Use Text row to label specific regions
   - `x=26500:y=50000:Safe Zone:green`
   - `x=27500:y=60000:Danger Zone:red`
5. **Save**: Recipe is automatically saved with all annotations

---

## TROUBLESHOOTING

**Text/markers not showing:**
- Check that coordinates match your plot range
- Verify X and Y are separated by `:` (colon)
- For markers, use format: `x=value:label`

**Font changes not applying:**
- Fonts only change when you redraw (press Enter or change plot)
- Default values: Axis 12, Labels 10, Points 8

**Annotation disappeared:**
- Check you didn't accidentally clear them
- Use "Clear" button cautiously
- Refresh page to restore from snapshot if lost

**Port 5059 not responding:**
- Verify server is running: `python3 fdv_chart.py 5059`
- Check no other process is using port 5059
- Try different port: `python3 fdv_chart.py 5060`

---

## KEYBOARD SHORTCUTS

| Action | Key |
|--------|-----|
| Add marker | Enter in Markers input |
| Add text annotation | Enter in Text input |
| Add font size | (Auto-applies on change) |
| Remove marker | Click ✕ on marker tag |
| Remove text | Click ✕ on text tag |
| Clear all markers | Click "Clear markers" button |
| Clear all text | Click "Clear" button |

---

**Rev3 Server**: http://localhost:5059  
**Version**: 1.0 (Rev3)  
**Last Updated**: April 21, 2026
