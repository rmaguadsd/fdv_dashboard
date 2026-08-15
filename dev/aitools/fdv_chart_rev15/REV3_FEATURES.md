# FDV Chart Rev3 - Enhancement Summary

## Overview
Rev3 has been enhanced with three major feature additions to the FDV Chart Parser:

## 1. Font Resizing Controls ✅

### Features Added:
- **Axis Font Size** - Adjust the font size of axis titles (default: 12px, range: 8-32px)
- **Label Font Size** - Adjust the font size of axis tick labels (default: 10px, range: 6-28px)
- **Point Font Size** - Adjust the font size of point labels (default: 8px, range: 4-20px)

### How to Use:
1. Look for the "Font" row in the plot control panel (marked with "🖥" icon)
2. Adjust any of the three font size sliders
3. Changes apply immediately when you modify the value

### Implementation Details:
- Font controls are integrated into `getAxisScale()` function
- Point labels use the `fdvPointLabels` plugin which reads the font-point value
- Changes persist within session snapshots

---

## 2. Text Annotations ✅

### Features Added:
- Add freeform text annotations anywhere within the chart
- Specify exact X and Y coordinates for text placement
- Optional color customization for annotations

### Format:
```
x=X_VALUE:y=Y_VALUE:TEXT[:COLOR]
```

### Examples:
```
x=100:y=0.5:Peak
x=50:y=200:Error:red
x=75:y=1000:Threshold:#3366cc
```

### How to Use:
1. Find the "Text" row in the plot control panel (marked with "📝" icon)
2. Enter text in the format: `x=100:y=0.5:My Label`
3. Optionally add a color as the last component
4. Press Enter to add the annotation
5. Click the ✕ button next to a tag to remove it
6. Use "Clear" button to remove all text annotations

### Implementation Details:
- Text annotations stored in `_texts` array
- Persisted in snapshots via `__texts` JSON field
- Each text has: x, y, text, and color properties
- Rendered using chart.js annotation plugin labels

---

## 3. Enhanced Marker System ✅

### Features Added:
- **Marker Labels** - Add descriptive labels to markers (e.g., "Threshold", "UCL")
- **Chart Targeting** - Optionally apply markers to specific charts only (for multi-chart views)

### Format:
```
x=VALUE[:MARKER_LABEL[:CHART_ITEM]]
y=VALUE[:MARKER_LABEL[:CHART_ITEM]]
```

### Examples:
```
x=10                          # Marker at x=10, no label
x=10:Threshold                # Marker at x=10 with label "Threshold"
x=10:Threshold:Chart1         # Marker at x=10 only on Chart1
y=0.5:Confidence Bound:all    # Horizontal marker on all charts
```

### How to Use:
1. Find the "Markers" row in the plot control panel (marked with "—" icon)
2. Enter marker in the format: `x=VALUE:LABEL:CHART`
3. Press Enter to add
4. Labels are displayed on the marker line in the chart
5. Chart item specification allows targeting specific charts
6. Click the ✕ button in a marker tag to remove it
7. Use "Clear markers" button to remove all

### Implementation Details:
- Enhanced `_parseMarkerExpr()` to parse new format
- Marker object contains: axis, value, markerLabel, chartItem, color
- Display labels updated in `_rebuildMarkerTags()`
- Annotations built with `_buildAnnotations()` using markerLabel

---

## Server Configuration

### Running Rev3:
```powershell
# Using the startup script
.\start_rev3.ps1

# Or directly with Python 3:
python3 fdv_chart.py 5059

# Port can be overridden:
python3 fdv_chart.py 5060
```

### Default Settings:
- **Port**: 5059 (Rev3 dedicated)
- **Store Directory**: D:\FDV\recipes
- **Python Version**: 3.12+ required (Python 2.7 not supported)

---

## Technical Architecture

### Client-Side Components:
- **HTML Controls**: Font controls row, Text annotation row, Enhanced marker row
- **JavaScript Functions**:
  - `_setPointLabels()` - Point label rendering
  - `_parseMarkerExpr()` - Marker expression parsing
  - `_parseTextExpr()` - Text expression parsing
  - `_buildAnnotations()` - Chart.js annotation building
  - `_buildTextAnnotations()` - Text annotation building
  - `getAxisScale()` - Axis configuration with fonts

### Data Persistence:
- All features persist through snapshots
- Snapshot keys: `__markers`, `__texts`, `__font_axis`, `__font_label`, `__font_point`
- Session restoration automatically loads all saved values

---

## Browser Compatibility
- Modern browsers with Canvas support (Chrome, Firefox, Edge)
- Requires JavaScript enabled
- Works with local file uploads and network paths

---

## Version Information
- **Rev3 Port**: 5059
- **Base Version**: Rev2 (1806 lines of Python)
- **Date Launched**: April 21, 2026
- **Python**: 3.12.8
- **Chart.js**: 4.4.3
- **Plugin**: chartjs-plugin-annotation 3.0.1

---

## Feature Comparison

| Feature | Rev1 | Rev2 | Rev3 |
|---------|------|------|------|
| Basic Plotting | ✅ | ✅ | ✅ |
| Markers (x/y lines) | ✅ | ✅ | ✅ |
| Marker Labels | ❌ | ❌ | ✅ |
| Marker Chart Targeting | ❌ | ❌ | ✅ |
| Text Annotations | ❌ | ❌ | ✅ |
| Resizable Fonts | ❌ | ❌ | ✅ |
| Split Charts | ✅ | ✅ | ✅ |
| AI Analysis (Ollama) | ✅ | ✅ | ✅ |
| Chat Interface | ✅ | ✅ | ✅ |

---

## Next Steps
To use Rev3:
1. Open http://localhost:5059 in your browser
2. Upload or select a log file
3. Use the Font controls to adjust text sizes
4. Add markers with: `x=VALUE:LABEL:CHART`
5. Add text annotations with: `x=X:y=Y:TEXT:COLOR`
6. All features work with existing snapshot/recipe system
