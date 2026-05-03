# Split-Chart Grouping - Visual Diagrams

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ CSV INPUT DATA                                              │
├─────────────────────────────────────────────────────────────┤
│ measurement | WL | pagetype                                 │
│ ─────────────────────────────────────────────────────────── │
│     42.5    | 0  |   UP                                     │
│     45.3    | 1  |   UP                                     │
│     48.2    | 2  |   UP                                     │
│     41.2    | 0  |   TP                                     │
│     44.8    | 1  |   TP                                     │
│     47.5    | 2  |   TP                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ User configures:                      │
        │ Y = measurement                       │
        │ X = WL                                │
        │ Add grouping: pagetype                │
        └───────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ _buildGroupedXAxis() PROCESSES:                             │
├─────────────────────────────────────────────────────────────┤
│ Input: plotPts, xCol='WL', xGroupDims=[{col:'pagetype'}]  │
│                                                             │
│ Process:                                                    │
│ 1. Extract WL values: 0, 1, 2, 0, 1, 2                    │
│ 2. Extract pagetype: UP, UP, UP, TP, TP, TP               │
│ 3. Build groupsSet:                                         │
│    { UP: {0: true, 1: true, 2: true},                      │
│      TP: {0: true, 1: true, 2: true} }                     │
│ 4. Sort x-values numerically: [0, 1, 2]                   │
│ 5. Sort groups by min x-value                              │
│                                                             │
│ Output:                                                     │
│ {                                                           │
│   groups: [                                                 │
│     { name: 'UP', xValues: ['0','1','2'] },               │
│     { name: 'TP', xValues: ['0','1','2'] }                │
│   ],                                                        │
│   groupMap: { '0':'UP', '1':'UP', '2':'UP', ... }         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ POINT DATA ENHANCED:                                        │
├─────────────────────────────────────────────────────────────┤
│ dp = {                                                      │
│   x: 0.15,          (jittered position for scatter)        │
│   y: 42.5,          (Y value)                              │
│   _xVal: '0',       (original X value)  ← NEW              │
│   _groupName: 'UP', (group membership)  ← NEW              │
│   _lbl: '[UP]'      (display label)     ← NEW              │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CHART OPTIONS CREATED:                                      │
├─────────────────────────────────────────────────────────────┤
│ Chart config:                                               │
│ {                                                           │
│   type: 'scatter',                                          │
│   data: { datasets: [...] },                               │
│   options: {                                                │
│     layout: {                                               │
│       padding: {                                            │
│         bottom: 50  ← Space for tier labels               │
│       }                                                     │
│     },                                                      │
│     scales: {                                               │
│       x: {                                                  │
│         ticks: {                                            │
│           callback: function(v) {                          │
│             return String(v);  ← Just '0','1','2'         │
│           }                                                 │
│         }                                                   │
│       }                                                     │
│     }                                                       │
│   }                                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ groupSeparators PLUGIN ACTIVATED:                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ PHASE 1: Draw Separator Lines                              │
│ For each group boundary:                                    │
│   Calculate: lineX = (maxXPrev + minXCurrent) / 2          │
│   Position:  pixelX = xScale.getPixelForValue(lineX)       │
│   Draw:      ctx.moveTo(pixelX, top)                       │
│              ctx.lineTo(pixelX, bottom)                    │
│              ctx.stroke() [dashed pattern]                 │
│                                                             │
│ PHASE 2: Render Group Tier Labels                          │
│ For each group:                                             │
│   Calculate: centerX = (minX + maxX) / 2                   │
│   Position:  pixelX = xScale.getPixelForValue(centerX)     │
│              pixelY = chartArea.bottom + 25                │
│   Draw:      Background box in #f0f0f0                     │
│              Text label in #333 bold 13px                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ FINAL RENDERED CHART:                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Y(measurement)                                              │
│ 50│                                                         │
│   │    UP    |    TP                                        │
│ 45│   • •    |   • •                                        │
│   │  •   •   |  •   •                                       │
│ 40│  •       •  •                                           │
│   └────────────────────                                     │
│     0 1 2 3 4 0 1 2 3 4  X(WL)                             │
│           UP        TP                                      │
│                                                             │
│ ───────────┼──────────── ← Dashed separator                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison: Before vs After

### BEFORE (Problematic)
```
CONFIGURATION:
Y = measurement
X = WL
Hierarchy by pagetype

AXIS DISPLAY:
X-axis: 0[UP] 1[UP] 2[UP] | 0[TP] 1[TP] 2[TP]
        ↓
        Confusing - category names in axis labels
        Hard to parse
        Points visually scattered

VISUAL:
        •   •       •   •       •
       •   • •     •   •      •   •
      • •          •           •
    ──────────────────────────────── ← No separation


ISSUES:
❌ Axis labels too long and confusing
❌ No visual separation between groups
❌ Hard to understand group boundaries
❌ Points appear continuously distributed
```

### AFTER (Split-Chart)
```
CONFIGURATION:
Y = measurement
X = WL
Add grouping: pagetype

AXIS DISPLAY:
X-axis: 0 1 2 3 4 | 0 1 2 3 4
Groups:    UP      |    TP
        ↓
        Clear - groups shown separately
        Easy to interpret
        Professional appearance

VISUAL:
        •   •       |   •   •       •
       •   •        |  •   •      •   •
      •            |  •           •
    ────────────────┼────────────────────
           UP       |       TP


BENEFITS:
✅ X-axis clean and readable
✅ Clear visual separation (dashed lines)
✅ Group names prominently displayed
✅ Professional split-chart appearance
✅ Intuitive to understand
```

---

## Layout Diagram

```
┌────────────────────────────────────────────────────────┐
│                 CHART CONTAINER                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │          CHART AREA                             │ │
│  │  (responsive: fills container)                  │ │
│  │                                                 │ │
│  │  Y-axis    Data area (plot)    Legend           │ │
│  │    ▲       • • • | • • •                        │ │
│  │    │      •   • | •   •                         │ │
│  │ 45├──────────────┼─────────────────             │ │
│  │    │     •      |  •                            │ │
│  │    │    • •     |• • •                          │ │
│  │ 40├────────────────────────                     │ │
│  │    │                                            │ │
│  │    └────────────────────► X-axis                │ │
│  │         0 1 2 3 4 | 0 1 2 3 4                   │ │
│  │                                                 │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌────────────────────────────────────────────────┐   │
│  │        GROUP TIER LABELS                       │   │
│  │    ┌─────────┐              ┌─────────┐       │   │
│  │    │    UP   │              │   TP    │       │   │
│  │    └─────────┘              └─────────┘       │   │
│  │  (positioned: centerX, y=bottom+25px)         │   │
│  │  (light gray bg, dark text, bold font)        │   │
│  └────────────────────────────────────────────────┘   │
│  (layout.padding.bottom = 50px)                       │
│                                                        │
│                                                        │ 
└────────────────────────────────────────────────────────┘

KEY MEASUREMENTS:
- Chart area: Responsive (fills container)
- Group label Y: chartArea.bottom + 25px
- Group label X: Centered per group (minX+maxX)/2
- Layout padding: 50px bottom for labels
- Dashed line: From top to bottom of chart area
```

---

## Separator Positioning Logic

```
GROUP 1 (UP):              GROUP 2 (TP):
X values: 0,1,2,3,4       X values: 0,1,2,3,4

min=0, max=4              min=0, max=4

Visual:
    0  1  2  3  4         0  1  2  3  4
    |  |  |  |  |         |  |  |  |  |
    UP UP UP UP UP        TP TP TP TP TP
    
                 ↑ Separator calculated at midpoint
              maxUP=4, minTP=0
              lineX = (4 + 0) / 2 = 2.0
              
Pixel position:
    pixelX = xScale.getPixelForValue(2.0)
    
Draw:
    ctx.moveTo(pixelX, chartTop)
    ctx.lineTo(pixelX, chartBottom)
    ctx.setLineDash([4, 4])  ← Dashed pattern
    ctx.stroke()


RESULT:
    0  1  2  | 3  4         0  1  2  3  4
                ↑              |
            Separator      (actually rendered)
            (midpoint)     
                          
    Actual visual:
    0  1  2         0  1  2  3  4
    ──────────┼──────────────
       UP    |      TP
```

---

## Code Flow Diagram

```
User clicks Render
        ↓
    drawScatterLine()
        ↓
    Plot points prepared
        ↓
    _buildGroupedXAxis()  ← Key function
        ├─ Collect points per group
        ├─ Sort x-values numerically
        ├─ Sort groups by min x-value
        └─ Return: { groups, groupMap }
        ↓
    Enhance points with _xVal, _groupName
        ↓
    Create chart datasets
        ↓
    Configure xAxisConfig
        ├─ ticks callback (simplified)
        └─ layout padding (bottom: 50px)
        ↓
    new Chart(ctx, { ... })
        ├─ Create chart instance
        └─ Register plugins (including groupSeparators)
        ↓
    Chart.js rendering pipeline
        ├─ Draw chart background
        ├─ Draw axes and ticks
        ├─ Plot data points (scatter)
        ├─ Draw legend
        ├─ afterDraw hook → groupSeparators plugin
        │   ├─ Draw separator lines
        │   └─ Draw group tier labels
        └─ Render complete chart
        ↓
    Display on screen
        ↓
    User sees split-chart with:
    - Data points
    - Dashed separators
    - Group labels below axis
    - Professional appearance
```

---

## Group Tier Label Rendering

```
For each group in xGrouped.groups:

1. CALCULATE POSITION
   ├─ Get xValues: ['0', '1', '2', '3', '4']
   ├─ Convert to numbers: [0, 1, 2, 3, 4]
   ├─ Find min: 0
   ├─ Find max: 4
   └─ Calculate center: (0 + 4) / 2 = 2.0

2. CONVERT TO PIXEL COORDINATES
   └─ pixelX = xScale.getPixelForValue(2.0)
      (Returns actual pixel position on canvas)

3. PREPARE TEXT DRAWING
   ├─ Text: "UP" (group.name)
   ├─ Font: "bold 13px Arial"
   ├─ Color: "#333"
   └─ Background: "#f0f0f0"

4. DRAW BACKGROUND BOX
   ├─ Measure text width: ctx.measureText("UP")
   ├─ Calculate box dimensions:
   │  ├─ padding: 4px
   │  ├─ width = textWidth + (padding × 2)
   │  └─ height = 18px
   ├─ Position box:
   │  ├─ x = pixelX - (textWidth / 2) - padding
   │  └─ y = labelY - 2
   └─ Fill: ctx.fillRect(x, y, w, h)

5. DRAW TEXT
   └─ ctx.fillText("UP", pixelX, labelY)

RESULT: Text centered at pixelX with background box
```

---

## Example Output to Console

```javascript
When chart renders with grouping:

[_buildGroupedXAxis] Built 2 groups: UP:0,1,2,3,4 | TP:0,1,2,3,4
[groupSeparators] Drawing separators for 2 groups
[groupSeparators] Group 1 UP - drawing separator at x=2.0 (pixel=234)
[groupSeparators] Drawing group tier labels at y=450
[groupSeparators] Group label "UP" at pixel x=150
[groupSeparators] Group label "TP" at pixel x=350

INTERPRETATION:
✅ Groups built successfully (2 groups, UP and TP)
✅ Each group has complete x-value set (0-4)
✅ Separator positioned at x=2.0 (midpoint)
✅ Rendered at pixel 234 (within chart area)
✅ Group labels positioned at x=150 (UP center) and x=350 (TP center)
✅ All rendering operations successful
```

---

## Summary

These diagrams show:

1. **Data flow**: CSV → Processing → Chart → Rendered output
2. **Visual transformation**: Confusing axis → Clear split-chart
3. **Layout components**: Chart area, tier labels, separators
4. **Positioning logic**: How midpoint and center calculations work
5. **Code execution**: Step-by-step rendering pipeline
6. **Console output**: What to expect in browser DevTools

The split-chart grouping creates a professional, visually clear representation of grouped data with distinct visual markers separating categories.
