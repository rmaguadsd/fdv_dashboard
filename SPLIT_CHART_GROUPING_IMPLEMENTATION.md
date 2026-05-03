# Split-Chart Grouping Implementation

## Overview

Implemented hierarchical x-axis grouping to support split-chart visualization where:
- **Primary X-axis**: Main measurement dimension (e.g., WL - Water Level)
- **Secondary grouping**: Category dimension (e.g., pagetype - UP, TP, etc.)
- **Visual result**: Single merged chart with clear visual separation between groups

## How It Works

### Example Configuration
```
Y = measurement
X = WL                (Primary X-axis)
X-Group = pagetype    (Secondary grouping layer)
```

### Visual Result
The chart displays:
```
X-axis labels:     0    1    2    3    4  |  0    1    2    3    4
Group tier labels:         UP                     TP
                      [separator line]
```

### Key Features

1. **Unified Chart**: All data points plotted on single chart with consistent X values
2. **Visual Separators**: Dashed vertical lines separate groups
3. **Group Labels**: Category names displayed below x-axis ticks
4. **Smart Positioning**: Group labels centered within their respective range

## Implementation Details

### 1. Data Structure (`_buildGroupedXAxis()`)

```javascript
// Returns:
{
  groups: [
    { name: 'UP', xValues: ['0', '1', '2', '3', '4'] },
    { name: 'TP', xValues: ['0', '1', '2', '3', '4'] }
  ],
  groupMap: {
    '0': 'UP', '1': 'UP', '2': 'UP', '3': 'UP', '4': 'UP',
    '0': 'TP', '1': 'TP', '2': 'TP', '3': 'TP', '4': 'TP'
  }
}
```

### 2. Point Data Enhancement

Each data point stores:
```javascript
{
  x: 1.5,              // Jittered position for scatter display
  y: 42.7,             // Y value
  _xVal: '1',          // Original X value (before jitter)
  _groupName: 'UP',    // Group membership
  _lbl: '[UP]'         // Label with group info
}
```

### 3. X-Axis Configuration

- **Ticks**: Display primary X values (0, 1, 2, 3, 4)
- **Formatting**: Group membership shown in point tooltips/labels
- **Layout**: Extra bottom padding (50px) for group tier labels

### 4. Group Separators Plugin

The `groupSeparators` Chart.js plugin:

**Draws separator lines:**
- Vertical dashed lines between group boundaries
- Positioned at midpoint between max of previous group and min of current group
- Only drawn within chart area boundaries

**Renders group tier labels:**
- Centered horizontally within each group's x-value range
- Light background box for readability
- Bold font at 13px size
- Positioned 25px below x-axis

### 5. Chart Layout

```javascript
layout: {
    padding: {
        bottom: 50px  // Extra space for group tier labels
    }
}
```

## UI Integration

### Adding X-Group Layer

The UI provides:
1. **Primary X dropdown**: Select main x-axis column (e.g., WL)
2. **Primary regex**: Optional filter on X values
3. **+ Add button**: Opens grouping layer configuration
4. **Group layers**: Stacked controls for each grouping dimension

```
X-axis controls:
┌─────────────────────────────────────┐
│ [WL ▼] [regex]  [+ Add]             │  Primary X config
├─────────────────────────────────────┤
│ [pagetype ▼] [regex] [× Remove]     │  Grouping layer 1
│ [status ▼] [regex] [× Remove]       │  Grouping layer 2
└─────────────────────────────────────┘
```

### Configuration Storage

Grouping configuration persists in:
```javascript
_xGroupDims = [
  { col: 'pagetype', rx: null }  // First grouping dimension
]
```

Currently only the first grouping dimension is used.

## Console Logging

Debug information available in browser console:

```javascript
// Building groups
[_buildGroupedXAxis] Built 2 groups: UP:0,1,2,3,4 | TP:0,1,2,3,4

// Drawing separators
[groupSeparators] Drawing separators for 2 groups
[groupSeparators] Group 1 UP - drawing separator at x=2.0 (pixel=234)

// Rendering labels
[groupSeparators] Drawing group tier labels at y=450
[groupSeparators] Group label "UP" at pixel x=150
[groupSeparators] Group label "TP" at pixel x=350
```

## Technical Changes

### Modified Functions

1. **`_buildGroupedXAxis()`** (Lines 2545-2619)
   - Groups points by first category dimension
   - Sorts x-values numerically within groups
   - Sorts groups by minimum x-value
   - Returns structure for plugin consumption

2. **Point data preparation** (Lines 5365-5390)
   - Stores `_xVal` and `_groupName` on each data point
   - Maintains separation of jitter offset and true value

3. **X-axis configuration** (Lines 5433-5450)
   - Simplified tick callback (shows only primary X)
   - Extra layout padding for group labels

4. **Chart options** (Lines 5497-5530)
   - Added `layout.padding.bottom` for grouped charts
   - Conditional padding based on group count

### New Plugin: `groupSeparators` (Lines 1208-1305)

Registered Chart.js plugin that:
- Draws dashed separator lines between groups
- Renders group tier labels with background boxes
- Performs bounds checking and coordinate validation
- Provides detailed console logging

## Visual Styling

### Separator Lines
- Color: `#ddd` (light gray)
- Width: 2px
- Pattern: Dashed (4px dash, 4px gap)

### Group Labels
- Background: `#f0f0f0` (very light gray)
- Text color: `#333` (dark gray)
- Font: Bold 13px Arial
- Padding: 4px
- Border: None

## Example Workflow

1. **Load data**: CSV with Y, X (WL), and grouping column (pagetype)
2. **Select Y column**: measurement
3. **Select X column**: WL
4. **Add grouping**: Click [+ Add], select pagetype
5. **Render chart**: 
   - All WL values (0-4) repeated for each pagetype
   - Dashed lines separate pagetype sections
   - Group labels (UP, TP) appear below axis

## Limitations & Future Enhancements

### Current Limitations
- Only first grouping dimension used (multi-level hierarchy not yet supported)
- Fixed group label positioning (50px below axis)
- Group order determined by minimum x-value (not user-configurable)

### Possible Enhancements
1. Support multiple grouping dimensions (nested hierarchies)
2. Customizable separator styling (color, width, pattern)
3. User-configurable group ordering
4. Group-level statistics in labels
5. Interactive group expand/collapse
6. Export grouped configuration with session

## Testing Checklist

- [ ] Load sample CSV with X, grouping, and Y columns
- [ ] Verify x-axis shows primary X values
- [ ] Verify group labels appear below axis
- [ ] Verify dashed separator lines visible
- [ ] Check point labels include group membership
- [ ] Test with 2+ grouping values
- [ ] Test point sizing and visibility
- [ ] Test tooltip display
- [ ] Verify console logs show group structure
- [ ] Test responsiveness (resize window)

## Files Modified

- `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev7\fdv_chart.html`
  - Lines 1208-1305: Enhanced groupSeparators plugin
  - Lines 5365-5390: Point data enhancement
  - Lines 5433-5450: X-axis configuration
  - Lines 5497-5530: Chart layout padding

## Session Persistence

Grouping configuration automatically saved/restored with session:
```javascript
_xGroupDims = JSON.parse(localStorage.getItem('xGroupDims') || '[]');
```

When user adds grouping layers, configuration persists across page reloads.
