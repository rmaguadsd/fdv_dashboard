# Split-Chart Grouping - Quick Visual Guide

## What Changed

### Before: No Grouping
```
Y-axis
  |     • •  •  •
  |   •  •   •  •  •
  |  •     •  •    •
  |___________________________
     0  1  2  3  4  X-axis (WL)
```

### After: With Pagetype Grouping
```
Y-axis
  |    UP    |    TP
  |     • •  |•  •
  |   •  •   |  •  •  •
  |  •     • |•  •    •
  |__________|__________
     0 1 2 3 4 0 1 2 3 4  X-axis (WL)
            UP       TP
        [Group Labels]
        ─── ───  ─── ───
     [Dashed Separators]
```

## How to Use

### Step 1: Load Data
- CSV file with columns: Y measurement, X (WL), grouping column (pagetype)
- Example:
  ```
  measurement,WL,pagetype
  42.5,0,UP
  45.3,1,UP
  48.2,2,UP
  41.2,0,TP
  44.8,1,TP
  47.5,2,TP
  ```

### Step 2: Configure Chart
1. Set Y = measurement
2. Set X = WL
3. Click **[+ Add]** button to add grouping
4. In new layer, select pagetype
5. Click Render

### Step 3: Interpret Results
- **X-axis ticks** (0, 1, 2, 3, 4): Primary measurement dimension
- **Dashed lines**: Visual separation between groups
- **Bottom labels** (UP, TP): Group category names
- **Points**: Color-coded by series, x/y positioned by measurements

## Key Components

### 1. Visual Separator Lines
```
      Dashed line
          ↓
Chart:   •  |  •
         •  |  •
      ──────┼──────
        UP  |  TP
```
- Divides chart into clear sections
- Positioned at group boundaries
- Dashed pattern (not solid)

### 2. Group Tier Labels
```
                UP              TP
           ┌─────────┐    ┌─────────┐
X-axis: 0 1 2 3 4  |  0 1 2 3 4
            [Centered under each group's range]
```
- Positioned below x-axis
- Centered under each group
- Light background for visibility

### 3. Data Point Positioning
```
Same X coordinate for all pagetype groups:
- WL=0, UP:  one set of y-values
- WL=0, TP:  another set of y-values
- WL=1, UP:  one set of y-values
- WL=1, TP:  another set of y-values
...
```

## Configuration Panel

### X-Axis Controls
```
┌─────────────────────────────────┐
│ Primary X Configuration          │
├─────────────────────────────────┤
│ Column: [WL ▼]                   │
│ Filter:  [regex input]           │
│          [+ Add grouping layer]  │
│                                  │
│ Grouping Layer 1                 │
│ Column: [pagetype ▼]             │
│ Filter: [regex input]            │
│         [× Remove]               │
│                                  │
│ Grouping Layer 2                 │
│ Column: [status ▼]               │
│ Filter: [regex input]            │
│         [× Remove]               │
└─────────────────────────────────┘
```

## Example Dataset

### Sample Data
```
measurement | WL | pagetype | date
────────────────────────────────────
     42.5   | 0  |   UP     | 2024-01-01
     45.3   | 1  |   UP     | 2024-01-02
     48.2   | 2  |   UP     | 2024-01-03
     41.2   | 0  |   TP     | 2024-01-01
     44.8   | 1  |   TP     | 2024-01-02
     47.5   | 2  |   TP     | 2024-01-03
```

### Rendered Chart
```
Y (measurement)
50 |         
   |    UP    |    TP
   |   •       |  •
45 |  • •      | • •
   | •    •    |•    •
40 |
   |___________|_________
     0 1 2 3 4 0 1 2 3 4  X (WL)
          UP         TP
```

## Troubleshooting

### Labels Don't Show
- Check browser console: `[groupSeparators]` messages
- Verify group column has distinct values
- Ensure proper x-value range

### Separators Missing
- Console should show: `Drawing separators for N groups`
- If N=1, only one group detected - check grouping column
- Verify data loaded correctly

### Chart Layout Issues
- Extra space added for labels (50px bottom padding)
- If labels overlap, adjust font size or group spacing
- Responsive: resize window to see dynamic adjustment

## Browser Console Debug

Enable console (F12) to see:
```javascript
[_buildGroupedXAxis] Built 2 groups: UP:0,1,2,3,4 | TP:0,1,2,3,4
[groupSeparators] Drawing separators for 2 groups
[groupSeparators] Group 1 UP - drawing separator at x=2.0 (pixel=234)
[groupSeparators] Drawing group tier labels at y=450
[groupSeparators] Group label "UP" at pixel x=150
[groupSeparators] Group label "TP" at pixel x=350
```

## Performance Notes

- Large datasets (10K+ points): May need sampling
- Many groups (10+): Consider color-coding instead
- Wide x-range: Separators automatically scale
- Responsive rendering: Updates on window resize

## Comparison with Alternatives

### Option 1: Multiple Separate Charts
- **Pro**: Clear separation
- **Con**: Hard to compare across groups
- **This approach**: Single chart with visual markers

### Option 2: Color-code by Group
- **Pro**: Simpler implementation
- **Con**: Hard to read with many series
- **This approach**: Combines color + spatial separation

### Option 3: Dropdown to Filter Groups
- **Pro**: Cleaner interface
- **Con**: Can't see all groups simultaneously
- **This approach**: Always visible, organized

## Next Steps

1. Test with your data
2. Adjust group ordering if needed
3. Customize separator styling (colors, patterns)
4. Consider adding multi-level grouping
5. Export configuration for reproducibility
