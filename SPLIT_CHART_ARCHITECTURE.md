# Architecture: Split-Chart vs Previous Approach

## Previous Approach (Hierarchical Labels)

### Structure
```javascript
// Old: Both dimensions used for point POSITIONING
X = WL       (primary positioning)
H = pagetype (hierarchical label, not positioning)

Result: Points clustered per unique WL value
        Labels show hierarchy within each X position
```

### Visual Result
```
X-axis:      0        1        2
             LP      LP      LP
          •   •   •   •   •   •
          UP  TP  UP  TP  UP  TP
[Wrong interpretation: points bunched]
```

### Code Pattern
```javascript
// Hierarchical: Group name shown in LABEL
dp._lbl = (dp._lbl ? dp._lbl + ' ' : '') + '[LP]';

// X-axis ticks: Show category in parentheses
callback: function(v) {
    var groupName = xGrouped.groupMap[v];
    return groupName ? v + ' [' + groupName + ']' : v;
}
```

### Issues
- Confusing x-axis (0 [LP], 1 [LP], 0 [UP], 1 [UP])
- No clear visual separation between categories
- Hard to distinguish independent sections
- Points visually cluttered

---

## New Approach (Split-Chart Grouping)

### Structure
```javascript
// New: Separate visual sections for each group
X = WL           (positioning within section)
Split by pagetype (section boundaries)

Result: X-axis values repeated per group
        Clear visual separation
        Group labels below axis
```

### Visual Result
```
X-axis:    0 1 2 3 4  |  0 1 2 3 4
Groups:       UP           TP
          ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
         • • •  |  • • •
         • •    |  • •
        [Clear split between groups]
```

### Code Pattern
```javascript
// Split: Points keep SAME x-positioning
// Each group shows complete x-range

// X-axis ticks: Show only primary dimension
callback: function(v) {
    return String(v);  // Just "0", "1", "2", etc.
}

// Group labels shown SEPARATELY below axis
// Plugin renders: "UP", "TP" with separators
```

### Advantages
- ✅ Clear visual separation
- ✅ X-axis unambiguous (0, 1, 2 repeated)
- ✅ Group labels positioned prominently
- ✅ Easier to interpret visually
- ✅ Natural split-chart appearance

---

## Key Differences

| Aspect | Previous | New |
|--------|----------|-----|
| **X values** | Unique per category | Repeated per group |
| **Axis labels** | Show categories: `0 [LP]`, `0 [UP]` | Show X only: `0`, `1`, `2` |
| **Group ID** | In bracket suffix | Below axis as tier label |
| **Separation** | Point label only | Visual line + tier label |
| **Visual style** | Single continuous axis | Split sections with markers |
| **Interpretation** | Hierarchical labels | Categorical split |
| **Use case** | Multi-level hierarchy | Split-view comparison |

---

## Implementation Comparison

### Previous: Tick Label Approach
```
User intent: Y vs X(WL), grouped by pagetype
Our code: Map each unique (WL, pagetype) combination
Result: Confusing x-axis with duplicates
```

### New: Separate Tier Approach
```
User intent: Y vs X(WL), split chart by pagetype
Our code: Keep X positioning, add visual markers
Result: Clear split-chart visualization
```

---

## Data Structure Mapping

### Previous
```javascript
groupMap = {
  '0': 'LP',
  '1': 'LP',
  '0': 'UP',   // Overwrites previous '0': 'LP'
  '1': 'UP'
}
// Problem: Can't distinguish LP vs UP for X=0
```

### New
```javascript
groups = [
  { name: 'UP', xValues: ['0', '1', '2', '3', '4'] },
  { name: 'TP', xValues: ['0', '1', '2', '3', '4'] }
]
groupMap = { used for quick lookup only }
// Clear: Each group has complete x-value set
```

---

## Plugin Approach

### Previous: None (handled in axis config)
- Problem: X-axis ticks had collision (showing `0 [LP]` and `0 [UP]`)
- Solution: Add to label suffix

### New: Custom `groupSeparators` Plugin
```javascript
Chart.register({
    id: 'groupSeparators',
    afterDraw: function(chart) {
        // 1. Draw dashed separator lines
        // 2. Render group tier labels
        // 3. Calculate positions
        // 4. Log debug info
    }
});
```

Benefits:
- ✅ Cleaner separation of concerns
- ✅ Dedicated rendering logic
- ✅ Easy to customize styling
- ✅ Proper integration with Chart.js lifecycle

---

## Migration Checklist

If transitioning from previous approach:

- [ ] Update `_buildGroupedXAxis()` to preserve x-value ranges per group ✓
- [ ] Enhance data point structure with `_xVal` and `_groupName` ✓
- [ ] Simplify x-axis tick callback (show X only) ✓
- [ ] Create `groupSeparators` plugin ✓
- [ ] Add layout padding for tier labels ✓
- [ ] Remove conflicting hierarchical logic
- [ ] Test with sample data
- [ ] Update user documentation
- [ ] Adjust UI controls if needed

---

## Summary

**Split-Chart Approach = Better UX**

Before:
```
Confusing: X shows category membership in label
Visual issue: Hard to see where one group ends and another begins
```

After:
```
Clear: X shows measurement values only
Visual: Dashed lines + tier labels create obvious sections
```

**Result**: Users immediately understand the grouping structure without reading labels.

---

## Use Cases

### When to use Split-Chart (NEW)
- Comparing same measurements across categories
- X-values common to all categories (WL 0-4 for both UP and TP)
- Want visual split-view effect
- Categories are major grouping factor

### When to use Hierarchical (OLD)
- True nested hierarchy (Country > State > City)
- Unique X-values per category
- Want textual label-based grouping
- Multiple nesting levels needed

**This implementation** is optimized for the split-chart use case (comparing measurements across categorical sections).
