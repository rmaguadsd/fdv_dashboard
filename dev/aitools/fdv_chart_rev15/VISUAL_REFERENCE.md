# FDV Chart Issue - Visual Reference Guide

---

## The Problem Visualized

### Case #1: Single Chart (No Split) - "Color by BLK"

```
Your Data File
┌─────────────────────────────────────┐
│ Row  │ DUT  │ BLK  │ RBER          │
├─────┼──────┼──────┼──────────────┤
│ 1   │ DUT1 │ 316  │ 1.2e-4        │ ✓ Has measurement
│ 2   │ DUT1 │ 338  │ (null)        │ ✗ NO measurement!
│ 3   │ DUT1 │ 416  │ 2.1e-4        │ ✓ Has measurement
│ 4   │ DUT1 │ 505  │ 0.9e-4        │ ✓ Has measurement
│ 5   │ DUT1 │ 195  │ 1.5e-4        │ ✓ Has measurement
│ 6   │ DUT1 │ 316  │ 1.3e-4        │ ✓ Has measurement
│ 7   │ DUT2 │ 316  │ 1.1e-4        │ ✓ Has measurement
│ 8   │ DUT2 │ 338  │ (null)        │ ✗ NO measurement!
│ ... │ ...  │ ...  │ ...           │
└─────┴──────┴──────┴──────────────┘

Chart Generation Process
┌────────────────────────────────────────────┐
│ Step 1: Discover all colors                │
│ ┌──────────────────────────────────────┐   │
│ │ Scan all rows (including null RBER)  │   │
│ │ Found colors: 195, 316, 338, 416, 505│   │
│ └──────────────────────────────────────┘   │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│ Step 2: Filter to valid points             │
│ ┌──────────────────────────────────────┐   │
│ │ Keep only rows with RBER != null     │   │
│ │ Result: 195, 316, 416, 505 ← 338 lost!  │
│ └──────────────────────────────────────┘   │
└────────────────────────────────────────────┘
                    ↓
Chart Legend
┌─────────────────┐
│ ○ 195           │ ← Present
│ ○ 316           │ ← Present
│ ○ 338           │ ← MISSING! ❌
│ ○ 416           │ ← Present
│ ○ 505           │ ← Present
└─────────────────┘

❌ Problem: User sees 5 blocks instead of 6
```

---

### Case #2: Split Chart by DUT - "Color by BLK"

```
Your Data File (DUT breakdown)
┌──────────────────────────────────────┐
│ DUT1 → BLK with RBER: 195, 316, 416, 505
│ DUT2 → BLK with RBER: 316, 338, 416    ← 195, 505 missing!
│ DUT3 → BLK with RBER: 195, 316, 338, 416, 505
│ DUT4 → BLK with RBER: 195, 316, 338, 416, 505
└──────────────────────────────────────┘

Split Chart Generation
┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ Tile 1: DUT1      │ Tile 2: DUT2      │ Tile 3: DUT3      │ Tile 4: DUT4      │
├───────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Legend:           │ Legend:           │ Legend:           │ Legend:           │
│ ○ 195             │ ○ 316             │ ○ 195             │ ○ 195             │
│ ○ 316             │ ○ 338             │ ○ 316             │ ○ 316             │
│ ○ 416             │ ○ 416             │ ○ 338             │ ○ 338             │
│ ○ 505             │                   │ ○ 416             │ ○ 416             │
│   (4 blocks)      │   (3 blocks) ❌   │ ○ 505             │ ○ 505             │
│                   │   MISSING 195,    │   (5 blocks)      │   (5 blocks)      │
│                   │   505!            │                   │                   │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┘

❌ Problem: DUT2 has different blocks than others!
   User sees inconsistent legends across tiles.
```

---

## Why This Happens

### The Filtering Chain

```
Input Data
    ↓
[Regex filter] ← User's active filters (e.g., "DUT1 only")
    ↓
filteredIndices[] ← All rows after user filters
(24 rows: 4 DUTs × 6 blocks)
    ↓
┌─────────────────────────────────────────┐
│ FORK 1: Color Discovery                 │ FORK 2: Point Plotting
│ ┌───────────────────────────────┐      │ ┌──────────────────────┐
│ │ Scan ALL filteredIndices      │      │ │ readFilteredFromMemory
│ │ Extract BLK values            │      │ │ Keep only:
│ │ Result: 195, 316, 338,        │      │ │  - X is numeric
│ │         416, 505              │      │ │  - Y is numeric
│ │ ✅ Complete set              │      │ │ Result: 195, 316,
│ │                               │      │ │         416, 505
│ │                               │      │ │ ❌ Missing 338!
│ └───────────────────────────────┘      │ └──────────────────────┘
└─────────────────────────────────────────┘
            ↓ Divergence ↓
        But only Fork 2
        is used for plotting!
            ↓
Legend ← Uses Fork 2 result
[195, 316, 416, 505]  ← Missing 338!
```

### The Core Issue

```
                Discovery Phase            Plotting Phase
                ┌────────────┐             ┌────────────┐
                │ All rows   │             │ Valid rows │
                │ (includes  │             │ (excludes  │
                │ null RBER) │             │ null RBER) │
                └────────────┘             └────────────┘
                      ↓                          ↓
                Colors found:             Colors plotted:
                195, 316, 338,            195, 316,
                416, 505                  416, 505
                      ↓                          ↓
                    ❌ Mismatch!
            Legend has more colors
             than points on chart
```

---

## The Four Solutions Compared

### Solution A: Unified Colors (Recommended)

```
Input
  ↓
Discover ALL colors once
  ↓
┌─────────────────────────────┐
│ Global Color Set            │
│ [195, 316, 338, 416, 505]   │
└─────────────────────────────┘
  ↓                   ↓
[Plot points]    [Legend colors]
[195, 316,   ]   [195, 316, 338 (n=0),
 416, 505]        416, 505]
  ↓                   ↓
  └─────→ Consistent! ←─┘

✓ Pros: Consistent, complete, honest
✗ Cons: Shows "(n=0)" groups in legend
```

### Solution B: Warning System

```
Current behavior (no change)
         ↓
Add console logging:
"⚠️ Color 338 discovered
   but has 0 plotted points"
         ↓
User sees warning in console
and understands what happened

✓ Pros: Non-invasive, helps debugging
✗ Cons: Doesn't fix the inconsistency
```

### Solution C: Pre-Filter

```
Input
  ↓
Filter to VALID rows first
  ↓
[195, 316, 416, 505]
(excludes 338)
  ↓
Discover colors
  ↓
[195, 316, 416, 505]
  ↓
Plot points
  ↓
Legend: [195, 316, 416, 505]
         ✓ Consistent!

✓ Pros: Clean, no empty groups
✗ Cons: Hides blocks that have no measurements
```

### Solution D: Toggle

```
User can choose:

[ ] Include all blocks
    ├─ ON  → Use Solution A (with "(n=0)")
    └─ OFF → Use Solution C (without empty)

✓ Pros: Flexible, user control
✗ Cons: More complex UI, user confusion
```

---

## Solution Comparison Matrix

| Criterion | A | B | C | D |
|-----------|---|---|---|---|
| Fixes inconsistency | ✅ | ❌ | ✅ | ✅ |
| Shows complete picture | ✅ | ❌ | ❌ | ✅ |
| Clean legend | ❌ | ❌ | ✅ | 🤝 |
| Easy to implement | 🟡 | ✅ | ✅ | ❌ |
| Matches user expectation | 🤔 | 🤔 | 🤔 | ✅ |
| **Recommended** | **✅** | — | — | — |

---

## Implementation Flow

### Solution A Step-by-Step

```
1. Compute global colors ONCE
   ┌─────────────────────────┐
   │ for each row:           │
   │   extract color value   │
   │   add to globalColors[] │
   │ END                     │
   │ Result: {195, 316,      │
   │          338, 416, 505} │
   └─────────────────────────┘
   Time: O(n) where n = total rows
         ≈ 25ms for 10k rows

2. Build split buckets
   ┌──────────────────────────┐
   │ for each row:            │
   │   determine split key    │
   │   (e.g., DUT)            │
   │   add to bucket[key]     │
   │ END                      │
   └──────────────────────────┘
   Time: Same as before

3. For each tile
   ┌──────────────────────────┐
   │ use globalColors[]       │
   │ (not bucket-specific)    │
   │ mark colors with (n=0)   │
   │ if no points for that    │
   │ color in this tile       │
   └──────────────────────────┘

Total time impact: +0% (faster actually!)
```

---

## Before & After Comparison

### Before Solution A

```
Case: Split by DUT, Color by BLK

Tile 1 (DUT1)        Tile 2 (DUT2)
Legend:              Legend:
○ 195                ○ 316
○ 316                ○ 338
○ 416                ○ 416
○ 505

❌ Different sets!
```

### After Solution A

```
Case: Split by DUT, Color by BLK

Tile 1 (DUT1)        Tile 2 (DUT2)
Legend:              Legend:
○ 195                ○ 195
○ 316                ○ 316
○ 338 (n=0)          ○ 338
○ 416                ○ 416
○ 505                ○ 505 (n=0)

✅ Same set in both!
   Missing measurements marked as (n=0)
```

---

## Debug Checklist Quick Reference

```
[ ] Verify root cause
    └─ Run console script → Check if block has 0 valid measurements

[ ] Choose solution
    ├─ A (unified colors) → Most comprehensive
    ├─ B (warnings) → Debugging only
    ├─ C (pre-filter) → If you want clean legend
    └─ D (toggle) → If you want user choice

[ ] Implement chosen solution
    └─ Follow IMPLEMENTATION_GUIDE.md

[ ] Test with 3 cases
    ├─ Case #1: Single chart
    ├─ Case #2: Split by DUT
    └─ Case #3: Split by tname

[ ] Verify console output
    └─ Check for warnings/errors

[ ] Run regression tests
    ├─ Test all chart types
    ├─ Test all split modes
    └─ Test with different measurements

[ ] Deploy & document
    └─ Update BLOCK_DATA_MISSING_FIX.md
```

---

## Key Insights

### Why You See Different Results in Different Cases

```
Case #1: Single chart (no split)
→ All rows are considered together
→ If ANY block has 0 valid measurements, it disappears
→ Result: Depends on global data completeness

Case #2: Split by DUT
→ Each DUT considered separately
→ If a block has 0 measurements IN THAT DUT, it disappears
→ Result: Different blocks per DUT!
→ This is the REAL problem

Case #3: Split by tname
→ Similar to Case #2 but splits by test name
→ Different tnames may have different DUTs
→ Result: Inconsistent DUT appearance
```

### The "Why Sometimes Right" Question

```
You see:
- Sometimes all 6 blocks ✓
- Sometimes 5 blocks ✗
- Sometimes 3-4 blocks ✗

This depends on:
1. Which blocks have measurements in which DUTs
2. The Y column you selected (RBER vs. other)
3. The interval filters (Min/Max)
4. The chart type (scatter needs both X & Y)

Example:
- If using column that's 100% populated → All blocks visible
- If using column that's 50% populated → ~3 blocks visible
- If filtering by range → Fewer blocks visible
```

---

## Questions This Guide Answers

**Q: Where does the data go?**
A: It doesn't go anywhere! It's still in the file, just not rendered in the chart.

**Q: Is this a bug?**
A: It's more of a design inconsistency. The code is doing what it was programmed to do, but the result is confusing.

**Q: Will I lose data if I implement Solution A?**
A: No. Solution A doesn't modify data, only how it's displayed.

**Q: Which solution should I pick?**
A: Solution A if you want consistent behavior, even with empty groups showing as "(n=0)".

**Q: How long until I can deploy?**
A: 30 minutes (Solution B), 3-4 hours (Solution A), 1 hour (Solution C).

---

## File Reference

| Document | Purpose |
|----------|---------|
| SUMMARY.md | This document — visual overview |
| DATA_CONSISTENCY_INVESTIGATION.md | Deep technical analysis |
| DIAGNOSTIC_CHECKLIST.md | Step-by-step verification |
| IMPLEMENTATION_GUIDE.md | Exact code changes |

---

**Status:** 🟢 Ready to diagnose and fix  
**Recommendation:** Start with diagnostic checklist, implement Solution A
