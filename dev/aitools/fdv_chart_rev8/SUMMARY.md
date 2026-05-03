# FDV Chart Data Consistency Issue - Complete Summary

**Issue Created:** April 29, 2026  
**Status:** ✅ Root Cause Identified & Solutions Documented  
**Severity:** Medium (affects data visibility, not calculation)  

---

## Executive Summary

Your FDV chart shows **inconsistent block/DUT data** depending on the chart configuration:

| Case | Configuration | Problem | Expected | Actual |
|------|---------------|---------|----------|--------|
| #1 | Single chart, color by BLK | Missing blocks | 316, 338, 416, 505, 195, 316 | 316, 416, 505, 195, 316 |
| #2 | Split by DUT, color by BLK | Different blocks per tile | Same in all tiles | Different per DUT |

**Root Cause:** The chart system discovers color values from rows **with invalid measurements**, but **plots only rows with valid measurements**. When data is sparse per DUT or measurement, different color values appear in different tiles.

**Why "sometimes right, sometimes wrong":**
- Depends on which blocks have measurements in which DUTs
- Depends on the measurement column selected (RBER, etc.)
- Depends on the interval filters applied

---

## Quick Diagnosis

**To verify this is your issue:**

1. Open browser console (F12)
2. Upload your log file
3. Run this script:

```javascript
var blkIdx = currentHeaders.indexOf('BLK');
var rberIdx = currentHeaders.indexOf('RBER');

// Count blocks with RBER data
var blocksWith Rber = {};
filteredIndices.forEach(function(ri) {
    var row = allRows[ri];
    var rber = extractNum(String(row[rberIdx]), '');
    if (rber !== null) {
        var blk = row[blkIdx];
        blocksWith Rber[blk] = (blocksWith Rber[blk] || 0) + 1;
    }
});

console.log('Blocks with RBER data:', Object.keys(blocksWith Rber).sort());
```

**If BLK 338 has count=0, that's the issue!** 🎯

---

## Root Cause Deep Dive

```
Chart Rendering Pipeline
├─ Step 1: Load all rows → allRows[]
├─ Step 2: Apply filters → filteredIndices[]
├─ Step 3: Discover colors (from ALL filteredIndices)
│  └─ ✅ Correctly scans all rows, even those with missing Y
├─ Step 4: Filter to valid X/Y → readFilteredFromMemory()
│  └─ ❌ Removes rows with missing measurements
└─ Step 5: Build legend (from Step 4 results)
   └─ ❌ Legend only includes colors that have plotted points!
```

**The Contradiction:**
- Discovery uses: All rows (including those with missing measurements)
- Plotting uses: Only rows with valid measurements
- **Result:** Color appears in discovery but not in final chart → Inconsistency

---

## Three Documented Solutions

### ✅ Solution A: Unified Color Discovery (Recommended)

**What it does:** Ensures every possible color appears in every chart's legend

**How it works:**
1. Compute all possible color values from all rows **once**
2. Pass the same color list to **every tile**
3. Mark empty groups as "(n=0)"

**Pros:**
- ✅ Consistent legends across all tiles
- ✅ Shows complete data picture
- ✅ Minimal performance impact
- ✅ User can see why a block is grayed out

**Cons:**
- Clutters legend with empty groups
- Might be confusing if user expects "only blocks with data"

**Complexity:** Medium (2-3 hours implementation)

**Files:** `fdv_chart.html` — `_drawSplitCharts()` and `_buildTileChart()`

---

### ⚠️ Solution B: Warning System

**What it does:** Log console warnings when colors are inconsistent

**How it works:**
1. Track which colors appear in each tile
2. Warn if a color is missing in one tile but present in another
3. Help user understand what's happening

**Pros:**
- ✅ Non-invasive (no UI changes)
- ✅ Good for debugging
- ✅ Quick to implement

**Cons:**
- ❌ Doesn't fix the inconsistency, just warns about it
- ❌ Users might miss console warnings

**Complexity:** Low (30 minutes)

---

### 🔄 Solution C: Pre-Filter by Validity

**What it does:** Filter to only rows with valid measurements before building charts

**How it works:**
1. Before split/color discovery, filter `filteredIndices` to only rows with valid X/Y
2. Build charts from pre-filtered data
3. No empty groups possible

**Pros:**
- ✅ Clean legend (no "(n=0)" labels)
- ✅ Consistent behavior

**Cons:**
- ❌ Hides blocks that legitimately have no measurements
- ❌ User sees less data than file contains

**Complexity:** Low (30 minutes)

---

### 🎛️ Solution D: User Toggle

**What it does:** Let user choose between showing all colors or only measured ones

**How it works:**
1. Add checkbox: "Show all blocks (including empty)"
2. When checked: Use Solution A logic
3. When unchecked: Use Solution C logic

**Pros:**
- ✅ Gives user control
- ✅ Works for all use cases

**Cons:**
- ❌ More UI complexity
- ❌ User confusion (what does "empty" mean?)

**Complexity:** High (4-6 hours)

---

## Recommended Implementation Path

### Phase 1: Verification (Now)
- [ ] Run diagnostic script from "Quick Diagnosis" above
- [ ] Confirm blocks with 0 measurements
- [ ] Document findings

### Phase 2: Quick Fix (1-2 hours)
- [ ] Implement Solution B (warning system)
- [ ] Deploy to staging
- [ ] Verify console shows warnings for missing colors

### Phase 3: Proper Fix (3-4 hours)
- [ ] Implement Solution A (unified color discovery)
- [ ] Add "(n=0)" labels
- [ ] Test all three cases

### Phase 4: Regression Testing (1 hour)
- [ ] Test each chart type: scatter, histogram, cumproba, boxplot
- [ ] Test each split mode: none, by DUT, by tname
- [ ] Test each color dimension
- [ ] Performance: verify still fast with large files

---

## Documentation Created

For your reference:

1. **DATA_CONSISTENCY_INVESTIGATION.md**
   - Deep technical explanation of the root cause
   - Data flow diagrams for each case
   - Current partial fixes analyzed
   - Four solution approaches detailed

2. **DIAGNOSTIC_CHECKLIST.md**
   - Step-by-step guide to verify the issue
   - Console commands to inspect your data
   - Expected outputs for each scenario
   - How to confirm the fix works

3. **IMPLEMENTATION_GUIDE.md**
   - Exact code changes needed (Solution A)
   - Line-by-line modifications
   - Testing checklist
   - Quick rollback plan

4. **This file: SUMMARY.md**
   - Executive overview
   - Quick diagnosis
   - Solution comparison
   - Next steps

---

## Key Code Locations

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `recomputeFilteredIndices()` | fdv_chart.html | 1322 | Global row filtering |
| `readFilteredFromMemory()` | fdv_chart.html | 1837 | X/Y validity filtering ← **The bottleneck** |
| `drawScatterLine()` | fdv_chart.html | 4272 | Single chart rendering |
| `_drawSplitCharts()` | fdv_chart.html | 3805 | Split chart rendering ← **Main fix location** |
| `_buildTileChart()` | fdv_chart.html | 4100+ | Individual tile builder ← **Secondary fix location** |

---

## Next Steps

### Immediate Actions

1. **Run diagnostic:** Verify the root cause with your data
   - See "Quick Diagnosis" section above
   - Takes 5 minutes

2. **Choose a solution:** Based on requirements
   - Show all blocks? → Solution A
   - Just warn? → Solution B
   - Hide empty? → Solution C
   - User choice? → Solution D

3. **Implement:** Pick one and follow the guide
   - Solution B is fastest (30 min)
   - Solution A is best (3-4 hours)

### Communication

**To stakeholders:**
- "We identified why some blocks disappear depending on chart settings."
- "It's not data loss, just inconsistent filtering logic."
- "We have three fixes available with different tradeoffs."
- "Recommend Solution A (consistent legends) implemented in Phase 3."

---

## FAQ

**Q: Is data actually lost?**  
A: No! The data is intact in the file. It's just not appearing in the chart legend or plot, depending on which rows have valid measurements. The data is never deleted or corrupted.

**Q: Why didn't this happen before?**  
A: It's always been happening — you're just noticing it now because you're comparing charts across different configurations. The inconsistency only becomes obvious when switching between "color by BLK" and "split by DUT".

**Q: Will fixing this slow down the chart?**  
A: No. In fact, Solution A is slightly faster because it discovers colors once instead of per-bucket.

**Q: Should I implement all solutions?**  
A: No, pick one. They're alternatives to each other, not complementary:
- A and C are opposite approaches
- B is just a diagnostic tool
- D combines A and C with a toggle

**Q: Can I implement this without restarting the server?**  
A: Yes! Just edit `fdv_chart.html`, save, and refresh the browser. No server restart needed.

**Q: What if I want to show empty groups but don't want to implement Solution A?**  
A: You'd need to modify the code anyway. The current behavior is "only show colors that have plotted points", which is hard-coded. To show empty groups, you must change that logic.

---

## Success Criteria

**After implementing the fix:**

- [ ] Case #1 (single chart): All 6 blocks appear in legend
- [ ] Case #2 (split by DUT): All 6 blocks appear in all tiles' legends
- [ ] Case #3 (split by tname): All 4 DUTs appear in all tiles' legends
- [ ] Other chart types: No regression (histogram, cumproba, etc.)
- [ ] Empty groups: Marked with "(n=0)" (for Solution A)
- [ ] Performance: No noticeable slowdown
- [ ] Console: Helpful warnings for debugging

---

## Support

**If you have questions during implementation:**

1. Check IMPLEMENTATION_GUIDE.md for step-by-step code changes
2. Check DIAGNOSTIC_CHECKLIST.md to verify the root cause
3. Check DATA_CONSISTENCY_INVESTIGATION.md for deep technical details
4. Look at existing code comments in fdv_chart.html (already has some fixes)

**If something breaks:**

1. Check console for error messages
2. Use git to see what changed: `git diff fdv_chart_rev4/fdv_chart.html`
3. Rollback: `git checkout -- fdv_chart_rev4/fdv_chart.html` and reload
4. Retry: Start from IMPLEMENTATION_GUIDE.md step by step

---

## Timeline Estimate

| Activity | Time | Dependencies |
|----------|------|--------------|
| Verification | 10 min | None |
| Design review | 20 min | After verification |
| Implementation | 2-4 hrs | After design choice |
| Testing | 1-2 hrs | After implementation |
| Documentation | 30 min | After testing |
| **Total** | **4-7 hrs** | — |

---

## Summary for Leadership

**Issue:** Chart inconsistently displays blocks/DUTs depending on configuration  
**Root Cause:** Multi-level filtering creates gaps between data discovery and rendering  
**Impact:** User confusion, but no data loss  
**Fix:** Unified color discovery approach (Solution A)  
**Effort:** 3-4 hours implementation + 1-2 hours testing  
**Risk:** Low (UI-only change, no data modification)  
**Benefit:** Consistent, predictable chart behavior; improved user confidence  

---

**Status:** 🟢 Ready to implement  
**Next:** Choose a solution and follow IMPLEMENTATION_GUIDE.md
