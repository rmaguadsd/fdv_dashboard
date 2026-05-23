# Z-Order Feature - Complete Documentation Index

## 📚 Documentation Files

This package includes comprehensive documentation for the new **Z-Order (rendering order control)** feature.

### Quick References

1. **ZORDER_QUICK_START.md** ⭐ **START HERE**
   - Quick visual overview
   - Common examples
   - Basic troubleshooting
   - **Best for:** First-time users, quick reference

2. **Z_ORDER_FEATURE.md**
   - Complete feature guide
   - Detailed examples and patterns
   - Technical details
   - **Best for:** Understanding all capabilities

3. **ZORDER_COMPLETE_GUIDE.md**
   - In-depth usage guide
   - Real-world scenarios
   - Advanced techniques
   - Step-by-step walkthroughs
   - **Best for:** Power users, learning new patterns

4. **ZORDER_IMPLEMENTATION.md**
   - Technical implementation details
   - Code walkthrough
   - Architecture explanation
   - **Best for:** Developers, understanding the code

5. **ZORDER_SUMMARY.md**
   - Executive summary
   - Quick feature list
   - Deployment checklist
   - **Best for:** Project overview, status tracking

---

## 🚀 Quick Start (30 seconds)

1. Open http://localhost:5059
2. Parse a CSV file with a color-by column
3. Look for **Z-Order** field in Plot Panel
4. Enter: `value1, value2, value3`
5. Click Plot
6. value3 renders on top ✓

---

## 🎯 What Is Z-Order?

**Z-Order controls which color-by groups render on top when data points overlap.**

### Example
- Default (alphabetical): A bottom, B middle, C top
- With Z-Order "C, A, B": C bottom, A middle, B top
- Result: Group rendering order is customizable

---

## 📍 Where to Find It

**Plot Panel** → Right after "Color by" selector

```
Color by: [column] +
Z-Order:  [text input] (last = top)
```

---

## 📋 Usage Summary

| Task | Action | Result |
|------|--------|--------|
| Put one group on top | Enter: `groupname` | Group renders on top |
| Control 3+ groups | Enter: `a, b, c` | Renders in that order (c = top) |
| Reset to default | Clear field, click Plot | Returns to alphabetical |
| Multi-dimensional | Enter: `val1~val2` | Works with compound keys |
| Partial list | Enter: `c, a` | b renders first, then a, then c |

---

## 🔍 Common Patterns

### Pattern 1: Emphasize One Group
```
Z-Order: focus_group
```
Focus group renders on top, others alphabetically underneath

### Pattern 2: Hide Background
```
Groups: background, data1, data2, data3
Z-Order: background, data1, data2, data3
```
Background renders first (under all data)

### Pattern 3: Comparative Visualization
```
Groups: control, treatment1, treatment2, highlight
Z-Order: control, treatment1, treatment2, highlight
```
Creates visual hierarchy: context → comparison → focus

---

## 💡 Key Concepts

### Rendering Order (Bottom to Top)
1. Groups **not listed** in z-order
2. Groups **in z-order**, in specified sequence
3. **Last group** in z-order renders **on top**

### Important Notes
- ✅ Comma-separated values (whitespace trimmed)
- ✅ Case-sensitive (must match legend exactly)
- ✅ Partial lists supported
- ✅ Empty field = alphabetical order
- ✅ Works with all chart types

---

## 🛠️ Implementation Details

### File Changed
- `dev/aitools/fdv_chart_rev11/fdv_chart.html`
- Lines 786-791: UI control added
- Lines 5095-5120: Sorting logic added
- Total: ~40 lines of new code

### How It Works
```
User Input (comma-separated)
    ↓
Parse & trim whitespace
    ↓
Sort groups by z-order preference
    ↓
Build Chart.js datasets in sorted order
    ↓
Chart.js renders bottom-to-top
```

### Compatibility
- ✅ All chart types (scatter, line, histogram, etc.)
- ✅ Multi-dimensional color-by
- ✅ Backward compatible (optional feature)
- ✅ No breaking changes

---

## 📊 Examples

### Example 1: Sales Data
```
Data: Regions (North, South, East, West) over time
Problem: West region (low sales) hidden under high-sales regions
Solution: Z-Order: North, South, East, West
Result: West line on top, clearly visible
```

### Example 2: Quality Control
```
Data: Defect types (category_a, category_b, category_c, critical)
Problem: Critical defects lost in crowd
Solution: Z-Order: category_a, category_b, category_c, critical
Result: Critical always visible
```

### Example 3: A/B Testing
```
Data: Baseline, ControlA, ControlB, TreatmentX
Problem: TreatmentX effects not clear
Solution: Z-Order: Baseline, ControlA, ControlB, TreatmentX
Result: TreatmentX on top for clear comparison
```

---

## ❓ FAQ

**Q: Does z-order affect data or just visualization?**  
A: Only visualization (rendering order). Data unchanged.

**Q: Can I reorder the legend?**  
A: No, legend is alphabetical. Z-order only affects rendering order.

**Q: Does z-order save automatically?**  
A: No, save with recipe or session to preserve.

**Q: What if I mistype a value?**  
A: It's treated as not-in-list (renders first, alphabetically).

**Q: Can I use regex patterns?**  
A: Not in current version, exact values only.

**Q: Does it work with filtered data?**  
A: Yes, z-order applies to visible (non-filtered) groups.

---

## 🔗 Related Features

### Companion Features
- **Color-by:** Selects which column to split by color
- **Multi-dim Color-by:** Combine multiple columns for compound coloring
- **Split-chart:** Split visualization by column (separate charts)
- **Sampling:** Reduce point count for performance

### Coordinate Features
- These features work independently
- Z-order layers on top of all other settings
- Combine for powerful visualizations

---

## 📈 Use Cases

1. **Highlight Important Data** - Emphasize focus group on top
2. **Hide Distractions** - Put background/noise underneath
3. **Visual Hierarchy** - Create clear visual organization
4. **Comparative Analysis** - Layer comparison data clearly
5. **Outlier Detection** - Keep rare events always visible
6. **Time Series** - Show most recent data on top
7. **Quality Metrics** - Emphasize critical measurements
8. **A/B Testing** - Treatment on top for comparison

---

## 🚨 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Z-order not working | Wrong spelling/case | Copy from legend exactly |
| Some groups unordered | Not all in z-order field | List all groups for full control |
| Still alphabetical | Empty z-order field | Enter at least one value |
| Group missing | Filtered out | Check intervals and filters |

---

## 📞 Support Resources

### Quick Help
- See: ZORDER_QUICK_START.md
- Visual diagrams and common examples

### Detailed Guidance
- See: ZORDER_COMPLETE_GUIDE.md
- Step-by-step walkthroughs and patterns

### Technical Information
- See: ZORDER_IMPLEMENTATION.md
- Code explanation and architecture

### Full Reference
- See: Z_ORDER_FEATURE.md
- Comprehensive feature documentation

---

## ✅ Feature Status

- ✅ **Development:** Complete
- ✅ **Testing:** Verified on localhost:5059
- ✅ **Documentation:** Comprehensive
- ✅ **Backward Compatibility:** Maintained
- ✅ **Production Ready:** Yes

---

## 📝 Version Info

- **Version:** 1.0
- **Release Date:** May 21, 2026
- **Status:** ✅ Production Ready
- **Server:** http://localhost:5059
- **Main File:** `dev/aitools/fdv_chart_rev11/fdv_chart.html`

---

## 🎓 Learning Path

### Beginner (5 minutes)
1. Read: ZORDER_QUICK_START.md
2. Try: Enter one value in z-order field
3. Test: Click Plot, see results

### Intermediate (15 minutes)
1. Read: Z_ORDER_FEATURE.md
2. Try: Multiple groups in z-order
3. Experiment: Different sort orders

### Advanced (30 minutes)
1. Read: ZORDER_COMPLETE_GUIDE.md
2. Try: Real-world scenarios
3. Master: Advanced techniques

### Technical (Developer)
1. Read: ZORDER_IMPLEMENTATION.md
2. Study: Code in fdv_chart.html lines 5095-5120
3. Understand: Sorting algorithm and data flow

---

## 🔄 Next Steps

1. **Try It:** Open chart at http://localhost:5059
2. **Experiment:** Test with your data
3. **Save It:** Save settings in recipe/session
4. **Explore:** Check advanced guides for patterns
5. **Share:** Other users can benefit from z-order

---

## 📮 Feedback

- Report issues or suggestions
- Propose new patterns
- Share use cases
- Contribute improvements

---

## 🏁 Getting Started Now

**Ready to use z-order? Follow this:**

1. **Open the Chart:** http://localhost:5059
2. **Parse Your File:** Upload and parse CSV
3. **Set Up Plot:**
   - X-axis: [select column]
   - Y-axis: [select column]
   - Color-by: [select column]
4. **Add Z-Order:**
   - Click Z-Order field
   - Type: `value1, value2, value3`
5. **Click Plot**
6. **Done!** value3 renders on top

**Need help?** See ZORDER_QUICK_START.md for more examples.

---

**Welcome to Z-Order! Enjoy enhanced control over your visualizations. 🚀**
