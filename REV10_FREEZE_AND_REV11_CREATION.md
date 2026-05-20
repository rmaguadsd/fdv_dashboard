# REV10 Freeze & REV11 Creation Summary

## Date: May 19, 2026

### Action Completed

✅ **REV10 Frozen** - Created snapshot `fdv_chart_rev10_frozen`
✅ **REV11 Created** - New development branch based on REV10

---

## Version Timeline

```
REV9 (Frozen: May 18)
    ↓
REV10 (Stable: May 18-19)
    ├─→ REV10_FROZEN (Snapshot: May 19, 10:07 AM)
    └─→ REV11 (New Dev: May 19, 10:08 AM)
```

---

## REV10 Summary (Frozen Version)

### Features & Fixes
1. ✅ **Session Loading** - Fixed "binding error" for all session sizes
2. ✅ **Backward Compatibility** - Auto-detects and converts old session format (34 cols → 33 cols)
3. ✅ **Batch Processing** - 50K-row transactions for large files (tested with 1.7GB+ sessions)
4. ✅ **Box Plot Enhancements**:
   - Outliers enabled by default (`outlierRadius: 3`)
   - "Data Points" checkbox controls scatter overlay visibility
   - Jitter disabled by default (checkbox unchecked)
   - Jitter positioning fixed: stays within category bounds (pixel-space offset)
5. ✅ **Plugin Guards** - Strict `=== true` checks for scatter visibility

### Key Improvements
- All 6 `SELECT *` queries fixed for column alignment
- Compatibility layer for old sessions
- Gray outliers now controllable via chart options
- Scatter jitter properly positioned

### Files Modified
- `dev/aitools/fdv_chart_rev10/fdv_chart.py` - Server (2347 lines)
- `dev/aitools/fdv_chart_rev10/fdv_chart.html` - Frontend with all fixes

---

## REV11 (New Development Version)

### Status
- ✅ Created: May 19, 2026, 10:08 AM
- ✅ Base: Copied from REV10
- ✅ Ready for: New features & enhancements

### Purpose
Forward development continues on REV11 while REV10 remains stable and frozen.

### Next Steps for REV11
- Potential UI enhancements
- Additional chart types
- Performance optimizations
- New features as requested

---

## Version Directory Structure

```
dev/aitools/
├── fdv_chart_rev1/           (Original)
├── fdv_chart_rev2/           (Deprecated)
├── fdv_chart_rev3-8/         (Deprecated)
├── fdv_chart_rev9/           (Deprecated - Original frozen as rev9_frozen)
├── fdv_chart_rev9_frozen/    (Snapshot)
├── fdv_chart_rev10/          (Stable - Production)
├── fdv_chart_rev10_frozen/   ✅ (Snapshot - Today)
└── fdv_chart_rev11/          ✅ (New Dev - Today)
```

---

## Production Status

### Current Production (Port 5059)
- **Version**: REV10
- **Status**: Active & Stable
- **Last Start**: May 19, 2026
- **Uptime**: Continuous

### Rollback Plan
If REV11 development causes issues:
1. Stop REV11 server
2. Switch to REV10_FROZEN (guaranteed snapshot)
3. Restart on same port

---

## REV10 Key Fixes Reference

### 1. Session Loading (Fixed 6 locations)
```python
# Before: SELECT * (34 cols, including auto-id)
# After: SELECT col1, col2, ... col33 (explicit columns)
```

### 2. Backward Compatibility Layer
```python
# Auto-detection in /store/register_session:
if len(row[0]) > len(headers):
    # Old format detected, remove first column
    remove_auto_id_column()
```

### 3. Box Plot Visualization
```javascript
// Outliers: enabled by default
outlierRadius: 3
outlierBackgroundColor: [colors]

// Data Points: toggled by checkbox
_scatterVisible: checkbox.checked

// Jitter: disabled by default
window._useScatterJitter = false

// Jitter positioning: fixed to category bounds
baseXPixel = xScale.getPixelForValue(pt.gIdx)
jitterPixelX = (pt.x - pt.gIdx) * categoryWidth
xPixel = baseXPixel + jitterPixelX
```

---

## Testing Checklist (REV10 Verified)

- ✅ Load sessions (all sizes)
- ✅ Old sessions auto-convert
- ✅ Large sessions (1.7GB+) process efficiently
- ✅ Box plot renders with outliers
- ✅ "Data Points" checkbox works
- ✅ Jitter disabled by default
- ✅ Jitter positioning correct
- ✅ All scatter overlay interactions work

---

## Migration Notes

### For REV11 Development
1. REV11 is a clean copy of REV10
2. All fixes are inherited
3. New features can be added without affecting REV10
4. REV10 remains production-stable

### For Future Versions
- Follow same pattern: Freeze → Copy → Develop
- Always maintain frozen snapshot for rollback
- Test thoroughly before promoting to production

---

## Documentation References

See related documentation:
- `BOXPLOT_OUTLIER_FIX.md` - Outlier control fix
- `JITTER_CONTROL_FIX.md` - Jitter default & positioning fix
- `REV10_LAUNCH_SUMMARY.md` - REV10 transition details
- Session loading fixes documented in code comments

---

## Contact & Support

For issues with REV10:
1. Check frozen snapshot status
2. Review documentation
3. Consider rolling back to previous stable version
4. Document issue for REV11 improvements

**REV10 is locked and stable. Use REV11 for new development.**
