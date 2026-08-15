# Font Control Feature - Setup & Usage Guide

## ✅ Feature Status
**Font Control has been fully implemented in fdv_chart_rev3.**

### What's Included:
1. **Font Control HTML Section** (lines 903-930 in fdv_chart.html)
   - Select Item dropdown: "Point Size", "Axis Font Size", "Legend Label Font Size"
   - Dynamic Size dropdown: populated based on selection
   - Current value display

2. **JavaScript Functions** (lines 2999-3065 in fdv_chart.html)
   - `_onFontItemSelect()`: Populates size options when item selected
   - `_applyFontSize()`: Applies size change and redraws chart

3. **Variables** (defined early in JavaScript)
   - `_pointSize` (default: 8)
   - `_axisFontSize` (default: 12)
   - `_legendFontSize` (default: 12)

## 🚀 How to Use

### Starting the Server:
```powershell
cd d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev3
& 'C:\Python312\python.exe' fdv_chart.py 5059
```

### Accessing the Application:
Open in browser: **http://localhost:5059**

### Using Font Control:
1. Load a data file and plot it
2. Look for the **"🖥️ Font Control:"** section in the plot controls
3. Select an item from the first dropdown:
   - "Point Size" (4-20px)
   - "Axis Font Size" (8-24px)
   - "Legend Label Font Size" (10-24px)
4. Select a size from the second dropdown
5. The chart updates immediately

## 📋 Troubleshooting

### If font controls don't appear:
1. Make sure you're on **http://localhost:5059** (not just localhost)
2. Load a data file first - controls are in the Plot Panel
3. Clear browser cache (Ctrl+Shift+Delete) and hard refresh (Ctrl+Shift+R)

### If changes don't take effect:
1. Open Developer Tools (F12)
2. Check Console for JavaScript errors
3. Verify `_onFontItemSelect()` and `_applyFontSize()` functions exist:
   ```javascript
   console.log(typeof _onFontItemSelect)  // should show "function"
   console.log(typeof _applyFontSize)     // should show "function"
   ```

### If server won't start:
Make sure you're using Python 3.12:
```powershell
C:\Python312\python.exe --version  # should show Python 3.12.x
```

## 📁 File Locations
- **HTML**: `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev3\fdv_chart.html` (6508 lines)
- **Python Server**: `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev3\fdv_chart.py`

## 🔍 Code References

### HTML (lines 903-930):
- Font Control section with two dropdowns
- IDs: `font-item-select`, `font-value-select`, `font-current-value`
- CSS styled with light gray background (#f9f9f9)

### JavaScript (lines 2999-3065):
- Both functions properly handle all three font types
- Automatically redraws chart with `drawPlot()` after size change
- Current value always displayed

---
**Last Updated**: 2026-04-21  
**Tested On**: Python 3.12.8, Windows PowerShell 5.1
