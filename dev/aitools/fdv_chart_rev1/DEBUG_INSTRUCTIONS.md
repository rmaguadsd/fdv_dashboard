# Debug Instructions for Dimension Restoration

## Objective
Find out exactly where the dimension restoration is failing by examining the browser console logs.

## Steps

### 1. Open Browser Console
- Open http://localhost:5059
- Press **F12** to open Developer Tools
- Click the **Console** tab

### 2. Test Dimension Restoration

**A. Parse a data file:**
- Use the Path/Files/Dir input to load a CSV file
- Wait for the table to populate

**B. Set dimensions:**
- Click the **+** button next to "Color by:"
- Select a column from the dropdown (e.g., "RESULT" or "DUT")
- Click the **+** button next to "Split-chart:"
- Select a column from the dropdown (e.g., "PAGETYPE")

**C. Create and save a session:**
- Select X and Y columns
- Click "Plot" to verify the plot uses your dimensions
- Enter a session name (e.g., "test123")
- Click "Save" in the Sessions panel
- Wait for the "Saved" confirmation

**D. Clear the dimensions:**
- Click the **×** button next to the Color-by dimension row
- Click the **×** button next to the Split-chart dimension row
- Verify both are now empty

**E. Load the session:**
- Select your saved session from the "Saved sessions" dropdown
- Click "Load"
- **Watch the Console for log messages**

### 3. Examine Console Output

Look for log messages that start with:
- `[_recipeApply]` - Initial restoration attempt
- `[populatePlotSelectors]` - Deferred restoration attempt
- `[_gdimSetDims]` - Setting dimension values
- `[_gdimChanged]` - Updating global arrays

### 4. Report These Details

**What you should see:**
1. Messages indicating dimensions are being restored
2. Messages showing the select options that are available
3. Messages showing the values being set

**What to report:**
- Does "Applying pending recipe snapshot" appear?
- Does "Restoring __colorDims from pending" appear?
- Does "[_gdimSetDims] select options available:" show options?
- What are those options?
- After setting, what does the select.value show?

### 5. Copy Console Output

**To copy all console output:**
1. Right-click in the console
2. Select "Save as..."
3. Or select all (Ctrl+A) and copy (Ctrl+C)
4. Paste the output so we can analyze it

## Key Clues

The logs will show us:
- Whether `_recipeApply()` is being called
- Whether dimensions are in the snapshot (`__colorDims`, `__scDims`)
- Whether `_gdimSetDims()` is being called
- Whether the select options contain the saved column names
- Whether values are being set successfully

## Most Likely Issues

1. **Old session format**: Sessions saved before dimension feature won't have `__colorDims` keys
2. **Headers not loaded**: Select options not populated when restoration is attempted
3. **Column name mismatch**: Select trying to set value to column that doesn't exist in options
4. **Timing issue**: Restoration happening before selects are built

