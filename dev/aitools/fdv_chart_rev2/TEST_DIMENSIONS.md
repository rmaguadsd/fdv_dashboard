# Testing Dimension Restoration

## Setup
1. Open http://localhost:5059
2. Press **F12** to open Developer Tools
3. Go to **Console** tab
4. **Clear any existing logs** (right-click → Clear)

## Test Steps

### Step 1: Parse Data
- Use the **Path/Files/Dir** input to load a CSV file
- Wait for the table to populate completely
- Verify data appears in the table

### Step 2: Add Color-by Dimension
- Locate the **"Color by:"** section (left panel)
- Click the **+** button next to it
- In the first row that appears:
  - Click the dropdown (showing "— col —")
  - Select a column (e.g., "RESULT", "DUT", or "PAGETYPE")
- **Check Console**: You should see `[_gdimChanged]` log message

### Step 3: Add Split-chart Dimension
- Locate the **"Split-chart:"** section (below Color by)
- Click the **+** button next to it
- In the first row that appears:
  - Click the dropdown (showing "— col —")
  - Select a different column (e.g., "PAGETYPE" or "Type")
- **Check Console**: You should see `[_gdimChanged]` log message

### Step 4: Create and Plot
- Select **X column** from the main dropdown
- Select **Y column** from the main dropdown
- Click **Plot** button
- Verify the plot appears and uses your selected dimensions

### Step 5: Save Session
- Scroll down to the **Sessions** panel
- Enter a **Session Name** (e.g., "TestSession_001")
- Click **Save** button
- **Check Console**: Look for `[_recipeSnapshot]` message showing:
  - `__colorDims`: Should NOT be empty `[]`
  - `__scDims`: Should NOT be empty `[]`
  - Example: `"[{\"col\":\"RESULT\",...}]"`

### Step 6: Clear Dimensions
- Click the **×** button on the Color-by dimension row (deletes it)
- Click the **×** button on the Split-chart dimension row (deletes it)
- Verify both sections now show only empty dropdowns

### Step 7: Load Session
- Find the **"Saved sessions"** dropdown
- Select your saved session name from the list
- Click **Load** button
- **EXPECTED**: Color-by and Split-chart dimensions should reappear

### Step 8: Check Console
After loading, look for these messages in Console:
1. `[_recipeApply] Restoring dimensions immediately` OR `Deferring dimension restoration`
2. `[_recipeApply] Found __colorDims:` with the JSON array
3. `[_recipeApply] Found __scDims:` with the JSON array
4. `[_gdimSetDims] Restoring color dims:` with the array
5. `[_gdimSetDims] Restoring sc dims:` with the array

## What to Report

### If it WORKS:
- Congratulations! Dimensions are being restored properly.
- Verify that the plot updates to use the restored dimensions.

### If it DOESN'T WORK:
Copy the console output and report:

1. **After saving (Step 5)**:
   - What does `[_recipeSnapshot]` show for `__colorDims` and `__scDims`?
   - Are they empty `[]` or do they have values?

2. **After loading (Step 7)**:
   - Do you see `[_recipeApply]` messages in the console?
   - Does it say "Restoring immediately" or "Deferring"?
   - Do the `__colorDims` and `__scDims` keys appear in the logs?
   - Are they showing the correct JSON data?

3. **Any error messages**?
   - Look for red text (errors) in the console

## Console Output Examples

### Good Save Log:
```
[_recipeSnapshot] Captured dims - color: "[{"col":"RESULT","colIdx":2,"rx":""}]" split: "[]" sc: "[{"col":"PAGETYPE","colIdx":5,"rx":""}]"
[_recipeSnapshot] Full snapshot: {...}
```

### Good Load Log:
```
[_recipeApply] Restoring dimensions immediately (headers loaded)
[_recipeApply] Found __colorDims: "[{"col":"RESULT","colIdx":2,"rx":""}]"
[_recipeApply] Calling _gdimSetDims for color with 1 dims
[_gdimSetDims] Restoring color dims: [{"col":"RESULT"...}]
[_gdimSetDims] select options available: ["— col —", "WL", "BLK", "SB", ...]
```

