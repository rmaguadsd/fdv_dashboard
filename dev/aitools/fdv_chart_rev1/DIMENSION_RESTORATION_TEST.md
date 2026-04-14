# Dimension Restoration Test Guide

## What to Test

This guide walks through testing the color-by and split-chart dimension restoration from saved sessions.

## Step 1: Load a Data File

1. Open http://localhost:5059
2. Parse a CSV file using the Path/Files/Dir input
3. Wait for the table to populate

## Step 2: Set Dimensions

1. **Color-by**: Click the "+" button next to "Color by:" and select a column
2. **Split-chart**: Click the "+" button next to "Split-chart:" and select a column

## Step 3: Create a Plot

1. Select X and Y columns
2. Click "Plot" button to generate a plot
3. Verify the plot uses the selected dimensions

## Step 4: Save a Session

1. Enter a session name in the "Session Name" field
2. Click "Save" button in the Sessions panel
3. You should see a confirmation message

## Step 5: Clear the Dimensions

1. Click the "×" button on the color-by dimension row to remove it
2. Click the "×" button on the split-chart dimension row to remove it
3. Verify both are now empty (only showing "— col —" dropdown)

## Step 6: Load the Session

1. Select your saved session from the "Saved sessions" dropdown
2. Click "Load" button
3. **EXPECTED RESULT**: The color-by and split-chart dimensions should reappear with the values you saved

## What Should Be Restored

- **Color-by dimensions**: Shows the column name in the select dropdown and any regex in the input field
- **Split-chart dimensions**: Shows the split-chart column in the select dropdown

## Troubleshooting

If dimensions don't restore:

1. **Check browser console** (F12 → Console tab):
   - Look for any JavaScript errors
   - Look for "Restoring X dims:" log messages (if debug logging is enabled)

2. **Verify session was saved**:
   - Delete the session and manually edit a saved one if possible
   - Check if `__colorDims` and `__scDims` keys exist in the snapshot

3. **Check if headers are loaded**:
   - The column dropdowns should show all columns from your CSV
   - If showing only "— col —", headers might not be loaded

## Expected Session Format

A properly saved session with dimensions should include keys like:
- `__colorDims`: `"[{\"col\":\"ColumnName\",\"colIdx\":2,\"rx\":\"regex\"}, ...]"`
- `__scDims`: `"[{\"col\":\"ColumnName\",\"colIdx\":3,\"rx\":\"\"}, ...]"`

