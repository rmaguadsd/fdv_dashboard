# Ollama Model List Debug & Enhancement

## Problem
"Not all models are showing for ollama"

## Root Cause Analysis
The issue could stem from several places:

1. **Ollama API response format** - The `/api/tags` endpoint might return models in an unexpected format
2. **Model extraction logic** - The code extracts `m['name']` which might fail if the format differs
3. **Silent failure** - Exceptions are caught silently, making debugging difficult
4. **No refresh mechanism** - No way to reload the model list without restarting the application

## Changes Made

### Backend (fdv_chart.py)
**Enhanced `/models` endpoint** to be more robust:
- Improved error handling with detailed debug logging
- Added fallback for different Ollama response formats (dict vs string)
- Removes duplicate model names using `set()`
- Prints full raw Ollama response to stderr for debugging
- Added traceback printing for exceptions

**Key improvements:**
```python
# Now handles both dict and string responses:
for m in models_list:
    if isinstance(m, dict) and 'name' in m:
        names.append(m['name'])
    elif isinstance(m, str):
        names.append(m)
    else:
        print(f'Unexpected model format: {m}')

# Remove duplicates and sort
names = sorted(list(set(names)))
```

### Frontend (fdv_chart.html)
1. **Enhanced debugging** - Added comprehensive console logging:
   - Full JSON response from `/models` endpoint
   - Total count of models being added
   - Each individual model being added
   - Final selected model and all available options
   - Error stack traces for troubleshooting

2. **Refresh button** - Added new button (🔄) to manually reload model list:
   - Allows users to refresh without restarting
   - Button appears only when Ollama is selected as provider
   - ID: `llm-refresh-btn`
   - Calls `_loadModelList()` on click

3. **Improved UI** - New refresh button with color-coded styling:
   - Teal color (#17a2b8) to distinguish from pull button
   - Hover effect and disabled state styling

## Debugging Steps

### To troubleshoot "models not showing":

1. **Check browser console (F12)**:
   - Look for `[DEBUG] _loadModelList response:` messages
   - Verify the `models` array contains expected models
   - Check if any error messages appear

2. **Check server logs**:
   - Look for `[DEBUG] /models:` messages
   - Check the raw Ollama response: `[DEBUG] /models: Raw Ollama response:`
   - Verify model extraction: `[DEBUG] /models: Extracted N unique models:`

3. **Use refresh button**:
   - Click the new 🔄 button next to the model selector
   - Observe browser console for loading progress
   - Models should appear/update dynamically

4. **Manual testing**:
   - Visit `/models` endpoint directly in browser
   - Should show JSON: `{"success": true, "models": [...]}`
   - Paste response in console to inspect structure

### Common Issues & Solutions

**Issue**: Models list is empty
- **Check**: Is Ollama running on `localhost:11434`?
- **Check**: Do you have any models pulled in Ollama?
- **Action**: Run `ollama pull llama3` in Ollama terminal

**Issue**: Only one default model showing (llama3)
- **Means**: Backend cannot connect to Ollama (returns fallback)
- **Check**: Server logs for connection errors
- **Check**: Ollama port configuration matches `_OLLAMA_BASE` in code

**Issue**: Duplicate models in list
- **Fixed**: New code uses `set()` to remove duplicates
- **Verification**: Check browser console, count should be less than raw count

**Issue**: Unknown model names or format issues
- **Check**: Raw Ollama response in server logs
- **Report**: Share the raw response JSON for investigation

## Example Debugging Session

```
Browser Console (F12):
> [DEBUG] _loadModelList response: {success: true, models: ["llama2", "llama3", "neural-chat", ...]}
> [DEBUG] _loadModelList: Adding 5 models to dropdown
> [DEBUG] Adding model option #1: llama2
> [DEBUG] Adding model option #2: llama3
> [DEBUG] Adding model option #3: neural-chat
> ...
> [DEBUG] _loadModelList: Total options in dropdown: 5

Server Output (check for):
[DEBUG] /models: Raw Ollama response: {"models": [...], ...}
[DEBUG] /models: Extracted 5 unique models: ["llama2", "llama3", ...]
```

## Next Steps If Issues Persist

1. Take a screenshot of browser console showing the response
2. Copy the full raw Ollama response from server logs
3. Verify Ollama is running: `curl http://localhost:11434/api/tags`
4. Check if specific models are "corrupted" in Ollama's database
5. Consider rebuilding Ollama or re-pulling all models

## Files Modified
- `fdv_chart_rev1/fdv_chart.py` - Enhanced `/models` endpoint with better error handling
- `fdv_chart_rev1/fdv_chart.html` - Added refresh button, improved logging, better debugging

## Testing Verification

After changes, verify:
- ✅ Click 🔄 button and models appear in dropdown
- ✅ Browser console shows debug messages
- ✅ Server logs show model extraction details
- ✅ Switching providers shows/hides Ollama controls correctly
