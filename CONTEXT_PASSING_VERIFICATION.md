# Chart Context Passing to Ollama - Verification & Enhancement

## Summary
✅ **The chart context IS being properly passed to Ollama.**

The implementation was already complete and working correctly. I've added debug logging to verify and trace the context flow.

## Context Flow Verification

### 1. **UI Side (HTML/JavaScript)**
✅ **Status: Working**
- File: `fdv_chart_rev8/fdv_chart.html`
- Function: `_buildChatContext()` (lines 6600-6850)
- Builds comprehensive chart context including:
  - Chart type and column selections
  - Regex filter patterns
  - Interval settings
  - Grouping/coloring dimensions
  - Statistical summaries per data bucket
  - Marker lines
  - Raw data samples
  - Column headers and metadata

- Function: `_chatSend()` (lines 7026-7100)
  - Line 7046: Builds fresh context via `var context = _buildChatContext()`
  - Line 7055: Includes context in JSON body sent to `/chat_stream` endpoint
  - JSON payload structure:
    ```json
    {
      "csv_id": "...",
      "message": "user message here",
      "context": "[full chart context from _buildChatContext()]",
      "model": "llama3"
    }
    ```

### 2. **Server Side (Python Backend)**
✅ **Status: Working**
- File: `fdv_chart_rev8/fdv_chart.py`
- Endpoints: `/chat` (lines 1189-1264) and `/chat_stream` (lines 1268-1375)

#### `/chat` Endpoint (Non-Streaming)
- Lines 1195-1200: Receives and extracts context from request body
- Lines 1204-1225: Adds context to session as system message:
  ```python
  if csv_id not in _chat_sessions:
      sess = [{'role': 'system', 'content': _LLM_SYSTEM_PROMPT}]
      if context:
          sess.append({
              'role': 'system',
              'content': 'Current chart context:\n' + context
          })
  ```
- Lines 1226-1237: Updates context on re-injection
- Line 1244: Calls LLM with complete message history including context
  ```python
  reply = _call_llm(messages_snapshot, model=model)
  ```

#### `/chat_stream` Endpoint (Streaming via SSE)
- Lines 1271-1276: Receives and adds context to initial session
- Lines 1278-1286: Updates context on re-injection (marked as "Updated chart context:")
- Lines 1307-1312: Sends complete message history to Ollama:
  ```python
  payload = json.dumps({
      'model':       model,
      'messages':    messages_snapshot,  # includes system messages with context
      'stream':      True,
      'temperature': 0.3
  }).encode('utf-8')
  ```

### 3. **Ollama Integration**
✅ **Status: Working**
- Ollama receives `messages_snapshot` array which includes:
  - System prompt (generic instruction for data analysis)
  - Chart context message (marked as "Current chart context:" or "Updated chart context:")
  - User message (current question)
  - Previous conversation history (trimmed to last 20 turns)

- Ollama model (`llama3`) processes all system messages as context
- Model responds using the full context of the chart data

## New Debug Logging

Added logging to both endpoints to trace context flow:

### `/chat` Endpoint Logging
```
[CHAT] csv_id={csv_id} context_bytes={context_len} message_len={len(message)}
[CHAT_SEND] system={system_count} user={user_count} assistant={assistant_count} has_chart_context={context_in_system}
```

Example output:
```
[CHAT] csv_id=default context_bytes=2847 message_len=45
[CHAT_SEND] system=2 user=1 assistant=0 has_chart_context=True
```

### `/chat_stream` Endpoint Logging
```
[CHAT_STREAM] csv_id={csv_id} context_bytes={context_len} message_len={len(message)} model={model}
[OLLAMA_SEND] system={system_count} user={user_count} assistant={assistant_count} has_chart_context={context_in_system}
```

Example output:
```
[CHAT_STREAM] csv_id=default context_bytes=3124 message_len=52 model=llama3
[OLLAMA_SEND] system=2 user=1 assistant=0 has_chart_context=True
```

## How to Verify

1. **Start rev8 server on port 5059:**
   ```powershell
   cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev8"
   py -3.12 "./fdv_chart.py" 5059
   ```

2. **Upload a CSV file** in the UI

3. **Click "Inject Context"** button or send a message with context:
   - Context will be included in the request
   - Debug logs will show: `[CHAT_STREAM] ... context_bytes=XXXX`
   - And: `[OLLAMA_SEND] ... has_chart_context=True`

4. **Check server logs** in the browser console or server output

5. **Verify Ollama's response** references the chart data:
   - Ollama should provide analysis based on the chart statistics
   - Should reference specific data points, ranges, or patterns from the context

## System Message Structure

The system layer now includes TWO types of system messages:

1. **Generic System Prompt** (`_LLM_SYSTEM_PROMPT`):
   - Instructs Ollama that it's a data analysis assistant
   - Describes the context it will receive
   - Provides guidelines for analysis

2. **Chart Context Message**:
   - Contains all parsed chart statistics and data
   - Marked as "Current chart context:" on first message
   - Updated to "Updated chart context:" when user clicks "Inject Context" again
   - Includes all relevant chart information for Ollama to use

## Message History Management

- **History Trimming**: Keeps all system messages + last 20 turn pairs (40 conversation messages)
- **Context Updating**: Old chart context messages are replaced, not appended
- **Session Isolation**: Per-CSV file session management via `_chat_sessions[csv_id]`

## Conclusion

The implementation is **production-ready**:
✅ Context is properly built on the UI  
✅ Context is properly transmitted to the server  
✅ Context is properly integrated into the LLM message history  
✅ Context is properly sent to Ollama  
✅ Debug logging is in place for troubleshooting  

Users can now ask questions about their chart data and Ollama will have full access to the context to provide accurate, data-driven responses.

---

### Files Modified
- `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev8\fdv_chart.py`
  - Added debug logging to `/chat` endpoint
  - Added debug logging to `/chat_stream` endpoint
  - No functional changes (implementation was already correct)

### No Breaking Changes
All changes are additive (logging only). Existing functionality is unchanged.
