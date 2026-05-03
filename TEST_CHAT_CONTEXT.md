# Quick Test Guide: Chart Context in Ollama Chat

## What You'll See

✅ **Context IS being passed to Ollama** - verified and working!

## Testing Steps

### 1. Start the Server
```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev8"
py -3.12 "./fdv_chart.py" 5059
```

Then navigate to: **http://localhost:5059/**

### 2. Load Chart Data
- Click "Choose File" and select a CSV/log file with data
- Click "View Results" to load the data
- You should see the chart render

### 3. Test Context Passing

#### Option A: Click "Inject Context" Button
1. In the chat area, click the "⟳ Inject context" button
2. You'll see: "✓ Context armed — current chart statistics will be sent with your next message"
3. Type a question like: "What are the main statistics for this data?"
4. Send the message

**Result**: Ollama will respond using the chart context data

#### Option B: Direct Chat with Context
1. Simply type a question in the chat box
2. Send it
3. Context is automatically included

**Result**: Ollama responds with data-aware answer

### 4. Verify Context is Working

In Ollama's response, you should see:
- References to specific data values from your chart
- Statistical analysis of the loaded data
- Comments about trends or patterns in the data
- NOT generic answers without data reference

### 5. Debug Logging

To see the context flow in detail:

**If running in terminal**, watch for messages like:
```
[CHAT_STREAM] csv_id=default context_bytes=3124 message_len=52 model=llama3
[OLLAMA_SEND] system=2 user=1 assistant=0 has_chart_context=True
```

This shows:
- ✅ Context received (context_bytes=3124)
- ✅ System messages include chart context (has_chart_context=True)
- ✅ Sent to Ollama with 2 system messages, 1 user message

## What the Context Includes

When you "Inject Context" or send a chat message, Ollama receives:

```
SYSTEM MESSAGE 1: You are a data analysis assistant...
SYSTEM MESSAGE 2: Current chart context:
  - Chart type: [histogram/line/scatter/etc]
  - Columns: X=[...], Y=[...]
  - Data grouping: [...]
  - Statistics: min/max/mean/std for each group
  - Filtered intervals: [...]
  - Raw data samples: [...]
  - Column headers and metadata
```

Ollama uses this to answer your questions accurately!

## Expected Behavior

| Your Question | Expected Response |
|---|---|
| "What is the range of values?" | Returns min/max from chart context |
| "Show me statistics" | Provides mean/std/quartiles from context |
| "Are there outliers?" | Identifies based on data in context |
| "What patterns do you see?" | Analyzes trends in the data |

## Troubleshooting

| Issue | Solution |
|---|---|
| "Ollama seems generic" | Click "Inject Context" button and try again |
| "Context bytes = 0" | Make sure data is loaded in the chart first |
| "has_chart_context=False" | The message didn't include context - try again |
| "Ollama connection error" | Make sure Ollama service is running on port 11434 |

---

**Key Point**: The context is **already working**! 🎉  
This guide just shows you how to use and verify it's working correctly.
