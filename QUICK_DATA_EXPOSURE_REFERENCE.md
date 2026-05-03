# Quick Reference: Data Exposure Channels

## 🎯 The 5 Main Channels

### 1️⃣ Automatic Chart Context (MOST IMPORTANT)
**What**: Every chat message includes real-time chart statistics
**Where**: Built in `_buildChatContext()` (HTML, lines 6600–6900)
**What Ollama gets**:
- Chart type, column names, regex patterns
- Active filters & intervals
- Stats for each group (n, mean, std, min, max)
- 20–50 sample raw data rows
- Reference markers
- Statistical summary from Summary panel

**Typical size**: 3–5 KB per message
**Latency**: ~100ms

---

### 2️⃣ Streaming Real-Time Tokens
**What**: Ollama's response arrives token-by-token (not all-at-once)
**Where**: `/chat_stream` endpoint in Python
**Browser sees**: Words appearing in 1–2 seconds instead of 10+ seconds

**Example flow**:
```
User: "Analyze variance by DUT"
  ↓ (3 sec wait)
Ollama: "I see 3 DU..."
Ollama: "I see 3 DUTs with si..."
Ollama: "I see 3 DUTs with significant variance..."
```

---

### 3️⃣ QUERY Tokens (Advanced)
**What**: Ollama requests computed aggregations by embedding `[QUERY: ...]` tokens

**Syntax**:
```
[QUERY: col=COLUMN, filter=EXPRESSION, group=BY_COLUMN, agg=mean|count|min|max|std|all]
```

**Ollama says**: "Let me check the 95th percentile..."
```
[QUERY: col=RBER, agg=max]
```

**Server responds**: 
```
Query Result: 0.98765
```

**Status**: ✅ Defined, ⏳ Evaluation pending, ⏳ Injection pending

---

### 4️⃣ Session History & Context Replacement
**What**: Ollama remembers prior chat turns (conversation context)
**How**: Server maintains message array per CSV file
**Max turns kept**: Last 20 user+assistant pairs (configurable)
**Context refresh**: When user clicks **Inject**, old context is replaced with new

**Memory management**:
```python
_CHAT_MAX_TURNS = 20  # Edit this to remember more/less
```

---

### 5️⃣ ConnectMaiGPT Cloud Integration
**What**: Alternative to local Ollama (uses gpt4 or claude-3-5-sonnet)
**When to use**: For complex reasoning tasks, or if Ollama unavailable
**Data sent**: Same chart context as Ollama, but to cloud API

**⚠️ Privacy note**: Data leaves your local system → use only with non-sensitive data

---

## 📊 What Data Reaches Ollama?

### Always Included (Unless no chart selected)
```
✓ Chart metadata (type, X column, Y column, filters)
✓ Active interval filters (X range, Y range)
✓ Grouping info (color-by, split-by columns)
✓ Statistics per group (count, mean, std, min, max)
✓ Reference marker lines
✓ Statistical summary data
```

### Conditionally Included
```
✓ Raw data sample (if user sets "Rows: 50" in chat menu)
  - Default: 50 rows, evenly spaced
  - User can change: 0, 20, 50, 100, 200, 500
✓ QUERY results (only if Ollama requests them)
✓ Prior chat messages (conversation history)
```

### NOT Included
```
✗ Raw binary data (all data read from file → kept in browser only)
✗ Plot image/PNG (chart is Canvas, not sent to LLM)
✗ User password/credentials
✗ Log file name (only CSV ID passed)
```

---

## 🚀 How to Use Effectively

### Step 1: Load Data & Create Chart
```
1. Upload CSV file
2. Select X column
3. Select Y column (or use histogram)
4. (Optional) Add filters, colors, split-chart
5. Plot renders
```

### Step 2: Open Chat
```
1. Click "💬 Chat" button in plot toolbar
2. Chat drawer appears
3. First message auto-includes chart context
```

### Step 3: Ask Question
```
User: "What's unusual about UNIT_02?"
```

### Step 4: Inject Updated Context (After Plot Changes)
```
1. Modify plot (new filter, new grouping, etc.)
2. Click "⟳ Inject" button
3. Next chat message includes UPDATED stats
```

---

## 🔧 Configuration Options

### Raw Data Sample Size
**Location**: Chat drawer menu, "Rows:" selector
```
none   → 0 rows (stats only)
20     → 20 rows (quick analysis)
50     → 50 rows (default, balanced)
100    → 100 rows (more detail)
200    → 200 rows (detailed)
500    → 500 rows (max verbosity)
```

### LLM Provider
**Location**: Chat drawer menu, "Provider:" selector
```
Ollama (default)       → localhost:11434, free, fast
ConnectMaiGPT (cloud)  → gpt4, claude-3-5-sonnet, slower
```

### Chat Font Size
**Location**: Chat drawer menu, "Font:" selector
```
XS → 0.75em (tiny)
S  → 0.82em (small)
M  → 0.90em (medium)
L  → 1.0em (large) ← NEW! larger than before
XL → 1.15em (huge)
```

### Context Max Turns (Edit Python Code)
**Location**: `fdv_chart.py` line ~40
```python
_CHAT_MAX_TURNS = 20  # Change to 10 for shorter memory, 50 for longer
```

---

## 📋 Data Exposure Checklist

Before asking Ollama a question, confirm:

- [ ] CSV file is loaded
- [ ] Chart is displayed (you see the plot)
- [ ] Chat drawer is open
- [ ] "Provider" is set to your preferred LLM
- [ ] (Optional) "Rows" is set to desired sample size
- [ ] (Optional, after plot changes) Clicked "⟳ Inject"

---

## 🐛 Troubleshooting

### Problem: Ollama Says "No Context"
**Solution**: 
```
1. Check: Is a chart displayed?
2. Check: Do you see "Current chart context:" in chat?
3. If not: Click "⟳ Inject" button to force send
```

### Problem: Stale Data (Ollama Referencing Old Stats)
**Solution**:
```
1. After changing plot → Click "⟳ Inject"
2. Then ask follow-up question
3. Ollama now sees updated stats
```

### Problem: Ollama Response Too Slow
**Solution**:
```
Option 1: Reduce "Rows" from 50 to 20
Option 2: Switch from streaming to non-streaming (in code)
Option 3: Use ConnectMaiGPT for faster cloud inference
```

### Problem: "QUERY" Tokens Not Working
**Solution**: This is a known pending feature. Workaround:
```
Ask Ollama directly: "What is the mean of column X by group Y?"
(Server will compute once QUERY feature is implemented)
```

---

## 📈 Performance Tips

| Goal | Adjustment |
|------|------------|
| Faster responses | Reduce "Rows: 50" → "Rows: 20" |
| Better analysis | Increase "Rows: 50" → "Rows: 200" |
| Longer memory | Edit `_CHAT_MAX_TURNS = 50` in Python |
| Shorter memory | Edit `_CHAT_MAX_TURNS = 10` in Python |
| Cloud LLM | Switch "Provider" to ConnectMaiGPT |

---

## 🔐 Privacy Matrix

| Data | Ollama (Local) | ConnectMaiGPT |
|------|---|---|
| CSV rows | ✅ Stays local | ❌ Sent to cloud |
| Filtered data | ✅ Stays local | ❌ Sent to cloud |
| Chart stats | ✅ Stays local | ❌ Sent to cloud |
| Chat history | ✅ Stays local | ❌ Sent to cloud (session scoped) |
| User identity | ✅ No tracking | ⚠️ Per ConnectMaiGPT policy |

---

## 📚 References

- **Full documentation**: `DATA_EXPOSURE_STRATEGY.md`
- **Browser code**: `fdv_chart_rev8/fdv_chart.html` lines 6600–7100
- **Server code**: `fdv_chart_rev8/fdv_chart.py` lines 1200–1350
- **Logs**: `fdv_chart_startup.log` (search for `[CHAT]` prefix)

---

**Version**: 1.0 | **Last Updated**: May 2, 2026
