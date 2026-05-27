# REV13 Chat Analysis Coverage

## Overview
REV13 includes a comprehensive **multi-modal AI analysis engine** with two backends (Ollama and ConnectMaiGPT) supporting both single-turn analysis and multi-turn interactive chat with real-time chart manipulation.

---

## 1. Analysis Modes & Prompting

### A. Single-Chart Analysis (`_buildChartPrompt`)
**When used:** Single chart with optional in-chart color groups  
**Context includes:**
- Chart type (scatter, histogram, cumproba, rcdf, etc.)
- Column names and regex patterns
- **Overall statistics:** n, min, max, mean, std, p5, p25, p50 (median), p75, p95
- **Per-group breakdown** (if color-by or split-by groups exist):
  - Each group labeled with its statistics
  - X and Y stats separately
- **Reference markers** (vertical/horizontal lines) with value, label, and position relative to data range
- **Active column filters** count
- **Data interval information**

**Task instruction:** 
```
Analyze this chart responding in bullet points (•).
- Cover: distribution shape, central tendency (mean/median), 
  spread (std/range), and outliers
- Use concrete numbers
- If multiple groups: dedicate one bullet per group + comparative bullet
- Do NOT truncate — be complete
```

### B. Split-Chart Comparative Analysis (`_buildSplitPrompt`)
**When used:** Split-charts active (one chart per unique value of split-by column)  
**Context includes:**
- Chart type and dimensions
- **Split-by column** name and number of tiles
- **Per-tile statistics** for each split:
  - Tile key (group identifier)
  - Sample count (n)
  - X and Y statistics
- Reference markers with axis and value
- Active filters and data interval

**Task instruction:**
```
Comparative analysis across N groups.
- One bullet minimum per group with key stats
- Add bullets for:
  * Ranking by median/mean
  * Widest spread (std deviation)
  * Extreme outliers
  * Groups crossing marker lines
- Cover every group — do not truncate
```

### C. Two-Dimensional Cross Analysis (`_buildCrossPrompt`)
**When used:** Both split-charts AND color-groups active (tile × color grid)  
**Context includes:**
- Chart type and dimensions
- Split-chart dimension (tile keys and count)
- Color-group dimension (color keys and count)
- **Cross-tabulated statistics:** [tile × color] combinations
  - Sample count per cell
  - X and Y stats per cell
- Reference markers
- Active filters and data interval

**Task instruction:**
```
Two-dimensional comparative analysis: [split-dimension × color-dimension]
- Within each tile: one bullet per color group with key stats and comparison
- Across tiles: one bullet per color group noting changes tile-to-tile
- Final bullet: the standout [tile × color] combination and why
- Note any marker line crossings
- Cover every group and tile — do not truncate
```

---

## 2. Statistical Coverage

All three analysis modes compute and send:

| Statistic | Formula | Use Case |
|-----------|---------|----------|
| **n** | Count | Sample size |
| **min** | Minimum value | Data range lower bound |
| **max** | Maximum value | Data range upper bound |
| **mean** | Sum / n | Central tendency |
| **std** | √(variance) | Spread/dispersion |
| **p5** | 5th percentile | Lower tail outliers |
| **p25** | 25th percentile (Q1) | Lower quartile |
| **p50** | Median | Robust central tendency |
| **p75** | 75th percentile (Q3) | Upper quartile |
| **p95** | 95th percentile | Upper tail outliers |

**Percentile calculation:** `sorted[floor(n * percentile)]` — discrete approximation for accurate data representation.

---

## 3. Interactive Chart Manipulation Commands

Chat responses can emit special tokens that modify the chart in real-time:

### A. Marker Addition
**Token:** `[MARKER: x=<value>:<label>]` or `[MARKER: y=<value>:<label>]`

Example responses:
```
[MARKER: x=27000:user_threshold]
• Added vertical marker at x=27000 as requested

[MARKER: y=0.99:confidence_bound]
• Highlighted 99% confidence interval
```

**Behavior:**
- Multiple markers can be emitted in one response
- Markers are added to existing ones
- Position context automatically calculated: below/within/above data range

### B. Marker Clearing
**Token:** `[CLEAR_MARKERS]`

Example response:
```
[CLEAR_MARKERS]
• Cleared all reference lines to reset the view
```

### C. Summary Refresh
**Token:** `[SUMMARY]`

Effect: Re-run the statistical summary table and refresh context for next analysis.

### D. Analysis Re-run
**Token:** `[ANALYZE]`

Effect: Re-run the AI analysis panel with current chart data.

---

## 4. Multi-Turn Chat Session Management

### Session Storage
- **Per-dataset basis:** Each CSV ID gets its own conversation history
- **Lock-protected:** Thread-safe session dictionary with mutex
- **History trimming:** Keeps last 20 turn pairs (40 messages) + all system prompts
  - Prevents unbounded growth
  - Older turns automatically discarded

### Context Injection Modes
1. **First message:** System prompt + optional initial context (chart statistics)
2. **Mid-conversation re-inject:** Replace existing context message with updated statistics
   - Triggered by user clicking "↻ Inject context" button
   - Allows user to push fresh chart state mid-conversation
3. **Session clear:** Reset history and re-initialize with fresh system prompt

### System Prompt
Universal system prompt for all turns:
```
You are a data analysis assistant in an engineering test-data viewer.
User gives you statistics from parsed log file charts.
ALWAYS respond in bullet points (•) — one per key finding.
Cover distribution shape, central tendency, spread, outliers.
Use engineering language and concrete numbers.
Never pad with generic disclaimers.
```

Plus inline instructions for MARKER, [SUMMARY], [ANALYZE] token execution.

---

## 5. Streaming vs. Non-Streaming Modes

### `/chat` Endpoint (Non-streaming)
- Single HTTP request → single response
- Session history maintained per CSV ID
- History trimming applied
- Full LLM response buffered before return
- Latency: 5-30s for llama3 depending on prompt size

### `/chat_stream` Endpoint (Streaming)
- **Server-Sent Events (SSE)** for token-by-token delivery
- Same session management as `/chat`
- Tokens arrive as generated (low-latency user feedback)
- Automatic history append post-stream
- Model selection per request

---

## 6. Supported Analysis Queries

### Inline Query Tokens
Chat supports dynamic data queries via:
```
[QUERY: col=COLUMN_NAME, filter=EXPRESSION, group=COLUMN_NAME, agg=mean]
```

**Parameters:**
- `col` — Column to aggregate
- `filter` — WHERE clause (e.g., `>0.001`, `==PASS`, `!=NULL`)
- `group` — GROUP BY column
- `agg` — Aggregation: `mean` (default), `count`, `min`, `max`, `std`, `all` (full stats)

**Example:**
```
[QUERY: col=RBER, filter=>0.001, group=DUT, agg=mean]
```

Effect: Groups by DUT, filters RBER > 0.001, returns mean per group.

---

## 7. LLM Backend Support

### A. Ollama (Default)
- **Local model:** llama3
- **API:** POST to `http://localhost:11434/api/chat`
- **Timeout:** 180 seconds (long conversations)
- **Temperature:** 0.3 (deterministic, factual)
- **Stream support:** Yes (native Ollama `stream=True`)

### B. ConnectMaiGPT (Enterprise)
- **Protocol:** MCP (Model Context Protocol)
- **Host:** `fmgnpsgautoplt01.elements.local`
- **Auth:** Token-based (`b9e1c4f2-6a3d-4f8b-9c2e-7d1a5b3e6f4c`)
- **Tool methods:**
  - `connectmaigpt` — session establishment
  - `askmaigpt` — single-turn analysis
- **Model selection:** GPT-4 or other MaiGPT-available models
- **Timeout:** 120 seconds

---

## 8. Error Handling & Logging

### Logging Destinations
1. **Startup log:** `fdv_chart_startup.log`
2. **Debug log:** `fdv_chart_debug.log`
3. **Server log:** `server.log` (in rev13 root)

### Chat-Specific Logs
```
[CHAT] csv_id=<ID> context_bytes=<N> message_len=<L>
[CHAT_SEND] system=<count> user=<count> assistant=<count> has_chart_context=<bool>
[CHAT_STREAM] csv_id=<ID> context_bytes=<N> message_len=<L> model=<model>
```

### Error Responses
- Empty message → ValueError
- LLM timeout → HTTP 200 with error field
- ConnectMaiGPT unavailable → fallback gracefully documented

---

## 9. UI Components

### Chat Drawer
- **Floating window** with draggable title bar (`#chat-grip`)
- **Resizable** via corner (`#chat-resize`, `#chat-resize-topleft`)
- **Message history** (`#chat-messages`) — scrollable, auto-scroll on new message
- **Input field** (`#chat-input`) — multiline textarea
- **Send button** (`#chat-send-btn`) — disabled during processing
- **Toggle button** (`#chat-toggle-btn`) in plot bar to show/hide drawer

### Header Controls
- **Clear history button** — reset conversation
- **Inject context button** (↻) — re-send current chart stats mid-conversation
- **Font size selector** — dropdown for text zoom (persisted in localStorage)
- **Title bar** with CSV ID or dataset name

### Message Styling
- **User messages** — blue background
- **Assistant messages** — gray background
- **System notes** — italicized, informational (e.g., "Context armed", "Chart updated")
- **Analysis panel** (`#analysis-text`) — separate rich-text area for summary output

---

## 10. Data Context Building

### Dynamic Context Assembly
Context is built on-demand using exact same bucketing logic as visual chart:

1. **Read current UI state:**
   - Selected columns (X, Y, color, split)
   - Regex patterns for each
   - Chart type
   - Active column filters
   - Data interval (min/max) settings

2. **Bucket data by group key:**
   - Apply filters (column-level filters + interval bounds)
   - Extract values using regex patterns
   - Group by dimension (color or split-by)
   - Skip null/unparseable values

3. **Compute statistics per bucket:**
   - Numeric extraction and sorting
   - Percentile calculation (discrete)
   - Mean, std deviation, min/max

4. **Format for LLM:**
   - Human-readable text with concrete numbers
   - One section per analysis mode (single, split, cross)
   - Reference marker context
   - Filter and interval metadata

**Key:** Context always reflects **current view state** including filters and transformations.

---

## 11. Chat Feature Checklist

| Feature | Status | Coverage |
|---------|--------|----------|
| **Single-chart analysis** | ✅ | Distribution, central tendency, spread, outliers |
| **Multi-group comparison** | ✅ | Per-group stats + cross-group rankings |
| **Split-chart analysis** | ✅ | Tile-by-tile breakdown + comparative bullets |
| **2D cross-analysis** | ✅ | Tile × color grid with per-cell stats |
| **Marker commands** | ✅ | Add x/y markers, clear, with context calculation |
| **Summary/analyze tokens** | ✅ | Re-run analysis, refresh context |
| **Multi-turn sessions** | ✅ | Per-CSV history, trimmed to 20 turn pairs |
| **Context re-injection** | ✅ | Mid-conversation chart updates |
| **Streaming responses** | ✅ | Token-by-token via SSE |
| **LLM selection** | ✅ | Ollama or ConnectMaiGPT per request |
| **Data queries** | ✅ | [QUERY: ...] token parsing and execution |
| **Percentile stats** | ✅ | p5, p25, median, p75, p95 + mean/std |
| **Error handling** | ✅ | Logged, user-facing errors, graceful fallback |
| **Logging** | ✅ | Startup, debug, server, chat-specific |

---

## 12. User Workflow Example

**Scenario:** Analyzing RBER (Read Bit Error Rate) split by DUT (Device Under Test)

1. **User loads CSV:** RBER column, splits by DUT
2. **System displays:** 5 tiles (one per DUT), each with scatter plot
3. **Chat initialized:** "📊 Ready to analyze — current chart statistics will be sent with your first message"
4. **User asks:** "Analyze these results"
5. **System:**
   - Builds split-chart context (5 tiles × RBER/count stats)
   - Sends to Ollama with system prompt + analysis instruction
   - Receives bullet-point response ranking DUTs by median RBER, noting outliers
6. **User asks:** "Mark the 99.5% percentile"
7. **AI responds:**
   ```
   [MARKER: y=0.00542:p995_threshold]
   • Added horizontal marker at y=0.00542 (99.5th percentile)
   • DUT-A and DUT-C exceed this threshold; investigate further
   ```
8. **System:**
   - Executes MARKER command
   - Redrawchart with new reference line
   - User sees visual context immediately
9. **User asks:** "Group by result type within DUT-A and average"
10. **AI responds:**
    ```
    [QUERY: col=RBER, filter=DUT==DUT-A, group=RESULT, agg=mean]
    • PASS: mean=0.00087
    • FAIL: mean=0.00156 (79% higher failure rate)
    • TIMEOUT: mean=0.00201 (highest error rate)
    ```

---

## Summary

**REV13 Chat Analysis** is a **sophisticated, multi-modal system** supporting:

✅ **Three analysis paradigms** (single-chart, split-chart, 2D-cross)  
✅ **Comprehensive statistics** (10-percentile breakdown)  
✅ **Real-time chart manipulation** (marker commands, context injection)  
✅ **Multi-turn conversation** (per-dataset session history, auto-trimming)  
✅ **Dual LLM backends** (Ollama default, ConnectMaiGPT enterprise)  
✅ **Streaming + non-streaming modes** (latency optimization)  
✅ **Dynamic data queries** (inline [QUERY:...] token support)  
✅ **Full logging** (startup, debug, server, chat-specific)  

**Coverage:** From exploratory analysis through detailed statistical investigation to actionable insights with interactive chart annotation.

