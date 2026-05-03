# Data Exposure to Ollama: Complete Documentation Index

## 📚 Documentation Suite

This is a comprehensive suite of guides explaining how the FDV Dashboard exposes chart and table data to Ollama. Choose the document that matches your role and needs.

---

## Quick Navigation

### 🎯 For End Users (How to Use the Chat Feature)

**Start here:** [`QUICK_DATA_EXPOSURE_REFERENCE.md`](./QUICK_DATA_EXPOSURE_REFERENCE.md)

**What you'll learn:**
- The 5 main channels data reaches Ollama
- What data gets sent (and what doesn't)
- How to use the chat interface effectively
- Configuration options (font size, rows, provider)
- Troubleshooting common issues

**Time to read:** 10 minutes

---

### 🔧 For Developers (Architecture & Implementation)

**Architecture Overview:** [`DATA_EXPOSURE_STRATEGY.md`](./DATA_EXPOSURE_STRATEGY.md)

**What you'll learn:**
- Complete data flow from UI to Ollama
- How automatic chart context is built
- Session management & history trimming
- QUERY token system (defined, pending implementation)
- Logging & observability
- Privacy & security considerations

**Time to read:** 20 minutes

---

**Code Walkthrough:** [`IMPLEMENTATION_CODE_WALKTHROUGH.md`](./IMPLEMENTATION_CODE_WALKTHROUGH.md)

**What you'll learn:**
- Exact code paths and function calls
- How `_buildChatContext()` works (lines-by-lines)
- Server-side context reception and processing
- Streaming response handling
- Complete data flow diagrams
- Session history management code

**Time to read:** 30 minutes (with code review)

---

### 📈 For Optimization & Troubleshooting

**Best Practices:** [`BEST_PRACTICES_OPTIMIZATION.md`](./BEST_PRACTICES_OPTIMIZATION.md)

**What you'll learn:**
- When to use "Inject Context"
- How to choose sample data size
- Regex pattern optimization
- Grouping strategies
- Model selection (Ollama vs ConnectMaiGPT)
- Performance tuning
- Common pitfalls & solutions
- Debug logging interpretation

**Time to read:** 25 minutes

---

## 📊 One-Page Overview

### The 5 Data Channels

```
1. AUTOMATIC CHART CONTEXT (Every Message)
   ├─ Sent automatically with each chat message
   ├─ Size: 2–10 KB
   ├─ Includes: Stats, sample rows, reference markers
   └─ Ollama quality: ⭐⭐⭐⭐⭐ (most important)

2. STREAMING REAL-TIME TOKENS (Response Display)
   ├─ Tokens arrive as generated
   ├─ Latency: 1–2 seconds to first token
   ├─ Display: Words appear in real-time
   └─ UX quality: ⭐⭐⭐⭐⭐ (very responsive)

3. QUERY TOKENS (Advanced Aggregations)
   ├─ Ollama requests computed data
   ├─ Status: ✅ Defined, ⏳ Evaluation pending
   ├─ Example: [QUERY: col=RBER, agg=mean]
   └─ Implementation: Ready in v1.1

4. SESSION HISTORY (Conversation Memory)
   ├─ Ollama remembers prior turns
   ├─ Max turns: Last 20 (configurable)
   ├─ Trimming: Automatic, prevents bloat
   └─ Context relevance: ⭐⭐⭐⭐ (good for follow-ups)

5. CLOUD LLM INTEGRATION (Alternative Provider)
   ├─ Uses gpt4 / claude-3-5-sonnet
   ├─ Same data exposed as Ollama
   ├─ Status: ✅ Fully implemented
   └─ Privacy: ⚠️ Data goes to cloud
```

---

## 🚀 Getting Started

### First Time: Load Data & Chat

```
Step 1: Load CSV file
        → Click "Upload" button
        
Step 2: Configure plot
        → Select X column
        → Select Y column
        → (Optional) Add filters, colors, grouping
        
Step 3: View chart
        → Plot appears with data
        
Step 4: Open chat
        → Click "💬 Chat" in toolbar
        
Step 5: Ask question
        → Type question in chat input
        → First message auto-includes context
        → Response arrives in 2–10 seconds
```

### After Plot Changes: Inject Context

```
Step 1: Modify plot
        → Change filters OR
        → Change columns OR
        → Change grouping OR
        → Zoom/pan
        
Step 2: Click "⟳ Inject" button
        → Chat updates with new context
        → Watch for "Sending..." note
        
Step 3: Ask follow-up
        → Ollama now sees updated stats
        → Response reflects new data
```

---

## 🎛️ Configuration Reference

### Chat Menu Options

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| **Font** | S (0.82em) | XS–XL | Chat message text size |
| **Provider** | Ollama | Ollama / ConnectMaiGPT | LLM source |
| **Model** | llama3 | llama3 / gemma4 / gpt4 / claude | Which model to use |
| **Rows** | 50 | 0, 20, 50, 100, 200, 500 | Raw data sample size |
| **Inject** | (button) | — | Force context update |
| **New** | (button) | — | Clear history, start fresh |

### Python Configuration

**File:** `fdv_chart_rev8/fdv_chart.py`

```python
_LLM_MODEL         = 'llama3'       # Line ~40: Default model
_LLM_TIMEOUT       = 180            # Line ~41: Response timeout (seconds)
_CHAT_MAX_TURNS    = 20             # Line ~42: History size (turn pairs)
_LLM_SYSTEM_PROMPT = '...'          # Line ~53: Custom instructions
_OLLAMA_BASE       = 'http://localhost:11434'  # Line ~44: Ollama endpoint
```

---

## 🔍 Data Flow at a Glance

```
Browser                    Server                 Ollama
─────────────────────────────────────────────────────────

User writes question
         │
         ├─→ _buildChatContext()  (builds stats)
         │
         ├─→ POST /chat          ────→ Receives request
         │   (context in body)          │
         │                             ├─→ Parses messages[]
         │                             │   (context in system[1])
         │                             │
         │                             ├─→ Generates response
         │                             │
         │                      ←───── Sends tokens or reply
         │
    Displays response
```

---

## ✅ Feature Status

| Feature | Status | Implementation |
|---------|--------|---|
| **Automatic Chart Context** | ✅ Complete | `_buildChatContext()` in HTML |
| **Streaming Responses** | ✅ Complete | `/chat_stream` in Python |
| **Session History** | ✅ Complete | `_chat_sessions` manager |
| **Ollama Integration** | ✅ Complete | `/api/chat` calls |
| **ConnectMaiGPT** | ✅ Complete | `/maigpt_chat` endpoint |
| **QUERY Token System** | 🔨 Defined | Ready for implementation |
| **Result Injection** | ⏳ Pending | After QUERY parsing |
| **Chart Export** | ⏳ Future | For vision models (v2.0) |
| **Multi-file Sessions** | ✅ Supported | Per-CSV session storage |
| **Debug Logging** | ✅ Complete | `fdv_chart_startup.log` |

---

## 📋 Checklist: Optimal Usage

**Before starting analysis:**
```
□ CSV file loaded successfully
□ Chart visible on screen
□ X and Y columns selected
□ No excessive filters applied (rows > 0)
□ Chat provider set correctly
□ Chat rows sample size appropriate
```

**During analysis:**
```
□ Click "⟳ Inject" after ANY plot modification
□ Wait for context send to complete before asking follow-up
□ Monitor response times (should be < 10 seconds typically)
□ Use "New" button to start fresh conversation if needed
```

**For optimal results:**
```
□ Use 50–100 sample rows (default is good)
□ Keep grouping to 5–15 groups (avoid >50)
□ Use local Ollama for privacy (gpt4 for complex analysis)
□ Save important findings (copy-paste to file)
□ Check logs if issues arise
```

---

## 🐛 Troubleshooting Quick Links

**"Ollama says no context"**
→ See: [QUICK_DATA_EXPOSURE_REFERENCE.md § Troubleshooting](./QUICK_DATA_EXPOSURE_REFERENCE.md#troubleshooting)

**"Response is stale (old data)"**
→ See: [BEST_PRACTICES_OPTIMIZATION.md § Pitfall 2](./BEST_PRACTICES_OPTIMIZATION.md#pitfall-2-ollama-refers-to-old-stats)

**"Response is very slow"**
→ See: [BEST_PRACTICES_OPTIMIZATION.md § Pitfall 3](./BEST_PRACTICES_OPTIMIZATION.md#pitfall-3-response-is-very-slow)

**"How to interpret logs?"**
→ See: [BEST_PRACTICES_OPTIMIZATION.md § Debugging & Logging](./BEST_PRACTICES_OPTIMIZATION.md#10-debugging--logging)

**"I want custom instructions"**
→ See: [BEST_PRACTICES_OPTIMIZATION.md § Custom System Prompt](./BEST_PRACTICES_OPTIMIZATION.md#12-advanced-custom-system-prompt)

---

## 📖 Reading Paths by Role

### 👤 Data Analyst (Non-Technical User)

```
1. Start: QUICK_DATA_EXPOSURE_REFERENCE.md (10 min)
   Learn: What data Ollama sees, how to use chat
   
2. Explore: Best Practices section on "Raw Data Sample Sizing" (5 min)
   Learn: How to optimize for your dataset
   
3. Reference: Keep troubleshooting section handy
   Use: When issues arise
```

**Total time to master:** 30 minutes

### 👨‍💻 Software Developer (Integration)

```
1. Start: DATA_EXPOSURE_STRATEGY.md (20 min)
   Learn: Complete architecture, all channels
   
2. Deep dive: IMPLEMENTATION_CODE_WALKTHROUGH.md (30 min)
   Learn: Exact code paths, function signatures
   
3. Optimize: BEST_PRACTICES_OPTIMIZATION.md § "Performance Optimization" (10 min)
   Learn: Tuning parameters, logging
```

**Total time to master:** 1 hour

### 🏗️ DevOps / System Engineer

```
1. Start: DATA_EXPOSURE_STRATEGY.md § "Logging & Observability" (5 min)
   Learn: What logs to expect, how to monitor
   
2. Reference: BEST_PRACTICES_OPTIMIZATION.md § "Debugging & Logging" (10 min)
   Learn: Log interpretation, diagnostics
   
3. Configure: BEST_PRACTICES_OPTIMIZATION.md § "Session Trimming Config" (5 min)
   Learn: Tuning for production
```

**Total time to master:** 30 minutes

### 🔬 Data Scientist (Advanced Analysis)

```
1. Start: DATA_EXPOSURE_STRATEGY.md § "QUERY Token System" (10 min)
   Learn: Advanced feature for custom aggregations
   
2. Master: IMPLEMENTATION_CODE_WALKTHROUGH.md § "Streaming Response" (15 min)
   Learn: Real-time token handling
   
3. Optimize: BEST_PRACTICES_OPTIMIZATION.md § "Complete guide" (25 min)
   Learn: All optimization techniques
```

**Total time to master:** 1 hour

---

## 🔗 File References

### HTML/JavaScript
- **Main file:** `fdv_chart_rev8/fdv_chart.html` (7613 lines)
  - Context building: Lines 6600–6900
  - Chat sending: Lines 7026–7100
  - Stream handling: Lines 7060–7090

### Python
- **Main file:** `fdv_chart_rev8/fdv_chart.py` (1874 lines)
  - Config: Lines 40–53
  - `/chat` endpoint: Lines 1180–1260
  - `/chat_stream` endpoint: Lines 1282–1341
  - Session management: Lines 1208–1238
  - Logging: Lines 1201–1203, 1245–1253, 1321–1328

### Logs
- **Log file:** `fdv_chart_startup.log`
  - Location: `d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart_startup.log`
  - Search for: `[CHAT]`, `[CHAT_SEND]`, `[OLLAMA_SEND]`

---

## 📚 Glossary

| Term | Definition |
|------|-----------|
| **Context** | Chart statistics, sample data, configuration info sent to Ollama |
| **Inject** | Button to force send updated context to Ollama |
| **Session** | Per-CSV conversation history stored on server |
| **Turn** | One user message + one assistant response pair |
| **QUERY token** | Ollama's request for computed aggregation (pending implementation) |
| **SSE** | Server-Sent Events (streaming protocol for real-time tokens) |
| **System message** | Instructions/context sent to LLM (not visible to user) |
| **User message** | User's question sent to LLM |
| **Assistant message** | LLM's response |

---

## 🎓 Learning Resources

### External References
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Server-Sent Events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- Chart.js: https://www.chartjs.org/docs/latest/

### Internal References
- Root README: `README.md`
- Setup guide: `LAUNCH.md`
- Plot features: `PLOT_FEATURES_GUIDE.md`

---

## 💡 Key Insights

1. **Context is built entirely client-side** in the browser before sending to server
   - Ensures privacy (can use local Ollama)
   - Allows smart sampling (20–500 rows as needed)
   - Mirrors chart bucketing logic exactly

2. **Ollama sees identical data as the plot**
   - Same filters applied
   - Same grouping logic
   - Same regex extraction
   - Result: LLM's analysis matches visualization

3. **Session history enables multi-turn analysis**
   - Ollama remembers prior questions/answers
   - Context can be updated without losing history
   - Configurable trimming prevents memory bloat

4. **Streaming provides real-time feedback**
   - First token in 1–2 seconds (vs 10+ seconds for full response)
   - Words appear incrementally (better UX)
   - No latency penalty (same computation time)

5. **QUERY tokens (pending) will enable dynamic aggregations**
   - Ollama can request exact calculations
   - Server computes on-demand
   - Results injected back into session
   - Enables reasoning over arbitrary aggregations

---

## 📞 Support & Feedback

### Reporting Issues
- Check: Logs in `fdv_chart_startup.log`
- Verify: All documentation sections applied
- Include: Steps to reproduce, browser console errors, log excerpts

### Feature Requests
- QUERY token implementation (high priority)
- Chart export for vision models
- Multi-file session management

### Questions?
- Consult: Appropriate documentation section
- Search: This index for keyword/role
- Check: Troubleshooting sections

---

## 📝 Document Maintenance

| Document | Last Updated | Status | Reviewer |
|----------|---|---|---|
| **DATA_EXPOSURE_STRATEGY.md** | May 2, 2026 | ✅ Current | — |
| **QUICK_DATA_EXPOSURE_REFERENCE.md** | May 2, 2026 | ✅ Current | — |
| **IMPLEMENTATION_CODE_WALKTHROUGH.md** | May 2, 2026 | ✅ Current | — |
| **BEST_PRACTICES_OPTIMIZATION.md** | May 2, 2026 | ✅ Current | — |
| **INDEX.md** (this file) | May 2, 2026 | ✅ Current | — |

---

**Version**: 1.0  
**Release Date**: May 2, 2026  
**Status**: ✅ Complete & Production Ready  
**Next Review**: December 2026
