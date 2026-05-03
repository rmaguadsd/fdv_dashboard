# Best Practices: Optimizing Data Exposure to Ollama

## Executive Summary

This guide covers **optimization strategies** to maximize the effectiveness of data exposure while maintaining performance, accuracy, and usability.

---

## 1. Context Freshness & Relevance

### ✅ Best Practice: Use "Inject" After Plot Changes

**When to use "⟳ Inject":**
- ✓ After changing X/Y columns
- ✓ After adding/removing filters
- ✓ After adjusting interval ranges
- ✓ After changing grouping (color-by, split-by)
- ✓ After modifying regex patterns
- ✓ After zooming into specific data range

**Why:**
- Ollama's analysis is only as good as the context it receives
- Stale context leads to outdated recommendations
- "Inject" replaces the old context in the session with fresh stats

**Example workflow:**
```
1. Plot shows X column = frequency, Y = RBER
   Ask: "What's the mean RBER?"
   Ollama: "Mean is 0.513"

2. You zoom: Y interval now [0.001, 0.1]
   Click "⟳ Inject"  ← CRITICAL STEP!
   
3. Ask: "What's the mean now?"
   Ollama: "Mean is 0.045" (filtered to new range)
   
Without step 2, Ollama would still think mean is 0.513 (stale)
```

---

## 2. Raw Data Sample Sizing

### How Sample Size Affects Context

| Size | Pros | Cons | Best For |
|------|------|------|----------|
| **0 (none)** | Small context (2KB) | No ground truth | Quick analysis, API limits |
| **20** | Compact (3KB) | Limited coverage | Fast iteration |
| **50** | Balanced (5KB) ✓ | Good coverage | Default, most cases |
| **100** | Detailed (8KB) | Larger LLM input | Complex patterns |
| **200** | Very detailed (10KB) | Slower responses | Critical analysis |
| **500** | Exhaustive (12KB) | Risk of bloat | Debugging only |

### Strategy: Adaptive Sampling

**Rule of thumb:**
```
If dataset size < 1000 rows:
  Use 50 rows (default)
  
If dataset size 1000-10K:
  Use 100 rows for complex grouping
  Use 50 rows for simple analysis
  
If dataset size > 10K:
  Use 50-100 rows (sufficient for statistical inference)
  Avoid 500 (diminishing returns)
```

### Implementation

```javascript
// In browser console or chat menu
_chatDataSampleSize = 100;  // Change at runtime

// Or edit in HTML:
<option value="100">100</option>  // Add/modify option
```

---

## 3. Regex Pattern Optimization

### Key Principle: Precision Extracts Better Data

**Bad regex (too loose):**
```
X regex: ".*"  (matches everything)
Problem: Extracts "test_12_failed_03.45" → whole string (not numeric!)
```

**Good regex (precise extraction):**
```
X regex: "(\d+\.?\d*)"  (extracts only numbers)
Result: "test_12_failed_03.45" → 12, then 03.45
```

### Common Patterns

```javascript
// Extract first decimal number
"(\d+\.?\d*)"

// Extract number after underscore
"_(\d+\.?\d*)"

// Extract number between colon and dot
":(\d+\.?\d*)\."

// Extract test result (PASS/FAIL)
"(PASS|FAIL)"

// Extract DUT identifier
"(UNIT_\d+)"

// Extract frequency with unit
"(\d+\.?\d*)\s*(GHz|MHz)"
```

### Testing Regexes

In browser console:
```javascript
// Test extraction function
var testStr = "UNIT_01_2.4GHz_0.523";
var rx = "UNIT_(\d+)";
var m = new RegExp(rx).exec(testStr);
console.log(m[1]);  // Output: "01"
```

---

## 4. Grouping Strategy

### Single-Level Grouping (Recommended)

**Good**: Color-by DUT only
```
Chart shows: 5 DUTs (5 lines/colors)
Context includes: Stats for each DUT
Ollama overhead: Minimal
```

**Code:**
```
Color column: DUT
Regex: "" (no extraction)
Split column: (empty)
Result: 5 groups
```

### Two-Level Grouping (Advanced)

**Good**: Split-by TestType, then color-by DUT
```
Chart shows: 3 tiles × 5 DUTs (cross product)
Context includes: Stats for each (tile, DUT) combination
Ollama overhead: 3 × 5 = 15 stats blocks
```

**Code:**
```
Split column: TestType
Color column: DUT
Result: 3 × 5 = 15 groups
```

### ⚠️ Avoid: Excessive Grouping

**Too many groups:**
```
100+ unique values in color column
Result: 100+ statistics blocks
Context becomes: 15-20 KB
Response time: +50% slower
Ollama performance: Diminishing returns
```

**Solution**: Pre-filter data
```
1. Load full dataset
2. Apply table filter: Only show TestType == "Functional"
3. Then color-by DUT (now maybe 10 DUTs instead of 50)
4. Chart has fewer groups, context is leaner
```

---

## 5. Model Selection & Performance

### Ollama Model Comparison

| Model | Speed | Quality | Context Limit | Cost |
|-------|-------|---------|---|---|
| **llama3** (default) | Fast | Good | 8KB context | Free (local) |
| **gemma4** | Faster | Okay | 8KB | Free (local) |
| **gpt4** (ConnectMaiGPT) | Slow | Excellent | Unlimited | Paid |
| **claude-3-5-sonnet** (ConnectMaiGPT) | Medium | Excellent | Unlimited | Paid |

### Recommendation

**For most use cases:**
- Use **Ollama + llama3** (default)
- Fast iteration, no API keys, privacy-preserving

**When llama3 struggles (e.g., reasoning about multiple variables):**
- Switch to **ConnectMaiGPT + gpt4** for one complex query
- Then return to Ollama for rapid follow-ups

**Example workflow:**
```
1. Quick analysis: Ollama llama3 (2-3 sec response)
2. User asks complex question
3. Switch to ConnectMaiGPT gpt4 (5-10 sec, but better reasoning)
4. Back to Ollama for quick follow-ups
```

---

## 6. Context Management

### Session Trimming Configuration

**Current setting:**
```python
_CHAT_MAX_TURNS = 20  # (in fdv_chart.py)
```

**Adjustments:**

**For short analyses** (typical):
```python
_CHAT_MAX_TURNS = 10  # Keep only 10 turns (20 messages)
Benefit: Faster responses, lower memory
Tradeoff: Less conversation history
```

**For investigative analysis** (deep dives):
```python
_CHAT_MAX_TURNS = 50  # Keep 50 turns (100 messages)
Benefit: Ollama remembers entire analysis thread
Tradeoff: Slower responses, more memory
```

**For debugging** (troubleshooting):
```python
_CHAT_MAX_TURNS = 100  # Keep everything
Benefit: Perfect for reproducing issues
Tradeoff: Very slow, only use when necessary
```

### How to Change

**Edit Python file:**
```python
# fdv_chart_rev8/fdv_chart.py, line ~40
_CHAT_MAX_TURNS = 20  ← Change this number

# Save, restart server
# Changes take effect immediately
```

### Monitor Session Size

**In browser console:**
```javascript
// Check current session size
console.log(_chat_sessions);  // See all sessions
```

**In Python logs:**
```
[CHAT_SEND] system=2 user=1 assistant=0 has_chart_context=True
                ↑ This grows as turns accumulate
```

---

## 7. QUERY Token Strategy (Future)

### When Query Tokens Will Be Useful

**Current workflow (manual):**
```
User: "What's the mean RBER per DUT?"
Ollama: "I need to compute that... (makes guess)"
```

**Future with QUERY tokens:**
```
Ollama: "Let me compute the exact aggregations..."
[QUERY: col=RBER, group=DUT, agg=mean]

Server: "Query Results:
  DUT_01: 0.234
  DUT_02: 0.456
  DUT_03: 0.123"

Ollama: "Based on the precise aggregations..."
```

### Best Practices (When Implemented)

✅ **Use QUERY for:**
- Precise aggregations (mean, std, count)
- Multi-column grouping
- Complex filters
- Percentiles (p5, p95)

❌ **Don't use QUERY for:**
- Simple trends visible in plot
- Qualitative observations
- Boolean/categorical analysis (use filters instead)

---

## 8. Privacy & Security

### Data Exposure Checklist

**Before sharing analysis with non-technical users:**

- [ ] Verify data is anonymized (no PII)
- [ ] If using ConnectMaiGPT, confirm non-sensitive data
- [ ] Check firewall rules (is Ollama on localhost only?)
- [ ] Confirm no passwords/credentials in log files
- [ ] Review chat history (any sensitive info logged?)

### How to Sanitize Data Before Analysis

**Option 1: Column Renaming**
```
Instead of: Employee_ID, Salary, SSN
Rename to:  Subject_A, Value_1, Value_2
Perform: Analysis
```

**Option 2: Data Filtering**
```
Load full dataset
Apply filter: ≠ null (remove rows with PII)
Analyze filtered data
```

**Option 3: Aggregation**
```
Instead of: Raw transaction data
Use: Summary statistics only
Send to Ollama: Means, medians, percentiles (not raw rows)
```

---

## 9. Performance Optimization

### Measure Baseline

**Before optimizing, measure:**
```
1. First token latency (how long until first word appears?)
2. Full response time (when does response complete?)
3. Context build time (how long to create context?)
4. Round-trip time (network delay?)
```

**In browser, use console timer:**
```javascript
console.time('context-build');
var ctx = _buildChatContext();
console.timeEnd('context-build');  // Output: "context-build: 234ms"

console.time('fetch');
fetch('/chat', {...}).then(...);
console.timeEnd('fetch');
```

### Optimization Checklist

| Issue | Solution | Benefit |
|-------|----------|---------|
| Large context (>10KB) | Reduce sample size: 50→20 | -40% context, +30% speed |
| Too many groups (>20) | Pre-filter dataset | -50% groups, -30% context |
| Slow Ollama response | Switch to gemma4 (faster) | +50% speed, -5% quality |
| Long session history | Edit `_CHAT_MAX_TURNS = 10` | +40% speed, -memory |
| Network latency | Use streaming (/chat_stream) | First token in 2sec vs 10sec |

---

## 10. Debugging & Logging

### Enable Debug Logging

**Already enabled by default:**
```python
# fdv_chart_rev8/fdv_chart.py
# Logs go to: d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart_startup.log
```

**Example log output:**
```
[CHAT] csv_id=default context_bytes=3124 message_len=52
[CHAT_SEND] system=2 user=1 assistant=0 has_chart_context=True
[OLLAMA_SEND] system=2 user=1 assistant=0 has_chart_context=True
[CHAT_STREAM] csv_id=default context_bytes=3124 message_len=52 model=llama3
```

### Interpret Logs

| Log Entry | Meaning | Action |
|-----------|---------|--------|
| `context_bytes=0` | No context sent | ❌ Problem: Click "Inject"? |
| `context_bytes=3124` | 3.1KB context | ✓ Normal |
| `has_chart_context=True` | Context in session | ✓ Good |
| `has_chart_context=False` | Context not found | ❌ Problem: Re-inject |
| `system=2` | 2 system messages | ✓ Normal (1 prompt + 1 context) |
| `system=5` | Many system messages | ⚠️ Warning: Multiple contexts? |

### View Live Logs

**In PowerShell (while server running):**
```powershell
Get-Content -Path "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart_startup.log" -Tail 20 -Wait
# Shows last 20 lines, updates live
```

**In terminal (while server running):**
```bash
tail -f "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart_startup.log"
```

---

## 11. Common Pitfalls & Solutions

### Pitfall 1: "Ollama Says It Has No Data"

**Symptom:**
```
Ollama: "I don't see any data to analyze."
```

**Root Cause:**
- Chart not displayed (X/Y columns not selected)
- Data filtered to 0 rows
- Context build failed

**Solution:**
```
1. Check: X column selected? Y column selected?
2. Check: "Rows passing table filters" > 0?
3. Click: "⟳ Inject" to force send context
4. View: Browser console for errors
```

### Pitfall 2: "Ollama Refers to Old Stats"

**Symptom:**
```
You zoom: "Now showing Y ∈ [0.001, 0.1]"
Ollama: "The mean is 0.513" (old value!)
```

**Root Cause:**
- Forgot to click "⟳ Inject"
- Old context still in session

**Solution:**
```
1. After ANY plot change → Click "⟳ Inject"
2. Wait for "Sending..." note
3. Then ask follow-up question
```

### Pitfall 3: "Response Is Very Slow"

**Symptom:**
```
10+ second wait for response
```

**Root Causes & Solutions:**
```
Cause 1: Large dataset (10K+ rows)
  Solution: Apply table filter to reduce
  
Cause 2: Large sample size (500 rows)
  Solution: Reduce to 50 rows in chat menu
  
Cause 3: Too many groups (50+)
  Solution: Pre-filter (e.g., TestType == "Functional")
  
Cause 4: Ollama overloaded
  Solution: Wait a moment, try again
  
Cause 5: Not using streaming
  Solution: Enable "Streaming" option (if available)
```

### Pitfall 4: "Chat History Lost"

**Symptom:**
```
Asked 5 questions, now Ollama doesn't remember Q1
```

**Root Cause:**
- Session trimmed (only keeps last 20 turns)
- New CSV file loaded (separate session)

**Solution:**
```
1. To remember longer: Edit _CHAT_MAX_TURNS = 50
2. To preserve session: Don't switch CSV files
3. To keep history: Save chat transcript manually
   (Copy-paste chat into text file)
```

---

## 12. Advanced: Custom System Prompt

### Current System Prompt

```python
_LLM_SYSTEM_PROMPT = (
    'You are a data analysis assistant embedded in an engineering test-data viewer. '
    'The user gives you statistics extracted from parsed log file charts. '
    'Provide concise, actionable insights. If the user marks something with [QUERY: ...], '
    'parse it as a request for computed aggregations.'
)
```

### Customization Ideas

**For different domains:**

**Engineering (default):**
```python
'You are an RF engineer analyzing test results. Focus on signal quality, yield trends, and anomalies.'
```

**Business/Sales:**
```python
'You are a business analyst. Focus on revenue trends, customer segments, and growth opportunities.'
```

**Scientific Research:**
```python
'You are a research scientist. Provide statistical rigor and hypothesis testing insights.'
```

### How to Change

**Edit Python file:**
```python
# fdv_chart_rev8/fdv_chart.py, line ~53
_LLM_SYSTEM_PROMPT = (
    'Your custom prompt here...'
)

# Save, restart server
```

---

## Summary: Best Practices Checklist

**Before any analysis:**
- [ ] Data is loaded and chart is visible
- [ ] X/Y columns are selected
- [ ] Appropriate table filters applied
- [ ] Chat "Provider" set correctly
- [ ] Chat "Rows" sample size appropriate (typically 50)

**During analysis:**
- [ ] After plot changes: Click "⟳ Inject"
- [ ] Before follow-up questions: Wait for context injection
- [ ] Monitor response times (adjust sample size if slow)
- [ ] Keep conversation focused (trim history if needed)

**After analysis:**
- [ ] Save important insights (copy-paste or screenshot)
- [ ] Document unusual findings
- [ ] If debugging issues: Check logs in fdv_chart_startup.log

---

**Version**: 1.0 | **Last Updated**: May 2, 2026
