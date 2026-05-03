# Implementation Guide: Data Exposure Code Walkthrough

## Overview

This guide walks through the actual source code that exposes chart and table data to Ollama. It explains **where** data is built, **how** it's formatted, and **when** it's sent.

---

## 1. Browser-Side: Context Building

### File: `fdv_chart_rev8/fdv_chart.html`

#### Entry Point: `_buildChatContext()` (Lines 6600–6900)

```javascript
function _buildChatContext() {
    // GUARDS: Check if data is loaded
    if (!currentCsvId || !filteredIndices || filteredIndices.length === 0) {
        return 'No chart data currently loaded.';
    }
    if (!document.getElementById('x-col').value) {
        return 'No X column selected — draw a plot first.';
    }

    // READS: All chart configuration (mirrors _analyzeChart exactly)
    var ct       = document.getElementById('chart-type').value;      // 'scatter', 'histogram', etc.
    var xCol     = document.getElementById('x-col').value;           // Column name
    var yCol     = document.getElementById('y-col').value;           
    var xRx      = document.getElementById('x-regex').value.trim();  // Regex pattern
    var yRx      = document.getElementById('y-regex').value.trim();
    var colCol   = document.getElementById('color-col').value;       // For grouping
    var colorRx  = document.getElementById('color-regex').value.trim();
    var splitCol = (document.getElementById('split-col') || {value:''}).value;
    var splitRx  = (document.getElementById('split-regex') || {value:''}).value.trim();
    
    // READS: Interval filters
    var iv = _getIntervals();  // { xLo, xHi, yLo, yHi }
    
    // READS: Data arrays (global scope)
    // currentHeaders   → column names
    // filteredIndices  → row indices passing table filters
    // allRows          → raw parsed CSV data

    // ── PHASE 1: Build header/metadata section ──
    var lines = [];
    lines.push('FDV Dashboard — chart context for ' + _selectedModel());
    lines.push('Chart type: ' + ct);
    lines.push('X column: ' + xCol + (xRx ? '  (regex: /' + xRx + '/)' : ''));
    if (!isXOnly) {
        lines.push('Y column: ' + yCol + (yRx ? '  (regex: /' + yRx + '/)' : ''));
    }
    lines.push('Rows in memory: ' + allRows.length + '   passing table filters: ' + filteredIndices.length);

    // ── PHASE 2: Build interval filter section ──
    var ivParts = [];
    if (iv.xLo || iv.xHi) {
        ivParts.push('X ∈ [' + (iv.xLo||'−∞') + ', ' + (iv.xHi||'+∞') + ']');
    }
    if (!isXOnly && (iv.yLo || iv.yHi)) {
        ivParts.push('Y ∈ [' + (iv.yLo||'−∞') + ', ' + (iv.yHi||'+∞') + ']');
    }
    if (ivParts.length) {
        lines.push('Data interval filter: ' + ivParts.join(';  '));
    }

    // ── PHASE 3: Determine grouping column (mirrors _analyzeChart) ──
    var inChartGroupCol = isXOnly ? (splitCol || colCol) : colCol;
    var inChartGroupRx  = isXOnly ? (splitCol ? splitRx : colorRx) : colorRx;
    
    if (inChartGroupCol) {
        lines.push((isXOnly ? 'Split-by' : 'Color-by') + ' column: ' + inChartGroupCol + 
                   (inChartGroupRx ? '  (regex: /' + inChartGroupRx + '/)' : ''));
    }

    // ── PHASE 4: Compute bucket statistics ──
    // The logic here is CRITICAL: it must mirror _analyzeChart exactly,
    // so Ollama sees identical bucketing as the plot.

    function bucketRows(keyFn) {
        // Returns: { keys: [...], buckets: {key1: [...], key2: [...]} }
        var buckets = {}, order = [];
        var xLo = iv.xLo !== '' ? parseFloat(iv.xLo) : NaN;
        var xHi = iv.xHi !== '' ? parseFloat(iv.xHi) : NaN;
        var yLo = (!isXOnly && iv.yLo !== '') ? parseFloat(iv.yLo) : NaN;
        var yHi = (!isXOnly && iv.yHi !== '') ? parseFloat(iv.yHi) : NaN;

        // Iterate filtered rows
        for (var ri = 0; ri < filteredIndices.length; ri++) {
            var row = allRows[filteredIndices[ri]];
            
            // Extract numeric values using regex
            var xi  = currentHeaders.indexOf(xCol);
            var yi  = isXOnly ? -1 : currentHeaders.indexOf(yCol);
            var xv  = extractNum(row[xi] != null ? String(row[xi]) : '', xRx);  // Regex extraction
            var yv  = isXOnly ? xv : extractNum(row[yi] != null ? String(row[yi]) : '', yRx);
            
            // Skip null values or interval mismatches
            if (xv === null || (!isXOnly && yv === null)) continue;
            if (!isNaN(xLo) && xv < xLo) continue;  // Below interval
            if (!isNaN(xHi) && xv > xHi) continue;  // Above interval
            if (!isNaN(yLo) && yv < yLo) continue;
            if (!isNaN(yHi) && yv > yHi) continue;
            
            // Add to bucket
            var key = keyFn(row) || '(blank)';
            if (!buckets[key]) { 
                buckets[key] = []; 
                order.push(key); 
            }
            buckets[key].push({ x: xv, y: isXOnly ? xv : yv });
        }

        // Sort keys for consistent output
        var keys = order.filter(function(k,i){ return order.indexOf(k)===i; })
                        .sort(function(a,b){ return a.localeCompare(b,undefined,{numeric:true}); });
        return { keys: keys, buckets: buckets };
    }

    function fmtBucket(pts) {
        // Format bucket as: n=1240, X: μ=0.523, σ=0.142, min=0.001, max=0.998
        var xs = pts.map(function(p){ return p.x; });
        var ys = isXOnly ? [] : pts.map(function(p){ return p.y; });
        var out = 'n=' + pts.length + ',  X: ' + _fmtStats(_descStats(xs));
        if (ys.length) out += ',  Y: ' + _fmtStats(_descStats(ys));
        return out;
    }

    // ── PHASE 5a: Split-chart mode (if applicable) ──
    if (scCol && _splitInsts.length > 0) {
        var scIdx = currentHeaders.indexOf(scCol);
        var gcIdx = inChartGroupCol ? currentHeaders.indexOf(inChartGroupCol) : -1;

        if (gcIdx >= 0) {
            // 2-D: tile × color group
            // [Complex bucketing logic that handles both dimensions]
            lines.push('\nCross-dimensional data: ' + tileKeys.length + ' tiles × ' + colorKeys.length + ' color groups');
            tileKeys.forEach(function(tk) {
                lines.push('\n  Tile [' + tk + ']:');
                colorKeys.forEach(function(ck) {
                    var cpk = tk + '\x00' + ck;
                    var pts = (crossBuckets[cpk] || {}).pts || [];
                    if (!pts.length) return;
                    lines.push('    [' + ck + '] ' + fmtBucket(pts));
                });
            });
        } else {
            // 1-D: tile comparison only
            var b1 = bucketRows(function(row) { return _splitChartKey(row, scIdx, scRx); });
            lines.push('\nSplit-chart tiles (' + b1.keys.length + ' tiles by ' + scCol + '):');
            b1.keys.forEach(function(k) {
                lines.push('  [' + k + '] ' + fmtBucket(b1.buckets[k]));
            });
        }

    } else if (inChartGroupCol) {
        // ── PHASE 5b: Single chart with in-chart grouping ──
        var gcIdx3 = currentHeaders.indexOf(inChartGroupCol);
        var b2 = bucketRows(function(row) { return inChartKey(row[gcIdx3]); });
        if (b2.keys.length > 1) {
            lines.push('\nIn-chart groups (' + b2.keys.length + ' groups by ' + inChartGroupCol + '):');
            b2.keys.forEach(function(k) {
                lines.push('  [' + k + '] ' + fmtBucket(b2.buckets[k]));
            });
        } else if (b2.keys.length === 1) {
            lines.push('\nOverall stats: ' + fmtBucket(b2.buckets[b2.keys[0]]));
        }

    } else {
        // ── PHASE 5c: Plain single chart (no grouping) ──
        var b3 = bucketRows(function() { return '__all__'; });
        var allPts = b3.buckets['__all__'] || [];
        if (allPts.length) {
            lines.push('\nOverall stats: ' + fmtBucket(allPts));
        } else {
            lines.push('\n(No numeric data passes current filters/intervals.)');
        }
    }

    // ── PHASE 6: Add reference markers ──
    if (_markers && _markers.length) {
        lines.push('\nReference marker lines:');
        _markers.forEach(function(m) {
            lines.push('  ' + (m.axis||'x').toUpperCase() + '=' + m.value + 
                      (m.label ? ' (' + m.label + ')' : ''));
        });
    }

    // ── PHASE 7: Add raw data sample (if enabled) ──
    var sampleN = _chatDataSampleSize;  // User setting: 0, 20, 50, 100, 200, 500
    if (sampleN > 0 && filteredIndices.length > 0) {
        var sampleCount = Math.min(sampleN, filteredIndices.length);
        var sampleStep  = Math.max(1, Math.floor(filteredIndices.length / sampleCount));
        
        lines.push('\nColumn headers: ' + currentHeaders.join(' | '));
        lines.push('Raw data sample (' + sampleCount + ' of ' + filteredIndices.length + ' filtered rows, evenly spaced):');
        
        var drawn = 0;
        for (var si = 0; si < filteredIndices.length && drawn < sampleCount; si += sampleStep) {
            var srow = allRows[filteredIndices[si]];
            lines.push('  ' + srow.map(function(c){ return c != null ? String(c) : ''; }).join(' | '));
            drawn++;
        }
    }

    // ── PHASE 8: Add QUERY token instructions ──
    lines.push('\nYou may request computed aggregations by embedding one or more QUERY tokens in your reply:');
    lines.push('[QUERY: col=COLUMN_NAME, filter=EXPRESSION, group=COLUMN_NAME, agg=mean|count|min|max|std|all]');
    lines.push('  col    — column to aggregate (required)');
    lines.push('  filter — optional row filter, same syntax as table filter (e.g. >0.001  or  ==PASS)');
    lines.push('  group  — optional column to group by');
    lines.push('  agg    — aggregation: mean (default), count, min, max, std, all (full stats)');
    lines.push('Example: [QUERY: col=RBER, filter=>0.001, group=DUT, agg=mean]');
    lines.push('Results will be injected as context before your next reply.');

    // ── PHASE 9: Append summary statistics (if available) ──
    if (typeof _lastSummaryText === 'string' && _lastSummaryText.trim()) {
        lines.push('');
        lines.push('--- Statistical Summary (from Summary panel) ---');
        lines.push(_lastSummaryText.trim());
    }

    // ── FINAL: Join and return ──
    return lines.join('\n');
}
```

### Key Functions Called

#### `extractNum(str, regex)` - Regex-based numeric extraction
```javascript
function extractNum(str, rx) {
    if (!rx || !str) return parseFloat(str);  // Direct parse if no regex
    try {
        var m = new RegExp(rx).exec(str);
        return m ? parseFloat(m[1] !== undefined ? m[1] : m[0]) : null;
    } catch(e) { return null; }
}
```

#### `_descStats(arr)` - Compute statistics
```javascript
function _descStats(arr) {
    if (arr.length === 0) return { n: 0, mean: NaN, std: NaN, min: NaN, max: NaN };
    var sum = arr.reduce((a,b) => a+b, 0);
    var mean = sum / arr.length;
    var sq_sum = arr.map(x => (x - mean) * (x - mean)).reduce((a,b) => a+b, 0);
    var std = Math.sqrt(sq_sum / arr.length);
    var sorted = arr.slice().sort((a,b) => a - b);
    return {
        n: arr.length,
        mean: mean,
        std: std,
        min: sorted[0],
        max: sorted[sorted.length-1]
    };
}
```

#### `_fmtStats(stats)` - Format for display
```javascript
function _fmtStats(s) {
    return 'μ=' + s.mean.toFixed(3) + ', σ=' + s.std.toFixed(3) + 
           ', min=' + s.min.toFixed(3) + ', max=' + s.max.toFixed(3);
}
```

---

## 2. Browser-to-Server: Sending Context

### File: `fdv_chart_rev8/fdv_chart.html`, Lines 7026–7100

#### `_chatSend()` - User clicks "Send" button

```javascript
function _chatSend() {
    var ta = document.getElementById('chat-input');
    if (!ta) return;
    
    var msg = ta.value.trim();
    if (!msg) return;
    if (_chatBusy) return;  // Prevent double-send

    _chatSetBusy(true);
    _chatAddMsg('user', msg);
    ta.value = '';

    // ── Build context if dirty (changed since last send) ──
    var ctx = '';
    if (_chatContextDirty) {
        ctx = _buildChatContext();  // ← THIS IS THE KEY CALL
        _chatContextDirty = false;  // Reset flag
    }

    // ── Determine provider ──
    var provider = _chatProvider();  // 'ollama' or 'connectmaigpt'

    // ── Choose endpoint based on provider ──
    if (provider === 'connectmaigpt') {
        // ConnectMaiGPT path (omitted for brevity)
    } else {
        // Ollama path (default)
        var useStream = document.getElementById('chat-stream-toggle') && 
                        document.getElementById('chat-stream-toggle').checked;
        
        var endpoint = useStream ? '/chat_stream' : '/chat';
        
        fetch(endpoint, {
            method:  'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                csv_id:   currentCsvId || '__default__',
                message:  msg,
                context:  ctx,        // ← Context in POST body
                model:    _selectedModel()
            })
        })
        .then(function(response) {
            if (useStream) {
                return _handleStreamResponse(response);  // SSE reader
            } else {
                return response.json().then(function(data) {
                    if (data.success) {
                        _chatAddMsg('assistant', data.reply);
                    } else {
                        _chatAddMsg('assistant', '❌ Error: ' + data.error);
                    }
                    _chatSetBusy(false);
                });
            }
        });
    }
}
```

### POST Body Example
```json
{
    "csv_id": "file_12345.csv",
    "message": "What's different about UNIT_02?",
    "context": "FDV Dashboard — chart context for llama3\nChart type: scatter\nX column: frequency\nY column: RBER\n...[full context from _buildChatContext]...",
    "model": "llama3"
}
```

---

## 3. Server-Side: Receiving & Processing Context

### File: `fdv_chart_rev8/fdv_chart.py`

#### `/chat` Endpoint (Lines 1180–1260)

```python
elif self.path == '/chat':
    # Non-streaming chat endpoint
    try:
        length  = int(self.headers.get('Content-Length', 0))
        body    = json.loads(self.rfile.read(length).decode('utf-8'))
        
        # ── EXTRACT from POST body ──
        csv_id  = body.get('csv_id', 'default') or 'default'
        message = body.get('message', '').strip()
        context = body.get('context', '').strip()  # ← Context from browser
        model   = body.get('model', '').strip() or _LLM_MODEL
        
        if not message:
            raise ValueError('Empty message')

        # ── LOG context reception (line 1201-1203) ──
        context_len = len(context) if context else 0
        with open(log_path, 'a') as f:
            f.write(f"[CHAT] csv_id={csv_id} context_bytes={context_len} message_len={len(message)}\n")

        # ── MANAGE session history (lines 1208-1228) ──
        with _chat_sessions_lock:
            if csv_id not in _chat_sessions:
                # NEW session: start with system prompt
                sess = [{'role': 'system', 'content': _LLM_SYSTEM_PROMPT}]
                if context:
                    # Add context as second system message
                    sess.append({
                        'role': 'system',
                        'content': 'Current chart context:\n' + context
                    })
                _chat_sessions[csv_id] = sess
            
            elif context:
                # EXISTING session: replace old context with new
                sess = _chat_sessions[csv_id]
                replaced = False
                for i, m in enumerate(sess):
                    if m['role'] == 'system' and (
                            m['content'].startswith('Current chart context:') or
                            m['content'].startswith('Updated chart context:')):
                        # Found existing context → replace it
                        sess[i] = {'role': 'system',
                                   'content': 'Updated chart context:\n' + context}
                        replaced = True
                        break
                
                if not replaced:
                    # No existing context → append new one
                    sess.append({'role': 'system',
                                 'content': 'Updated chart context:\n' + context})

            # Append user message
            _chat_sessions[csv_id].append({'role': 'user', 'content': message})

            # ── TRIM history (lines 1234-1238) ──
            # Keep all system messages + last _CHAT_MAX_TURNS turn pairs
            sess = _chat_sessions[csv_id]
            system_msgs = [m for m in sess if m['role'] == 'system']
            conv_msgs   = [m for m in sess if m['role'] != 'system']
            max_conv    = _CHAT_MAX_TURNS * 2  # Each turn = user + assistant
            if len(conv_msgs) > max_conv:
                conv_msgs = conv_msgs[-max_conv:]  # Keep only last N turns
            _chat_sessions[csv_id] = system_msgs + conv_msgs

            messages_snapshot = list(_chat_sessions[csv_id])

        # ── LOG message structure (lines 1245-1253) ──
        system_count = sum(1 for m in messages_snapshot if m['role'] == 'system')
        user_count = sum(1 for m in messages_snapshot if m['role'] == 'user')
        assistant_count = sum(1 for m in messages_snapshot if m['role'] == 'assistant')
        context_in_system = any('chart context' in m.get('content', '').lower() 
                               for m in messages_snapshot if m['role'] == 'system')
        with open(log_path, 'a') as f:
            f.write(f"[CHAT_SEND] system={system_count} user={user_count} assistant={assistant_count} has_chart_context={context_in_system}\n")

        # ── CALL LLM (lines 1256-1258) ──
        reply = _call_llm(messages_snapshot, model=model)

        # Append assistant response to session
        with _chat_sessions_lock:
            _chat_sessions[csv_id].append({'role': 'assistant', 'content': reply})
            turn = sum(1 for m in _chat_sessions[csv_id] if m['role'] == 'user')

        # Return reply to browser
        _send_json(self, 200, {'success': True, 'reply': reply, 'turn': turn})

    except Exception as e:
        _send_json(self, 200, {'success': False, 'error': str(e)})
```

#### Session Structure Example

After first message:
```python
_chat_sessions['default'] = [
    {'role': 'system', 'content': 'You are a data analysis assistant...'},
    {'role': 'system', 'content': 'Current chart context:\nFDV Dashboard — chart context for llama3\nChart type: scatter\n...'},
    {'role': 'user', 'content': 'What are the key findings?'},
    {'role': 'assistant', 'content': 'Based on the chart context...[response]...'}
]
```

---

## 4. Server-Side: Calling Ollama

### File: `fdv_chart_rev8/fdv_chart.py`, Lines 1330–1370

#### `_call_llm()` - Send to Ollama

```python
def _call_llm(messages, model=None):
    """
    Send messages (with context) to Ollama and get reply.
    
    Args:
        messages: List of {'role': '...', 'content': '...'} dicts
        model: Model name (e.g., 'llama3', 'gemma4')
    
    Returns:
        Response string from Ollama
    """
    if not model:
        model = _LLM_MODEL
    
    payload = json.dumps({
        'model':       model,
        'messages':    messages,        # ← Includes context in system message!
        'stream':      False,
        'temperature': 0.3
    }).encode('utf-8')
    
    req = urllib.request.Request(
        _OLLAMA_BASE + '/api/chat',     # ← Ollama endpoint
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as r:
            response_json = json.loads(r.read().decode('utf-8'))
            return response_json.get('message', {}).get('content', '')
    except Exception as e:
        return f"LLM Error: {str(e)}"
```

#### Payload Sent to Ollama

```json
{
    "model": "llama3",
    "messages": [
        {
            "role": "system",
            "content": "You are a data analysis assistant embedded in an engineering test-data viewer..."
        },
        {
            "role": "system",
            "content": "Current chart context:\nFDV Dashboard — chart context for llama3\nChart type: scatter\nX column: frequency\n...[full statistics and sample data]..."
        },
        {
            "role": "user",
            "content": "What are the key findings?"
        }
    ],
    "stream": false,
    "temperature": 0.3
}
```

**This is where Ollama receives the data!** The context is in the system message.

---

## 5. Streaming Response Handling

### File: `fdv_chart_rev8/fdv_chart.py`, Lines 1282–1341

#### `/chat_stream` Endpoint

```python
elif self.path == '/chat_stream':
    try:
        length  = int(self.headers.get('Content-Length', 0))
        body    = json.loads(self.rfile.read(length).decode('utf-8'))
        csv_id  = body.get('csv_id', 'default') or 'default'
        message = body.get('message', '').strip()
        context = body.get('context', '').strip()
        model   = body.get('model', '').strip() or _LLM_MODEL
        
        if not message:
            raise ValueError('Empty message')

        # Log context (line 1282-1284)
        context_len = len(context) if context else 0
        with open(log_path, 'a') as f:
            f.write(f"[CHAT_STREAM] csv_id={csv_id} context_bytes={context_len} message_len={len(message)} model={model}\n")

        # Build/update session (same logic as /chat)
        with _chat_sessions_lock:
            if csv_id not in _chat_sessions:
                sess = [{'role': 'system', 'content': _LLM_SYSTEM_PROMPT}]
                if context:
                    sess.append({'role': 'system',
                                 'content': 'Current chart context:\n' + context})
                _chat_sessions[csv_id] = sess
            elif context:
                sess = _chat_sessions[csv_id]
                replaced = False
                for i, m in enumerate(sess):
                    if m['role'] == 'system' and (
                            m['content'].startswith('Current chart context:') or
                            m['content'].startswith('Updated chart context:')):
                        sess[i] = {'role': 'system',
                                   'content': 'Updated chart context:\n' + context}
                        replaced = True
                        break
                if not replaced:
                    sess.append({'role': 'system',
                                 'content': 'Updated chart context:\n' + context})

            _chat_sessions[csv_id].append({'role': 'user', 'content': message})

            # Trim history
            sess        = _chat_sessions[csv_id]
            system_msgs = [m for m in sess if m['role'] == 'system']
            conv_msgs   = [m for m in sess if m['role'] != 'system']
            max_conv    = _CHAT_MAX_TURNS * 2
            if len(conv_msgs) > max_conv:
                conv_msgs = conv_msgs[-max_conv:]
            _chat_sessions[csv_id] = system_msgs + conv_msgs
            messages_snapshot = list(_chat_sessions[csv_id])

        # Log message structure (line 1321-1328)
        system_count = sum(1 for m in messages_snapshot if m['role'] == 'system')
        user_count = sum(1 for m in messages_snapshot if m['role'] == 'user')
        assistant_count = sum(1 for m in messages_snapshot if m['role'] == 'assistant')
        context_in_system = any('chart context' in m.get('content', '').lower() 
                               for m in messages_snapshot if m['role'] == 'system')
        with open(log_path, 'a') as f:
            f.write(f"[OLLAMA_SEND] system={system_count} user={user_count} assistant={assistant_count} has_chart_context={context_in_system}\n")

        # ── Send SSE headers ──
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        # ── Stream from Ollama ──
        payload = json.dumps({
            'model':       model,
            'messages':    messages_snapshot,     # ← With context!
            'stream':      True,
            'temperature': 0.3
        }).encode('utf-8')
        
        req = urllib.request.Request(
            _OLLAMA_BASE + '/api/chat',
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        full_reply = []
        with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as r:
            # Read streaming response line-by-line
            for line in r:
                data = json.loads(line.decode('utf-8'))
                token = data.get('message', {}).get('content', '')
                
                if token:
                    full_reply.append(token)
                    # Send SSE frame
                    event_data = json.dumps({'response': token})
                    self.wfile.write(f'data: {event_data}\n\n'.encode('utf-8'))
                    self.wfile.flush()
                
                if data.get('done'):
                    break

        # Store full response
        with _chat_sessions_lock:
            full_text = ''.join(full_reply)
            _chat_sessions[csv_id].append({'role': 'assistant', 'content': full_text})

    except Exception as e:
        self.send_response(500)
        self.end_headers()
```

---

## 6. Browser-Side: Receiving Streaming Response

### File: `fdv_chart_rev8/fdv_chart.html`, Lines 7060–7090

#### `_handleStreamResponse()` - Real-time token display

```javascript
function _handleStreamResponse(response) {
    var div = _chatAddMsg('assistant', '');  // Create empty message div
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';

    function read() {
        reader.read().then(function(result) {
            if (result.done) {
                _chatSetBusy(false);
                return;
            }

            buffer += decoder.decode(result.value, {stream: true});
            var lines = buffer.split('\n');
            buffer = lines.pop();  // Keep incomplete line

            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                
                // Parse SSE frame: "data: {...JSON...}"
                if (line.startsWith('data: ')) {
                    try {
                        var evt = JSON.parse(line.slice(6));
                        if (evt.response) {
                            // Append token to message div in real-time
                            div.textContent += evt.response;
                            _chatScrollBottom();
                        }
                    } catch(e) {}
                }
            }

            read();  // Recurse for next chunk
        });
    }

    read();  // Start reading
}
```

---

## 7. Data Flow Diagram (Complete)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Browser (JavaScript)                          │
│                                                                  │
│ 1. User clicks "Send"                                           │
│    ↓                                                            │
│ 2. _chatSend() calls                                           │
│    ↓                                                            │
│ 3. _buildChatContext()  ← BUILDS CONTEXT (3-10 KB)            │
│    ├─ Reads: chart config, filters, intervals                 │
│    ├─ Reads: currentHeaders, filteredIndices, allRows         │
│    ├─ Computes: bucketed stats (mean, std, min, max)         │
│    ├─ Samples: 20-200 raw data rows (evenly spaced)          │
│    └─ Returns: 300-3000 line string with full context        │
│    ↓                                                            │
│ 4. POST /chat or /chat_stream                                 │
│    ├─ message:  user's question                              │
│    ├─ context:  output from _buildChatContext()              │
│    └─ model:    'llama3' or 'gemma4'                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ (Network: ~100ms)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Server (Python)                               │
│                                                                  │
│ 1. Parse POST body (json.loads)                               │
│    ├─ csv_id:  file identifier                               │
│    ├─ message: user question                                 │
│    ├─ context: chart context string (3-10 KB)               │
│    └─ model:   LLM model name                                │
│    ↓                                                           │
│ 2. Log [CHAT] context_bytes=3124                             │
│    ↓                                                           │
│ 3. Manage session history (_chat_sessions_lock)              │
│    ├─ NEW session? Create with system prompt                │
│    ├─ Existing session? Replace old context if present      │
│    ├─ Append context as system message: "Current chart..." │
│    ├─ Append user message                                    │
│    └─ Trim history (keep last 20 turns)                     │
│    ↓                                                           │
│ 4. Create messages_snapshot (list of message dicts)         │
│    ├─ messages[0]: system prompt                            │
│    ├─ messages[1]: system prompt with context ← KEY!        │
│    ├─ messages[2]: user question                            │
│    └─ messages[3]: prior assistant response (if exists)     │
│    ↓                                                           │
│ 5. Log [CHAT_SEND] system=2 user=1 assistant=0             │
│                    has_chart_context=True                    │
│    ↓                                                           │
│ 6. Call _call_llm(messages_snapshot, model='llama3')        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ (Network: Ollama API at localhost:11434)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Ollama (LLM Model)                             │
│                                                                  │
│ 1. Receive POST /api/chat                                     │
│    {                                                          │
│      "model": "llama3",                                       │
│      "messages": [                                            │
│        {                                                      │
│          "role": "system",                                    │
│          "content": "You are a data analysis assistant..."   │
│        },                                                     │
│        {                                                      │
│          "role": "system",                                    │
│          "content": "Current chart context:\n              │
│                      FDV Dashboard...\n                       │
│                      Chart type: scatter\n                    │
│                      X column: frequency\n                    │
│                      [full context from browser]"            │
│        },                                                     │
│        {                                                      │
│          "role": "user",                                      │
│          "content": "What are key findings?"                 │
│        }                                                      │
│      ],                                                       │
│      "stream": true,                                          │
│      "temperature": 0.3                                       │
│    }                                                          │
│    ↓                                                           │
│ 2. Process context (llama3 understands chart stats)         │
│    ↓                                                           │
│ 3. Generate tokens (if stream=true, send token-by-token)    │
│    "Based on the chart..."                                   │
│    "Based on the chart context..."                           │
│    "Based on the chart context, I see..."                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ (Stream: tokens arrive ~500/sec)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Server (SSE Wrapper)                          │
│                                                                  │
│ 1. For each token from Ollama:                               │
│    ├─ Parse JSON line                                         │
│    ├─ Extract token content                                   │
│    └─ Send SSE frame: "data: {response: 'token'}\n\n"       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ (Network: SSE stream)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Browser (SSE Listener)                        │
│                                                                  │
│ 1. _handleStreamResponse() reads stream                       │
│    ↓                                                           │
│ 2. For each SSE frame received:                              │
│    ├─ Parse: "data: {response: 'token'}"                    │
│    ├─ Extract: token = "token"                              │
│    ├─ Append to message div: div.textContent += token       │
│    ├─ Scroll to bottom                                       │
│    └─ Repeat (continuous until done)                         │
│    ↓                                                           │
│ 3. User sees words appearing in real-time (1-2 sec)         │
│    "Based"                                                    │
│    "Based on"                                                 │
│    "Based on the"                                             │
│    "Based on the chart"                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Context is built entirely from client-side data** in `_buildChatContext()`
   - Reads: chart config, filters, column names, raw rows
   - Computes: statistics that mirror the plot
   - Formats: 300–3000 line text string

2. **Context is sent to server in POST body** (plaintext, not encrypted unless HTTPS)
   - Server logs arrival: `[CHAT] context_bytes=3124`
   - Server adds to session as system message

3. **Server sends context to Ollama in /api/chat request**
   - Context is in messages[1] (second system message)
   - Ollama uses it to understand user's data

4. **Streaming provides real-time feedback**
   - Tokens arrive ~500/sec
   - Browser appends incrementally
   - User sees response within 1–2 seconds

5. **Session history is trimmed** to avoid memory bloat
   - Keeps all system messages (unlimited)
   - Keeps last 20 user+assistant turn pairs (configurable)

---

**Documentation Version**: 1.0 | **Last Updated**: May 2, 2026
