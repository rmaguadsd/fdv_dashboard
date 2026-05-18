# Rev9 Hanging on Large Files - Root Cause Analysis

## Summary
Rev9 will hang when parsing huge files due to **unbounded memory consumption** during file upload and parsing. The issue occurs at multiple points in the request handling pipeline.

---

## Issue #1: Unbounded Memory Read in `/parse` Endpoint (CRITICAL)
**Location**: `fdv_chart.py` lines 924-930
```python
content_len = int(self.headers.get('Content-Length', 0))
if content_len > 2 * 1024 * 1024 * 1024:  # Only checks >2GB
    raise ValueError('File too large (>2 GB)')

body = self.rfile.read(content_len)  # ← ENTIRE FILE READ INTO RAM
```

**Problem**: 
- The entire file is read into memory before processing (line 924)
- With a 1-2 GB file, this allocates 1-2 GB of RAM instantly
- Python then tries to split the multipart boundary across this huge buffer (line 939)
- Another copy is created when `file_content` is extracted (line 948)
- Total memory usage: 3-4x the file size during parsing

**Result**: Memory pressure causes thrashing, GC pauses, and eventual hang/crash

---

## Issue #2: Unbounded Memory Read in `/parse_multi` Endpoint (CRITICAL)
**Location**: `fdv_chart.py` lines 1044-1048
```python
content_len = int(self.headers.get('Content-Length', 0))
if content_len > 4 * 1024 * 1024 * 1024:  # Only checks >4GB
    raise ValueError('Upload too large (>4 GB)')

body = self.rfile.read(content_len)  # ← ENTIRE MULTI-FILE UPLOAD READ INTO RAM
```

**Problem**: 
- Same issue as Issue #1, but for multi-file uploads
- If uploading 3 x 500MB files, allocates 1.5 GB instantly
- Then splits all files again, creating more copies
- Total: 4-5 GB memory for 1.5 GB upload

---

## Issue #3: Inefficient Multipart Parsing
**Location**: `fdv_chart.py` lines 939-965
```python
parts_list = body.split(boundary_bytes)  # ← Splits entire 1-2 GB buffer into list

file_content = None
for part in parts_list:  # ← Iterates hundreds of thousands of parts
    if b'name="file"' in part and b'filename=' in part:
        lines = part.split(b'\r\n')  # ← More string operations on huge buffers
        for i, line in enumerate(lines):
            if line == b'':
                file_content = b'\r\n'.join(lines[i+1:-1])  # ← Creates new bytes object
```

**Problem**:
- `.split()` creates a list of all parts in memory
- For a 1 GB file, this can create thousands of strings
- Each `.split()`, `.join()` operation copies data
- Regex matching on each part adds CPU overhead

**Result**: O(n) memory copies, quadratic time complexity

---

## Issue #4: No Progress/Timeout Handling
**Location**: `fdv_chart.py` lines 463-495 (parse job)

**Problem**:
- Parse jobs have no timeout mechanism
- Large files with 100M+ rows can take hours to parse
- Browser request timeout (typically 30-60 seconds) causes orphaned threads
- No progress reporting for large operations

**Result**: Hangs appear to freeze completely; threads pile up

---

## Issue #5: Parsing Loop Has No Batching/Yielding
**Location**: `fdv_chart.py` lines 410-455
```python
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line_num += 1
        line_stripped = line.rstrip('\n')

        if not line_stripped.strip():
            continue

        # Apply regex filters
        if compiled_include:
            if not compiled_include.search(line_stripped):
                continue
        if compiled_exclude:
            if compiled_exclude.search(line_stripped):
                continue

        # Auto-detect and parse line type
        if 'FDV OUTPUT' in line_stripped:
            row = _parse_output_line(line_stripped, line_num)
        elif 'FDV POLL' in line_stripped:
            row = _parse_poll_line(line_stripped, line_num)
        else:
            continue

        if row:
            rows.append(row)  # ← Unbounded list growth
```

**Problem**:
- `rows` list grows unbounded in memory
- For 100M lines, stores ~30-50 GB of Python objects
- No periodic persistence or batching
- No progress updates to browser

**Result**: Memory exhaustion, CPU thrashing from GC

---

## Recommended Fixes (Priority Order)

### Fix #1: Streaming Multipart Upload (CRITICAL)
Replace the entire `body = self.rfile.read(content_len)` approach with streaming:

```python
# Stream directly to temp file instead of reading into RAM
temp_path = Path(tempfile.gettempdir()) / ('fdv_upload_' + uuid.uuid4().hex + '.log')
max_chunk = 256 * 1024  # 256 KB chunks
total_read = 0

with open(temp_path, 'wb') as tmp:
    while total_read < content_len:
        chunk_size = min(max_chunk, content_len - total_read)
        chunk = self.rfile.read(chunk_size)
        if not chunk:
            break
        tmp.write(chunk)
        total_read += len(chunk)
```

**Benefit**: Memory usage drops from 2-4 GB to 256 KB

---

### Fix #2: Streaming Multipart Parsing (CRITICAL)
Parse multipart boundary while streaming, not after loading:

```python
# Use a streaming multipart parser that yields chunks
# Instead of: parts_list = body.split(boundary_bytes)
# Do: for part_header, part_content in stream_multipart(self.rfile, boundary):
```

**Benefit**: Constant memory regardless of file size

---

### Fix #3: Add Row Batching to Parse Loop (HIGH)
Batch rows and checkpoint progress:

```python
MAX_BATCH = 50000
batch = []

for line in f:
    # ... parse line ...
    if row:
        batch.append(row)
        if len(batch) >= MAX_BATCH:
            rows.extend(batch)
            batch = []
            # Optional: update progress
            
rows.extend(batch)  # flush remainder
```

**Benefit**: Allows progress reporting, reduces peak memory

---

### Fix #4: Add Timeout/Cancellation (MEDIUM)
```python
def _run_parse_job(job_id, file_path, ...):
    import signal
    
    def timeout_handler(signum, frame):
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'timeout'
            parse_jobs[job_id]['error'] = 'Parse job exceeded 5-minute timeout'
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes
    
    try:
        headers, rows = parse_log_file(...)
    finally:
        signal.alarm(0)
```

**Benefit**: Prevents orphaned threads, clear user feedback

---

### Fix #5: Limit Initial Result Size (MEDIUM)
```python
# Current: returns all rows
# New: return only first 500 with "more available"

PREVIEW = 500
result = {
    'success': True,
    'csv_id': csv_id,
    'headers': headers,
    'rows': rows[:PREVIEW],
    'total_rows': len(rows),
    'has_more': len(rows) > PREVIEW,
    'message': f'Showing first {PREVIEW} of {len(rows)} rows'
}
```

**Benefit**: UI stays responsive, let user download full CSV if needed

---

## Quick Summary Table

| Issue | Severity | Cause | Effect | Fix |
|-------|----------|-------|--------|-----|
| Full file to RAM | CRITICAL | `rfile.read(content_len)` | 2-4 GB memory | Stream to disk |
| Multipart split | CRITICAL | `body.split(boundary)` | Huge list creation | Stream parse |
| Unbounded row list | HIGH | `rows.append(row)` loop | 30-50 GB for 100M rows | Batch + checkpoint |
| No timeout | MEDIUM | No signal/timer | Orphaned threads | Add `signal.alarm()` |
| Full results returned | MEDIUM | Returns all rows | 1 GB JSON payloads | Preview mode |

---

## Testing Recommendations

1. **Upload a 1 GB test file** → should not use >512 MB RAM (with Fix #1)
2. **Parse 100M row file** → should complete in <10 minutes (with Fix #2-3)
3. **Check memory graph** → should stay flat during upload (with streaming)
4. **Cancel job mid-parse** → should kill thread cleanly (with Fix #4)
5. **Test timeout** → job should stop at 5 minutes (with Fix #4)

---

## Files to Modify
- `fdv_chart_rev9/fdv_chart.py`: All fixes apply here

## Implementation Priority
1. Fix #1 (streaming upload) — **blocks all large files**
2. Fix #3 (batching) — **fixes parsing hang**
3. Fix #4 (timeout) — **prevents zombie threads**
4. Fix #2 (multipart) — **nice-to-have optimization**
5. Fix #5 (preview) — **UX improvement**
