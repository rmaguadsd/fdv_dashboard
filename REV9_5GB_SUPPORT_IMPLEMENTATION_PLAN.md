# REV9 5GB Support Implementation Plan

**Objective**: Enable REV9 to safely parse files up to 5GB without hanging or OOM

**Current Limit**: ~1GB (before memory pressure / timeout issues)  
**Target**: 5GB files with graceful handling

---

## Executive Summary

REV9 currently **cannot reliably parse 5GB files** due to three critical bottlenecks:

1. **Unbounded row list growth** (30-50 GB for 100M rows)
2. **10-minute parse timeout** (insufficient for large files)
3. **No streaming/pagination of results**

**To support 5GB, implement these 4 changes** (Priority Order):

| Priority | Change | Impact | Effort | Timeline |
|----------|--------|--------|--------|----------|
| **P1** | Row batching + SQLite cache | Eliminates 30-50 GB allocation | Medium | 1-2 days |
| **P2** | Extend timeout to 60+ minutes | Allows time to parse 5GB | Low | 1 hour |
| **P3** | Streaming result pagination | Allows UI to show data faster | Medium | 1-2 days |
| **P4** | Progress reporting | Better UX | Low | 1 day |

---

## Current State Analysis

### **Problem #1: Unbounded Row List (CRITICAL)**

**Location**: `parse_log_file()` line 428

```python
rows = []
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # ... parse line ...
        if row:
            rows.append(row)  # ← Grows unbounded
```

**Impact on 5GB**:
```
5 GB file
↓
~50M parsed rows (assuming 10% match rate)
↓
50M rows × 20 KB/row = 1 TB? NO, actually ~50 GB
↓
System runs out of memory or swap thrashing
```

**Why problematic**:
- All rows held in memory simultaneously
- No pagination/streaming
- GC thrashing on large allocations
- Browser timeout (30-60s) but parse continues for 10+ minutes

**Solution**: Implement **row batching with SQLite cache**

---

### **Problem #2: 10-Minute Timeout (INSUFFICIENT)**

**Location**: `_run_parse_job()` line 478

```python
MAX_PARSE_TIME = 600  # 10 minutes
```

**Time needed for 5GB**:
```
5 GB file
Parsing speed: ~100K-200K rows/sec (with regex)
Time needed: 5GB / 200 MB/sec = 25 seconds (pure I/O)
            + parsing overhead: 5-10x
            = 125-250 seconds = 2-4 minutes

WORST CASE (complex regex):
5 GB / 10 MB/sec (complex regex) = 500 seconds = 8+ minutes
(But currently timeout is only 10 min, cutting it close)

RECOMMENDATION: Extend to 60 minutes
```

**Why problematic**:
- 10 minutes insufficient for 5GB with complex regex
- No margin for slow storage or system load
- Browser times out before server

**Solution**: Extend to **60-minute timeout** (configurable)

---

### **Problem #3: No Result Streaming (UX ISSUE)**

**Current Behavior**:
```
User uploads 5GB file
↓
System parses silently (no progress)
↓
10-60 minutes pass
↓
First response: 500 preview rows
↓
User can download full CSV (all 50M rows as JSON) 💥 OOM
```

**Why problematic**:
- Browser can't serialize 50M rows to JSON
- No way to get partial results
- User sees nothing for long time

**Solution**: Implement **paginated/streaming results**

---

## Implementation Plan

### **PHASE 1: Row Batching + SQLite Cache (P1 - CRITICAL)**

**Goal**: Prevent unbounded row list growth by checkpointing to disk

**Implementation**:

#### Step 1.1: Create SQLite Schema

```python
import sqlite3
from pathlib import Path
import tempfile

def create_cache_db(cache_id):
    """Create SQLite database for caching parsed rows"""
    cache_dir = Path(tempfile.gettempdir()) / 'fdv_parse_cache'
    cache_dir.mkdir(exist_ok=True)
    
    db_path = cache_dir / f'cache_{cache_id}.db'
    conn = sqlite3.connect(str(db_path))
    
    # Create table matching HEADERS
    col_defs = ', '.join([
        f'"{col}" TEXT' for col in HEADERS
    ])
    
    conn.execute(f'''
        CREATE TABLE rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {col_defs}
        )
    ''')
    
    # Create index on SourceFile for quick queries
    conn.execute('CREATE INDEX idx_source ON rows(SourceFile)')
    
    conn.commit()
    return db_path, conn
```

#### Step 1.2: Modify parse_log_file() to Use Batching

```python
def parse_log_file(file_path, regex_include=None, regex_exclude=None, source_name=None, cache_id=None):
    """
    Parse log file with row batching to prevent unbounded memory growth.
    
    Instead of:
        rows = []
        for line in file:
            rows.append(parse(line))
        return rows  # Full dataset
    
    Now:
        rows_batch = []
        for line in file:
            rows_batch.append(parse(line))
            if len(rows_batch) >= BATCH_SIZE:
                db_insert(rows_batch)
                rows_batch = []
        
        return (headers, cache_id, total_count)  # Metadata + cache reference
    """
    
    BATCH_SIZE = 50000  # Insert 50K rows at a time
    
    if not cache_id:
        cache_id = 'cache_' + uuid.uuid4().hex[:8]
    
    db_path, conn = create_cache_db(cache_id)
    
    # ... same regex setup ...
    
    batch = []
    total_rows = 0
    
    try:
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
                
                # Parse line
                if 'FDV OUTPUT' in line_stripped:
                    row = _parse_output_line(line_stripped, line_num)
                elif 'FDV POLL' in line_stripped:
                    row = _parse_poll_line(line_stripped, line_num)
                else:
                    continue
                
                if row:
                    batch.append(row)
                    total_rows += 1
                    
                    # ← KEY CHANGE: Batch insert to DB
                    if len(batch) >= BATCH_SIZE:
                        _batch_insert_rows(conn, batch)
                        batch = []
                        
                        # Optional: Update progress
                        if cache_id in parse_jobs:
                            with parse_jobs_lock:
                                parse_jobs[cache_id]['rows_parsed'] = total_rows
        
        # Flush remainder
        if batch:
            _batch_insert_rows(conn, batch)
        
        conn.close()
        
    except Exception as e:
        conn.close()
        raise
    
    elapsed_parse = time.time() - start_parse
    debug_log(f"[parse_log_file] Cached {total_rows} rows in {elapsed_parse:.2f}s")
    
    # Return metadata instead of rows
    return HEADERS, cache_id, total_rows
```

#### Step 1.3: Implement Batch Insert

```python
def _batch_insert_rows(conn, rows_batch):
    """Insert batch of rows into SQLite database"""
    
    if not rows_batch:
        return
    
    # Build INSERT statement
    cols = ', '.join([f'"{col}"' for col in HEADERS])
    placeholders = ', '.join(['?' for _ in HEADERS])
    
    sql = f'INSERT INTO rows ({cols}) VALUES ({placeholders})'
    
    # Convert rows to tuples
    data = [tuple(row) for row in rows_batch]
    
    conn.executemany(sql, data)
    conn.commit()
```

#### Step 1.4: Implement Pagination Query

```python
def get_cached_rows(cache_id, offset=0, limit=500):
    """Retrieve paginated rows from cache database"""
    
    cache_dir = Path(tempfile.gettempdir()) / 'fdv_parse_cache'
    db_path = cache_dir / f'cache_{cache_id}.db'
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Query with pagination
    cols = ', '.join([f'"{col}"' for col in HEADERS])
    sql = f'SELECT {cols} FROM rows ORDER BY id LIMIT {limit} OFFSET {offset}'
    
    cursor = conn.execute(sql)
    rows = [dict(row) for row in cursor.fetchall()]
    
    # Get total count
    total = conn.execute('SELECT COUNT(*) FROM rows').fetchone()[0]
    
    conn.close()
    
    return rows, total
```

**Result**: 
- ✅ Only 50K rows in memory at any time (~1 GB peak)
- ✅ 5GB file uses constant memory (~512 MB)
- ✅ All rows accessible via pagination
- ⏱️ Parse time: 5-15 minutes (depends on regex)

---

### **PHASE 2: Extend Parse Timeout (P2 - HIGH)**

**Goal**: Allow 60+ minutes for large file parsing

**Implementation**:

#### Step 2.1: Make Timeout Configurable

```python
# Instead of hardcoded:
MAX_PARSE_TIME = 600  # 10 minutes

# Make it configurable by file size:
def calculate_timeout(file_size_bytes):
    """Calculate timeout based on file size"""
    
    # Conservative estimate: 100 MB per minute
    # (accounts for complex regex, slow storage)
    mb = file_size_bytes / (1024 * 1024)
    timeout_seconds = int((mb / 100) * 60)
    
    # Minimum 10 min, Maximum 120 min
    timeout_seconds = max(600, min(7200, timeout_seconds))
    
    return timeout_seconds

# Usage in _run_parse_job():
def _run_parse_job(job_id, file_path, ...):
    file_size = os.path.getsize(file_path)
    MAX_PARSE_TIME = calculate_timeout(file_size)
    
    # 5 GB file: timeout = 5000 MB / 100 MB/min × 60 = 3000 sec = 50 min ✓
```

#### Step 2.2: Progress Tracking During Parse

```python
# In _run_parse_job(), pass progress callback:
def _run_parse_job(job_id, file_path, ...):
    MAX_PARSE_TIME = calculate_timeout(os.path.getsize(file_path))
    
    def progress_callback(rows_parsed, elapsed):
        with parse_jobs_lock:
            parse_jobs[job_id]['rows_parsed'] = rows_parsed
            parse_jobs[job_id]['elapsed_seconds'] = elapsed
            parse_jobs[job_id]['progress_pct'] = min(
                100,
                int((elapsed / MAX_PARSE_TIME) * 100)
            )
    
    headers, cache_id, total_rows = parse_log_file(
        file_path,
        regex_include=regex_include,
        regex_exclude=regex_exclude,
        source_name=source_name,
        cache_id=job_id,
        progress_callback=progress_callback  # ← Pass callback
    )
```

**Result**:
- ✅ 5GB files get 50 minutes to parse
- ✅ UI can show progress in real-time
- ✅ No more mysterious 10-minute timeout for large files

---

### **PHASE 3: Streaming Result Pagination (P3 - MEDIUM)**

**Goal**: Return results in pages instead of all-at-once

**Implementation**:

#### Step 3.1: Modify Result Response Format

```python
# Before:
result = {
    'success': True,
    'csv_id': csv_id,
    'headers': headers,
    'rows': rows[:PREVIEW],  # Only 500 rows
    'total_rows': len(rows),
    'has_more': len(rows) > PREVIEW
}

# After: Reference cache instead of holding data
result = {
    'success': True,
    'cache_id': cache_id,  # ← Reference to SQLite DB
    'headers': headers,
    'rows': get_cached_rows(cache_id, offset=0, limit=500)[0],  # First 500
    'total_rows': total_rows,
    'page_size': 500,
    'current_page': 0,
    'has_more': total_rows > 500
}
```

#### Step 3.2: Add Pagination Endpoint

```python
def do_GET(self):
    # ... existing code ...
    
    elif self.path.startswith('/get_rows/'):
        # GET /get_rows/<cache_id>?offset=0&limit=500
        try:
            cache_id = self.path.split('/')[2]
            query = urlparse(self.path).query
            params = parse_qs(query)
            
            offset = int(params.get('offset', ['0'])[0])
            limit = int(params.get('limit', ['500'])[0])
            
            # Ensure sane limits
            offset = max(0, offset)
            limit = min(10000, max(1, limit))  # Max 10K rows per request
            
            rows, total = get_cached_rows(cache_id, offset, limit)
            
            _send_json(self, 200, {
                'success': True,
                'rows': rows,
                'offset': offset,
                'limit': limit,
                'total': total,
                'has_more': offset + limit < total
            })
            
        except Exception as e:
            _send_json(self, 400, {'success': False, 'error': str(e)})
```

#### Step 3.3: Add CSV Download with Streaming

```python
elif self.path.startswith('/download_csv/'):
    # GET /download_csv/<cache_id>
    try:
        cache_id = self.path.split('/')[2]
        
        # Stream CSV from SQLite
        cache_dir = Path(tempfile.gettempdir()) / 'fdv_parse_cache'
        db_path = cache_dir / f'cache_{cache_id}.db'
        
        conn = sqlite3.connect(str(db_path))
        
        # Send CSV headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv; charset=utf-8')
        self.send_header('Content-Disposition', f'attachment; filename="{cache_id}.csv"')
        self.end_headers()
        
        # Stream rows
        writer = csv.DictWriter(self.wfile, fieldnames=HEADERS)
        writer.writeheader()
        
        # Query in chunks to avoid memory spike
        CHUNK_SIZE = 50000
        offset = 0
        
        while True:
            cursor = conn.execute(f'''
                SELECT * FROM rows 
                ORDER BY id 
                LIMIT {CHUNK_SIZE} OFFSET {offset}
            ''')
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            for row in rows:
                writer.writerow(dict(row))
                self.wfile.flush()
            
            offset += CHUNK_SIZE
        
        conn.close()
        
    except Exception as e:
        _send_json(self, 400, {'success': False, 'error': str(e)})
```

**Result**:
- ✅ UI can fetch first 500 rows in seconds
- ✅ User can scroll/paginate through results
- ✅ Can download full CSV without OOM
- ✅ Server uses constant memory during streaming

---

### **PHASE 4: Progress Reporting (P4 - NICE-TO-HAVE)**

**Goal**: Real-time progress feedback to user

**Implementation**:

#### Step 4.1: Add Progress Polling Endpoint

```python
elif self.path.startswith('/job_status/'):
    # GET /job_status/<job_id>
    try:
        job_id = self.path.split('/')[2]
        
        with parse_jobs_lock:
            if job_id not in parse_jobs:
                _send_json(self, 404, {'success': False, 'error': 'Job not found'})
                return
            
            job = parse_jobs[job_id]
            
            status = {
                'state': job['state'],
                'rows_parsed': job.get('rows_parsed', 0),
                'elapsed_seconds': job.get('elapsed_seconds', 0),
                'progress_pct': job.get('progress_pct', 0),
            }
            
            if job['state'] == 'done':
                status['cache_id'] = job['result']['cache_id']
                status['total_rows'] = job['result']['total_rows']
            elif job['state'] == 'error':
                status['error'] = job['error']
            
            _send_json(self, 200, {'success': True, **status})
        
    except Exception as e:
        _send_json(self, 400, {'success': False, 'error': str(e)})
```

#### Step 4.2: Update JavaScript to Poll Progress

```javascript
async function pollJobStatus(jobId) {
    const maxWaitTime = 3600000; // 1 hour max
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
        const resp = await fetch(`/job_status/${jobId}`);
        const data = await resp.json();
        
        if (!data.success) {
            showError('Job error: ' + data.error);
            break;
        }
        
        const { state, rows_parsed, elapsed_seconds, progress_pct } = data;
        
        // Update UI
        document.getElementById('progress-bar').style.width = progress_pct + '%';
        document.getElementById('progress-text').textContent = 
            `Parsed ${rows_parsed} rows (${progress_pct}%) - ${elapsed_seconds}s elapsed`;
        
        if (state === 'done') {
            showResults(data.cache_id, data.total_rows);
            break;
        } else if (state === 'error') {
            showError(data.error);
            break;
        }
        
        // Poll every 2 seconds
        await new Promise(r => setTimeout(r, 2000));
    }
}
```

**Result**:
- ✅ User sees live progress: "Parsed 5M rows (25%) - 120s elapsed"
- ✅ Knows how long to wait
- ✅ Can cancel if needed (future enhancement)

---

## Memory Impact Analysis

### **Before (Current)**

```
5 GB file upload
↓
Stream to disk: 512 KB ✓
↓
Parse 50M rows into memory: 50M × 20 KB = 1 TB? NO, ~50 GB
↓
Returned as JSON: HUGE OOM
↓
PEAK MEMORY: 50 GB ❌
```

### **After (With Batching)**

```
5 GB file upload
↓
Stream to disk: 512 KB ✓
↓
Parse in batches (50K rows per batch):
  - Read 50K from file: 100 MB
  - Parse 50K: 1 GB
  - Insert to SQLite: 1 GB
  - Clear memory: back to 512 KB
  - Repeat 1000x
↓
Result queried from SQLite: 1 MB per page ✓
↓
PEAK MEMORY: ~1-2 GB ✓✓✓
```

**Memory Reduction**: 50 GB → 2 GB (96% reduction)

---

## Timeline & Effort

| Phase | Task | Effort | Time | Complexity |
|-------|------|--------|------|-----------|
| P1 | SQLite schema | 2 hours | 2h | Medium |
| P1 | Row batching | 3 hours | 3h | Medium |
| P1 | Batch insert | 1 hour | 1h | Low |
| P1 | Pagination query | 1 hour | 1h | Low |
| **P1 TOTAL** | **4 core changes** | **7 hours** | **1 day** | **Medium** |
| P2 | Dynamic timeout | 1 hour | 1h | Low |
| P2 | Progress tracking | 1 hour | 1h | Low |
| **P2 TOTAL** | **2 changes** | **2 hours** | **2h** | **Low** |
| P3 | Pagination endpoint | 2 hours | 2h | Low |
| P3 | CSV streaming | 1 hour | 1h | Low |
| **P3 TOTAL** | **2 changes** | **3 hours** | **3h** | **Low** |
| P4 | Progress endpoint | 1 hour | 1h | Low |
| P4 | JS polling | 1 hour | 1h | Low |
| **P4 TOTAL** | **2 changes** | **2 hours** | **2h** | **Low** |
| **ALL** | **Full 5GB support** | **14 hours** | **2-3 days** | **Medium** |

---

## Implementation Checklist

### **Phase 1: Row Batching**
- [ ] Create `create_cache_db()` function
- [ ] Modify `parse_log_file()` to accept `cache_id` parameter
- [ ] Implement batch insertion logic (50K rows per batch)
- [ ] Update return value to `(headers, cache_id, total_rows)`
- [ ] Create `get_cached_rows(cache_id, offset, limit)` function
- [ ] Test with 1GB file
- [ ] Test with 5GB file
- [ ] Verify memory stays under 2GB

### **Phase 2: Extended Timeout**
- [ ] Create `calculate_timeout(file_size_bytes)` function
- [ ] Update `_run_parse_job()` to use dynamic timeout
- [ ] Add progress callback to parse_log_file()
- [ ] Update parse_jobs to track progress_pct
- [ ] Test timeout on slow system

### **Phase 3: Pagination**
- [ ] Modify result format to use cache_id reference
- [ ] Create `/get_rows/<cache_id>` endpoint
- [ ] Create `/download_csv/<cache_id>` endpoint
- [ ] Implement CSV streaming
- [ ] Test pagination with 1M+ rows
- [ ] Test large CSV download

### **Phase 4: Progress Reporting**
- [ ] Create `/job_status/<job_id>` endpoint
- [ ] Update UI to poll progress
- [ ] Add progress bar to HTML
- [ ] Test progress updates during parse
- [ ] Test on slow 5GB parse

---

## Testing Plan

### **Test 1: 1GB File (Baseline)**
```
File: 1GB test log
Expected: 10M rows parsed
Memory: <2GB
Time: 2-3 minutes
Result: ✅ Should work
```

### **Test 2: 5GB File (Main Goal)**
```
File: 5GB test log
Expected: 50M rows parsed
Memory: <2GB (constant)
Time: 10-15 minutes
Result: ✅ Should not OOM or hang
```

### **Test 3: Complex Regex on 5GB**
```
File: 5GB with complex regex
Expected: 25M rows parsed (50% match)
Memory: <2GB
Time: 20-30 minutes
Result: ✅ Should complete or timeout gracefully
```

### **Test 4: Pagination**
```
Parse 1GB file
Fetch rows:
  - Offset 0, limit 500: <100ms
  - Offset 1M, limit 500: <200ms
  - Offset 10M, limit 500: <300ms
Result: ✅ Should be fast regardless of offset
```

### **Test 5: CSV Download**
```
Parse 5GB file
Download full CSV:
  - Should stream without loading all to RAM
  - Should complete in reasonable time
Result: ✅ Should not OOM during download
```

### **Test 6: Progress Polling**
```
Parse 5GB file
Poll progress every 2 seconds:
  - Should show rows_parsed increasing
  - Should show elapsed_seconds increasing
  - Should show progress_pct 0-100%
Result: ✅ Should provide real-time feedback
```

---

## Risk Mitigation

### **Risk 1: SQLite Performance Degrades with Large DB**
**Mitigation**:
- Create index on SourceFile for quick queries
- Use BATCH_SIZE of 50K for fast inserts
- Vacuum database after parsing to optimize

### **Risk 2: Disk Space for Cache Database**
**Mitigation**:
- 5GB file = ~50GB cache DB (worst case)
- Need ~100GB free disk space
- Auto-cleanup cache after download or 24 hours
- Add warning if disk space low

### **Risk 3: Query Performance on Large SQLite DB**
**Mitigation**:
- Use LIMIT/OFFSET with ORDER BY id (fastest)
- Pre-create index for common queries
- Consider sharding if >1B rows needed

### **Risk 4: Thread Safety with SQLite**
**Mitigation**:
- SQLite handles multi-threaded access automatically
- Use timeout=30 for connection locks
- Don't share connection between threads

---

## Configuration Parameters

Add to REV9 config:

```python
# Row batching
BATCH_SIZE = 50000              # Rows per batch insert
CACHE_DIR = Path(tempfile.gettempdir()) / 'fdv_parse_cache'

# Timeouts
MIN_TIMEOUT = 600               # 10 minutes minimum
MAX_TIMEOUT = 7200              # 2 hours maximum
PARSE_RATE_MB_PER_MIN = 100     # Conservative estimate for complex regex

# Pagination
PAGE_SIZE = 500                 # Default rows per page
MAX_PAGE_SIZE = 10000           # Max rows per request

# Cleanup
CACHE_TTL_HOURS = 24            # Delete cache after 24 hours
```

---

## Conclusion

To support **5GB file parsing**, REV9 needs:

1. **✅ Row Batching + SQLite Cache** (P1) - Reduces 50GB allocation to 2GB
2. **✅ Extended Timeout** (P2) - Allows 50+ minutes for parsing
3. **✅ Pagination/Streaming** (P3) - Fast UI + safe downloads
4. **✅ Progress Reporting** (P4) - Better UX

**Total effort**: 2-3 days of development  
**Total complexity**: Medium  
**Result**: Safe, reliable 5GB file support with <2GB peak memory

**Estimated completion**: 1 week (including testing)
