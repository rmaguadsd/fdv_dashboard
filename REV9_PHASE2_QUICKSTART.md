# Phase 2: Dynamic Timeouts & Pagination - Quick Start Guide

**Objective**: Enable 5GB file support by implementing dynamic timeouts and efficient pagination

---

## Phase 2a: Dynamic Timeout Calculation (30 minutes)

### Location
File: `dev/aitools/fdv_chart_rev9/fdv_chart.py`  
Current: Line 583 `MAX_PARSE_TIME = 600  # 10 minutes max parse time`

### Changes Required

**1. Add file size detection** (before parse_log_file call):
```python
import os
file_size_gb = os.path.getsize(file_path) / (1024**3)

# Calculate timeout: 10 min base + 10 min per GB
# 1GB = 20 min, 5GB = 60 min
calculated_timeout = 600 + int(file_size_gb * 600)
MAX_PARSE_TIME = max(600, min(calculated_timeout, 3600))  # Cap at 60 min
```

**2. Update parse job**:
- Replace line: `MAX_PARSE_TIME = 600`
- With: Dynamic calculation above

**3. Add logging**:
```python
debug_log("Parse timeout: {0}s for {1}GB file".format(MAX_PARSE_TIME, file_size_gb))
```

**4. Update _run_parse_multi_job** similarly (line ~668)

---

## Phase 2b: CSV Download Endpoint (45 minutes)

### Location
File: `dev/aitools/fdv_chart_rev9/fdv_chart.py`  
Add new endpoint in RequestHandler.do_GET around line 1100

### Implementation

```python
elif self.path.startswith('/download_csv/'):
    # Parse csv_id from path: /download_csv/csv_XXXXX
    csv_id = self.path.split('/')[-1]
    
    if csv_id not in parsed_cache:
        self.send_error(404)
        return
    
    cached = parsed_cache[csv_id]
    headers = cached['headers']
    
    # Stream from SQLite
    if cached.get('is_sqlite'):
        cache_id = cached.get('cache_id')
        db_path = f"{CACHE_DIR}/{cache_id}.db"
        
        # Create CSV in memory buffer
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        
        # Stream rows from SQLite in batches
        if Path(db_path).exists():
            db = sqlite3.connect(db_path, check_same_thread=False)
            db.row_factory = sqlite3.Row
            
            # Fetch in 10K batches to manage memory
            offset = 0
            batch_size = 10000
            while True:
                cursor = db.execute(
                    'SELECT * FROM rows LIMIT ? OFFSET ?',
                    (batch_size, offset)
                )
                rows = cursor.fetchall()
                if not rows:
                    break
                for row in rows:
                    writer.writerow(row)
                offset += batch_size
            
            db.close()
    
    # Send as file download
    csv_data = output.getvalue().encode('utf-8')
    self.send_response(200)
    self.send_header('Content-Type', 'text/csv; charset=utf-8')
    self.send_header('Content-Disposition', 
                     'attachment; filename="data_{0}.csv"'.format(csv_id))
    self.send_header('Content-Length', len(csv_data))
    self.end_headers()
    self.wfile.write(csv_data)
```

---

## Phase 2c: Enhanced Pagination Endpoint (30 minutes)

### Location
Enhance existing `/rows` endpoint around line 938

### Current Implementation Problem
```python
all_rows = parsed_cache[csv_id]['rows']
chunk = all_rows[offset:offset + limit]  # ← Loads ALL rows to RAM!
```

### New Implementation
```python
elif self.path.startswith('/rows'):
    from urllib.parse import urlparse, parse_qs
    qs = parse_qs(urlparse(self.path).query)
    csv_id = qs.get('csv_id', [''])[0]
    offset = int(qs.get('offset', ['0'])[0])
    limit = int(qs.get('limit', ['1000'])[0])
    
    if csv_id not in parsed_cache:
        _send_json(self, 404, {'success': False, 'error': 'csv_id not found'})
        return
    
    cached = parsed_cache[csv_id]
    headers = cached['headers']
    
    # Query only needed rows from SQLite
    chunk = []
    total = 0
    
    if cached.get('is_sqlite'):
        cache_id = cached.get('cache_id')
        db_path = f"{CACHE_DIR}/{cache_id}.db"
        
        if Path(db_path).exists():
            db = sqlite3.connect(db_path, check_same_thread=False)
            db.row_factory = sqlite3.Row
            
            # Get total count
            cursor = db.execute('SELECT COUNT(*) FROM rows')
            total = cursor.fetchone()[0]
            
            # Get only requested chunk (efficient!)
            cursor = db.execute(
                'SELECT * FROM rows LIMIT ? OFFSET ?',
                (limit, offset)
            )
            chunk = [list(row) for row in cursor.fetchall()]
            db.close()
    else:
        all_rows = cached.get('rows', [])
        total = len(all_rows)
        chunk = all_rows[offset:offset + limit]
    
    resp = json.dumps({
        'success': True,
        'rows': chunk,
        'offset': offset,
        'total': total,
        'has_more': (offset + limit) < total
    }).encode()
    
    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', len(resp))
    self.end_headers()
    self.wfile.write(resp)
```

**Key Improvement**: 
- Old: Loads 100M rows to RAM, slices → 2-5 seconds wait
- New: Direct SQLite LIMIT OFFSET query → 10-100ms wait

---

## Phase 2d: Progress Reporting Endpoint (60 minutes)

### Location
New endpoint in RequestHandler.do_GET around line 1150

### Implementation

```python
elif self.path.startswith('/job_status/'):
    job_id = self.path.split('/')[-1]
    
    with parse_jobs_lock:
        job = parse_jobs.get(job_id)
    
    if not job:
        _send_json(self, 404, {'success': False, 'error': 'job_id not found'})
        return
    
    state = job.get('state')
    result = {
        'success': True,
        'job_id': job_id,
        'state': state,  # 'pending', 'running', 'done', 'error'
        'start_time': job.get('start_time'),
        'elapsed_seconds': time.time() - job.get('start_time', time.time())
    }
    
    if state == 'done':
        result['result'] = job.get('result')
    elif state == 'error':
        result['error'] = job.get('error')
    elif state == 'running':
        # Could add progress estimates here
        result['status'] = 'Parsing file...'
    
    _send_json(self, 200, result)
```

---

## Testing Checklist for Phase 2

- [ ] Dynamic timeout: 10-minute job completes successfully
- [ ] Dynamic timeout: 30-minute job (3GB) not killed prematurely  
- [ ] CSV download: Small file (<100K rows) exports correctly
- [ ] CSV download: Large file (10M rows) exports without crashing
- [ ] Pagination: offset=0, limit=1000 returns fast (<100ms)
- [ ] Pagination: offset=50M, limit=1000 returns fast (<100ms)
- [ ] Job status: Returns 'running', 'done', or 'error' correctly
- [ ] Memory usage during pagination: <200MB for 100M row database

---

## Implementation Order

1. **Start with Dynamic Timeout** (easiest, highest value)
   - Unblocks all 5GB testing
   - Single variable change
   - No new endpoints needed

2. **Add Pagination Enhancement** (medium effort, high value)
   - Fixes UI responsiveness for large datasets
   - Reuses existing endpoint
   - SQLite query already optimized

3. **Add CSV Download** (medium effort, medium value)
   - Enables data export
   - New endpoint
   - Streaming required

4. **Add Job Status** (lower effort, medium value)
   - Helps with UX
   - Simple status tracking
   - Can add progress estimates later

---

## Time Estimate

- **Phase 2a (Timeout)**: 30 min
- **Phase 2b (CSV)**: 45 min
- **Phase 2c (Pagination)**: 30 min
- **Phase 2d (Progress)**: 60 min
- **Testing & Integration**: 60 min
- **Total**: ~4 hours

---

## Expected Outcomes After Phase 2

✅ 5GB files parseable in <2 hours  
✅ Memory usage capped at 200MB  
✅ Pagination responsive (<100ms)  
✅ CSV downloads work  
✅ Job progress visible to users  

---

**Ready to implement Phase 2? Continue or review Phase 1 first?**
