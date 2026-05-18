# REV9 5GB Support - Implementation Roadmap

**Goal**: Transform REV9 from 1GB-limited to 5GB-capable in 3 phases

---

## Quick Reference: What Needs to Change

```
CURRENT STATE:
File (5GB) → Upload ✅ → Parse to RAM (50GB) ❌ → DEAD

NEW STATE:
File (5GB) → Upload ✅ → Parse to SQLite (2GB peak) ✅ → Return paginated results ✅
```

---

## The 4 Changes (Ranked by Impact)

### **#1: Row Batching + SQLite Cache** (CRITICAL)
**Status**: Not implemented  
**Location**: `parse_log_file()` function  
**Lines Changed**: ~100 lines  
**Impact**: 96% memory reduction (50GB → 2GB)  
**Effort**: 1 day  
**Difficulty**: Medium  

**What it does**:
- Removes unbounded row list
- Batches 50K rows at a time
- Saves to SQLite instead of RAM
- Enables 5GB file support

**Code changes**:
```python
# BEFORE: rows = []
# AFTER:  Parse in 50K batches, save to SQLite
```

---

### **#2: Dynamic Timeout** (HIGH)
**Status**: Partially implemented  
**Location**: `_run_parse_job()` function  
**Lines Changed**: ~30 lines  
**Impact**: Large files don't timeout prematurely  
**Effort**: 2 hours  
**Difficulty**: Low  

**What it does**:
- Calculates timeout based on file size
- 5GB gets ~50 min instead of 10 min
- Small files still fast

**Formula**:
```python
timeout = (file_size_mb / 100) * 60
# Examples:
# 1 GB = 1000 MB → 10 min
# 5 GB = 5000 MB → 50 min
```

---

### **#3: Pagination & CSV Streaming** (MEDIUM)
**Status**: Not implemented  
**Location**: `do_GET()` method + new endpoints  
**Lines Changed**: ~300 lines  
**Impact**: Fast UI response + safe downloads  
**Effort**: 1 day  
**Difficulty**: Low  

**What it does**:
- `/get_rows/<id>?offset=N&limit=500` - paginated results
- `/download_csv/<id>` - stream full CSV
- First 500 rows in <1 second
- Can download 50M rows without OOM

---

### **#4: Progress Reporting** (NICE-TO-HAVE)
**Status**: Not implemented  
**Location**: `do_GET()` + JavaScript  
**Lines Changed**: ~150 lines  
**Impact**: Better UX + user transparency  
**Effort**: 4 hours  
**Difficulty**: Low  

**What it does**:
- `/job_status/<id>` endpoint
- Returns: rows_parsed, elapsed_seconds, progress_pct
- UI polls every 2 seconds
- Shows live progress bar

---

## Phase-by-Phase Rollout

### **Phase 1: Row Batching (Day 1)**

**Objective**: Reduce memory from 50GB to 2GB

**Deliverables**:
- ✅ SQLite batching working
- ✅ 5GB files can parse (no OOM)
- ✅ All rows saved to cache DB

**Testing**:
- 1GB file: Memory <1GB, completes in 60s
- 5GB file: Memory <2GB, completes in <15 min

**Risk**: Medium (database schema, transaction handling)

**Effort**: 1 full day

---

### **Phase 2: Timeouts & Pagination (Day 2)**

**Objective**: Make results accessible

**Deliverables**:
- ✅ Dynamic timeout (50 min for 5GB)
- ✅ /get_rows endpoint (paginate results)
- ✅ /download_csv endpoint (stream CSV)
- ✅ First 500 rows in <1 second

**Testing**:
- Pagination: offset 0, 500, 1000, etc.
- CSV download: no OOM, all rows present
- Timeout: 5GB file gets 50 min

**Risk**: Low (straightforward endpoints)

**Effort**: 1 full day

---

### **Phase 3: Progress & Polish (Day 3)**

**Objective**: Better UX

**Deliverables**:
- ✅ /job_status endpoint (progress)
- ✅ Progress bar in UI
- ✅ Documentation complete
- ✅ Edge cases handled

**Testing**:
- Progress updates during parse
- Large file behavior
- Error handling

**Risk**: Low (UI enhancements)

**Effort**: 0.5 day

---

## Implementation Sequence

```
DAY 1: PHASE 1 (Row Batching)
├─ 09:00 - SQLite schema creation
├─ 10:00 - Modify parse_log_file() for batching
├─ 11:00 - Implement batch insert function
├─ 12:00 - Test with 1GB file
├─ 13:00 - LUNCH
├─ 14:00 - Test with 5GB file
├─ 15:00 - Fix any issues
├─ 16:00 - Code review
└─ 17:00 - DONE ✅

DAY 2: PHASE 2 (Timeouts & Pagination)
├─ 09:00 - Calculate dynamic timeout function
├─ 10:00 - Implement /get_rows endpoint
├─ 11:00 - Implement /download_csv endpoint
├─ 12:00 - Test pagination
├─ 13:00 - LUNCH
├─ 14:00 - Test CSV download on 5GB
├─ 15:00 - Performance optimization
├─ 16:00 - Code review
└─ 17:00 - DONE ✅

DAY 3: PHASE 3 (Progress & Polish)
├─ 09:00 - /job_status endpoint
├─ 10:00 - JavaScript progress polling
├─ 11:00 - Progress bar in HTML
├─ 12:00 - Edge case testing
├─ 13:00 - LUNCH
├─ 14:00 - Documentation update
├─ 15:00 - Integration testing
├─ 16:00 - Final validation
└─ 17:00 - DONE ✅

DAY 4: TESTING (Not development)
├─ Comprehensive 5GB file test
├─ Pagination stress test
├─ CSV download validation
├─ Memory profiling
└─ Production readiness checklist
```

---

## Code Files to Modify

| File | Lines | Changes | Status |
|------|-------|---------|--------|
| `fdv_chart.py` | 45 | Add imports (sqlite3, contextmanager) | ✏️ |
| `fdv_chart.py` | 48 | Add CACHE_DIR, cache utils | ✏️ |
| `fdv_chart.py` | 305 | Modify parse_log_file() signature | ✏️ |
| `fdv_chart.py` | 410 | Replace unbounded list with batching | ✏️ |
| `fdv_chart.py` | 470 | Add _batch_insert_rows() function | ✏️ |
| `fdv_chart.py` | 510 | Add get_cached_rows() function | ✏️ |
| `fdv_chart.py` | 475 | Add calculate_timeout() function | ✏️ |
| `fdv_chart.py` | 478 | Update _run_parse_job() | ✏️ |
| `fdv_chart.py` | 515 | Update result format | ✏️ |
| `fdv_chart.py` | 750+ | Add /get_rows endpoint | ✏️ |
| `fdv_chart.py` | 800+ | Add /download_csv endpoint | ✏️ |
| `fdv_chart.py` | 850+ | Add /job_status endpoint | ✏️ |
| `fdv_chart.html` | ? | Add progress bar UI | ✏️ |
| `fdv_chart.html` | ? | Add progress polling JS | ✏️ |

**Total Lines of Code**: ~800 lines (new + modified)

---

## Success Metrics

### **Before Implementation**
```
5GB file test:
✗ Upload: OK (streaming)
✗ Parse: FAILS (50GB memory)
✗ Timeout: Hangs (10 min insufficient)
✗ Result: Inaccessible
✗ Peak memory: 50 GB
Result: ❌ BROKEN
```

### **After Implementation**
```
5GB file test:
✓ Upload: OK (streaming)
✓ Parse: Works (15 min)
✓ Timeout: OK (50 min available)
✓ Result: Paginated + downloadable
✓ Peak memory: 2 GB
Result: ✅ WORKING
```

---

## Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Max file size | 5 GB | 1 GB | ↑ 5x |
| Peak memory | 2 GB | 50 GB | ↓ 96% |
| Parse time (5GB) | <15 min | N/A | ✓ |
| UI response | <1 sec | 30-60s | ↑ 60x |
| CSV download | Works | OOM | ✓ |
| Timeout | 50 min | 10 min | ↑ 5x |

---

## Configuration Changes

Add to config:

```python
# Row batching
BATCH_SIZE = 50000              # Rows per batch
CACHE_DIR = Path(temp) / 'fdv_parse_cache'

# Timeouts
MIN_TIMEOUT = 600               # 10 min
MAX_TIMEOUT = 7200              # 120 min
PARSE_RATE_MB_PER_MIN = 100     # Conservative estimate

# Pagination
PAGE_SIZE = 500                 # Rows per page
MAX_PAGE_SIZE = 10000           # Hard limit

# Cleanup
CACHE_TTL_HOURS = 24            # Delete old cache
CLEANUP_INTERVAL_HOURS = 6      # Run cleanup every 6 hours
```

---

## Risk Mitigation Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| SQLite locks | Medium | High | Connection timeout + retry logic |
| Disk full | Low | Critical | Disk monitoring + auto-cleanup |
| Query slow | Low | Medium | Index optimization |
| Thread safety | Low | High | Context managers + timeouts |
| Cache corruption | Very Low | High | Integrity checks on startup |

---

## Testing Checklist

### **Unit Tests**
- [ ] `_batch_insert_rows()` - SQL correctness
- [ ] `get_cached_rows()` - Pagination logic
- [ ] `calculate_timeout()` - Timeout calculation
- [ ] SQLite DB creation - Schema validation

### **Integration Tests**
- [ ] 100MB file end-to-end
- [ ] 1GB file end-to-end
- [ ] 5GB file end-to-end
- [ ] Pagination across all sizes

### **Performance Tests**
- [ ] Memory profile during 5GB parse
- [ ] Parse time measurement
- [ ] Query performance on large DB
- [ ] CSV download throughput

### **Stress Tests**
- [ ] Concurrent 5GB uploads
- [ ] Rapid pagination requests
- [ ] Large offset queries
- [ ] Disk space exhaustion

### **Edge Cases**
- [ ] Empty file
- [ ] No matching rows
- [ ] Invalid regex
- [ ] Timeout scenarios
- [ ] Disk full scenarios
- [ ] Cache corruption

---

## Dependencies & Prerequisites

### **System Requirements**
- Python 3.6+ (sqlite3 standard library)
- 100GB+ free disk space (for cache DBs)
- 8GB+ RAM (recommended)

### **Python Dependencies**
- None new (sqlite3 is built-in)

### **Testing Resources**
- 5GB test log file (create or use sample)
- Disk space for cache (100GB)
- Time for 5GB parse test (15 minutes)

---

## Documentation Updates Required

- [ ] Update user guide with 5GB limitation
- [ ] Add /get_rows endpoint documentation
- [ ] Add /download_csv endpoint documentation
- [ ] Add /job_status endpoint documentation
- [ ] Add cache cleanup documentation
- [ ] Update troubleshooting guide
- [ ] Add performance tuning guide

---

## Rollback Plan

If deployment issues occur:

**Immediate Actions**:
1. Disable new endpoints (/get_rows, /download_csv, /job_status)
2. Revert to in-memory caching (original behavior)
3. Clear all cache files from disk

**Root Cause Analysis**:
1. Check `fdv_chart_debug.log`
2. Check SQLite error logs
3. Check disk usage
4. Verify file permissions

**Recovery**:
```bash
# Clear cache
rm -rf /tmp/fdv_parse_cache/*

# Restart server
# Server falls back to in-memory caching
```

---

## Communication Plan

### **Before Implementation**
- [ ] Announce maintenance window
- [ ] Prepare rollback procedures
- [ ] Test on staging environment

### **During Implementation**
- [ ] Daily progress updates
- [ ] Share any blockers
- [ ] Request stakeholder feedback

### **After Implementation**
- [ ] Release notes with new capabilities
- [ ] User documentation
- [ ] Performance benchmarks
- [ ] Support guidelines

---

## Conclusion

To support 5GB file parsing in REV9:

**What**: 4 core changes to handle large files efficiently  
**Why**: Enable 5x larger files without memory issues  
**How**: SQLite batching, dynamic timeouts, pagination, progress  
**When**: 2-3 days development + 1 day testing  
**Who**: One developer, with code review  
**How Much**: 800 lines of new/modified code  
**Risk**: Low-Medium (manageable with proper testing)  
**Value**: High (major capability expansion)  

**Recommendation**: ✅ **PROCEED WITH IMPLEMENTATION**

---

## Next Steps

1. ✅ **Review** - Read implementation plan
2. ✅ **Estimate** - Developer capacity check
3. ✅ **Prepare** - Create test files, staging env
4. ✅ **Implement** - Follow phase-by-phase plan
5. ✅ **Test** - Comprehensive testing
6. ✅ **Deploy** - Production release
7. ✅ **Monitor** - Watch for issues
8. ✅ **Document** - Update guides

**Estimated Total Time**: 1 week (dev + testing + validation)

**Go Live Date**: Next sprint + 1 week
