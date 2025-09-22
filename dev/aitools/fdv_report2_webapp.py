#!/usr/bin/env python
"""
Standalone FDV Report web app (v2):
- Input: one or more files, or a selected directory (process all files under it)
- Parse FDV OUTPUT lines into rows
- Report 1: table of RBER stats by fdvtest split by PR, VCC, TM, TEMP
- Report 2: clicking an fdvtest shows testname-level stats within that fdvtest
- Report 3: selecting one or more testnames shows WL vs RBER variability plots (read_eimpro_plot-like), faceted by DUT and colored by pagetype

Flow:
  input files -> fdvtest report -> click fdvtest -> testname report -> select testnames -> plots
"""
from __future__ import annotations
import os
import io
import uuid
import secrets
import tempfile
from pathlib import Path
import sqlite3
import atexit
from typing import Dict, List, Tuple
import re
import threading
import time
import json
import urllib.request
import urllib.error

# Headless matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")

from flask import Flask, render_template, request, redirect, url_for, flash, Response, send_file, jsonify

# Ensure local aitools is importable
_HERE = Path(__file__).parent
if str(_HERE) not in os.sys.path:
    os.sys.path.insert(0, str(_HERE))

# Persistent snapshots: write static HTML to D:\ by default so it can be referenced later
def _persist_base_dir() -> Path:
    """Return base directory for persistent HTML snapshots.
    Priority: FDV_PERSIST_DIR env -> D:\\fdv_tmp\\persist -> C:\\fdv_tmp\\persist -> system temp.
    """
    base = (os.environ.get('FDV_PERSIST_DIR') or '').strip() or r'D:\\fdv_tmp\\persist'
    try:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        try:
            p = Path(r'C:\\fdv_tmp\\persist')
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return Path(tempfile.gettempdir()) / 'fdv_persist'

def _persist_write(appname: str, token: str, html: str, filename: str = 'index.html') -> Path:
    """Write HTML snapshot under <base>/<appname>/<token>/<filename> and return full path."""
    root = _persist_base_dir() / appname / token
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    out = root / filename
    try:
        out.write_text(html, encoding='utf-8')
    except Exception:
        pass
    return out

# ---------------- Lightweight SQLite persistence (rows on D:) ----------------
def _sqlite_db_path(token: str) -> Path:
    base = _persist_base_dir() / 'report2' / token
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base / 'rows.db'

def _sqlite_connect(token: str) -> sqlite3.Connection:
    with _SQLITE_LOCK:
        if token in _SQLITE_CONNS:
            return _SQLITE_CONNS[token]
        dbp = _sqlite_db_path(token)
        conn = sqlite3.connect(str(dbp), check_same_thread=False)
        try:
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            conn.execute('PRAGMA temp_store=MEMORY;')
            # Performance-tuning pragmas (safe on modern SQLite):
            # - Increase mmap/cache sizes when possible to reduce I/O on large jobs.
            #   mmap_size is in bytes; cache_size negative means KB units.
            try:
                conn.execute('PRAGMA mmap_size=268435456;')  # 256 MB
            except Exception:
                pass
            try:
                conn.execute('PRAGMA cache_size=-200000;')   # ~200 MB
            except Exception:
                pass
        except Exception:
            pass
        _SQLITE_CONNS[token] = conn
        return conn

def _sqlite_close_all() -> None:
    with _SQLITE_LOCK:
        for k, c in list(_SQLITE_CONNS.items()):
            try:
                c.close()
            except Exception:
                pass
            try:
                del _SQLITE_CONNS[k]
            except Exception:
                pass

atexit.register(_sqlite_close_all)

def _sqlite_init(token: str) -> None:
    try:
        con = _sqlite_connect(token)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS rows (
                token TEXT,
                fdv   TEXT,
                pr    TEXT,
                vcc   TEXT,
                tm    TEXT,
                temp  TEXT,
                line_number INTEGER,
                fuseid TEXT,
                rber REAL,
                pass_fail TEXT,
                source_file TEXT,
                fdv_file TEXT,
                testname TEXT,
                raw_line TEXT
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_rows_token ON rows(token);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_rows_token_split ON rows(token, fdv, pr, vcc, tm, temp);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_rows_token_file ON rows(token, source_file);")
        # Help median queries: accelerate ORDER BY rber within a split by indexing rber after the split keys
        con.execute("CREATE INDEX IF NOT EXISTS idx_rows_token_split_rber ON rows(token, fdv_file, pr, vcc, tm, temp, rber);")
        # Ensure idempotent updates per source file and line; enables fast UPSERT on reprocessing
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_rows_token_src_line ON rows(token, source_file, line_number);")
        con.commit()
    except Exception:
        pass

def _sqlite_insert_rows(token: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        con = _sqlite_connect(token)
        cur = con.cursor()
        to_ins = []
        for rr in rows:
                try:
                    fdv, pr, vcc, tm, temp = _get_split_tuple(rr)
                except Exception:
                    fdv = (rr.get('fdv_file') or rr.get('fdv') or '')
                    pr = str(rr.get('pr') or '')
                    vcc = str(rr.get('vcc') or '')
                    tm = str(rr.get('tm') or '')
                    temp = str(rr.get('temp') or '')
                # line number
                ln = None
                try:
                    for lk in ('line_number','lineno','line','line_no','line_num','linenum','_line','_lineno','line_idx','lineindex'):
                        if lk in rr and rr.get(lk) not in (None, ''):
                            try:
                                ln = int(str(rr.get(lk)).strip())
                                break
                            except Exception:
                                ln = int(str(rr.get(lk)).strip(), 0)
                                break
                except Exception:
                    ln = None
                # fuseid / rber / pass_fail
                try:
                    fuseid = (_get_fuseid(rr) or '').strip()
                except Exception:
                    fuseid = (rr.get('fuseid') or rr.get('FuseID') or rr.get('FUSEID') or '')
                try:
                    rber = _get_rber(rr)
                except Exception:
                    try:
                        rber = float(rr.get('rber')) if rr.get('rber') not in (None, '') else None
                    except Exception:
                        rber = None
                pf = str(rr.get('pass_fail') or rr.get('PASS_FAIL') or rr.get('status') or '')
                src = (rr.get('source_file') or '').strip()
                fdv_file = (rr.get('fdv_file') or rr.get('fdv') or '').strip()
                tname = (rr.get('testname') or rr.get('tname') or '').strip()
                raw = (rr.get('raw_line') or rr.get('raw') or '')
                to_ins.append((token, fdv, pr, vcc, tm, temp, ln, fuseid, rber, pf, src, fdv_file, tname, raw))
        # Use UPSERT to avoid duplicates and refresh fields when present.
        # Keep existing values when the new value is NULL or empty, to preserve previously parsed/enriched data.
        cur.executemany(
            """
            INSERT INTO rows(token, fdv, pr, vcc, tm, temp, line_number, fuseid, rber, pass_fail, source_file, fdv_file, testname, raw_line)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(token, source_file, line_number) DO UPDATE SET
                fdv       = CASE WHEN COALESCE(excluded.fdv, '') <> '' THEN excluded.fdv ELSE rows.fdv END,
                pr        = CASE WHEN COALESCE(excluded.pr, '') <> '' THEN excluded.pr ELSE rows.pr END,
                vcc       = CASE WHEN COALESCE(excluded.vcc, '') <> '' THEN excluded.vcc ELSE rows.vcc END,
                tm        = CASE WHEN COALESCE(excluded.tm, '') <> '' THEN excluded.tm ELSE rows.tm END,
                temp      = CASE WHEN COALESCE(excluded.temp, '') <> '' THEN excluded.temp ELSE rows.temp END,
                fuseid    = CASE WHEN COALESCE(excluded.fuseid, '') <> '' THEN excluded.fuseid ELSE rows.fuseid END,
                rber      = COALESCE(excluded.rber, rows.rber),
                pass_fail = CASE WHEN COALESCE(excluded.pass_fail, '') <> '' THEN excluded.pass_fail ELSE rows.pass_fail END,
                fdv_file  = CASE WHEN COALESCE(excluded.fdv_file, '') <> '' THEN excluded.fdv_file ELSE rows.fdv_file END,
                testname  = CASE WHEN COALESCE(excluded.testname, '') <> '' THEN excluded.testname ELSE rows.testname END,
                raw_line  = CASE WHEN COALESCE(excluded.raw_line, '') <> '' THEN excluded.raw_line ELSE rows.raw_line END
            ;
            """,
            to_ins
        )
        # Defer commit slightly; rely on caller’s cadence to amortize I/O
        con.commit()
        return len(to_ins)
    except Exception:
        return 0


def _stats_from_sqlite(token: str, *, limit: float = 0.0, passfail_mode: bool = False) -> List[Dict[str, object]]:
    """Build aggregated stats from persisted rows.db to cover all processed files, independent of in-memory trimming.

    Computes per-(fdv,pr,vcc,tm,temp): count, pass_n, fail_n and basic rber stats (min/max/mean/stdev) when available.
    Also fills lightweight fields for vector, unit info (FUSEIDS summary), and a coarse testtime label/start/end by
    inspecting file modification times and a representative testname. This avoids heavy rescans and keeps live updates fast.
    """
    agg: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    con = None
    # Cache file mtimes to avoid repeating expensive stat calls
    _mtime_cache: Dict[str, float] = {}
    try:
        con = _sqlite_connect(token)
        cur = con.cursor()
        # Columns: token, fdv, pr, vcc, tm, temp, line_number, fuseid, rber, pass_fail, source_file, fdv_file, testname, raw_line
        # Include source_file and testname to derive unit info and testtime cheaply.
        cur.execute("SELECT fdv_file, pr, vcc, tm, temp, rber, pass_fail, fuseid, source_file, testname FROM rows WHERE token= ?", (token,))
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            for fdv_file, pr, vcc, tm, temp, rber, pf, fuseid, source_file, testname in rows:
                try:
                    k = (str(fdv_file or ''), str(pr or ''), str(vcc or ''), str(tm or ''), str(temp or ''))
                    st = agg.get(k)
                    if st is None:
                        st = {
                            'fdv_file': k[0], 'pr': k[1], 'vcc': k[2], 'tm': k[3], 'temp': k[4],
                            'count': 0, 'pass_n': 0, 'fail_n': 0,
                            '_sum': 0.0, '_sumsq': 0.0, '_n_rber': 0,
                            'min': None, 'max': None,
                            # Vector populated at finalize from count (rows are FDV OUTPUT-derived)
                            'vector': '', 'median': '', 'stdev': '', 'mean': '',
                            # Unit Info via comments; testtime filled from coarse file mtimes
                            'comments': '', 'testtime_label': '', 'test_start': '', 'test_end': '',
                            '_fuseids': set(), 'valid_fuseid_count': 0,
                            '_start_ts': None, '_end_ts': None, '_tname': '',
                        }
                        agg[k] = st
                    st['count'] = int(st['count']) + 1  # type: ignore[index]
                    # fuseid tracking for Unit Count
                    try:
                        if fuseid:
                            st['_fuseids'].add(str(fuseid))  # type: ignore[index]
                    except Exception:
                        pass
                    # Track representative testname for label
                    try:
                        if testname and not st.get('_tname'):
                            st['_tname'] = str(testname)  # type: ignore[index]
                    except Exception:
                        pass
                    # Track earliest and latest file mtimes across contributing raw files
                    try:
                        sf = str(source_file or '')
                        if sf:
                            ts = _mtime_cache.get(sf)
                            if ts is None:
                                try:
                                    ts = Path(sf).stat().st_mtime
                                except Exception:
                                    ts = 0.0
                                _mtime_cache[sf] = ts
                            if ts and ts > 0:
                                st_s = st.get('_start_ts')
                                st_e = st.get('_end_ts')
                                if st_s is None or float(ts) < float(st_s):
                                    st['_start_ts'] = float(ts)
                                if st_e is None or float(ts) > float(st_e):
                                    st['_end_ts'] = float(ts)
                    except Exception:
                        pass
                    # PASS/FAIL accounting
                    pf_u = str(pf or '').upper()
                    if passfail_mode:
                        if pf_u == 'PASS':
                            st['pass_n'] = int(st['pass_n']) + 1  # type: ignore[index]
                        elif pf_u == 'FAIL':
                            st['fail_n'] = int(st['fail_n']) + 1  # type: ignore[index]
                    # rber aggregates
                    try:
                        if rber is not None:
                            rv = float(rber)
                            st['_n_rber'] = int(st['_n_rber']) + 1  # type: ignore[index]
                            st['_sum'] = float(st['_sum']) + rv  # type: ignore[index]
                            st['_sumsq'] = float(st['_sumsq']) + (rv * rv)  # type: ignore[index]
                            # numeric min/max
                            mn = st['min']
                            mx = st['max']
                            if mn is None or rv < float(mn):
                                st['min'] = rv
                            if mx is None or rv > float(mx):
                                st['max'] = rv
                            if not passfail_mode:
                                # Threshold classification in numeric mode
                                if rv <= limit:
                                    st['pass_n'] = int(st['pass_n']) + 1  # type: ignore[index]
                                else:
                                    st['fail_n'] = int(st['fail_n']) + 1  # type: ignore[index]
                    except Exception:
                        pass
                except Exception:
                    continue
    except Exception:
        pass
    # finalize aggregates
    out: List[Dict[str, object]] = []
    for k, st in agg.items():
        try:
            n_r = int(st.get('_n_rber', 0))
            if n_r > 0:
                s = float(st.get('_sum', 0.0))
                ss = float(st.get('_sumsq', 0.0))
                mean = s / n_r
                var = (ss - (s * s) / n_r) / (n_r - 1) if n_r > 1 else 0.0
                if var < 0:
                    var = 0.0
                st['mean'] = mean
                st['stdev'] = (var ** 0.5)
                # Compute exact median using a targeted SQL query scoped to this group
                try:
                    if con is not None:
                        fdv_file, pr, vcc, tm, temp = (str(st.get('fdv_file') or ''), str(st.get('pr') or ''), str(st.get('vcc') or ''), str(st.get('tm') or ''), str(st.get('temp') or ''))
                        cur_m = con.cursor()
                        if n_r % 2 == 1:
                            off = n_r // 2
                            cur_m.execute(
                                """
                                SELECT rber FROM rows
                                WHERE token=? AND fdv_file=? AND pr=? AND vcc=? AND tm=? AND temp=? AND rber IS NOT NULL
                                ORDER BY rber
                                LIMIT 1 OFFSET ?
                                """,
                                (token, fdv_file, pr, vcc, tm, temp, off)
                            )
                            row = cur_m.fetchone()
                            if row and row[0] is not None:
                                st['median'] = float(row[0])
                            else:
                                st['median'] = ''
                        else:
                            off = max(0, (n_r // 2) - 1)
                            cur_m.execute(
                                """
                                SELECT rber FROM rows
                                WHERE token=? AND fdv_file=? AND pr=? AND vcc=? AND tm=? AND temp=? AND rber IS NOT NULL
                                ORDER BY rber
                                LIMIT 2 OFFSET ?
                                """,
                                (token, fdv_file, pr, vcc, tm, temp, off)
                            )
                            vals = cur_m.fetchall()
                            if vals and len(vals) == 2 and vals[0][0] is not None and vals[1][0] is not None:
                                st['median'] = (float(vals[0][0]) + float(vals[1][0])) / 2.0
                            elif vals and len(vals) >= 1 and vals[0][0] is not None:
                                st['median'] = float(vals[0][0])
                            else:
                                st['median'] = ''
                    else:
                        st['median'] = ''
                except Exception:
                    st['median'] = ''
            else:
                st['mean'] = ''
                st['stdev'] = ''
                st['median'] = ''
            # finalize valid_fuseid_count
            try:
                st['valid_fuseid_count'] = len(st.get('_fuseids', set()))
            except Exception:
                st['valid_fuseid_count'] = 0
            # Ensure min/max are strings or numbers compatible with template
            if st.get('min') is None:
                st['min'] = ''
            if st.get('max') is None:
                st['max'] = ''
            # Populate vector as total row count for the group
            try:
                st['vector'] = str(int(st.get('count', 0)))
            except Exception:
                st['vector'] = str(st.get('count', '') or '')
            # Build a concise Unit Info into comments: ordered distinct fuseids (truncated)
            try:
                fids = sorted([str(x) for x in (st.get('_fuseids') or set())])
                if fids:
                    max_show = 40
                    shown = fids[:max_show]
                    extra = len(fids) - len(shown)
                    txt = "FUSEIDS: " + ", ".join(shown)
                    if extra > 0:
                        txt += f" (+{extra} more)"
                    st['comments'] = txt
                else:
                    st['comments'] = ''
            except Exception:
                st['comments'] = ''
            # Build coarse testtime label/start/end from earliest/latest file mtimes and representative testname
            try:
                st_ts = st.get('_start_ts')
                en_ts = st.get('_end_ts')
                start_str = ''
                end_str = ''
                secs = ''
                if st_ts and en_ts and float(st_ts) > 0 and float(en_ts) > 0:
                    import time as _t
                    lt_s = _t.localtime(float(st_ts))
                    lt_e = _t.localtime(float(en_ts))
                    start_str = f"{lt_s.tm_year:04d}_{lt_s.tm_mon}_{lt_s.tm_mday} {lt_s.tm_hour:02d}:{lt_s.tm_min:02d}:{lt_s.tm_sec:02d}"
                    end_str = f"{lt_e.tm_year:04d}_{lt_e.tm_mon}_{lt_e.tm_mday} {lt_e.tm_hour:02d}:{lt_e.tm_min:02d}:{lt_e.tm_sec:02d}"
                    dsecs = int(max(0, float(en_ts) - float(st_ts)))
                    secs = str(dsecs)
                st['test_start'] = start_str
                st['test_end'] = end_str
                # testtime_label uses guide-derived testname and fdvtest basename
                ln_disp = ''
                try:
                    tn = str(st.get('_tname') or '')
                    if tn:
                        ln_disp = _derive_testname_guide(tn)
                except Exception:
                    ln_disp = ''
                try:
                    import os as _os
                    fdvtest = _os.path.basename(str(st.get('fdv_file') or ''))
                    if fdvtest.lower().endswith('.fdv'):
                        fdvtest = fdvtest[:-4]
                except Exception:
                    fdvtest = ''
                lbl = ''
                if ln_disp or secs:
                    lbl = f"{ln_disp}::{fdvtest} = {secs}".strip()
                st['testtime_label'] = lbl
            except Exception:
                pass
        except Exception:
            pass
        # drop internal fields
        st.pop('_sum', None)
        st.pop('_sumsq', None)
        st.pop('_n_rber', None)
        st.pop('_fuseids', None)
        st.pop('_start_ts', None)
        st.pop('_end_ts', None)
        st.pop('_tname', None)
        # Format numeric fields like original renderer
        try:
            if isinstance(st.get('mean'), float):
                st['mean'] = f"{st['mean']:.6g}"
            if isinstance(st.get('stdev'), float):
                st['stdev'] = f"{st['stdev']:.6g}"
            if isinstance(st.get('median'), float):
                st['median'] = f"{st['median']:.6g}"
            if isinstance(st.get('min'), float):
                st['min'] = f"{st['min']:.6g}"
            if isinstance(st.get('max'), float):
                st['max'] = f"{st['max']:.6g}"
        except Exception:
            pass
        out.append(st)
    # order by previously seen file order if available
    try:
        order = (CACHE.get(token, {}) or {}).get('fdv_order', []) or []
        idx = {f: i for i, f in enumerate(order)}
        out.sort(key=lambda r: (idx.get(str(r.get('fdv_file') or ''), 10**9), str(r.get('fdv_file') or ''), str(r.get('pr') or '')))
    except Exception:
        out.sort(key=lambda r: (str(r.get('fdv_file') or ''), str(r.get('pr') or '')))
    return out

# ---------------- Encoding helpers / diagnostics ----------------
def _detect_file_encoding(fp: Path) -> str:
    """Heuristic detect encoding. Returns 'utf-16' if a high fraction of NUL bytes present early, else 'utf-8'."""
    try:
        with open(fp, 'rb') as fb:
            head = fb.read(4096)
        if not head:
            return 'utf-8'
        # UTF-16 LE often shows 0x00 in every other byte for ASCII subset
        nul_frac = head.count(b'\x00') / max(1, len(head))
        if nul_frac > 0.10:  # threshold
            return 'utf-16'
        # BOM checks
        if head.startswith(b'\xff\xfe') or head.startswith(b'\xfe\xff'):
            return 'utf-16'
    except Exception:
        return 'utf-8'
    return 'utf-8'

def _convert_utf16_to_temp_utf8(fp: Path) -> Path | None:
    """Convert a UTF-16 file to a temporary UTF-8 copy for downstream parser that only supports UTF-8.
    Returns new temp file path or None if conversion failed.
    """
    try:
        enc = 'utf-16'
        with open(fp, 'r', encoding=enc, errors='strict') as f:
            data = f.read()
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"fdv_utf8_{fp.name}"
        with open(tmp, 'w', encoding='utf-8', errors='replace') as out:
            out.write(data)
        return tmp
    except Exception as e:  # pragma: no cover
        try:
            print(f"[encoding] Failed to convert {fp} from UTF-16 -> UTF-8: {e}")
        except Exception:
            pass
        return None

# Try to use process_fdv for parsing; fall back to a minimal reader if missing
try:
    import process_fdv as pfdv  # type: ignore
    from process_fdv.core import derive_testname  # type: ignore
    # For fine-grained progress, import process_file directly
    from process_fdv.core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT  # type: ignore
except Exception:  # pragma: no cover
    pfdv = None
    def derive_testname(tname: str) -> str:
        return (tname or '').strip()
    process_file = None  # type: ignore
    PREFIX_DEFAULT = "FDV OUTPUT"  # type: ignore
    IGNORE_VALUE_DEFAULT = None  # type: ignore


# --------------- Guide-compliant extractors (tname-driven) ---------------

_RE_WL = re.compile(r"(?<![A-Z0-9])WL\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", re.IGNORECASE)
_RE_PAGE = re.compile(r"(?<![A-Z0-9])(?:PG|PAGE|PHYPAGE)\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", re.IGNORECASE)
_RE_BLK = re.compile(r"(?<![A-Z0-9])(BLK|BLOCK)\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", re.IGNORECASE)
_RE_PAGETYPE = re.compile(r"(?<![A-Z0-9])(PGTYPE|PAGETYPE)\s*[_:\-\s]?\s*([A-Z0-9]+)(?![A-Z0-9])", re.IGNORECASE)
_RE_PLANE_ADDR = re.compile(r"(?<![A-Z0-9])P(\d+)(?![A-Z0-9])", re.IGNORECASE)
_RE_DECK = re.compile(r"(?<![A-Z0-9])(UD|LD|MD)(?![A-Z0-9])", re.IGNORECASE)
_RE_STEP = re.compile(r"(?<![A-Z0-9])STEP\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", re.IGNORECASE)
_RE_PAGEMAP = re.compile(r"(?<![A-Z0-9])(QLC|TLC|MLC|SSLC|DSLC|SLC)(?![A-Z0-9])", re.IGNORECASE)


def _derive_testname_guide(tname: str) -> str:
    """Derive 'testname' from tname per guide: prefix part before BLK/PAGE/PG/PGTYPE/DECK/WL/SB/pagemap.
    Keeps underscores and case as-is from the original tname.
    """
    raw = (tname or '').strip()
    if not raw:
        return ''
    up = raw.upper()
    # Find earliest index of any parameter token; testname is substring before that
    indices = []
    for rex in (
        _RE_BLK,
        _RE_PAGE,
        _RE_PAGETYPE,
        _RE_DECK,
        _RE_WL,
        _RE_PAGEMAP,
        re.compile(r"(?<![A-Z0-9])SB[_:\-\s]?[0-9]+(?![A-Z0-9])", re.IGNORECASE),
    ):
        m = rex.search(up)
        if m:
            indices.append(m.start())
    cut = min(indices) if indices else -1
    if cut <= 0:
        # As a fallback, strip trailing numeric tokens if any, else return raw
        parts = raw.split('_')
        # remove trailing fields that look like NAME_123 or NAME:123 patterns
        trimmed = []
        for p in parts:
            trimmed.append(p)
        return '_'.join(trimmed)
    return raw[:cut].rstrip('_: -')


def _extract_pagetype_from_tname_only(r: Dict[str, str]) -> str:
    tn = (r.get('tname', '') or '')
    if not tn:
        return ''
    m = _RE_PAGETYPE.search(tn)
    if m:
        return (m.group(2) or '').upper()
    # Also try simple presence of known pagemap tokens as pagetype when explicit key missing
    m2 = _RE_PAGEMAP.search(tn)
    if m2:
        return (m2.group(1) or '').upper()
    return ''


def _extract_deck_from_tname_only(r: Dict[str, str]) -> str:
    tn = (r.get('tname', '') or '')
    if not tn:
        return ''
    m = _RE_DECK.search(tn)
    return (m.group(1).upper() if m else '')


def _extract_step_from_tname_only(r: Dict[str, str]) -> int | None:
    tn = (r.get('tname', '') or '')
    if not tn:
        return None
    m = _RE_STEP.search(tn)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _extract_pagemap_from_any(r: Dict[str, str]) -> str:
    # Prefer explicit field if present
    v = _first_nonempty_str(r, ['pagemap', 'product_type', 'product', 'pagemap_type'], '')
    if v:
        return v.upper()
    # Try tname, fdv_file, and path
    for s in ((r.get('tname') or ''), (r.get('fdv_file') or ''), (r.get('file') or '')):
        m = _RE_PAGEMAP.search(s or '')
        if m:
            return (m.group(1) or '').upper()
    return ''


# --------------- FUSEID validation & site extraction ---------------
# FUSEID format: K<6 digits>_<int>_<int>_<int>; the first integer must be > 0; the last two may be negative
_RE_FUSEID = re.compile(r"^K\d{6}_(\d+)_(-?\d+)_(-?\d+)$", re.IGNORECASE)

def _is_valid_fuseid(fid: str) -> bool:
    """Return True if fid matches K<6 digits>_<int>_<int>_<int> with the first underscore-int > 0.
    Example: K123456_1_0_0 is valid; K123456_0_1_2 is invalid.
    """
    if not fid:
        return False
    m = _RE_FUSEID.match(fid.strip())
    if not m:
        return False
    try:
        first_int = int(m.group(1))
        return first_int > 0
    except Exception:
        return False


def _extract_site_from_filename(fp: str) -> str:
    """Return site number from filename like 'Output_site111_...' or 'Output_111_...'; else ''."""
    if not fp:
        return ''
    base = Path(fp).name
    up = base.upper()
    m = re.search(r"OUTPUT[_-]SITE(\d+)", up, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"OUTPUT[_-](\d+)_", up, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    return ''


def _compute_phypage(page_val: int | None) -> int | None:
    if page_val is None:
        return None
    try:
        return int(page_val) & 0x1FFF
    except Exception:
        return None


def _scan_fuseid_and_pr_from_file(file_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Scan a raw log file and return (fuseid_by_dut, pr_by_dut) dicts.
    - FUSEID: capture order of appearance; first is DUT1, second DUT2, etc. Keep last seen per DUT.
    - PR: for 'PR' rows, capture the last hex token (e.g., '13') and convert to decimal.
    Only lines beginning with 'FDV OUTPUT', 'FDV POLL', 'ECHO: FUSEID:', 'Test Start Date', or 'Test End Date' (after whitespace) are considered.
    """
    fuseid_by_dut: dict[str, str] = {}
    pr_by_dut: dict[str, str] = {}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            dut_counter = 0
            for line in f:  # Iterate through each line in the file
                up_head = line.lstrip().upper()
                if not (
                    up_head.startswith('FDV OUTPUT')
                    or up_head.startswith('FDV POLL')
                    or up_head.startswith('ECHO: FUSEID:')
                    or up_head.startswith('TEST START DATE')
                    or up_head.startswith('TEST END DATE')
                ):
                    continue
                up = line.upper()
                # FUSEID lines
                mfid = re.search(r"FUSEID\s*:\s*([A-Z0-9_\-\+]+)", up)
                if mfid:
                    dut_counter += 1
                    fid_raw = mfid.group(1)
                    fuseid_by_dut[str(dut_counter)] = fid_raw
                # PR lines: detect tname '::PR' and a DUTn token
                if '::PR' in up and 'DUT' in up:
                    mdut = re.search(r"DUT\s*(\d+)", up)
                    dkey = (mdut.group(1) if mdut else '')
                    hex_matches = re.findall(r"\|[^|]*?:([0-9A-F]{1,2})(?=\||,|\s|$)", up)
                    if dkey and hex_matches:
                        try:
                            pr_dec = int(hex_matches[-1], 16)
                            pr_by_dut[dkey] = str(pr_dec)
                        except Exception:
                            pass
    except Exception:
        pass
    return fuseid_by_dut, pr_by_dut


def _apply_guide_annotations(rows: List[Dict[str, str]]) -> None:
    """Enrich rows in-place to align with guide_to_fdvlog.txt.
    - Derive testname, pagetype, pagemap, deck, step, plane_group, wl/page/phypage, blk, plane_addr.
    - Fill missing FUSEID and PR per file by scanning raw file.
    """
    # Build per-file DUT maps
    by_file: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    files = sorted({(r.get('fdv_file') or '').strip() for r in rows if (r.get('fdv_file') or '').strip()})
    for fp in files:
        if not fp:
            continue
        fuse_map, pr_map = _scan_fuseid_and_pr_from_file(fp)
        by_file[fp] = (fuse_map, pr_map)
    for r in rows:
        # testname
        if not (r.get('testname') or '').strip():
            tn_raw = (r.get('tname') or '').strip()
            if tn_raw:
                r['testname'] = _derive_testname_guide(tn_raw)
        # pagetype
        if not (r.get('pagetype') or '').strip():
            pt = _extract_pagetype_from_tname_only(r)
            if pt:
                r['pagetype'] = pt
        # pagemap/product type
        if not (r.get('pagemap') or r.get('product_type')):
            pm = _extract_pagemap_from_any(r)
            if pm:
                r['pagemap'] = pm
                r['product_type'] = pm
        # deck
        if not (r.get('deck') or '').strip():
            dk = _extract_deck_from_tname_only(r)
            if dk:
                r['deck'] = dk
        # step
        if not (r.get('step') or '').strip():
            stp = _extract_step_from_tname_only(r)
            if stp is not None:
                r['step'] = str(stp)
        # plane group (SP/MP)
        pg = _plane_from_tname_or_default(r)
        if pg and not (r.get('plane_group') or '').strip():
            r['plane_group'] = pg
        # canonical WL/PAGE already handled later; ensure PHYPAGE as 13 LSB of PAGE when missing
        ph = None
        # If an explicit phypage-like key exists, keep it
        if (r.get('phypage') or r.get('phy_page') or r.get('phy_page_canonical')):
            pass
        else:
            try:
                pg_val = _extract_page_value(r)
                ph = _compute_phypage(pg_val)
                if ph is not None:
                    r['phy_page'] = str(ph)
                    r['phypage_canonical'] = str(ph)
            except Exception:
                pass
        # Fill FUSEID/PR from per-file map when missing
        fp = (r.get('fdv_file') or '').strip()
        dut = (r.get('dut_id') or '').strip()
        if fp and dut and fp in by_file:
            fuse_map, pr_map = by_file.get(fp, ({}, {}))
            if not (r.get('fuseid') or '').strip():
                fid = fuse_map.get(dut)
                if fid:
                    r['fuseid'] = fid
            if not (r.get('pr') or '').strip():
                v = pr_map.get(dut)
                if v:
                    r['pr'] = v


def _read_rows_from_paths(paths: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if pfdv is not None and getattr(pfdv, 'read_dir_rows', None):
        # process one file at a time to honor order; reuse read_dir_rows for simplicity if dir
        for p in paths:
            if p.is_dir():
                for q in p.iterdir():
                    if q.is_file():
                        rows.extend(pfdv.read_dir_rows(q.parent if q.is_dir() else Path(q).parent))  # type: ignore
                        break
            else:
                # Use process_file directly for single file
                try:
                    from process_fdv.core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT  # type: ignore
                    r, _kept, _markers = process_file(p, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT)
                    rows.extend(r)
                except Exception:
                    continue
        return rows
    # Fallback: naive filter of FDV OUTPUT lines (very limited)
    for p in paths:
        if p.is_dir():
            for q in p.iterdir():
                if q.is_file():
                    rows.extend(_read_rows_from_paths([q]))
        else:
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    for i, line in enumerate(f, start=1):
                        _ls = line.lstrip()
                        if not _ls.startswith('FDV OUTPUT'):
                            continue
                        # Ignore lines with MONITOR or SHMOO in FDV OUTPUT tail (case-insensitive)
                        _u = _ls.upper()
                        if 'MONITOR' in _u or 'SHMOO' in _u:
                            # Requirement: ignore any 'FDV OUTPUT' line containing MONITOR or SHMOO
                            continue
                        rows.append({'raw_line': line.rstrip('\n'), 'line_number': str(i)})
            except Exception:
                continue
    return rows


def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _plane_from_tname_or_default(r: Dict[str, str]) -> str:
    """Return 'SP' or 'MP' if present as a single token in tname; else 'NP'."""
    tname = (r.get('tname','') or '').strip().upper()
    if tname:
        tokens = [tok for tok in re.split(r"[^A-Z0-9]+", tname) if tok]
        for tok in tokens:
            if tok in ('SP','MP'):
                return tok
    # No explicit plane token found; treat as wildcard (empty)
    return ''


# --------- Field alias helpers ---------
def _first_nonempty_str(r: Dict[str, str], keys: List[str], default: str = '') -> str:
    for k in keys:
        if k in r and r.get(k) is not None:
            v = str(r.get(k)).strip()
            if v:
                return v
    return default


def _get_rber(r: Dict[str, str]) -> float | None:
    for k in ('rber','RBER','raw_ber','RAW_BER','ber','BER','bit_error_rate','error_rate'):
        if k in r:
            v = _to_float(r.get(k))
            if v is not None:
                return v
    return None


def _get_fuseid(r: Dict[str, str]) -> str:
    return _first_nonempty_str(r, ['fuseid','fuse_id','fid','chipid','chip_id','device_id'], '')


def _get_split_tuple(r: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    fdv = _first_nonempty_str(r, ['fdv_file','fdv','file','filepath','filename'], '')
    pr = _first_nonempty_str(r, ['pr','PR'], '') or 'XX'
    vcc = _first_nonempty_str(r, ['vcc','VCC','vcc_mv'], '')
    tm = _first_nonempty_str(r, ['tm','TM'], '')
    temp = _first_nonempty_str(r, ['temp','TEMP','temperature'], '')
    return (fdv, pr, vcc, tm, temp)


def _extract_wl_value(r: Dict[str, str]) -> int | None:
    """Return WL as an integer if available.

    Rule per guide_to_fdvlog.txt:
    - If tname contains WL_10 (or WL10 / WL-10 / WL 10), take 10 as WL.
    - Otherwise, check explicit WL-like fields.
    """
    # Prefer tname token per spec
    tname = (r.get('tname', '') or '')
    if tname:
        try:
            tn = tname.upper()
            # Use custom boundaries: underscore is a word char in \\b, so prefer non-alnum guards
            m = re.search(r"(?<![A-Z0-9])WL\\s*[_\\-\\s]?\\s*([0-9]+)(?![A-Z0-9])", tn)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    # Fallback to explicit WL fields if parser already provided one
    wl_keys = [
        # include canonical annotations first if present
        'wl_canonical',
        'wl','WL','Wl','wordline','WORDLINE','word_line','WORD_LINE',
        'wlidx','wl_index','wladdr','wl_addr','wladdress','wl_address',
        'wordline_idx','wordline_index','wl_index_dec','wordline_dec','wl_dec'
    ]
    for k in wl_keys:
        if k in r and r.get(k) not in (None, ''):
            try:
                return int(float(str(r.get(k)).strip()))
            except Exception:
                try:
                    return int(str(r.get(k)).strip(), 0)
                except Exception:
                    continue
    return None


def _extract_page_value(r: Dict[str, str]) -> int | None:
    """Return PAGE as an integer if available.

    Rule per guide_to_fdvlog.txt:
    - If tname contains PG_44 / PAGE_44 / PHYPAGE_44, take 44 as PAGE.
    - Otherwise, check explicit PAGE-like fields.
    """
    # Prefer tname token per spec
    tname = (r.get('tname', '') or '')
    if tname:
        try:
            tn = tname.upper()
            # Custom boundaries to allow underscores after the digits
            m = re.search(r"(?<![A-Z0-9])(?:PG|PAGE|PHYPAGE)\s*[_\-\s]?\s*([0-9]+)(?![A-Z0-9])", tn)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    # Fallback to explicit page fields
    pg_keys = [
        'page_canonical',
        'page','PAGE','Page','page_idx','pageindex','pg','pgidx','page_addr','pageaddr',
        'page_address','pageno','page_no','pagenumber','page_num','pgno','pg_no',
        'pgindex','pg_index','pgaddr','pg_addr','page_address_dec'
    ]
    for k in pg_keys:
        if k in r and r.get(k) not in (None, ''):
            try:
                return int(float(str(r.get(k)).strip()))
            except Exception:
                try:
                    return int(str(r.get(k)).strip(), 0)
                except Exception:
                    continue
    return None


def _extract_wl_or_page(r: Dict[str, str], *, allow_page_fallback: bool = True) -> float | None:
    """Extract WL (preferred) or, if allowed, PAGE from a parsed row using the FDV guide rules.

    - WL from tname token WL_<n> first; then explicit WL fields.
    - If not found and allow_page_fallback=True, PAGE from tname token (PG|PAGE|PHYPAGE)_<n> first; then explicit PAGE fields.
    Returns a float for plotting convenience.
    """
    wl = _extract_wl_value(r)
    if wl is not None:
        return float(wl)
    if not allow_page_fallback:
        return None
    pg = _extract_page_value(r)
    if pg is not None:
        return float(pg)
    return None


def _extract_plane_addr(r: Dict[str, str]) -> str:
    """Extract plane address like 'P0'..'P7' if available.

    Priority:
    - Explicit fields: plane_addr, planeaddress, plane_address
    - Token in tname: P<digits>
    - Derive from block number (blk & 0x7)
    Returns '' if unavailable.
    """
    # Explicit fields (include several aliases; ignore SP/MP tokens)
    for k in ('plane_addr', 'planeaddress', 'plane_address', 'planeaddr', 'plane_id', 'planeid', 'plane_no', 'plane_num', 'planeindex', 'plane_idx'):
        v = (r.get(k, '') or '').strip().upper()
        if v and v not in ('SP', 'MP'):
            m = re.match(r"^P?(\d{1,2})$", v)
            if m:
                try:
                    n = int(m.group(1))
                    if 0 <= n <= 99:
                        return f"P{n}"
                except Exception:
                    pass
                # Note: no side-effects here; only parse and return plane address.
    # From tname token
    tn = (r.get('tname', '') or '')
    if tn:
        up = tn.upper()
        m = re.search(r"(?<![A-Z0-9])P(\d+)(?![A-Z0-9])", up)
        if m:
            try:
                n = int(m.group(1))
                if 0 <= n <= 99:
                    return f"P{n}"
            except Exception:
                pass
    # From raw_line token P<digits> if present; also handle PLANE[...] entries
    rl = (r.get('raw_line', '') or '')
    if rl:
        up = rl.upper()
        m = re.search(r"(?<![A-Z0-9])P(\d+)(?![A-Z0-9])", up)
        if m:
            try:
                n = int(m.group(1))
                if 0 <= n <= 99:
                    return f"P{n}"
            except Exception:
                pass
        m3 = re.search(r"(?<![A-Z0-9])PLANE(?:[_\s]?(ADDR|ADDRESS|ID|NO|NUM))?\s*[:=]?\s*(\d{1,2})(?![A-Z0-9])", up)
        if m3:
            try:
                n = int(m3.group(2))
                if 0 <= n <= 99:
                    return f"P{n}"
            except Exception:
                pass
    # Derive from block if numeric (consider many common keys)
    blk_keys = (
        'blk', 'blk_canonical', 'block', 'bl',
        'block_addr', 'blk_addr', 'blockaddress', 'block_address',
        'blkindex', 'blk_index', 'block_idx', 'block_index',
        'blockno', 'block_no', 'blocknum', 'block_num', 'blocknumber', 'block_number',
        'blkno', 'blk_no', 'blknum', 'blk_num',
        'blockid', 'block_id',
        'blkaddr', 'blk_addr_dec', 'block_addr_dec', 'block_address_dec', 'blk_dec', 'block_dec',
        'pbn', 'phy_block', 'phyblk', 'phy_blk', 'phyblock'
    )
    for bk in blk_keys:
        bv = r.get(bk)
        if bv is None:
            continue
        blk = str(bv).strip()
        if not blk:
            continue
        try:
            n = int(float(blk))
        except Exception:
            try:
                n = int(blk, 0)
            except Exception:
                continue
        if n >= 0:
            return f"P{(n & 0x7)}"
    return ''


def _extract_blk_value(r: Dict[str, str]) -> int | None:
    """Extract block address as an integer if present in the row.

    Tries common keys like 'blk', 'block', 'bl', 'block_addr', and parses decimal or base-prefixed strings.
    Returns None if unavailable.
    """
    blk_keys = (
        'blk', 'blk_canonical', 'block', 'bl',
        'block_addr', 'blk_addr', 'blockaddress', 'block_address',
        'blkindex', 'blk_index', 'block_idx', 'block_index',
        # common alternates
        'blockno', 'block_no', 'blocknum', 'block_num', 'blocknumber', 'block_number',
        'blkno', 'blk_no', 'blknum', 'blk_num',
        'blockid', 'block_id',
        'blkaddr', 'blk_addr_dec', 'block_addr_dec', 'block_address_dec', 'blk_dec', 'block_dec',
        # NAND-specific aliases
        'pbn', 'phy_block', 'phyblk', 'phy_blk', 'phyblock'
    )
    # 1) Prefer explicit fields if present
    for k in blk_keys:
        if k in r and r.get(k) not in (None, ''):
            s = str(r.get(k)).strip()
            try:
                return int(float(s))
            except Exception:
                try:
                    return int(s, 0)
                except Exception:
                    continue
    # 2) Derive from tname token e.g., BLK_778 or BLOCK_778
    tn = (r.get('tname', '') or '')
    if tn:
        up = tn.upper()
        m = re.search(r"(?<![A-Z0-9])(BLK|BLOCK)\s*[_\-\s]?\s*([0-9]+)(?![A-Z0-9])", up)
        if m:
            try:
                return int(m.group(2))
            except Exception:
                pass
    # 3) As a last resort, scan raw_line if available
    rl = (r.get('raw_line', '') or '')
    if rl:
        up = rl.upper()
        m = re.search(r"(?<![A-Z0-9])(BLK|BLOCK)\s*[_\-\s]?\s*([0-9]+)(?![A-Z0-9])", up)
        if m:
            try:
                return int(m.group(2))
            except Exception:
                pass
        # Also consider 'BLOCK_NO' or 'BLOCKNUM=123' and 'PBN=123'
        m2 = re.search(r"BLOCK[_\s]?NO\s*[:=]?\s*([0-9]+)", up)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                pass
        m3 = re.search(r"(?<![A-Z0-9])PBN\s*[:=]?\s*([0-9]+)", up)
        if m3:
            try:
                return int(m3.group(1))
            except Exception:
                pass
    return None


def stats_by_fdv_with_splits(rows: List[Dict[str, str]], *, limit: float = 0.0, passfail_mode: bool = False) -> List[Dict[str, str]]:
    """Group by (fdv_file, PR, VCC, TM, TEMP) and compute RBER stats.
    Applies FUSEID validation: rows with invalid/missing FUSEIDs are ignored from stats,
    and a per-(fdv,pr,vcc,tm,temp) comments field lists ignored DUT@site.
    """
    # Helper: robust PASS/FAIL extraction from an FDV OUTPUT line.
    def _extract_pf(raw_line: str) -> str | None:
        try:
            if not raw_line:
                return None
            up = raw_line.upper()
            # Fast path: look for ',PASS,' / ',FAIL,' exact delimited occurrences
            # (use these only if not immediately followed by alphabetic continuation forming a larger token)
            # We'll still prefer structured parsing below.
            # Structured parse: after the first "]:" split, comma tokens => DUTn, PASS/FAIL/MONITOR,...
            if ']: ' in raw_line:
                after = raw_line.split(']:', 1)[1]
            else:
                # Sometimes there's no space
                after = raw_line.split(']:', 1)[-1]
            after = after.strip()
            # Remove leading prefixes like 'DUT1,' etc only by splitting
            parts = [p.strip() for p in after.split(',')]
            if len(parts) >= 2:
                # parts[0] expected: DUT1 / DUT2 ...; parts[1] expected: PASS / FAIL / MONITOR
                p1 = parts[1].upper()
                if p1 in ('PASS','FAIL'):
                    return p1
                if p1 == 'MONITOR':
                    return 'MONITOR'
            # Fallback: explicit delimiter search ensuring trailing comma (avoid FAILCOUNT_ONLY)
            if ',FAIL,' in up:
                return 'FAIL'
            if ',PASS,' in up:
                return 'PASS'
            # Secondary fallback: standalone PASS/FAIL tokens bounded by non-alnum or start/end.
            # Avoid matching substrings like FAILCOUNT, PASSRATE, etc.
            # We deliberately restrict to lines that contain "FDV OUTPUT" to reduce false positives.
            try:
                if 'FDV OUTPUT' in up:
                    import re as _re_pf
                    # Word boundary with custom class ensuring previous/next not A-Z0-9 underscore
                    if _re_pf.search(r'(?<![A-Z0-9_])FAIL(?![A-Z0-9_])', up):
                        return 'FAIL'
                    if _re_pf.search(r'(?<![A-Z0-9_])PASS(?![A-Z0-9_])', up):
                        return 'PASS'
            except Exception:
                pass
            return None
        except Exception:
            return None
    # Global MONITOR/SHMOO filter:
    # Count MONITOR/SHMOO lines toward vector but drop from RBER stats.
    # If a group has only MONITOR/SHMOO lines, we'll inject a synthetic row later so UI doesn't appear empty.
    # Pre-compute fdv_output_counts on the full raw set (before filtering) for VECTOR semantics
    from collections import defaultdict as _dd_pre
    fdv_output_counts: Dict[Tuple[str,str,str,str,str], int] = _dd_pre(int)
    for __r in rows:
        try:
            fdv_f0 = _first_nonempty_str(__r, ['fdv_file','fdv','file','filepath','filename'], '')
            if not fdv_f0:
                continue
            pr_f0 = _first_nonempty_str(__r, ['pr','PR'], '') or 'XX'
            vcc_f0 = _first_nonempty_str(__r, ['vcc','VCC','vcc_mv'], '')
            tm_f0 = _first_nonempty_str(__r, ['tm','TM'], '')
            temp_f0 = _first_nonempty_str(__r, ['temp','TEMP','temperature'], '')
            key0 = (fdv_f0, pr_f0, vcc_f0, tm_f0, temp_f0)
            rl0 = (__r.get('raw_line') or '').lstrip().upper()
            if rl0.startswith('FDV OUTPUT'):
                fdv_output_counts[key0] += 1
        except Exception:
            pass
    try:
        _orig_len = len(rows)
        _filtered = []
        _skip = 0
        _ms_lines_to_add: list[Tuple[Tuple[str,str,str,str,str], int]] = []
        # Pre-map for start/end capture even if group only appears in filtered rows
        _early_aux: Dict[Tuple[str,str,str,str,str], Tuple[str,str]] = {}
        groups_seen_any: set[Tuple[str,str,str,str,str]] = set()
        groups_seen_real: set[Tuple[str,str,str,str,str]] = set()
        for _r in rows:
            fdv_f = _first_nonempty_str(_r, ['fdv_file','fdv','file','filepath','filename'], '')
            pr_f = _first_nonempty_str(_r, ['pr','PR'], '') or 'XX'
            vcc_f = _first_nonempty_str(_r, ['vcc','VCC','vcc_mv'], '')
            tm_f = _first_nonempty_str(_r, ['tm','TM'], '')
            temp_f = _first_nonempty_str(_r, ['temp','TEMP','temperature'], '')
            key_for_ms: Tuple[str,str,str,str,str] | None = (fdv_f, pr_f, vcc_f, tm_f, temp_f) if fdv_f else None
            if key_for_ms:
                groups_seen_any.add(key_for_ms)
            # NOTE: Raw FDV OUTPUT line counting now happens in the pre-pass above; do not increment here.
            # Early capture of test_start/test_end
            try:
                ts_early = (_r.get('test_start') or '').strip()
                te_early = (_r.get('test_end') or '').strip()
                if key_for_ms and (ts_early or te_early):
                    prev = _early_aux.get(key_for_ms)
                    if prev:
                        # earliest start, latest end
                        ts_prev, te_prev = prev
                        if ts_early and (not ts_prev or ts_early < ts_prev):
                            ts_prev = ts_early
                        if te_early and (not te_prev or te_early > te_prev):
                            te_prev = te_early
                        _early_aux[key_for_ms] = (ts_prev, te_prev)
                    else:
                        _early_aux[key_for_ms] = (ts_early, te_early)
            except Exception:
                pass
            # classify MONITOR/SHMOO
            is_ms = False
            for _v in _r.values():
                if isinstance(_v, str):
                    _upv = _v.upper()
                    if 'MONITOR' in _upv or 'SHMOO' in _upv:
                        is_ms = True
                        break
            if is_ms:
                _skip += 1
                if key_for_ms:
                    _ms_lines_to_add.append((key_for_ms,1))
                continue
            _filtered.append(_r)
            if key_for_ms:
                groups_seen_real.add(key_for_ms)
        if _skip:
            try:
                print(f"[stats_by_fdv_with_splits] filtered {_skip} MONITOR/SHMOO rows (of {_orig_len}) while counting them")
            except Exception:
                pass
        _placeholder_only = groups_seen_any - groups_seen_real
        if _placeholder_only:
            try:
                print(f"[stats_by_fdv_with_splits] monitor-only groups detected: {len(_placeholder_only)}")
            except Exception:
                pass
        rows = _filtered
    except Exception:
        pass
    from collections import defaultdict
    groups: Dict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    monitor_shmoo_counts: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    # Pre-credit MONITOR/SHMOO lines discovered in global filter above and merge early start/end if not set later
    try:
        for _k,_inc in locals().get('_ms_lines_to_add', []):  # type: ignore[index]
            monitor_shmoo_counts[_k] += _inc
        # Seed aux_by_key with early start/end for groups not yet present
        for _k, (_ts,_te) in locals().get('_early_aux', {}).items():  # type: ignore[index]
            if _k not in aux_by_key:
                aux_by_key[_k] = ('','','', _ts, _te)
            else:
                prev = aux_by_key[_k]
                ts_prev, te_prev = prev[3], prev[4]
                new_ts = _ts if (_ts and (not ts_prev or _ts < ts_prev)) else ts_prev
                new_te = _te if (_te and (not te_prev or _te > te_prev)) else te_prev
                if new_ts != ts_prev or new_te != te_prev:
                    aux_by_key[_k] = (prev[0], prev[1], prev[2], new_ts, new_te)
    except Exception:
        pass
    planes_seen: Dict[Tuple[str, str, str, str, str], set] = defaultdict(set)
    testtime_by_key: Dict[Tuple[str, str, str, str, str], str] = {}
    # aux_by_key holds (fdvlistname, fdvtestrun, testtime_seconds, test_start, test_end)
    aux_by_key: Dict[Tuple[str, str, str, str, str], Tuple[str, str, str, str, str]] = {}
    # Count pagemap occurrences per key
    pagemap_counts: Dict[Tuple[str, str, str, str, str], Dict[str, int]] = defaultdict(dict)
    # Track ignored (dut, site) per key due to invalid fuseid
    ignored_by_key: Dict[Tuple[str, str, str, str, str], List[Tuple[str, str]]] = defaultdict(list)
    # Track invalid/ignored units with raw (possibly invalid) fuseid for later display in unit_info
    invalid_units_by_key: Dict[Tuple[str, str, str, str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    # Track all DUT IDs observed (regardless of fuseid validity) to enable fallback Unit Count
    dut_ids_all_by_key: Dict[Tuple[str, str, str, str, str], set] = defaultdict(set)
    # Track valid fuseids (for primary Unit Count)
    valid_fuseids_by_key: Dict[Tuple[str, str, str, str, str], set] = defaultdict(set)
    # Count pagemap occurrences per key to pick a representative value
    pagemap_counts: Dict[Tuple[str, str, str, str, str], Dict[str, int]] = defaultdict(dict)
    # Track DUT->FUSEID mapping and distinct valid fuseids per key
    dut_fid_by_key: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = defaultdict(dict)
    valid_fuseids_set_by_key: Dict[Tuple[str, str, str, str, str], set] = defaultdict(set)
    # Track all DUT ids observed (even if fuseid invalid/missing) for fallback unit counting
    dut_ids_all_by_key: Dict[Tuple[str, str, str, str, str], set] = defaultdict(set)
    # Preserve ordered distinct valid fuseids (first-seen order) per key for display parity with count
    ordered_valid_fuseids_by_key: Dict[Tuple[str, str, str, str, str], list] = defaultdict(list)
    # When passfail_mode is requested we will classify PASS/FAIL using log tokens instead of numeric threshold.
    if passfail_mode:
        pass_token_counts: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
        fail_token_counts: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    # Track earliest and latest raw_line (and their strings) per key for fallback start/end
    earliest_raw_by_key: Dict[Tuple[str,str,str,str,str], str] = {}
    latest_raw_by_key: Dict[Tuple[str,str,str,str,str], str] = {}
    earliest_seen_order: Dict[Tuple[str,str,str,str,str], int] = {}
    order_counter = 0
    for r in rows:
        # Skip PR monitor rows and obvious non-FDV tests
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        fdv = _first_nonempty_str(r, ['fdv_file','fdv','file','filepath','filename'], '')
        if not fdv:
            continue
        fdv_l = fdv.lower()
        if 'poweron' in fdv_l or 'powerup' in fdv_l:
            continue
        pr = _first_nonempty_str(r, ['pr','PR'], '') or 'XX'
        vcc = _first_nonempty_str(r, ['vcc','VCC','vcc_mv'], '')
        tm = _first_nonempty_str(r, ['tm','TM'], '')
        temp = _first_nonempty_str(r, ['temp','TEMP','temperature'], '')
        # Enforce FUSEID validity; if invalid or missing, record and skip this row from stats
        fid = _get_fuseid(r)
        key = (fdv, pr, vcc, tm, temp)
        # Track DUT id regardless of fuseid validity for fallback unit count
        try:
            _dut_any = (r.get('dut_id') or '').strip()
            if _dut_any:
                dut_ids_all_by_key[key].add(_dut_any)
        except Exception:
            pass
        # Record earliest/latest raw_line for fallback timestamp population
        try:
            rl_fallback = (r.get('raw_line') or '').strip()
            if rl_fallback:
                order_counter += 1
                if key not in earliest_raw_by_key:
                    earliest_raw_by_key[key] = rl_fallback
                    earliest_seen_order[key] = order_counter
                # Always update latest
                latest_raw_by_key[key] = rl_fallback
        except Exception:
            pass
        if not _is_valid_fuseid(fid):
            # Ensure start/end captured before skipping
            try:
                ts_invalid = (r.get('test_start') or '').strip()
                te_invalid = (r.get('test_end') or '').strip()
                if ts_invalid or te_invalid:
                    prev = aux_by_key.get(key)
                    if prev:
                        # earliest start, latest end
                        prev_ts, prev_te = prev[3], prev[4]
                        new_ts = ts_invalid if (ts_invalid and (not prev_ts or ts_invalid < prev_ts)) else prev_ts
                        new_te = te_invalid if (te_invalid and (not prev_te or te_invalid > prev_te)) else prev_te
                        if new_ts != prev_ts or new_te != prev_te:
                            aux_by_key[key] = (prev[0], prev[1], prev[2], new_ts, new_te)
                    else:
                        aux_by_key[key] = ('','','', ts_invalid, te_invalid)
            except Exception:
                pass
            # Even if fuseid invalid, in passfail_mode we still want to capture PASS/FAIL tokens
            if passfail_mode:
                try:
                    rl_tok = (r.get('raw_line') or '')
                    pf_tok = _extract_pf(rl_tok)
                    if pf_tok == 'FAIL':
                        fail_token_counts[key] += 1  # type: ignore[name-defined]
                    elif pf_tok == 'PASS':
                        pass_token_counts[key] += 1  # type: ignore[name-defined]
                except Exception:
                    pass
            try:
                dut = (r.get('dut_id') or '').strip()
                site = _extract_site_from_filename(fdv)
                if dut:
                    ignored_by_key[key].append((dut, site))
                    invalid_units_by_key[key].append((dut, site, fid or ''))
            except Exception:
                pass
            continue  # skip invalid fuseid rows from numeric stats
        # Record valid mapping
        try:
            dut = (r.get('dut_id') or '').strip()
            if dut and fid:
                dut_fid_by_key[key][dut] = fid
                valid_fuseids_set_by_key[key].add(fid)
                # Maintain first-seen ordered list of distinct fuseids
                if fid not in ordered_valid_fuseids_by_key[key]:
                    ordered_valid_fuseids_by_key[key].append(fid)
        except Exception:
            pass
        # If test_start/test_end present on this row and not yet captured for key, capture earliest start and latest end
        try:
            ts = (r.get('test_start') or '').strip()
            te = (r.get('test_end') or '').strip()
            if ts:
                prev = aux_by_key.get(key)
                if prev:
                    # keep earliest start
                    if prev[3] and ts < prev[3]:
                        aux_by_key[key] = (prev[0], prev[1], prev[2], ts, prev[4])
                else:
                    aux_by_key[key] = ('', '', '', ts, '')
            if te:
                prev = aux_by_key.get(key)
                if prev:
                    # keep latest end
                    if prev[4] and te > prev[4]:
                        aux_by_key[key] = (prev[0], prev[1], prev[2], prev[3], te)
                    elif not prev[4]:
                        aux_by_key[key] = (prev[0], prev[1], prev[2], prev[3], te)
                else:
                    aux_by_key[key] = ('', '', '', '', te)
        except Exception:
            pass
        # Track pagemap/product type
        try:
            pm = _extract_pagemap_from_any(r)
            if pm:
                d = pagemap_counts.setdefault(key, {})
                d[pm] = d.get(pm, 0) + 1
        except Exception:
            pass
        # Track pagemap/product type
        try:
            pm = _extract_pagemap_from_any(r)
            if pm:
                d = pagemap_counts.setdefault(key, {})
                d[pm] = d.get(pm, 0) + 1
        except Exception:
            pass
        rv = _get_rber(r)
        if rv is None:
            # Even if we cannot parse an RBER value we might still record token PASS/FAIL (rare case);
            # skip adding to numeric groups but allow token classification below.
            pass
        else:
            groups[key].append(rv)
        # Token-based PASS/FAIL collection (only when mode active)
        if passfail_mode:
            try:
                rl = (r.get('raw_line') or '')
                pf_tok = _extract_pf(rl)
                if pf_tok == 'FAIL':
                    fail_token_counts[key] += 1
                elif pf_tok == 'PASS':
                    pass_token_counts[key] += 1
            except Exception:
                pass
        # Track pagemap/product type
        try:
            pm = _extract_pagemap_from_any(r)
            if pm:
                d = pagemap_counts.setdefault(key, {})
                d[pm] = d.get(pm, 0) + 1
        except Exception:
            pass
        # Record the most specific non-empty testtime label if present
        tl = (r.get('testtime_label') or '').strip()
        if tl and key not in testtime_by_key:
            testtime_by_key[key] = tl
        # Keep aux fields in case we need to synthesize a label
        if key not in aux_by_key:
            aux_by_key[key] = (
                (r.get('fdvlistname') or '').strip(),
                (r.get('fdvtestrun') or '').strip(),
                (r.get('testtime_seconds') or '').strip(),
                (r.get('test_start') or '').strip(),
                (r.get('test_end') or '').strip(),
            )
        # Track unique planes present (derived strictly from tname)
        pl = _plane_from_tname_or_default(r)
        planes_seen[key].add(pl)
    out: List[Dict[str, str]] = []
    # Union of keys so that fdv tests with only invalid rows still show up
    all_keys = set(dut_fid_by_key.keys()) | set(ignored_by_key.keys()) | set(groups.keys())
    # Fallback: if aux_by_key missing start/end for a key, attempt to derive from earliest/latest raw_line
    import re as _re_dt_norm
    _re_dt = _re_dt_norm.compile(r"(\d{4}_[0-1]?\d_[0-3]?\d)\s+([0-2]?\d:[0-5]?\d:[0-5]?\d)")
    for _k in all_keys:
        if _k not in aux_by_key:
            aux_by_key[_k] = ('','','','', '')
        prev = aux_by_key[_k]
        ts_prev, te_prev = prev[3], prev[4]
        changed = False
        # Derive placeholder start/end from earliest/latest raw lines only if missing
        if (not ts_prev) and earliest_raw_by_key.get(_k):
            m = _re_dt.search(earliest_raw_by_key[_k])
            if m:
                ts_prev = f"{m.group(1)} {m.group(2)}"
                changed = True
        if (not te_prev) and latest_raw_by_key.get(_k):
            m = _re_dt.search(latest_raw_by_key[_k])
            if m:
                te_prev = f"{m.group(1)} {m.group(2)}"
                changed = True
        if changed:
            aux_by_key[_k] = (prev[0], prev[1], prev[2], ts_prev, te_prev)
    for key in sorted(all_keys, key=lambda k: (k[0], k[1]=="XX", k[1], k[2], k[3], k[4])):
        fdv, pr, vcc, tm, temp = key
        raw_vals = groups.get(key, [])
        # Filter out None values defensively (minimal runner may emit None RBERs)
        vals = sorted([x for x in raw_vals if x is not None])
        n = len(vals)
        import statistics as _stats
        pls = planes_seen.get(key, set())
        # If all rows agree on a single plane, use it; else wildcard
        if len(pls) == 1:
            po_val = next(iter(pls))
        else:
            po_val = ''  # wildcard
        po_str = (po_val or '*')
        # PASS/FAIL determination
        if passfail_mode:
            # Strict token-based mode: ONLY use PASS/FAIL tokens; do not fall back to numeric threshold.
            pt = pass_token_counts.get(key, 0) if 'pass_token_counts' in locals() else 0
            ft = fail_token_counts.get(key, 0) if 'fail_token_counts' in locals() else 0
            pass_n = pt
            fail_n = ft
            pf_mode = 'token'
            # count is total token-classified rows (pass+fail). If zero, we still show 0/0.
            n_tokens = pt + ft
        else:
            if n:
                if limit is None:
                    # No numeric threshold supplied; treat all as fail (or neutral) => 0 passes.
                    pass_n = 0
                    fail_n = n
                else:
                    pass_n = sum(1 for x in vals if (x is not None and x < limit))
                    fail_n = n - pass_n
            else:
                pass_n = 0
                fail_n = 0
            pf_mode = 'limit'
            n_tokens = n
        # Synthesize label if missing; also attempt to derive missing duration seconds from start/end
        label = testtime_by_key.get(key, '')
        if not label:
            ln, rn, secs, ts, te = aux_by_key.get(key, ('', '', '', '', ''))
            # If duration seconds missing but we have start/end timestamps, compute it
            if (not secs) and ts and te:
                try:
                    import re as _re_dt_calc
                    _m_ts = _re_dt_calc.search(r"(\d{4})_(\d{1,2})_(\d{1,2})\s+([0-2]?\d:[0-5]?\d:[0-5]?\d)", ts)
                    _m_te = _re_dt_calc.search(r"(\d{4})_(\d{1,2})_(\d{1,2})\s+([0-2]?\d:[0-5]?\d:[0-5]?\d)", te)
                    if _m_ts and _m_te:
                        from datetime import datetime as _dt_calc
                        def _to_dt(_m):
                            yyyy = int(_m.group(1)); mm = int(_m.group(2)); dd = int(_m.group(3))
                            hh, mi, ss = [int(x) for x in _m.group(4).split(':')]
                            return _dt_calc(yyyy, mm, dd, hh, mi, ss)
                        _dts = _to_dt(_m_ts); _dte = _to_dt(_m_te)
                        _dur = int((_dte - _dts).total_seconds())
                        if _dur < 0: _dur = 0
                        secs = str(_dur)
                        # Write back into aux_by_key so later consumers see it
                        prev_aux = aux_by_key.get(key, ('', '', '', ts, te))
                        aux_by_key[key] = (prev_aux[0], prev_aux[1], secs, ts, te)
                except Exception:
                    pass
            # Use fdvtest (basename without extension) instead of run name for display
            import os as _os
            fdvtest = _os.path.basename(fdv)
            if fdvtest.lower().endswith('.fdv'):
                fdvtest = fdvtest[:-4]
            if ln or fdvtest or secs:
                label = f"{ln}::{fdvtest} = {secs}".strip()
        # Build comments with ignored DUT@site information
        ignored = ignored_by_key.get(key, [])
        ignored_parts = []
        if ignored:
            # compress duplicates
            seen_ig = set()
            for (d, s) in ignored:
                lab = f"DUT{d}@site{s}" if s else f"DUT{d}"
                if lab not in seen_ig:
                    seen_ig.add(lab)
                    ignored_parts.append(lab)
        # Build VALID listing similar to poll webapp: DUTn@site:FuseID
        site_label = _extract_site_from_filename(fdv)
        valid_map = dut_fid_by_key.get(key, {})
        valid_items = []
        for d in sorted(valid_map.keys(), key=lambda s: int(s) if s.isdigit() else 9999):
            vf = valid_map[d]
            lbl = f"DUT{d}{('@' + site_label) if site_label else ''}:{vf}"
            valid_items.append(lbl)
        # Build unit listing with explicit category labels:
        #   VALID: <site>:<dut>:<fuseid>, ... | INVALID: <site>:<dut>:<raw_fuseid>, ...
        # Always include a site token; if extraction fails, use 'xx'.
        def _norm_site(val: str | None) -> str:
            v = (val or '').strip()
            return v if v else 'xx'
        site_file = site_label  # site extracted from filename once per fdv_file
        valid_units: list[str] = []
        for d in sorted(valid_map.keys(), key=lambda s: int(s) if s.isdigit() else 9999):
            vf = valid_map[d]
            valid_units.append(f"{_norm_site(site_file)}:{d}:{vf}")
        invalid_units_render: list[str] = []
        inv_list = invalid_units_by_key.get(key, [])
        if inv_list:
            for (d_i, s_i, raw_fid) in sorted(inv_list, key=lambda t: int(t[0]) if t[0].isdigit() else 9999):
                shown_fid = (raw_fid or '-')
                invalid_units_render.append(f"{_norm_site(site_file or s_i)}:{d_i}:{shown_fid}")
        unit_info_parts: list[str] = []
        # Add back FUSEIDS list (ordered distinct valid fuseids) plus INVALID section
        _ordered_ids = ordered_valid_fuseids_by_key.get(key, [])
        if _ordered_ids:
            unit_info_parts.append("FUSEIDS: " + ", ".join(_ordered_ids))
        if invalid_units_render:
            unit_info_parts.append("INVALID: " + ", ".join(invalid_units_render))
        unit_info_str = " | ".join(unit_info_parts)
        base_comment = ("VALID: " + ", ".join(valid_items)) if valid_items else ''
        ig_comment = ("IGNORED (invalid FUSEID): " + ", ".join(sorted(ignored_parts))) if ignored_parts else ''
        comments = " | ".join([c for c in (base_comment, ig_comment) if c])
        valid_count = len(valid_fuseids_set_by_key.get(key, set())) if valid_fuseids_set_by_key.get(key) else 0
        # Updated semantics (2025-09-14, user request revision):
        #   VECTOR column = total number of raw lines beginning with 'FDV OUTPUT'
        #   captured globally (fdv_output_counts) BEFORE filtering for validity,
        #   regardless of PASS/FAIL classification.
        #   Monitor/Shmoo lines are still detected and excluded from PASS/FAIL counts
        #   but no longer added to VECTOR explicitly; their raw lines already count
        #   if they start with 'FDV OUTPUT'.
        count_total = pass_n + fail_n
        vector_total = fdv_output_counts.get(key, 0)
        # Safe fail percentage: if no PASS/FAIL lines at all, define as 0.0
        fail_pct_val: float = 0.0
        try:
            if count_total > 0 and fail_n > 0:
                fail_pct_val = (fail_n / count_total) * 100.0
        except Exception:
            fail_pct_val = 0.0
        out.append({
            'fdv_file': fdv,
            'pr': pr,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'plane_op': po_str,
            'comments': comments,
            'valid_fuseid_count': str(valid_count),
            'pagemap': (sorted(pagemap_counts.get(key, {}).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                        if pagemap_counts.get(key) else ''),
            'testtime_label': label,
            # Preserve actual derived start/end (no label substitution) so blanks remain blanks
            'test_start': (aux_by_key.get(key, ('','','','',''))[3] if key in aux_by_key else ''),
            'test_end': (aux_by_key.get(key, ('','','','',''))[4] if key in aux_by_key else ''),
            'count': str(count_total),
            'vector': str(vector_total),
            'pass': str(pass_n),
            'fail': str(fail_n),
            'pass_n': str(pass_n),
            'fail_n': str(fail_n),
            'fail_pct': f"{fail_pct_val:.1f}",
            'min': (f"{vals[0]:.6g}" if n else ''),
            'max': (f"{vals[-1]:.6g}" if n else ''),
            'mean': (f"{(sum(vals)/n):.6g}" if n else ''),
            'stdev': (f"{(_stats.stdev(vals) if n>=2 else 0.0):.6g}" if n else ''),
            'median': (f"{_stats.median(vals):.6g}" if n else ''),
            'pf_mode': pf_mode,
            'unit_info': unit_info_str,
        })
    try:
        _tok_rows = sum(1 for r in out if r.get('pf_mode') == 'token')
        print(f"[stats_by_fdv_with_splits] produced {len(out)} rows; token-mode rows={_tok_rows}; with unit info in {sum(1 for r in out if r.get('valid_fuseid_count') and r.get('valid_fuseid_count')!='0')} rows")
    except Exception:
        pass
    # Inject synthetic rows for monitor-only groups absent from out
    try:
        for _k in locals().get('_placeholder_only', []):  # type: ignore[index]
            if not any(r for r in out if (r.get('fdv_file'), r.get('pr'), r.get('vcc'), r.get('tm'), r.get('temp')) == _k):
                fdv, pr, vcc, tm, temp = _k
                ms_ct = monitor_shmoo_counts.get(_k, 0)
                out.append({
                    'fdv_file': fdv,
                    'pr': pr,
                    'vcc': vcc,
                    'tm': tm,
                    'temp': temp,
                    'plane_op': '',
                    'comments': 'MONITOR/SHMOO only',
                    'valid_fuseid_count': '0',
                    'pagemap': '',
                    'testtime_label': '',
                    'test_start': '',
                    'test_end': '',
                    'count': '0',
                    'vector': str(ms_ct),
                    'pass': '0',
                    'fail': '0',
                    'pass_n': '0',
                    'fail_n': '0',
                    'fail_pct': '0.0',
                    'min': '',
                    'max': '',
                    'mean': '',
                    'stdev': '',
                    'median': '',
                    'pf_mode': False,
                    'unit_info': '',
                })
    except Exception:
        pass
    return out


def stats_by_testname(rows: List[Dict[str, str]], fdv_file: str) -> List[Dict[str, str]]:
    from collections import defaultdict
    vals: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        if (r.get('fdv_file','') or '') != (fdv_file or ''):
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip()
        if not tn:
            raw = (r.get('tname','') or '').strip()
            tn = derive_testname(raw) if raw else ''
        if not tn or ('poweron' in tn.lower() or 'powerup' in tn.lower()):
            continue
        # Ignore rows with invalid/missing FUSEID
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        pr = (r.get('pr','') or '').strip() or 'XX'
        fid = (r.get('fuseid','') or '').strip()
        vcc = (r.get('vcc','') or '').strip()
        tm = (r.get('tm','') or '').strip()
        temp = (r.get('temp','') or '').strip()
        rv = _to_float(r.get('rber'))
        if rv is None:
            continue
        vals[(tn, pr, fid, vcc, tm, temp)].append(rv)
    def _pr_key(p: str):
        if p == 'XX':
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    for (tn, pr, fid, vcc, tm, temp) in sorted(vals.keys(), key=lambda k: (k[0], _pr_key(k[1]), k[2], k[3], k[4], k[5])):
        v = sorted(vals[(tn, pr, fid, vcc, tm, temp)])
        n = len(v)
        import statistics as _stats
        out.append({
            'testname': tn,
            'pr': pr,
            'fuseid': fid,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'count': str(n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
    })
    return out


def stats_by_testname_multi(rows: List[Dict[str, str]], fdv_files: List[str]) -> List[Dict[str, str]]:
    """Aggregate testname RBER stats limited to selected fdv_files.

    Note: This legacy variant filters only by fdv_file. Prefer stats_by_testname_selected.
    """
    sel = set([f for f in fdv_files if f])
    from collections import defaultdict
    vals: Dict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        fdv = (r.get('fdv_file','') or '')
        if fdv not in sel:
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip()
        if not tn:
            raw = (r.get('tname','') or '').strip()
            tn = derive_testname(raw) if raw else ''
        if not tn or ('poweron' in tn.lower() or 'powerup' in tn.lower()):
            continue
        # Ignore rows with invalid/missing FUSEID
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        pr = (r.get('pr','') or '').strip() or 'XX'
        fid = (r.get('fuseid','') or '').strip()
        vcc = (r.get('vcc','') or '').strip()
        tm = (r.get('tm','') or '').strip()
        temp = (r.get('temp','') or '').strip()
        rv = _to_float(r.get('rber'))
        if rv is None:
            continue
        vals[(tn, pr, fid, vcc, tm, temp)].append(rv)
    def _pr_key(p: str):
        if p == 'XX':
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    import statistics as _stats
    for (tn, pr, fid, vcc, tm, temp) in sorted(vals.keys(), key=lambda k: (k[0], _pr_key(k[1]), k[2], k[3], k[4], k[5])):
        v = sorted(vals[(tn, pr, fid, vcc, tm, temp)])
        n = len(v)
        out.append({
            'testname': tn,
            'pr': pr,
            'fuseid': fid,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'count': str(n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
        })
    return out


def _parse_fdv_selector(val: str) -> Tuple[str, str, str, str, str]:
    """Parse 'fdv|pr|vcc|tm|temp' or just 'fdv'. Missing parts become empty strings."""
    parts = (val or '').split('|')
    fdv = parts[0] if len(parts) >= 1 else ''
    pr = parts[1] if len(parts) >= 2 else ''
    vcc = parts[2] if len(parts) >= 3 else ''
    tm = parts[3] if len(parts) >= 4 else ''
    temp = parts[4] if len(parts) >= 5 else ''
    return (fdv, pr, vcc, tm, temp)


def stats_by_testname_selected(rows: List[Dict[str, str]], selectors: List[str], *, limit: float = 0.0, passfail_mode: bool = False) -> List[Dict[str, str]]:
    """Aggregate testname RBER stats limited to selected fdv rows (fdv,pr,vcc,tm,temp),
    further split by plane_group and operation so users can select precise rows including plane/operation.
    """
    keyset = set(_parse_fdv_selector(s) for s in selectors if s)
    from collections import defaultdict
    # Group by (testname, pr, fuseid, vcc, tm, temp, plane, op)
    vals: Dict[Tuple[str, str, str, str, str, str, str, str], List[float]] = defaultdict(list)
    plane_addr_sets: Dict[Tuple[str, str, str, str, str, str, str, str], set] = defaultdict(set)
    blk_addr_sets: Dict[Tuple[str, str, str, str, str, str, str, str], set] = defaultdict(set)
    # Track pagemap per (testname, pr, fuseid, vcc, tm, temp, plane, op)
    pagemap_counts: Dict[Tuple[str, str, str, str, str, str, str, str], Dict[str, int]] = defaultdict(dict)
    for r in rows:
        k = _get_split_tuple(r)
        # Accept match if fdv matches and either exact split matches OR selector omits split parts
        # Build a set of acceptable selectors where empty split fields are wildcards
        matched = False
        for (fdv, pr, vcc, tm, temp) in keyset:
            if not fdv:
                continue
            if fdv != k[0]:
                continue
            if pr and pr != k[1]:
                continue
            if vcc and vcc != k[2]:
                continue
            if tm and tm != k[3]:
                continue
            if temp and temp != k[4]:
                continue
            matched = True
            break
        if not matched:
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
        if not tn or ('poweron' in tn.lower() or 'powerup' in tn.lower()):
            continue
        pr = k[1]
        fid = _get_fuseid(r)
        vcc = k[2]
        tm = k[3]
        temp = k[4]
        # Derive plane strictly from tname
        plane = _plane_from_tname_or_default(r)
        op = _first_nonempty_str(r, ['operation','op','readtype'], '').upper()
        rv = _get_rber(r)
        if rv is None:
            continue
        kkey = (tn, pr, fid, vcc, tm, temp, plane, op)
        vals[kkey].append(rv)
        # Count pagemap
        try:
            pm = _extract_pagemap_from_any(r)
            if pm:
                d = pagemap_counts.setdefault(kkey, {})
                d[pm] = d.get(pm, 0) + 1
        except Exception:
            pass
        # Prefer previously annotated canonical plane/block if present
        if r.get('plane_addr_canonical'):
            pa = str(r.get('plane_addr_canonical'))
        else:
            pa = _extract_plane_addr(r)
        if pa:
            plane_addr_sets[kkey].add(pa)
        if r.get('blk_canonical'):
            try:
                bv = int(str(r.get('blk_canonical')))
            except Exception:
                bv = _extract_blk_value(r)
        else:
            bv = _extract_blk_value(r)
        if bv is not None:
            blk_addr_sets[kkey].add(bv)
    def _pr_key(p: str):
        if p == 'XX':
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    import statistics as _stats
    for (tn, pr, fid, vcc, tm, temp, plane, op) in sorted(vals.keys(), key=lambda k: (k[0], _pr_key(k[1]), k[2], k[3], k[4], k[5], k[6], k[7])):
        v = sorted(vals[(tn, pr, fid, vcc, tm, temp, plane, op)])
        n = len(v)
        # Default numeric threshold classification
        pass_n = sum(1 for x in v if x < limit)
        fail_n = n - pass_n
        pf_mode = 'limit'
        if passfail_mode:
            # Reconstruct token counts by scanning contributing rows (filter again)
            pt = 0; ft = 0
            try:
                import re as _re
                for r in rows:
                    ksplit = _get_split_tuple(r)
                    if ksplit != (ksplit[0], ksplit[1], ksplit[2], ksplit[3], ksplit[4]):
                        pass
                    if (r.get('tname','') or '').strip().upper() == 'PR':
                        continue
                    _tn = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
                    if _tn != tn:
                        continue
                    if _get_fuseid(r) != fid:
                        continue
                    # plane/op check
                    if _plane_from_tname_or_default(r) != plane:
                        continue
                    rop = _first_nonempty_str(r, ['operation','op','readtype'], '').upper()
                    if rop != op:
                        continue
                    rl = (r.get('raw_line') or '')
                    up = rl.upper()
                    if _re.search(r'(?<![A-Z0-9])FAIL(?![A-Z0-9])', up):
                        ft += 1
                    elif _re.search(r'(?<![A-Z0-9])PASS(?![A-Z0-9])', up):
                        pt += 1
            except Exception:
                pass
            if (pt + ft) > 0:
                pass_n = pt
                fail_n = ft
                n = pt + ft
                pf_mode = 'token'
        pa_set = plane_addr_sets.get((tn, pr, fid, vcc, tm, temp, plane, op), set())
        pa_disp = ''
        if pa_set:
            try:
                pa_disp = ','.join(sorted(pa_set, key=lambda s: int(s[1:]) if (isinstance(s, str) and s.startswith('P') and s[1:].isdigit()) else s))
            except Exception:
                pa_disp = ','.join(sorted(pa_set))
        blk_set = blk_addr_sets.get((tn, pr, fid, vcc, tm, temp, plane, op), set())
        blk_disp = ''
        if blk_set:
            try:
                blk_disp = ','.join(str(x) for x in sorted(blk_set))
            except Exception:
                blk_disp = ','.join(str(x) for x in blk_set)
    out.append({
            'testname': tn,
            'pr': pr,
            'fuseid': fid,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            # Include pagemap in testname table
            'pagemap': (sorted(pagemap_counts.get((tn, pr, fid, vcc, tm, temp, plane, op), {}).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                        if pagemap_counts.get((tn, pr, fid, vcc, tm, temp, plane, op)) else ''),
            'plane': plane,
            'op': op,
            'plane_op': (plane or '*'),
            'plane_addr': pa_disp,
            'blk_addr': blk_disp,
            'count': str(n),
            'pass': str(pass_n),
            'fail': str(fail_n),
            'pass_n': str(pass_n),
            'fail_n': str(fail_n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
            'pf_mode': pf_mode,
        })
    return out

# ---------------------------------------------------------------------------
# Split-by-DUT (FUSEID) stats for selected FDV test rows
# Group each selected (fdv,pr,vcc,tm,temp) by valid FUSEID and compute RBER stats.
# ---------------------------------------------------------------------------
def stats_by_dut_selected(rows: List[Dict[str, str]], selectors: List[str], *, limit: float | None = None) -> List[Dict[str, str]]:
    keyset = set(_parse_fdv_selector(s) for s in selectors if s)
    from collections import defaultdict
    # (fdv, pr, vcc, tm, temp, fuseid) -> list of rber values
    vals: Dict[Tuple[str,str,str,str,str,str], List[float]] = defaultdict(list)
    first_start: Dict[Tuple[str,str,str,str,str,str], str] = {}
    last_end: Dict[Tuple[str,str,str,str,str,str], str] = {}
    for r in rows:
        k = _get_split_tuple(r)  # (fdv, pr, vcc, tm, temp)
        matched = False
        for (fdv, pr, vcc, tm, temp) in keyset:
            if (not fdv or k[0] == fdv) and (not pr or k[1] == pr) and (not vcc or k[2] == vcc) and (not tm or k[3] == tm) and (not temp or k[4] == temp):
                matched = True
                break
        if not matched:
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        fid = _get_fuseid(r)
        if not _is_valid_fuseid(fid):
            continue
        rv = _get_rber(r)
        if rv is None:
            continue
        key = (k[0], k[1], k[2], k[3], k[4], fid)
        vals[key].append(rv)
        ts = (r.get('test_start') or '').strip()
        te = (r.get('test_end') or '').strip()
        if ts and not first_start.get(key):
            first_start[key] = ts
        if te:
            last_end[key] = te
    out: List[Dict[str, str]] = []
    import statistics as _stats
    for key in sorted(vals.keys(), key=lambda k: (k[0], k[1]=='XX', k[1], k[2], k[3], k[4], k[5])):
        fdv, pr, vcc, tm, temp, fid = key
        v = sorted(vals[key])
        n = len(v)
        pass_n = sum(1 for x in v if x < limit)
        fail_n = n - pass_n
        out.append({
            'fdv_file': fdv,
            'pr': pr,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'fuseid': fid,
            'count': str(n),
            'pass_n': str(pass_n),
            'fail_n': str(fail_n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
            'test_start': first_start.get(key, ''),
            'test_end': last_end.get(key, ''),
        })
    return out


def _build_variability_records(rows: List[Dict[str, str]], selectors: List[str], entries: List[Tuple[str, ...]], *, allow_page_fallback: bool = True, allow_missing_wl: bool = False):
    recs: List[Dict] = []
    # Build parsed selector tuples once
    parsed = [_parse_fdv_selector(x) for x in selectors]
    for s in entries:
        # tuple shape: (testname, pr, fuseid, vcc, tm, temp, plane, op)
        tn = s[0]
        pr = s[1] if len(s) > 1 else ''
        fid = s[2] if len(s) > 2 else ''
        sel_vcc = s[3] if len(s) > 3 else ''
        sel_tm = s[4] if len(s) > 4 else ''
        sel_temp = s[5] if len(s) > 5 else ''
        sel_plane = s[6].upper() if len(s) > 6 and s[6] else ''
        sel_op = s[7].upper() if len(s) > 7 and s[7] else ''

        for idx, r in enumerate(rows):
            rf = (r.get('fdv_file', '') or '')
            # row-level split key
            rk = _get_split_tuple(r)
            # match any selector (empty fields are wildcards)
            ok = False
            for (sf, spr, svcc, stm, stemp) in parsed:
                if sf and sf != rk[0]:
                    continue
                if spr and spr != rk[1]:
                    continue
                if svcc and svcc != rk[2]:
                    continue
                if stm and stm != rk[3]:
                    continue
                if stemp and stemp != rk[4]:
                    continue
                ok = True
                break
            if not ok:
                continue
            if (r.get('tname', '') or '').strip().upper() == 'PR':
                continue
            tnr_raw = (r.get('testname', '') or '').strip()
            tnr = tnr_raw if tnr_raw else derive_testname((r.get('tname', '') or '').strip())
            if (tnr or '').strip().lower() != (tn or '').strip().lower():
                continue
            # If selection's PR/FuseID are empty, treat them as wildcards (testname-only filter)
            _rf, row_pr, row_vcc, row_tm, row_temp = rk
            row_fid = _get_fuseid(r)
            if pr and row_pr != pr:
                continue
            # Soften FuseID: only exclude when row has a different non-empty fid
            if fid:
                if row_fid and row_fid != fid:
                    continue
            if sel_vcc and row_vcc != sel_vcc:
                continue
            if sel_tm and row_tm != sel_tm:
                continue
            if sel_temp and row_temp != sel_temp:
                continue
            # Derive plane strictly from tname; default NP when missing
            row_plane = _plane_from_tname_or_default(r)
            row_op = _first_nonempty_str(r, ['operation', 'op', 'readtype'], '').upper()
            # Treat empty or '*' (and legacy 'NP') as wildcard
            if sel_plane and sel_plane not in ('*', 'NP') and row_plane != sel_plane:
                continue
            if sel_op and sel_op not in ('*', 'NP') and row_op != sel_op:
                continue
            wl = _extract_wl_or_page(r, allow_page_fallback=allow_page_fallback)
            rber = _get_rber(r)
            if rber is None:
                continue
            if rber <= 0:
                rber = 1e-12
            if wl is None:
                # If WL (or PAGE when allowed) is unavailable, include only when explicitly allowed (data view).
                if not allow_missing_wl:
                    continue
                wl = -1.0
            recs.append({
                'testname': tn,
                'WL': wl,
                'RBER': rber,
                'pagetype': (r.get('pagetype', '') or '').strip(),
                'readtype': (r.get('operation', '') or '').strip().upper() or 'READ',
                'dut': f"DUT{(r.get('dut_id', '') or '').strip() or '?'}",
                'plane': row_plane,
                'op': row_op,
                'plane_addr': _extract_plane_addr(r),
                'blk': _extract_blk_value(r),
                '_idx': r.get('_idx', idx),
                'line_number': r.get('line_number', ''),
            })
    return recs


def _mk_app() -> Flask:
    app = Flask(__name__, template_folder=str(_HERE / 'templates'))
    app.secret_key = os.environ.get('FDV_REPORT2_SECRET', 'dev-secret')
    # Jinja filters
    @app.template_filter('basename')
    def _jinja_basename(value: object) -> str:
        """Return just the filename component from a possibly Windows path.

        Handles both \\ and / separators without relying on Jinja's split filter.
        """
        try:
            s = str(value or '')
            if not s:
                return ''
            s = s.replace('\\', '/')
            if '/' in s:
                return s.rsplit('/', 1)[-1]
            return s
        except Exception:
            return ''
    # Configure a reliable temp directory for multipart parsing/uploads.
    # Prefer D:\\fdv_tmp for all temp usage; allow override via FDV_REPORT2_TMPDIR.
    # Fall back gracefully to other candidates only if D: (or override) is unavailable.
    try:
        import tempfile as _tempfile
        from pathlib import Path as _Path
        import shutil as _shutil
        override_tmp = (os.environ.get('FDV_REPORT2_TMPDIR') or r'D:\\fdv_tmp').strip()
        best_path = None
        best_free = -1
        try:
            p = _Path(override_tmp)
            p.mkdir(parents=True, exist_ok=True)
            _ = _shutil.disk_usage(str(p))
            best_path = p
            best_free = int(_[2])
        except Exception:
            best_path = None
        if best_path is None:
            for pth in ['Z:\\fdv_tmp', 'C:\\fdv_tmp']:
                try:
                    q = _Path(pth)
                    q.mkdir(parents=True, exist_ok=True)
                    total, used, free = _shutil.disk_usage(str(q))
                    if free > best_free:
                        best_free = int(free)
                        best_path = q
                except Exception:
                    continue
        if best_path is not None:
            os.environ['FDV_REPORT2_TMPDIR'] = str(best_path)
            os.environ['TMP'] = str(best_path)
            os.environ['TEMP'] = str(best_path)
            try:
                mplcfg = _Path(str(best_path)) / 'mplconfig'
                mplcfg.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault('MPLCONFIGDIR', str(mplcfg))
            except Exception:
                pass
            try:
                _tempfile.tempdir = str(best_path)
            except Exception:
                pass
            try:
                app.config['UPLOAD_FOLDER'] = str(best_path)
            except Exception:
                pass
    except Exception:
        pass
    return app

# Initialize Flask app once (must be before any @app.route usage)
app = _mk_app()
# Fast caches
CACHE: Dict[str, Dict] = {}
# Job indirection: expose stable job ids separate from internal parse tokens
JOBS: Dict[str, Dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
JOB_NAMES_LOCK = threading.Lock()
# Cancellation flags per token (used by prodmode 'Done' to end early after current file)
CANCEL_FLAGS: Dict[str, bool] = {}

def _job_names_path() -> Path:
    p = _persist_base_dir() / 'report2' / 'job_names.json'
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

def _load_job_names() -> Dict[str, str]:
    p = _job_names_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8') or '{}')
    except Exception:
        pass
    return {}

def _save_job_names(d: Dict[str, str]) -> None:
    p = _job_names_path()
    try:
        p.write_text(json.dumps(d, indent=2, sort_keys=True), encoding='utf-8')
    except Exception:
        pass

JOB_NAMES: Dict[str, str] = _load_job_names()
try:
    _JOB_NAMES_PATH = _job_names_path()
    JOB_NAMES_MTIME: float | None = (_JOB_NAMES_PATH.stat().st_mtime if _JOB_NAMES_PATH.exists() else None)
except Exception:
    JOB_NAMES_MTIME = None  # type: ignore

def _maybe_reload_job_names() -> None:
    """Reload job names file if it changed on disk (multi-process safety)."""
    global JOB_NAMES_MTIME
    try:
        p = _job_names_path()
        if not p.exists():
            return
        mt = p.stat().st_mtime
        if JOB_NAMES_MTIME is None or mt > JOB_NAMES_MTIME:
            data = _load_job_names()
            with JOB_NAMES_LOCK:
                # merge new entries / updates
                for k, v in data.items():
                    JOB_NAMES[k] = v
            JOB_NAMES_MTIME = mt
    except Exception:
        pass

# Fast snapshot caches for progress page / table so clients can render instantly
SNAPSHOTS: Dict[str, Dict[str, object]] = {}  # token -> {'fdvtable_html': str, 'updated': float, 'limit_key': str}

# --- Performance configuration (env-tunable) ---
_ALLOWED_EXTS: set[str] = set([e.strip() for e in (os.environ.get('FDV_ALLOWED_EXT', '.txt,.log,.out') or '').lower().split(',') if e.strip()]) or {'.txt', '.log', '.out'}
_EXCLUDE_DIRS: set[str] = set([d.strip() for d in (os.environ.get('FDV_EXCLUDE_DIRS', '__pycache__,.git,ledger') or '').split(',') if d.strip()]) or {'__pycache__', '.git', 'ledger'}
_SNAPSHOT_INTERVAL_SECS: float = float(os.environ.get('FDV_SNAPSHOT_INTERVAL', '8.0'))
_SNAPSHOT_MIN_ROWS_DELTA: int = int(os.environ.get('FDV_SNAPSHOT_MIN_DELTA', '2000'))

# --- SQLite connection cache (per token) ---
_SQLITE_CONNS: Dict[str, sqlite3.Connection] = {}
_SQLITE_LOCK = threading.Lock()
SNAP_LOCK = threading.Lock()

# ---------------- LLM summary support (optional, async, cached) ----------------
# Config via environment variables:
#   FDV2_SUMMARY_ENABLED: '1' to enable UI and API (default: '1')
#   FDV2_SUMMARY_AUTO: '1' to auto-generate after parsing completes (default: '0')
#   FDV2_LLM_BACKEND: 'ollama' to use local Ollama HTTP API; anything else disables LLM and uses heuristic summary (default: 'ollama')
#   FDV2_LLM_MODEL: model name for Ollama (default: 'llama3.2')
#   FDV2_LLM_TIMEOUT: HTTP timeout seconds (default: '25')

SUMMARY_JOBS_RUNNING: Dict[str, bool] = {}
SUMMARY_LOCK = threading.Lock()

def _summary_enabled() -> bool:
    return str(os.environ.get('FDV2_SUMMARY_ENABLED', '1')).strip() not in ('0', 'false', 'off', 'no')

def _summary_auto() -> bool:
    return str(os.environ.get('FDV2_SUMMARY_AUTO', '0')).strip() in ('1', 'true', 'on', 'yes')

def _summary_backend() -> str:
    return (os.environ.get('FDV2_LLM_BACKEND') or 'ollama').strip().lower()

def _summary_model() -> str:
    return (os.environ.get('FDV2_LLM_MODEL') or 'llama3.2').strip()

def _summary_timeout() -> float:
    try:
        return float(os.environ.get('FDV2_LLM_TIMEOUT') or '25')
    except Exception:
        return 25.0

def _summary_path(token: str, limit_key: str) -> Path:
    base = _persist_base_dir() / 'report2' / token
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Normalize limit key for filename
    lk = (limit_key or 'none').replace('/', '_').replace('\\', '_').replace(':', '_')
    return base / f'summary_{lk}.json'

def _summary_load(token: str, limit_key: str) -> Dict[str, object] | None:
    try:
        p = _summary_path(token, limit_key)
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8') or '{}')
    except Exception:
        return None
    return None

def _summary_save(token: str, limit_key: str, payload: Dict[str, object]) -> None:
    try:
        p = _summary_path(token, limit_key)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass

def _compact_stats_for_prompt(stats: List[Dict[str, object]], *, max_rows: int = 30) -> str:
    """Return a compact, stable text encoding of key stats for the LLM prompt.
    Format: one row per line sorted by descending fail_n then fail_pct, limited to max_rows.
    Each line: <basename(fdv)> | PR=<pr> VCC=<vcc> TM=<tm> TEMP=<temp> | count=<count> pass=<pass_n> fail=<fail_n> fail%=<fail_pct>
    """
    def _base(s: str) -> str:
        try:
            s = s.replace('\\', '/');
            return s.rsplit('/', 1)[-1]
        except Exception:
            return s
    # Sort by fail_n desc, then fail_pct desc
    try:
        ord_stats = sorted(stats, key=lambda r: (-int((r.get('fail_n') or r.get('fail') or 0)), -float(str(r.get('fail_pct') or '0') or 0)))
    except Exception:
        ord_stats = stats[:]
    lines: List[str] = []
    for r in ord_stats[:max_rows]:
        try:
            fdv = _base(str(r.get('fdv_file') or ''))
            pr = str(r.get('pr') or '')
            vcc = str(r.get('vcc') or '')
            tm = str(r.get('tm') or '')
            temp = str(r.get('temp') or '')
            cnt = str(r.get('count') or '')
            pas = str(r.get('pass_n') or r.get('pass') or '')
            fai = str(r.get('fail_n') or r.get('fail') or '')
            fp = str(r.get('fail_pct') or '')
            pm = str(r.get('pagemap') or '')
            line = f"{fdv} | PR={pr} VCC={vcc} TM={tm} TEMP={temp} PAGEMAP={pm} | count={cnt} pass={pas} fail={fai} fail%={fp}"
            lines.append(line)
        except Exception:
            continue
    return "\n".join(lines)

def _build_summary_prompt(compact_rows: str) -> str:
    return (
        "You are a concise NAND validation assistant. Analyze the following FDV test summary table rows and produce a short, actionable summary for engineers. "
        "Focus on: overall health, highest failure concentrations, notable PR/VCC/TM/TEMP or pagemap patterns, and any obvious next steps. "
        "Be specific but brief (5-10 bullet points max). Avoid repeating raw numbers excessively; highlight the key outliers.\n\n"
        "Rows (top by failures):\n" + compact_rows + "\n\n"
        "Output format:\n- One-line headline\n- 3-8 bullet points\n- Optional short note on data coverage if limited\n"
    )

def _ollama_generate(prompt: str, *, model: str, timeout: float) -> str | None:
    """Call local Ollama HTTP API; return response text or None on failure."""
    try:
        url = os.environ.get('OLLAMA_URL') or 'http://127.0.0.1:11434/api/generate'
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.2, 'num_ctx': 4096}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return None
        try:
            obj = json.loads(data.decode('utf-8', errors='replace'))
            # Ollama returns {'response': '...'}
            return obj.get('response') or None
        except Exception:
            # Some backends may return plain text
            return data.decode('utf-8', errors='replace')
    except Exception:
        return None

def _heuristic_summary(stats: List[Dict[str, object]]) -> str:
    """Fallback quick summary without LLM. Emphasize top failing rows and coverage."""
    if not stats:
        return "No data available to summarize yet."
    try:
        total_rows = len(stats)
        total_count = sum(int(r.get('count') or 0) for r in stats)
        total_fail = sum(int(r.get('fail_n') or r.get('fail') or 0) for r in stats)
        total_pass = sum(int(r.get('pass_n') or r.get('pass') or 0) for r in stats)
        fail_pct = (100.0 * total_fail / max(1, total_pass + total_fail)) if (total_pass + total_fail) > 0 else 0.0
        # Top 3 by fail_n
        tops = sorted(stats, key=lambda r: -int(r.get('fail_n') or r.get('fail') or 0))[:3]
        lines = [
            f"Overall: {total_rows} groups; total vectors={total_count}; fail%={fail_pct:.1f}.",
        ]
        for i, r in enumerate(tops, start=1):
            fdv = str(r.get('fdv_file') or '')
            pr = str(r.get('pr') or '')
            vcc = str(r.get('vcc') or '')
            tm = str(r.get('tm') or '')
            temp = str(r.get('temp') or '')
            cnt = str(r.get('count') or '')
            fai = str(r.get('fail_n') or r.get('fail') or '')
            fp = str(r.get('fail_pct') or '')
            pm = str(r.get('pagemap') or '')
            lines.append(f"Top {i}: {os.path.basename(fdv)} | PR={pr} VCC={vcc} TM={tm} TEMP={temp} {('PM='+pm) if pm else ''} | fail={fai}/{cnt} ({fp}%).")
        if total_rows > 3:
            lines.append("See table for additional groups; investigate top failures first, then compare across PR/VCC/TM/TEMP.")
        return "Summary (fallback):\n- " + "\n- ".join(lines)
    except Exception:
        return "Summary unavailable due to an internal error."

def _generate_summary_text(token: str, limit_key: str) -> Dict[str, object]:
    """Compute summary text (LLM or heuristic) for the given token and limit key, using persisted SQLite stats when available."""
    # Determine effective limit and passfail mode from limit_key
    lr = (limit_key or '').strip().lower()
    passfail_mode = lr in ('', 'none', 'default')
    if passfail_mode:
        limit_val = 1e9
    else:
        try:
            limit_val = float(lr)
        except Exception:
            limit_val = 1e9
            passfail_mode = True
    # Prefer persisted SQLite aggregates for completeness
    try:
        stats = _stats_from_sqlite(token, limit=limit_val, passfail_mode=passfail_mode)
    except Exception:
        stats = []
    compact = _compact_stats_for_prompt(stats, max_rows=30)
    backend = _summary_backend()
    model = _summary_model()
    text: str | None = None
    used_backend = 'heuristic'
    if backend == 'ollama' and _summary_enabled() and compact:
        prompt = _build_summary_prompt(compact)
        text = _ollama_generate(prompt, model=model, timeout=_summary_timeout())
        if text:
            used_backend = 'ollama'
    if not text:
        text = _heuristic_summary(stats)
        used_backend = 'heuristic'
    payload = {
        'ok': True,
        'token': token,
        'limit': limit_key,
        'backend': used_backend,
        'model': (model if used_backend == 'ollama' else ''),
        'generated_at': time.time(),
        'summary': text,
        'rows_used': min(len(stats), 30)
    }
    return payload

def _ensure_summary_async(token: str, limit_key: str) -> None:
    """Start a summary generation job if not already running or available."""
    # If already exists on disk, do nothing
    try:
        if _summary_load(token, limit_key):
            return
    except Exception:
        pass
    job_key = f"{token}:{limit_key}"
    with SUMMARY_LOCK:
        if SUMMARY_JOBS_RUNNING.get(job_key):
            return
        SUMMARY_JOBS_RUNNING[job_key] = True
    def _worker():
        try:
            payload = _generate_summary_text(token, limit_key)
            _summary_save(token, limit_key, payload)
        except Exception:
            try:
                err = {'ok': False, 'token': token, 'limit': limit_key, 'error': 'summary generation failed', 'generated_at': time.time()}
                _summary_save(token, limit_key, err)
            except Exception:
                pass
        finally:
            with SUMMARY_LOCK:
                SUMMARY_JOBS_RUNNING.pop(job_key, None)
    threading.Thread(target=_worker, name=f"fdv-summarize-{token[:6]}", daemon=True).start()

def _create_job_id(token: str, limit_raw: str) -> str:
    for _ in range(5):
        jid = secrets.token_urlsafe(8).replace('-', '').replace('_', '')[:10]
        with JOBS_LOCK:
            if jid not in JOBS:
                JOBS[jid] = {'token': token, 'created_at': time.time(), 'limit_raw': limit_raw or '', 'name': ''}
                return jid
    fallback = token[:10]
    with JOBS_LOCK:
        JOBS.setdefault(fallback, {'token': token, 'created_at': time.time(), 'limit_raw': limit_raw or '', 'name': ''})
    return fallback

def _resolve_job_token(job_id: str) -> str | None:
    with JOBS_LOCK:
        rec = JOBS.get(job_id)
        return rec.get('token') if rec else None

def _comments_path(token: str) -> Path:
    base = _persist_base_dir() / 'report2' / token
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base / 'comments.json'

def _load_comments(token: str) -> Dict[str, str]:
    p = _comments_path(token)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}

def _save_comments(token: str, data: Dict[str, str]) -> None:
    try:
        p = _comments_path(token)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding='utf-8')
    except Exception:
        pass

# Reuse same mechanism for dispositions (comment column) to distinguish logically if needed later
def _dispositions_path(token: str) -> Path:
    base = _persist_base_dir() / 'report2' / token
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base / 'dispositions.json'

def _load_dispositions(token: str) -> Dict[str, str]:
    p = _dispositions_path(token)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}

def _save_dispositions(token: str, data: Dict[str, str]) -> None:
    try:
        p = _dispositions_path(token)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding='utf-8')
    except Exception:
        pass


def _list_files(paths: List[Path]) -> List[Path]:
    """Return all files under the given paths, searching subdirectories recursively.

    Notes:
    - Preserves existing behavior of accepting any file extension; filtering is handled
      downstream by the parser which looks for "FDV OUTPUT" lines.
    - Traversal errors (permissions, broken links) are ignored to keep the UI responsive.
    """
    files: List[Path] = []
    for p in paths:
        try:
            if p.is_file():
                # Filter by extension if configured
                try:
                    if _ALLOWED_EXTS and p.suffix and p.suffix.lower() not in _ALLOWED_EXTS:
                        continue
                except Exception:
                    pass
                files.append(p)
            elif p.is_dir():
                # Walk recursively; sort filenames for stable order
                for root, _dirs, fnames in os.walk(p):
                    try:
                        root_path = Path(root)
                    except Exception:
                        continue
                    # Skip excluded directories
                    try:
                        parts_lower = {seg.lower() for seg in root_path.parts}
                        if any((d.lower() in parts_lower) for d in _EXCLUDE_DIRS):
                            continue
                    except Exception:
                        pass
                    for name in sorted(fnames):
                        fp = root_path / name
                        try:
                            if fp.is_file():
                                if _ALLOWED_EXTS:
                                    try:
                                        if fp.suffix and fp.suffix.lower() not in _ALLOWED_EXTS:
                                            continue
                                    except Exception:
                                        pass
                                files.append(fp)
                        except Exception:
                            # Skip unreadable entries
                            continue
        except Exception:
            continue
    return files


def _start_parse_job(token: str, files: List[Path], used_dir: str | None, limit_raw: str = '', *, prodmode: bool = False, ledger_map: Dict[str, Path] | None = None) -> None:
    """Spawn a background job to parse files with progress updates stored in CACHE[token].

    When prodmode is True and a ledger_map is provided, after each file finishes processing,
    the corresponding ledger .ready file will be renamed to .done.
    """
    def job():
        try:
            # Allowed line prefixes per requirement; all other lines ignored early.
            _ALLOWED_PREFIXES_CS = (
                'Test Start Date',
                'Test End Date',
                'ECHO: FUSEID',
                'FDV OUTPUT',
            )
            _ALLOWED_PREFIXES = tuple(p.upper() for p in _ALLOWED_PREFIXES_CS)
            def _allowed_line(s: str) -> bool:
                ls = s.lstrip()
                up = ls.upper()
                return any(up.startswith(p) for p in _ALLOWED_PREFIXES)
            # Determine if we should run in streaming prodmode (watch ledger and keep going until cancel)
            stream_prodmode = bool(prodmode and used_dir and Path(used_dir).is_dir())
            total_bytes = 0
            sizes: List[int] = []
            # Optional pre-count of lines for more accurate progress; disabled by default to avoid double I/O on large files.
            # Enable by setting env FDV_PRECOUNT_LINES=1.
            precount = str(os.environ.get('FDV_PRECOUNT_LINES', '0')).strip().lower() in ('1','true','yes','on')
            line_counts: List[int] = []
            total_lines = 0
            if not stream_prodmode:
                for fp in files:
                    try:
                        sz = fp.stat().st_size
                    except Exception:
                        sz = 0
                    sizes.append(sz)
                    total_bytes += sz
                    lc = 0
                    if precount:
                        try:
                            with open(fp, 'r', encoding='utf-8', errors='replace') as _lfc:
                                for lc, _ in enumerate(_lfc, start=1):
                                    pass
                        except Exception:
                            lc = 0
                    line_counts.append(lc)
                    total_lines += lc
            progress = {
                'files_total': len(files),
                'files_done': 0,
                'current_file': '',
                'current_index': 0,
                'percent': 0.0,
                'lines': 0,
                'lines_total': total_lines,
                # Estimated total lines (same as lines_total; kept for external clients wanting a stable key name)
                'expected_overall_lines': total_lines,
                'file_lines_total': 0,
                'file_lines_done': 0,
                'file_percent': 0.0,
                'file_bytes_total': 0,
                'file_bytes_done': 0,
                'bytes_total': total_bytes,
                'bytes_done': 0,
                # Estimated current file lines (updated when file starts)
                'expected_file_lines': 0,
            }
            # Persist initial raw limit selection for downstream partial stats ('' or 'none' => token/passfail-from-logs mode)
            CACHE[token] = {
                'status': 'running',
                'progress': progress,
                'rows': [],
                'dir': used_dir,
                'limit_raw': limit_raw,
            }
            all_rows: List[Dict[str, str]] = []
            # Initialize on-disk storage for rows
            try:
                _sqlite_init(token)
            except Exception:
                pass
            # Keep a live reference to rows in CACHE so other endpoints can compute partial stats
            try:
                if token in CACHE:
                    CACHE[token]['rows'] = all_rows
                    CACHE[token]['limit_raw'] = limit_raw
                else:
                    CACHE[token] = {'status': 'running', 'progress': progress, 'rows': all_rows, 'dir': used_dir, 'limit_raw': limit_raw}
            except Exception:
                pass
            bytes_done_prev = 0
            lines_done_prev = 0
            # Normalize ledger map to string paths for quick lookup
            _ledger_map = {}
            try:
                if ledger_map:
                    for k, v in ledger_map.items():
                        _ledger_map[str(Path(k))] = Path(v)
            except Exception:
                _ledger_map = {}

            # Streaming prodmode branch: keep watching ledger and process files as .ready appear, until cancelled.
            if stream_prodmode:
                processed_ready: set[str] = set()
                # Seed processed set with any provided map entries to avoid reprocessing
                for _p in _ledger_map.values():
                    try:
                        processed_ready.add(str(Path(_p)))
                    except Exception:
                        pass
                # Work queue initialized from provided files
                work_queue: List[Path] = list(files)
                def _list_non_ledger_files(root: Path) -> List[Path]:
                    try:
                        out: List[Path] = []
                        for p in root.rglob('*'):
                            try:
                                if p.is_file():
                                    # skip ledger directory
                                    if (root / 'ledger') in p.parents:
                                        continue
                                    out.append(p)
                            except Exception:
                                continue
                        return out
                    except Exception:
                        return []
                def _discover_new_ready() -> None:
                    nonlocal work_queue
                    try:
                        led_dir = Path(used_dir) / 'ledger'  # type: ignore[arg-type]
                        current: list[Path] = []
                        if led_dir.is_dir():
                            # Discovery only: do not rename here; only list .ready files
                            current = [p for p in led_dir.iterdir() if p.is_file() and p.suffix.lower() == '.ready']
                        # Build/update mapping for any unseen .ready
                        _allowed_exts = {'.txt', '.log', '.out'}
                        for rf in current:
                            key_ready = str(rf)
                            if key_ready in processed_ready:
                                continue
                            stem = rf.stem
                            # Try to find a matching file by stem under used_dir
                            chosen: Path | None = None
                            try:
                                all_files = _list_non_ledger_files(Path(used_dir))  # type: ignore[arg-type]
                                candidates = [f for f in all_files if f.stem == stem and (f.suffix.lower() in _allowed_exts or f.suffix == '')]
                                chosen = candidates[0] if candidates else None
                            except Exception:
                                chosen = None
                            if chosen and chosen.is_file():
                                _ledger_map[str(chosen)] = rf
                                work_queue.append(chosen)
                                # Do not mark processed_ready yet; only after processing+rename
                    except Exception:
                        pass
                current_index = 0
                while True:
                    # Cancel requested by 'Done' button -> finalize
                    try:
                        if CANCEL_FLAGS.get(token):
                            break
                    except Exception:
                        pass
                    if not work_queue:
                        # No work right now: refresh discovery and idle
                        _discover_new_ready()
                        try:
                            progress['current_file'] = '(waiting for ledger)'
                            progress['file_lines_total'] = 0
                            progress['file_lines_done'] = 0
                            progress['file_percent'] = 0.0
                            progress['files_total'] = progress.get('files_done', 0)  # reflect that total grows over time
                            CACHE[token]['progress'] = progress
                        except Exception:
                            pass
                        time.sleep(1.0)
                        continue
                    # Pop next file to process
                    fp = work_queue.pop(0)
                    current_index += 1
                    progress['current_file'] = str(fp)
                    progress['current_index'] = current_index
                    # Determine current file size (bytes); avoid a separate pre-count pass over lines
                    try:
                        file_size = fp.stat().st_size
                    except Exception:
                        file_size = 0
                    # Use bytes-based estimates for progress
                    progress['file_lines_total'] = 0
                    progress['expected_file_lines'] = 0
                    progress['file_lines_done'] = 0
                    progress['file_bytes_total'] = file_size
                    progress['file_bytes_done'] = 0
                    progress['file_percent'] = 0.0
                    # Provide an estimated files_total for streaming mode so UI can display progress
                    try:
                        est_total = max(1, progress.get('files_done', 0) + len(work_queue) + 1)
                    except Exception:
                        est_total = max(1, progress.get('files_done', 0) + 1)
                    progress['files_total'] = est_total
                    last_lineno = {'n': 0}
                    last_snapshot_time = {'t': 0.0}
                    def _cb_stream(lineno: int, pct: float) -> None:
                        last_lineno['n'] = lineno
                        bytes_curr = 0
                        try:
                            bytes_curr = int((pct / 100.0) * file_size)
                        except Exception:
                            bytes_curr = 0
                        total_done = bytes_done_prev + bytes_curr
                        lines_done_now = lines_done_prev + lineno
                        # In streaming mode totals are unknown; estimate overall percent from files_done,
                        # number of pending files, and current file percent so the UI can reflect progress.
                        try:
                            files_done_now = int(progress.get('files_done', 0))
                        except Exception:
                            files_done_now = 0
                        try:
                            pending = len(work_queue)
                        except Exception:
                            pending = 0
                        est_files_total = max(1, files_done_now + pending + 1)  # include current file
                        overall_pct = ((files_done_now + (pct / 100.0)) / float(est_files_total)) * 100.0
                        if overall_pct < 0.0:
                            overall_pct = 0.0
                        if overall_pct > 100.0:
                            overall_pct = 100.0
                        # Estimate target totals (lines) without pre-counting:
                        # - per-file target: scale current lines by file percent
                        # - overall target: scale overall lines by overall percent
                        try:
                            exp_file = int(round(lineno / max(pct / 100.0, 1e-6))) if pct > 0 else 0
                        except Exception:
                            exp_file = 0
                        try:
                            exp_overall = int(round(lines_done_now / max(overall_pct / 100.0, 1e-6))) if overall_pct > 0 else 0
                        except Exception:
                            exp_overall = 0
                        progress.update({
                            'percent': overall_pct,
                            'lines': lines_done_now,
                            'file_lines_done': lineno,
                            'file_bytes_done': bytes_curr,
                            'file_bytes_total': file_size,
                            'file_percent': pct,
                            'bytes_done': total_done,
                            # keep an estimated files_total for UI display
                            'files_total': est_files_total,
                            'expected_overall_lines': exp_overall,
                            'expected_file_lines': exp_file,
                        })
                        try:
                            if token not in CACHE:
                                CACHE[token] = {'status': 'running', 'progress': progress, 'rows': [], 'dir': used_dir}
                            else:
                                CACHE[token]['progress'] = progress
                        except Exception:
                            pass
                        try:
                            if CANCEL_FLAGS.get(token):
                                raise RuntimeError('cancelled')
                        except Exception:
                            pass
                        now = time.time()
                        # Throttle fdvtable snapshots by time and growth
                        if now - last_snapshot_time['t'] > _SNAPSHOT_INTERVAL_SECS:
                            last_snapshot_time['t'] = now
                            try:
                                lr = (CACHE.get(token, {}).get('limit_raw') or '').strip().lower()
                                pf_mode = (lr in ('', 'none', 'default'))
                                limit_val = 1e9 if pf_mode else float(lr)
                                # Aggregate over all rows so previously processed files remain visible in the table
                                # Only rebuild if row growth since last snapshot is significant
                                rows_total = len(all_rows)
                                last_rows = 0
                                try:
                                    last_rows = int(progress.get('_last_snapshot_rows', 0) or 0)
                                except Exception:
                                    last_rows = 0
                                if rows_total - last_rows < _SNAPSHOT_MIN_ROWS_DELTA:
                                    raise RuntimeError('skip-snapshot-small-delta')
                                stats_small = stats_by_fdv_with_splits(all_rows, limit=limit_val, passfail_mode=pf_mode) if all_rows else []
                                table_html = ''
                                try:
                                    table_html = render_template('fdv2_report_table.html', token=token, stats=stats_small, used_dir=used_dir, limit=(None if pf_mode else limit_val))
                                except Exception:
                                    table_html = ''
                                if table_html:
                                    with SNAP_LOCK:
                                        SNAPSHOTS[token] = {'fdvtable_html': table_html, 'updated': now, 'limit_key': lr or 'none'}
                                        try:
                                            progress['_last_snapshot_rows'] = rows_total
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    # Parse the file (normal path)
                    try:
                        if process_file is not None:
                            r, _kept, _markers = process_file(fp, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT, progress=True, progress_cb=_cb_stream)  # type: ignore[arg-type]
                            # Defensive post-filter: remove any rows whose raw line includes MONITOR or SHMOO
                            try:
                                _filt = []
                                for _row in r:
                                    _rl = (_row.get('raw_line') or _row.get('raw') or '')
                                    _u = _rl.upper()
                                    if 'MONITOR' in _u or 'SHMOO' in _u:
                                        continue
                                    _filt.append(_row)
                                r = _filt
                            except Exception:
                                pass
                        else:
                            r = []
                            _candidate_fp = fp
                            try:
                                enc_guess = _detect_file_encoding(fp)
                                if enc_guess == 'utf-16':
                                    try:
                                        print(f"[encoding] Detected possible UTF-16 file: {fp}")
                                    except Exception:
                                        pass
                                    tmp_conv = _convert_utf16_to_temp_utf8(fp)
                                    if tmp_conv and tmp_conv.exists():
                                        _candidate_fp = tmp_conv
                            except Exception:
                                pass
                            try:
                                with open(_candidate_fp, 'r', encoding='utf-8', errors='replace') as f:
                                    for i, line in enumerate(f, start=1):
                                        _ls = line.lstrip()
                                        if _ls.startswith('FDV OUTPUT'):
                                            try:
                                                import re as _re_mon
                                                if _re_mon.search(r"\b(MONITOR|SHMOO)\b", _ls, _re_mon.IGNORECASE):
                                                    continue
                                            except Exception:
                                                _uu = _ls.upper()
                                                if 'MONITOR' in _uu or 'SHMOO' in _uu:
                                                    continue
                                            r.append({'raw_line': _ls.rstrip('\n'), 'line_number': str(i), 'fdv_file': str(fp)})
                                        if i % 100000 == 0:
                                            _cb_stream(i, 0.0)
                            except Exception as e:  # pragma: no cover
                                try:
                                    print(f"[fallback-parse] Failed reading {fp}: {e}")
                                except Exception:
                                    pass
                    except Exception:
                        r = r if 'r' in locals() and isinstance(r, list) else []
                    # Ensure fdv_file is set
                    try:
                        for rr in r:
                            if not (rr.get('fdv_file') or rr.get('fdv')):
                                rr['fdv_file'] = str(fp)
                    except Exception:
                        pass
                    # Attach time labels (reuse logic from non-stream path)
                    # Compute run/list info
                    def _extract_run_parts_from_filename(p: Path) -> Tuple[str, str]:
                        name = p.name
                        up = name
                        import re as _re
                        m = _re.search(r"_fdvrun_(.+?)_tb_set_utility_([^\.]+)", up, flags=_re.IGNORECASE)
                        if m:
                            return (m.group(1), m.group(2))
                        m2 = _re.search(r"fdvrun[_\-]([^\-]+?)[_\-]tb_set_utility[_\-]([^\.]+)", up, flags=_re.IGNORECASE)
                        if m2:
                            return (m2.group(1), m2.group(2))
                        return ('', '')
                    run_name, list_name = _extract_run_parts_from_filename(fp)
                    start_dt = None
                    end_dt = None
                    list_name_from_marker = ''
                    start_raw_str = ''
                    end_raw_str = ''
                    try:
                        from datetime import datetime as _dt
                        tm = next((m for m in (_markers or []) if isinstance(m, dict) and m.get('type') == 'test_time'), None)
                        if tm:
                            list_name_from_marker = (tm.get('list_name') or '').strip()
                            start_raw_str = (tm.get('start_raw') or '').strip()
                            end_raw_str = (tm.get('end_raw') or '').strip()
                            s_iso = (tm.get('start_iso') or '').strip()
                            e_iso = (tm.get('end_iso') or '').strip()
                            if s_iso:
                                try:
                                    start_dt = _dt.fromisoformat(s_iso)
                                except Exception:
                                    start_dt = None
                            if e_iso:
                                try:
                                    end_dt = _dt.fromisoformat(e_iso)
                                except Exception:
                                    end_dt = None
                    except Exception:
                        start_dt = start_dt or None
                        end_dt = end_dt or None
                    if not start_raw_str and not end_raw_str:
                        try:
                            import time as _t
                            _mt = fp.stat().st_mtime
                            _lt = _t.localtime(_mt)
                            start_raw_str = f"{_lt.tm_year:04d}_{_lt.tm_mon}_{_lt.tm_mday} {_lt.tm_hour:02d}:{_lt.tm_min:02d}:{_lt.tm_sec:02d}"
                            end_raw_str = start_raw_str
                        except Exception:
                            pass
                    if not start_raw_str and not end_raw_str:
                        try:
                            import time as _t2
                            _mt2 = fp.stat().st_mtime
                            _lt2 = _t2.localtime(_mt2)
                            start_raw_str = f"{_lt2.tm_year:04d}_{_lt2.tm_mon}_{_lt2.tm_mday} {_lt2.tm_hour:02d}:{_lt2.tm_min:02d}:{_lt2.tm_sec:02d}"
                            end_raw_str = start_raw_str
                        except Exception:
                            pass
                    def _fmt_duration_secs(s_dt, e_dt) -> int:
                        try:
                            if not s_dt or not e_dt:
                                return -1
                            delta = e_dt - s_dt
                            secs = int(delta.total_seconds())
                            if secs < 0:
                                secs = 0
                            return secs
                        except Exception:
                            return -1
                    dur_secs = _fmt_duration_secs(start_dt, end_dt)
                    testtime_label = ''
                    try:
                        ln_disp = (list_name or '').strip()
                        if not ln_disp and list_name_from_marker:
                            ln_tmp = list_name_from_marker
                            try:
                                import re as _re
                                ln_tmp = _re.sub(r"^\d+_", "", ln_tmp)
                                ln_tmp = _re.sub(r"(?i)^tb_set_utility_", "", ln_tmp)
                            except Exception:
                                pass
                            ln_disp = ln_tmp.strip()
                        if ln_disp or (dur_secs is not None and dur_secs >= 0):
                            secs_txt = str(dur_secs if dur_secs is not None and dur_secs >= 0 else '')
                            testtime_label = f"{ln_disp}:: = {secs_txt}".strip()
                    except Exception:
                        testtime_label = ''
                    try:
                        for rr in r:
                            fdvtest = (rr.get('fdv_file') or '').strip()
                            if fdvtest:
                                import os as _os
                                fdvtest = _os.path.basename(fdvtest)
                                if fdvtest.lower().endswith('.fdv'):
                                    fdvtest = fdvtest[:-4]
                            if testtime_label or (fdvtest and (dur_secs is not None and dur_secs >= 0)):
                                try:
                                    ln_disp_local = testtime_label.split('::')[0] if testtime_label else (list_name or '').strip()
                                except Exception:
                                    ln_disp_local = (list_name or '').strip()
                                secs_txt = str(dur_secs if dur_secs is not None and dur_secs >= 0 else '')
                                rr['testtime_label'] = f"{(ln_disp_local or '').strip()}::{fdvtest} = {secs_txt}".strip()
                                if dur_secs is not None and dur_secs >= 0:
                                    rr['testtime_seconds'] = str(dur_secs)
                            if start_raw_str:
                                rr['test_start'] = start_raw_str
                            if end_raw_str:
                                rr['test_end'] = end_raw_str
                            if list_name:
                                rr['fdvlistname'] = list_name
                            if run_name:
                                rr['fdvtestrun'] = run_name
                    except Exception:
                        pass
                    all_rows.extend(r)
                    # Persist this batch to D:\ and trim memory footprint
                    try:
                        if r:
                            _sqlite_insert_rows(token, r)
                            if len(all_rows) > 50000:
                                all_rows[:] = all_rows[-20000:]
                    except Exception:
                        pass
                    # finalize this file's contribution
                    lines_done_prev += last_lineno['n']
                    bytes_done_prev += file_size
                    progress['files_done'] = progress.get('files_done', 0) + 1
                    progress['bytes_done'] = bytes_done_prev
                    progress['lines'] = lines_done_prev
                    # Update cache and partial stats
                    try:
                        if token in CACHE:
                            CACHE[token]['progress'] = progress
                            CACHE[token]['progress']['last_file_completed_at'] = time.time()
                            try:
                                lr = (CACHE[token].get('limit_raw') or '').strip().lower()
                                if lr in ('', 'none', 'default'):
                                    limit_for_stats = 1e9
                                    pf_mode = True
                                else:
                                    try:
                                        limit_for_stats = float(lr)
                                        pf_mode = False
                                    except Exception:
                                        limit_for_stats = 1e9
                                        pf_mode = True
                                partial_stats = stats_by_fdv_with_splits(all_rows, limit=limit_for_stats, passfail_mode=pf_mode)
                                seen_pf = set()
                                fdv_order_pf: List[str] = []
                                for _r in partial_stats:
                                    _f = _r.get('fdv_file','') or ''
                                    if _f and _f not in seen_pf:
                                        seen_pf.add(_f)
                                        fdv_order_pf.append(_f)
                                CACHE[token]['stats'] = partial_stats
                                CACHE[token]['fdv_order'] = fdv_order_pf
                            except Exception:
                                pass
                        else:
                            CACHE[token] = {'status': 'running', 'progress': progress, 'rows': all_rows, 'dir': used_dir}
                    except Exception:
                        pass
                    # Rename ledger .ready to .done
                    try:
                        key = str(Path(fp))
                        if key in _ledger_map:
                            ready_path = _ledger_map.get(key)
                            if ready_path and ready_path.exists():
                                done_path = ready_path.with_suffix('.done')
                                try:
                                    os.replace(str(ready_path), str(done_path))
                                except Exception:
                                    try:
                                        ready_path.rename(done_path)
                                    except Exception:
                                        pass
                                processed_ready.add(str(ready_path))
                    except Exception:
                        pass
                    # After processing this item, discover new ones immediately (reduces idle time)
                    _discover_new_ready()
                # Cancelled: finalize stats and mark done
                try:
                    for i, rr in enumerate(all_rows):
                        if '_idx' not in rr:
                            rr['_idx'] = i
                    try:
                        _apply_guide_annotations(all_rows)
                    except Exception:
                        pass
                    for i, rr in enumerate(all_rows):
                        try:
                            wl_v = _extract_wl_value(rr)
                            if wl_v is not None:
                                rr['wl_canonical'] = str(wl_v)
                        except Exception:
                            pass
                        try:
                            pg_v = _extract_page_value(rr)
                            if pg_v is not None:
                                rr['page_canonical'] = str(pg_v)
                        except Exception:
                            pass
                        try:
                            pa = _extract_plane_addr(rr)
                            if pa:
                                rr['plane_addr'] = pa
                                rr['plane_addr_canonical'] = pa
                        except Exception:
                            pass
                        try:
                            bv = _extract_blk_value(rr)
                            if bv is not None:
                                rr['blk'] = str(bv)
                                rr['blk_addr'] = str(bv)
                                rr['blk_canonical'] = str(bv)
                        except Exception:
                            pass
                    # Final stats (using current limit mode)
                    lr_final = (limit_raw or '').strip().lower()
                    if lr_final in ('', 'none', 'default'):
                        limit_for_stats_final = 1e9
                        pf_mode_final = True
                    else:
                        try:
                            limit_for_stats_final = float(lr_final)
                            pf_mode_final = False
                        except Exception:
                            limit_for_stats_final = 1e9
                            pf_mode_final = True
                    stats = stats_by_fdv_with_splits(all_rows, limit=limit_for_stats_final, passfail_mode=pf_mode_final)
                    seen = set()
                    fdv_order: List[str] = []
                    for r in stats:
                        f = r.get('fdv_file','') or ''
                        if f and f not in seen:
                            seen.add(f)
                            fdv_order.append(f)
                    CACHE[token].update({'rows': all_rows[-20000:], 'stats': stats, 'fdv_order': fdv_order, 'status': 'done', 'limit_raw': limit_raw})
                    try:
                        with JOBS_LOCK:
                            for jid, rec in JOBS.items():
                                if rec.get('token') == token:
                                    if 'ended_at' not in rec:
                                        rec['ended_at'] = time.time()
                                    break
                    except Exception:
                        pass
                except Exception:
                    pass
                return
            # In non-stream mode, when prodmode flag is set, process only files that have a corresponding ledger .ready entry
            files_effective: List[Path] = files
            try:
                if prodmode and _ledger_map:
                    files_effective = [fp for fp in files if str(Path(fp)) in _ledger_map]
            except Exception:
                files_effective = files
            for idx, fp in enumerate(files_effective, start=1):
                # Honor cancel (from 'Done' button) between files
                try:
                    if CANCEL_FLAGS.get(token):
                        break
                except Exception:
                    pass
                progress['current_file'] = str(fp)
                progress['current_index'] = idx
                file_size = sizes[idx - 1] if idx - 1 < len(sizes) else 0
                file_lines_total = line_counts[idx - 1] if idx - 1 < len(line_counts) else 0
                progress['file_lines_total'] = file_lines_total
                progress['expected_file_lines'] = file_lines_total
                progress['file_lines_done'] = 0
                progress['file_bytes_total'] = file_size
                progress['file_bytes_done'] = 0
                progress['file_percent'] = 0.0
                last_lineno = {'n': 0}

                last_snapshot_time = {'t': 0.0}
                def _cb(lineno: int, pct: float) -> None:
                    last_lineno['n'] = lineno
                    # Estimate bytes processed in this file using pct and file_size
                    bytes_curr = 0
                    try:
                        bytes_curr = int((pct / 100.0) * file_size)
                    except Exception:
                        bytes_curr = 0
                    total_done = bytes_done_prev + bytes_curr
                    # Prefer line-based overall percentage when we know totals
                    lines_done_now = lines_done_prev + lineno
                    if total_lines > 0:
                        overall_pct = (float(lines_done_now) / float(total_lines) * 100.0)
                    else:
                        overall_pct = (float(total_done) / float(total_bytes) * 100.0) if total_bytes > 0 else 0.0
                    progress.update({
                        'percent': overall_pct,
                        'lines': lines_done_now,
                        'file_lines_done': lineno,
                        'file_bytes_done': bytes_curr,
                        'file_bytes_total': file_size,
                        'file_percent': pct,
                        'bytes_done': total_done,
                        # Keep expected values stable (only change if unknown (0) and we can approximate)
                        'expected_overall_lines': total_lines or progress.get('expected_overall_lines', 0),
                        'expected_file_lines': file_lines_total or progress.get('expected_file_lines', 0),
                    })
                    # Store back into CACHE for polling clients (tolerate missing entry)
                    try:
                        if token not in CACHE:
                            CACHE[token] = {
                                'status': 'running',
                                'progress': progress,
                                'rows': [],
                                'dir': used_dir,
                            }
                        else:
                            CACHE[token]['progress'] = progress
                    except Exception:
                        pass
                    # Allow cancel mid-file: signal via exception that outer parse_file may catch
                    try:
                        if CANCEL_FLAGS.get(token):
                            raise RuntimeError('cancelled')
                    except Exception:
                        pass
                    # Throttled partial stats snapshot for fast table refresh (every ~1.2s)
                    now = time.time()
                    # Throttle snapshot rebuilds; also skip if rows grew only slightly
                    if now - last_snapshot_time['t'] > max(1.2, _SNAPSHOT_INTERVAL_SECS):
                        last_snapshot_time['t'] = now
                        try:
                            # Build small subset stats (limit/passfail mode per current limit_raw)
                            lr = (CACHE.get(token, {}).get('limit_raw') or '').strip().lower()
                            pf_mode = (lr in ('', 'none', 'default'))
                            limit_val = 1e9 if pf_mode else float(lr)
                            # Aggregate over all rows so previously processed files remain visible in the table
                            rows_total = len(all_rows)
                            last_rows = int(progress.get('_last_snapshot_rows', 0) or 0)
                            if rows_total - last_rows < _SNAPSHOT_MIN_ROWS_DELTA:
                                raise RuntimeError('skip-snapshot-small-delta')
                            stats_small = stats_by_fdv_with_splits(all_rows, limit=limit_val, passfail_mode=pf_mode) if all_rows else []
                            table_html = ''
                            try:
                                table_html = render_template('fdv2_report_table.html', token=token, stats=stats_small, used_dir=used_dir, limit=(None if pf_mode else limit_val))
                            except Exception:
                                table_html = ''
                            if table_html:
                                with SNAP_LOCK:
                                    SNAPSHOTS[token] = {'fdvtable_html': table_html, 'updated': now, 'limit_key': lr or 'none'}
                                    try:
                                        progress['_last_snapshot_rows'] = rows_total
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                try:
                    if process_file is not None:
                        r, _kept, _markers = process_file(fp, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT, progress=True, progress_cb=_cb)  # type: ignore[arg-type]
                        # Defensive post-filter: remove any rows whose raw line still contains MONITOR or SHMOO
                        try:
                            _filt = []
                            for _row in r:
                                _rl = (_row.get('raw_line') or _row.get('raw') or '')
                                _u = _rl.upper()
                                if 'MONITOR' in _u or 'SHMOO' in _u:
                                    continue
                                _filt.append(_row)
                            r = _filt
                        except Exception:
                            pass
                    else:
                        # Fallback: very slow path without structured parsing
                        r = []
                        _candidate_fp = fp
                        try:
                            enc_guess = _detect_file_encoding(fp)
                            if enc_guess == 'utf-16':
                                try:
                                    print(f"[encoding] Detected possible UTF-16 file: {fp}")
                                except Exception:
                                    pass
                                tmp_conv = _convert_utf16_to_temp_utf8(fp)
                                if tmp_conv and tmp_conv.exists():
                                    _candidate_fp = tmp_conv
                        except Exception:
                            pass
                        try:
                            with open(_candidate_fp, 'r', encoding='utf-8', errors='replace') as f:
                                for i, line in enumerate(f, start=1):
                                    _ls = line.lstrip()
                                    if _ls.startswith('FDV OUTPUT'):
                                        # Skip any FDV OUTPUT line that includes MONITOR or SHMOO (case-insensitive)
                                        try:
                                            import re as _re_mon
                                            if _re_mon.search(r"\b(MONITOR|SHMOO)\b", _ls, _re_mon.IGNORECASE):
                                                continue
                                        except Exception:
                                            _uu = _ls.upper()
                                            if 'MONITOR' in _uu or 'SHMOO' in _uu:
                                                continue
                                        r.append({'raw_line': _ls.rstrip('\n'), 'line_number': str(i), 'fdv_file': str(fp)})
                                    if i % 100000 == 0:
                                        _cb(i, 0.0)
                        except Exception as e:  # pragma: no cover
                            try:
                                print(f"[fallback-parse] Failed reading {fp}: {e}")
                            except Exception:
                                pass
                except Exception as e_parse:
                    # If cancelled mid-file, proceed with whatever rows gathered so far
                    r = r if 'r' in locals() and isinstance(r, list) else []
                # Ensure fdv_file is set for all rows (some parsers may omit)
                try:
                    for rr in r:
                        if not (rr.get('fdv_file') or rr.get('fdv')):
                            rr['fdv_file'] = str(fp)
                except Exception:
                    pass
                # Derive fdv list/run names from filename
                def _extract_run_parts_from_filename(p: Path) -> Tuple[str, str]:
                    name = p.name
                    up = name
                    import re as _re
                    m = _re.search(r"_fdvrun_(.+?)_tb_set_utility_([^\.]+)", up, flags=_re.IGNORECASE)
                    if m:
                        return (m.group(1), m.group(2))
                    # Fallbacks: hyphens or different separators
                    m2 = _re.search(r"fdvrun[_\-]([^\-]+?)[_\-]tb_set_utility[_\-]([^\.]+)", up, flags=_re.IGNORECASE)
                    if m2:
                        return (m2.group(1), m2.group(2))
                    return ('', '')
                run_name, list_name = _extract_run_parts_from_filename(fp)
                # Compute Test Start/End and duration from parser markers (single pass)
                start_dt = None
                end_dt = None
                list_name_from_marker = ''
                start_raw_str = ''
                end_raw_str = ''
                try:
                    from datetime import datetime as _dt
                    tm = next((m for m in (_markers or []) if isinstance(m, dict) and m.get('type') == 'test_time'), None)
                    if tm:
                        list_name_from_marker = (tm.get('list_name') or '').strip()
                        start_raw_str = (tm.get('start_raw') or '').strip()
                        end_raw_str = (tm.get('end_raw') or '').strip()
                        s_iso = (tm.get('start_iso') or '').strip()
                        e_iso = (tm.get('end_iso') or '').strip()
                        if s_iso:
                            try:
                                start_dt = _dt.fromisoformat(s_iso)
                            except Exception:
                                start_dt = None
                        if e_iso:
                            try:
                                end_dt = _dt.fromisoformat(e_iso)
                            except Exception:
                                end_dt = None
                except Exception:
                    start_dt = start_dt or None
                    end_dt = end_dt or None
                # Fallback: if no marker-derived raw strings, use file mtime as both start/end (better than blank)
                if not start_raw_str and not end_raw_str:
                    try:
                        import time as _t
                        _mt = fp.stat().st_mtime
                        _lt = _t.localtime(_mt)
                        start_raw_str = f"{_lt.tm_year:04d}_{_lt.tm_mon}_{_lt.tm_mday} {_lt.tm_hour:02d}:{_lt.tm_min:02d}:{_lt.tm_sec:02d}"
                        end_raw_str = start_raw_str
                    except Exception:
                        pass
                # Fallback: if neither raw start nor end captured, use file mtime once
                if not start_raw_str and not end_raw_str:
                    try:
                        import time as _t2
                        _mt2 = fp.stat().st_mtime
                        _lt2 = _t2.localtime(_mt2)
                        start_raw_str = f"{_lt2.tm_year:04d}_{_lt2.tm_mon}_{_lt2.tm_mday} {_lt2.tm_hour:02d}:{_lt2.tm_min:02d}:{_lt2.tm_sec:02d}"
                        end_raw_str = start_raw_str
                    except Exception:
                        pass
                # Build label string: <fdvlistname>::<fdvtest> = seconds
                def _fmt_duration_secs(s_dt, e_dt) -> int:
                    try:
                        if not s_dt or not e_dt:
                            return -1
                        delta = e_dt - s_dt
                        secs = int(delta.total_seconds())
                        if secs < 0:
                            secs = 0
                        return secs
                    except Exception:
                        return -1
                dur_secs = _fmt_duration_secs(start_dt, end_dt)
                # We'll attach a per-row label using the row's fdv_file (fdvtest)
                # Compute the fdvlist display name once here
                testtime_label = ''
                try:
                    # Prefer filename-derived list; fallback to marker-derived list name
                    ln_disp = (list_name or '').strip()
                    if not ln_disp and list_name_from_marker:
                        # strip any numeric prefix like '34_' and leading 'tb_set_utility_'
                        ln_tmp = list_name_from_marker
                        try:
                            import re as _re
                            ln_tmp = _re.sub(r"^\d+_", "", ln_tmp)
                            ln_tmp = _re.sub(r"(?i)^tb_set_utility_", "", ln_tmp)
                        except Exception:
                            pass
                        ln_disp = ln_tmp.strip()
                    # Only build labels when we have a duration value available (>=0 includes 0s)
                    if ln_disp or (dur_secs is not None and dur_secs >= 0):
                        secs_txt = str(dur_secs if dur_secs is not None and dur_secs >= 0 else '')
                        testtime_label = f"{ln_disp}:: = {secs_txt}".strip()
                except Exception:
                    testtime_label = ''
                # Attach label to each row for this file for downstream grouping
                try:
                    for rr in r:
                        # Derive fdvtest (basename without extension) from rr['fdv_file']
                        fdvtest = (rr.get('fdv_file') or '').strip()
                        if fdvtest:
                            import os as _os
                            fdvtest = _os.path.basename(fdvtest)
                            # strip extension if present
                            if fdvtest.lower().endswith('.fdv'):
                                fdvtest = fdvtest[:-4]
                        # Build and attach label if we have either list name or duration
                        if testtime_label or (fdvtest and (dur_secs is not None and dur_secs >= 0)):
                            try:
                                ln_disp_local = testtime_label.split('::')[0] if testtime_label else (list_name or '').strip()
                            except Exception:
                                ln_disp_local = (list_name or '').strip()
                            secs_txt = str(dur_secs if dur_secs is not None and dur_secs >= 0 else '')
                            rr['testtime_label'] = f"{(ln_disp_local or '').strip()}::{fdvtest} = {secs_txt}".strip()
                            if dur_secs is not None and dur_secs >= 0:
                                rr['testtime_seconds'] = str(dur_secs)
                        # Attach raw start/end strings for visibility
                        if start_raw_str:
                            rr['test_start'] = start_raw_str
                        if end_raw_str:
                            rr['test_end'] = end_raw_str
                        if list_name:
                            rr['fdvlistname'] = list_name
                        # Keep run_name in case it's useful elsewhere, but label uses fdvtest
                        if run_name:
                            rr['fdvtestrun'] = run_name
                except Exception:
                    pass
                all_rows.extend(r)
                # Persist this batch and trim memory
                try:
                    if r:
                        _sqlite_insert_rows(token, r)
                        if len(all_rows) > 50000:
                            all_rows[:] = all_rows[-20000:]
                except Exception:
                    pass
                # finalize this file's contribution
                lines_done_prev += last_lineno['n']
                bytes_done_prev += file_size
                progress['files_done'] = idx
                progress['bytes_done'] = bytes_done_prev
                progress['lines'] = lines_done_prev
                # Finalize overall percent using lines if available
                if total_lines > 0:
                    progress['percent'] = (float(lines_done_prev) / float(total_lines) * 100.0)
                else:
                    progress['percent'] = (float(bytes_done_prev) / float(total_bytes) * 100.0) if total_bytes > 0 else progress.get('percent', 0.0)
                try:
                    if token in CACHE:
                        CACHE[token]['progress'] = progress
                        # Mark a checkpoint to let the UI know a file finished so it can refresh the table
                        CACHE[token]['progress']['last_file_completed_at'] = time.time()
                        # Compute and store partial stats for faster fdvtable responses
                        try:
                            # Determine pass/fail mode based on stored limit_raw
                            lr = (CACHE[token].get('limit_raw') or '').strip().lower()
                            if lr in ('', 'none', 'default'):
                                limit_for_stats = 1e9
                                pf_mode = True
                            else:
                                try:
                                    limit_for_stats = float(lr)
                                    pf_mode = False
                                except Exception:
                                    limit_for_stats = 1e9
                                    pf_mode = True
                            partial_stats = stats_by_fdv_with_splits(all_rows, limit=limit_for_stats, passfail_mode=pf_mode)
                            # Build ordered unique fdv list
                            seen_pf = set()
                            fdv_order_pf: List[str] = []
                            for _r in partial_stats:
                                _f = _r.get('fdv_file','') or ''
                                if _f and _f not in seen_pf:
                                    seen_pf.add(_f)
                                    fdv_order_pf.append(_f)
                            CACHE[token]['stats'] = partial_stats
                            CACHE[token]['fdv_order'] = fdv_order_pf
                        except Exception:
                            pass
                    else:
                        CACHE[token] = {
                            'status': 'running',
                            'progress': progress,
                            'rows': all_rows,
                            'dir': used_dir,
                        }
                except Exception:
                    pass
                # In prodmode, after finishing a file, rename its ledger .ready to .done if mapped
                try:
                    if prodmode and _ledger_map:
                        key = str(Path(fp))
                        if key in _ledger_map:
                            ready_path = _ledger_map.get(key)
                            if ready_path and ready_path.exists():
                                done_path = ready_path.with_suffix('.done')
                                try:
                                    os.replace(str(ready_path), str(done_path))
                                except Exception:
                                    # Attempt fallback rename via Path
                                    try:
                                        ready_path.rename(done_path)
                                    except Exception:
                                        pass
                except Exception:
                    pass
            # After parsing, annotate rows with indices for later raw-line lookup
            for i, rr in enumerate(all_rows):
                if '_idx' not in rr:
                    rr['_idx'] = i
            # Apply guide-compliant annotations (testname, pagetype, pagemap, deck, step, plane_group, PHYPAGE; fill FUSEID/PR per file)
            try:
                _apply_guide_annotations(all_rows)
            except Exception:
                pass
                # Apply canonical WL/PAGE extraction per guide for downstream use
            for i, rr in enumerate(all_rows):
                # Apply canonical WL/PAGE extraction per guide for downstream use
                try:
                    wl_v = _extract_wl_value(rr)
                    if wl_v is not None:
                        rr['wl_canonical'] = str(wl_v)
                except Exception:
                    pass
                try:
                    pg_v = _extract_page_value(rr)
                    if pg_v is not None:
                        rr['page_canonical'] = str(pg_v)
                except Exception:
                    pass
                # Apply canonical plane and block extraction
                try:
                    pa = _extract_plane_addr(rr)
                    if pa:
                        rr['plane_addr'] = pa
                        rr['plane_addr_canonical'] = pa
                except Exception:
                    pass
                try:
                    bv = _extract_blk_value(rr)
                    if bv is not None:
                        rr['blk'] = str(bv)
                        rr['blk_addr'] = str(bv)
                        rr['blk_canonical'] = str(bv)
                except Exception:
                    pass
            # Final stats using selected limit_raw (token/log mode vs threshold)
            lr_final = (limit_raw or '').strip().lower()
            if lr_final in ('', 'none', 'default'):
                limit_for_stats_final = 1e9
                pf_mode_final = True
            else:
                try:
                    limit_for_stats_final = float(lr_final)
                    pf_mode_final = False
                except Exception:
                    limit_for_stats_final = 1e9
                    pf_mode_final = True
            stats = stats_by_fdv_with_splits(all_rows, limit=limit_for_stats_final, passfail_mode=pf_mode_final)
            # Build ordered unique fdv list
            seen = set()
            fdv_order: List[str] = []
            for r in stats:
                f = r.get('fdv_file','') or ''
                if f and f not in seen:
                    seen.add(f)
                    fdv_order.append(f)
            CACHE[token].update({'rows': all_rows[-20000:], 'stats': stats, 'fdv_order': fdv_order, 'status': 'done', 'limit_raw': limit_raw})
            # Mark job end time
            try:
                with JOBS_LOCK:
                    for jid, rec in JOBS.items():
                        if rec.get('token') == token:
                            if 'ended_at' not in rec:
                                rec['ended_at'] = time.time()
                            break
            except Exception:
                pass
        except Exception as e:
            try:
                if token in CACHE:
                    CACHE[token].update({'status': 'error', 'error': str(e)})
                else:
                    CACHE[token] = {'status': 'error', 'error': str(e), 'progress': {}, 'rows': [], 'dir': used_dir}
                # Mark job end time (error)
                try:
                    with JOBS_LOCK:
                        for jid, rec in JOBS.items():
                            if rec.get('token') == token:
                                if 'ended_at' not in rec:
                                    rec['ended_at'] = time.time()
                                break
                except Exception:
                    pass
            except Exception:
                pass

    th = threading.Thread(target=job, name=f"fdv-parse-{token[:6]}", daemon=True)
    th.start()


@app.route('/', methods=['GET','POST'])
def report_home():
    if request.method == 'POST':
        dirpath = (request.form.get('dirpath') or '').strip()
        prod_raw = (request.form.get('prodmode') or '').strip().lower()
        prodmode = prod_raw in ('1','true','on','yes')
        used_dir = None
        try:
            file_list: List[Path] = []
            ledger_map: Dict[str, Path] = {}
            if dirpath:
                dp = Path(dirpath)
                if not dp.exists() or not dp.is_dir():
                    flash('Directory not found or not a directory.')
                    return redirect(url_for('report_home'))
                used_dir = str(dp)
                if prodmode:
                    # One-time normalization: when enabling prodmode from landing page,
                    # convert any lingering .done markers back to .ready so they can be processed again.
                    # This is the ONLY place such renames occur.
                    led_dir = dp / 'ledger'
                    try:
                        if led_dir.is_dir():
                            cnt = 0
                            for p in led_dir.iterdir():
                                try:
                                    if p.is_file() and p.suffix.lower() == '.done':
                                        target = p.with_suffix('.ready')
                                        if not target.exists():
                                            p.rename(target)
                                            cnt += 1
                                except Exception:
                                    continue
                            try:
                                if cnt:
                                    app.logger.info("prodmode normalization: renamed %d .done -> .ready in %s", cnt, str(led_dir))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Ledger-based selection: scan for *.ready files only
                    ready_files: List[Path] = []
                    try:
                        if led_dir.is_dir():
                            # Only list .ready files
                            ready_files = sorted([p for p in led_dir.iterdir() if p.is_file() and p.suffix.lower() == '.ready'])
                    except Exception:
                        ready_files = []
                    # Build an index of files in dp (recursive) by stem
                    all_files = _list_files([dp])
                    stem_index: Dict[str, List[Path]] = {}
                    for f in all_files:
                        # exclude ledger directory files from processing
                        try:
                            if 'ledger' in f.parts and (dp / 'ledger') in f.parents:
                                continue
                        except Exception:
                            pass
                        stem_index.setdefault(f.stem, []).append(f)
                    selected: List[Path] = []
                    # Accept multiple raw log extensions commonly used
                    _allowed_exts = {'.txt', '.log', '.out'}
                    for rf in ready_files:
                        base = rf.stem  # filename without .ready
                        candidates = [c for c in stem_index.get(base, []) if c.suffix.lower() in _allowed_exts or c.suffix == '']
                        chosen = candidates[0] if candidates else None
                        if chosen is not None and chosen.is_file():
                            selected.append(chosen)
                            ledger_map[str(chosen)] = rf
                    if not selected:
                        flash('No matching ledger .ready files found under directory/ledger.')
                        return redirect(url_for('report_home'))
                    file_list = selected
                else:
                    file_list = _list_files([dp])
            else:
                # When prodmode is enabled, we require a directory with a ledger folder; do not accept uploads
                if prodmode:
                    flash('Production mode requires a directory with a ledger folder (uploads are not allowed).')
                    return redirect(url_for('report_home'))
                files = request.files.getlist('dirfiles') or request.files.getlist('files')
                if not files:
                    flash('Please enter a directory or select one or more files.')
                    return redirect(url_for('report_home'))
                try:
                    # Always create upload temp under configured tempdir (D:\\fdv_tmp by default)
                    tmp_dir = Path(tempfile.mkdtemp(prefix='fdv_run_', dir=tempfile.gettempdir()))
                except OSError as e:
                    # Likely out of disk space or unwritable temp dir
                    try:
                        app.logger.error('Failed to create temp dir for uploads: %s', e)
                    except Exception:
                        pass
                    flash('Cannot create temporary folder for uploads (disk full or unwritable TEMP). Free up space or set FDV_REPORT2_TMPDIR to a writable path and retry.')
                    return redirect(url_for('report_home'))
                saved: List[Path] = []
                try:
                    for i, f in enumerate(files):
                        name = f.filename or f"fdv_{i}.txt"
                        base = Path(name).name or f"fdv_{i}.txt"
                        dst = tmp_dir / f"{i:05d}_{base}"
                        # Write uploaded content to disk; catch disk errors
                        try:
                            dst.write_bytes(f.read())
                        except OSError as e:
                            try:
                                app.logger.error('Failed saving upload %s: %s', base, e)
                            except Exception:
                                pass
                            flash('Failed writing uploaded file (disk full?). Free up space and retry.')
                            return redirect(url_for('report_home'))
                        saved.append(dst)
                except Exception:
                    flash('Unexpected error saving uploads. Please retry.')
                    return redirect(url_for('report_home'))
                file_list = saved
                used_dir = str(tmp_dir)
            if not file_list:
                flash('No files found to parse.')
                return redirect(url_for('report_home'))
            # Capture user-provided RBER limit (blank or 'none' => pass/fail-from-logs mode)
            lim_raw = (request.form.get('limit') or '').strip()
            user_jobname = (request.form.get('jobname') or '').strip()
            token = uuid.uuid4().hex
            CACHE[token] = {'status': 'queued', 'progress': {'files_total': len(file_list), 'files_done': 0, 'percent': 0.0, 'lines': 0}, 'dir': used_dir, 'prodmode': prodmode}
            CANCEL_FLAGS[token] = False
            _start_parse_job(token, file_list, used_dir, lim_raw, prodmode=prodmode, ledger_map=ledger_map if ledger_map else None)
            job_id = _create_job_id(token, lim_raw)
            # Assign job name (user provided or derive default)
            try:
                if user_jobname:
                    derived = user_jobname
                else:
                    derived = ''
                    if used_dir:
                        derived = Path(used_dir).name
                    if not derived:
                        bases = [p.name for p in file_list[:3]]
                        if bases:
                            derived = ','.join(bases)
                    if len(derived) > 60:
                        derived = derived[:57] + '...'
                    if not derived:
                        derived = 'job'
                with JOB_NAMES_LOCK:
                    if job_id not in JOB_NAMES or not JOB_NAMES.get(job_id):
                        JOB_NAMES[job_id] = derived
                        try:
                            with JOBS_LOCK:
                                JOBS.get(job_id, {})['name'] = derived
                        except Exception:
                            pass
                        _save_job_names(JOB_NAMES)
                # Persist prodmode on job
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]['prodmode'] = prodmode
            except Exception:
                pass
            # Resolve job name to display on progress page
            try:
                job_name_disp = (JOB_NAMES.get(job_id) or derived or user_jobname or '').strip()
            except Exception:
                job_name_disp = (derived or user_jobname or '')
            return render_template('fdv2_progress.html', token=token, job_id=job_id, limit_raw=lim_raw, prodmode=prodmode, job_name=job_name_disp)
        except Exception as e:
            flash(f"Failed to start parsing: {e}")
            return redirect(url_for('report_home'))
    # GET: allow token to show the previous fdvtest results table or kick off parsing when dirpath is provided
    # Optional GET-based analyze path to avoid multipart parsing/temp issues
    dirpath_q = (request.args.get('dirpath') or '').strip()
    prod_raw_q = (request.args.get('prodmode') or '').strip().lower()
    prodmode_q = prod_raw_q in ('1','true','on','yes')
    if dirpath_q:
        try:
            dp = Path(dirpath_q)
            if not dp.exists() or not dp.is_dir():
                flash('Directory not found or not a directory.')
                return redirect(url_for('report_home'))
            used_dir = str(dp)
            lim_raw = (request.args.get('limit') or '').strip()
            user_jobname = (request.args.get('jobname') or '').strip()
            ledger_map: Dict[str, Path] = {}
            if prodmode_q:
                led_dir = dp / 'ledger'
                # One-time normalization when enabling prodmode from landing page via GET:
                # rename any .done -> .ready in ledger (only here).
                try:
                    if led_dir.is_dir():
                        cnt = 0
                        for p in led_dir.iterdir():
                            try:
                                if p.is_file() and p.suffix.lower() == '.done':
                                    target = p.with_suffix('.ready')
                                    if not target.exists():
                                        p.rename(target)
                                        cnt += 1
                            except Exception:
                                continue
                        try:
                            if cnt:
                                app.logger.info("prodmode normalization (GET): renamed %d .done -> .ready in %s", cnt, str(led_dir))
                        except Exception:
                            pass
                except Exception:
                    pass
                ready_files: List[Path] = []
                try:
                    if led_dir.is_dir():
                        # Only list .ready files
                        ready_files = sorted([p for p in led_dir.iterdir() if p.is_file() and p.suffix.lower() == '.ready'])
                except Exception:
                    ready_files = []
                all_files = _list_files([dp])
                stem_index: Dict[str, List[Path]] = {}
                for f in all_files:
                    try:
                        if 'ledger' in f.parts and (dp / 'ledger') in f.parents:
                            continue
                    except Exception:
                        pass
                    stem_index.setdefault(f.stem, []).append(f)
                selected: List[Path] = []
                for rf in ready_files:
                    base = rf.stem
                    candidates = [c for c in stem_index.get(base, []) if c.suffix.lower() in _allowed_exts or c.suffix == '']
                    chosen = candidates[0] if candidates else None
                    if chosen is not None and chosen.is_file():
                        selected.append(chosen)
                        ledger_map[str(chosen)] = rf
                if not selected:
                    flash('No matching ledger .ready files found under directory/ledger.')
                    return redirect(url_for('report_home'))
                file_list = selected
            else:
                file_list = _list_files([dp])
            if not file_list:
                flash('No files found to parse in the directory.')
                return redirect(url_for('report_home'))
            token = uuid.uuid4().hex
            CACHE[token] = {'status': 'queued', 'progress': {'files_total': len(file_list), 'files_done': 0, 'percent': 0.0, 'lines': 0}, 'dir': used_dir, 'prodmode': prodmode_q}
            CANCEL_FLAGS[token] = False
            _start_parse_job(token, file_list, used_dir, lim_raw, prodmode=prodmode_q, ledger_map=ledger_map if ledger_map else None)
            job_id = _create_job_id(token, lim_raw)
            try:
                if user_jobname:
                    derived = user_jobname
                else:
                    derived = ''
                    if used_dir:
                        derived = Path(used_dir).name
                    if not derived:
                        bases = [p.name for p in file_list[:3]]
                        if bases:
                            derived = ','.join(bases)
                    if len(derived) > 60:
                        derived = derived[:57] + '...'
                    if not derived:
                        derived = 'job'
                with JOB_NAMES_LOCK:
                    if job_id not in JOB_NAMES or not JOB_NAMES.get(job_id):
                        JOB_NAMES[job_id] = derived
                        try:
                            with JOBS_LOCK:
                                JOBS.get(job_id, {})['name'] = derived
                        except Exception:
                            pass
                        _save_job_names(JOB_NAMES)
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]['prodmode'] = prodmode_q
            except Exception:
                pass
            # Resolve job name to display on progress page
            try:
                job_name_disp = (JOB_NAMES.get(job_id) or derived or user_jobname or '').strip()
            except Exception:
                job_name_disp = (derived or user_jobname or '')
            return render_template('fdv2_progress.html', token=token, job_id=job_id, limit_raw=lim_raw, prodmode=prodmode_q, job_name=job_name_disp)
        except Exception as e:
            flash(f"Failed to start parsing: {e}")
            return redirect(url_for('report_home'))
    # Display previous results by token with dynamic limit / token mode ('none')
    tok = (request.args.get('token') or '').strip()
    # Track the raw incoming limit (query) and how we sourced the effective limit
    lim_raw = (request.args.get('limit') or '').strip()
    limit_source = 'query' if ('limit' in request.args) else ''
    # If no limit provided in query, try session cached value (preserve numeric threshold across reloads)
    if (not lim_raw) and tok and tok in CACHE:
        try:
            cached_lr = (CACHE.get(tok, {}).get('limit_raw') or '').strip()
            if cached_lr and cached_lr.lower() not in ('none', ''):
                lim_raw = cached_lr
                limit_source = 'cache'
        except Exception:
            pass
    if not lim_raw and not limit_source:
        limit_source = 'default'
    passfail_mode = False
    if lim_raw.lower() in ('', 'none', 'default'):
        # Token mode: ignore numeric threshold entirely
        limit_for_stats = 0.0  # unused when passfail_mode=True
        limit_template = None
        passfail_mode = True
    else:
        try:
            limit_for_stats = float(lim_raw)
            limit_template = limit_for_stats
        except Exception:
            limit_for_stats = 0.0
            limit_template = None
            passfail_mode = True
    if tok and tok in CACHE:
        data = CACHE.get(tok, {})
        rows = data.get('rows', [])
        stats = stats_by_fdv_with_splits(rows, limit=limit_for_stats, passfail_mode=passfail_mode) if rows else []
        # Persist updated numeric limit back into session cache when user supplied one
        try:
            if tok in CACHE and lim_raw and lim_raw.lower() not in ('none','', 'default'):
                CACHE[tok]['limit_raw'] = lim_raw
        except Exception:
            pass
        # Only expose a snapshot permalink when the job is complete
        status_val = (data.get('status') or '').strip().lower()
        persist_url = url_for('report2_persist', token=tok) if status_val == 'done' else None
        try:
            app.logger.info("report_home GET token=%s limit_raw='%s' source=%s passfail_mode=%s effective_limit=%s", tok, lim_raw, limit_source, passfail_mode, (None if passfail_mode else limit_for_stats))
        except Exception:
            pass
        # Try to find associated job id
        job_id_for_token = None
        try:
            with JOBS_LOCK:
                for _jid, _rec in JOBS.items():
                    if _rec.get('token') == tok:
                        job_id_for_token = _jid
                        break
        except Exception:
            pass
        # Load comments (cached in memory if present, else from disk)
        try:
            if 'comments' not in data:
                data['comments'] = _load_comments(tok)
        except Exception:
            pass
        comments_map = data.get('comments', {}) or {}
        try:
            if 'dispositions' not in data:
                data['dispositions'] = _load_dispositions(tok)
        except Exception:
            pass
        dispositions_map = data.get('dispositions', {}) or {}
        # Resolve job name for banner
        job_name_val = ''
        try:
            if job_id_for_token:
                with JOBS_LOCK:
                    rec = JOBS.get(job_id_for_token)
                    if rec:
                        job_name_val = (rec.get('name') or '').strip()
            if (not job_name_val) and job_id_for_token:
                job_name_val = (JOB_NAMES.get(job_id_for_token) or '').strip()
        except Exception:
            job_name_val = job_name_val
        # Optionally kick off auto-summary in background when parsing is complete
        try:
            if status_val == 'done' and _summary_enabled() and _summary_auto():
                _ensure_summary_async(tok, (lim_raw or 'none').lower())
        except Exception:
            pass
        html = render_template('fdv2_report.html', token=tok, job_id=job_id_for_token, stats=stats, used_dir=data.get('dir'), fdv_order=data.get('fdv_order', []), limit=limit_template, persist_url=persist_url, limit_raw_string=lim_raw, limit_source=limit_source, comments=comments_map, dispositions=dispositions_map, prodmode=data.get('prodmode', False), job_name=job_name_val, summary_enabled=_summary_enabled())
        # Persist a snapshot only once parsing is done
        if status_val == 'done':
            try:
                _persist_write('report2', tok, html)
            except Exception:
                pass
        return html
    try:
        app.logger.info("report_home GET (no token) limit_raw='%s' source=%s passfail_mode=%s effective_limit=%s", lim_raw, limit_source, passfail_mode, (None if passfail_mode else limit_for_stats))
    except Exception:
        pass
    return render_template('fdv2_report.html', token='', job_id=None, stats=[], used_dir=None, fdv_order=[], limit=None, limit_raw_string=lim_raw, limit_source=limit_source, comments={}, dispositions={}, prodmode=False, summary_enabled=_summary_enabled())


@app.route('/status/<token>')
def report_status(token: str):
    data = CACHE.get(token)
    if not data:
        return Response(json.dumps({'status': 'missing'}), mimetype='application/json')
    # Only expose minimal info
    out = {
        'status': data.get('status', 'unknown'),
        'progress': data.get('progress', {}),
        'error': data.get('error')
    }
    return Response(json.dumps(out), mimetype='application/json')

@app.route('/job/<job_id>/status')
def job_status(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return Response(json.dumps({'status': 'missing'}), mimetype='application/json', status=404)
    return report_status(token)

@app.route('/job/<job_id>/progress')
def job_progress(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return redirect(url_for('report_home'))
    data = CACHE.get(token)
    if not data:
        return redirect(url_for('report_home'))
    limit_raw = ''
    job_name = ''
    try:
        with JOBS_LOCK:
            rec = JOBS.get(job_id)
            if rec:
                limit_raw = rec.get('limit_raw','')  # type: ignore[arg-type]
                # If prodmode wasn't persisted on JOBS record (older runs), fallback to cache value
                if 'prodmode' in rec:
                    prodmode = bool(rec.get('prodmode', False))
                else:
                    prodmode = bool(data.get('prodmode', False))
                try:
                    job_name = (rec.get('name') or '').strip()
                except Exception:
                    job_name = job_name
            else:
                prodmode = bool(data.get('prodmode', False))
    except Exception:
        prodmode = bool(data.get('prodmode', False))
    # Fallback to JOB_NAMES file
    try:
        if not job_name:
            nm = JOB_NAMES.get(job_id)
            if nm:
                job_name = nm
    except Exception:
        pass
    return render_template('fdv2_progress.html', token=token, job_id=job_id, limit_raw=limit_raw, prodmode=prodmode, job_name=job_name)

@app.route('/api/job/<job_id>/done', methods=['POST'])
def api_job_done(job_id: str):
    """Mark a job as done in prodmode: set cancel flag so parsing stops after current file, then redirect client to report.

    Returns JSON with report_url.
    """
    token = _resolve_job_token(job_id)
    if not token:
        return jsonify({'ok': False, 'error': 'job not found'}), 404
    # Set cancel flag (reader will stop between files; progress_cb will also check flag)
    CANCEL_FLAGS[token] = True
    # Also mark job record ended if not already
    try:
        with JOBS_LOCK:
            rec = JOBS.get(job_id)
            if rec and 'ended_at' not in rec:
                rec['ended_at'] = time.time()
    except Exception:
        pass
    return jsonify({'ok': True, 'report_url': url_for('job_report', job_id=job_id)})

@app.route('/job/<job_id>/report')
def job_report(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return redirect(url_for('report_home'))
    lr = request.args.get('limit')
    if lr:
        return redirect(url_for('report_home', token=token, limit=lr))
    return redirect(url_for('report_home', token=token))

@app.route('/api/summary/<token>', methods=['GET', 'POST'])
def api_summary(token: str):
    """Get or trigger generation of a summary for the current token and limit.

    Query/body params:
      - limit: string; '' or 'default' => pass/fail-from-logs; numeric => threshold
      - force: when POST and force=1, re-generate ignoring cached file
    Returns JSON with summary text and metadata.
    """
    if not _summary_enabled():
        return jsonify({'ok': False, 'error': 'summary disabled'}), 400
    lim_raw = (request.values.get('limit') or '').strip().lower()
    if lim_raw in ('default',):
        lim_raw = ''
    # For GET, try to load cached summary; if missing, start async job and return pending
    force = False
    if request.method == 'POST':
        force = str(request.values.get('force', '0')).strip().lower() in ('1','true','on','yes')
    if not force:
        cached = _summary_load(token, lim_raw or 'none')
        if cached:
            return Response(json.dumps(cached), mimetype='application/json')
    # No cached summary or force requested: start async and reply with pending
    _ensure_summary_async(token, lim_raw or 'none')
    return jsonify({'ok': True, 'pending': True, 'token': token, 'limit': lim_raw or 'none'})

@app.route('/embed/summary/<token>')
def embed_summary(token: str):
    """Return a small HTML snippet containing the current summary (if available) or a placeholder."""
    if not _summary_enabled():
        return Response('<div class="small">Summary disabled.</div>', mimetype='text/html')
    lim_raw = (request.args.get('limit') or '').strip().lower()
    if lim_raw in ('default',):
        lim_raw = ''
    cached = _summary_load(token, lim_raw or 'none')
    if cached and cached.get('summary'):
        txt = str(cached.get('summary') or '')
        backend = cached.get('backend') or ''
        model = cached.get('model') or ''
        meta = f"<div class=\"small\" style=\"color:#666;\">source: {backend}{(' / ' + model) if model else ''}</div>" if backend else ''
        body = txt.replace('\n', '<br/>')
        return Response(f"<div class=\"llm-summary\"><div style=\"font-weight:600;margin-bottom:4px;\">Report Summary</div><div>{body}</div>{meta}</div>", mimetype='text/html')
    # Not available yet
    return Response('<div class="small">Summary not ready…</div>', mimetype='text/html')

@app.route('/api/launch', methods=['POST'])
def api_launch():
    """Launch a parsing job via JSON or multipart form.

    Accepts either a directory path (dirpath) or uploaded files (files field).
    Optional field 'limit'. Returns JSON with job and status URLs.
    """
    try:
        lim_raw = (request.form.get('limit') or '').strip()
    except Exception:
        lim_raw = ''
    try:
        user_jobname = (request.form.get('jobname') or '').strip()
    except Exception:
        user_jobname = ''
    dirpath = (request.form.get('dirpath') or '').strip()
    used_dir: str | None = None
    file_list: List[Path] = []
    try:
        if dirpath:
            dp = Path(dirpath)
            if not dp.exists() or not dp.is_dir():
                return Response(json.dumps({'error': 'directory not found'}), mimetype='application/json', status=400)
            used_dir = str(dp)
            file_list = _list_files([dp])
        else:
            files = request.files.getlist('files')
            if not files:
                return Response(json.dumps({'error': 'no files provided'}), mimetype='application/json', status=400)
            tmp_dir = Path(tempfile.mkdtemp(prefix='fdv_run_', dir=tempfile.gettempdir()))
            saved: List[Path] = []
            for i, f in enumerate(files):
                name = f.filename or f"fdv_{i}.txt"
                base = Path(name).name or f"fdv_{i}.txt"
                dst = tmp_dir / f"{i:05d}_{base}"
                dst.write_bytes(f.read())
                saved.append(dst)
            used_dir = str(tmp_dir)
            file_list = saved
        if not file_list:
            return Response(json.dumps({'error': 'no files found'}), mimetype='application/json', status=400)
        token = uuid.uuid4().hex
        CACHE[token] = {'status': 'queued', 'progress': {'files_total': len(file_list), 'files_done': 0, 'percent': 0.0, 'lines': 0}, 'dir': used_dir}
        _start_parse_job(token, file_list, used_dir, lim_raw)
        job_id = _create_job_id(token, lim_raw)
        # Derive a default short job name from directory or files
        try:
            if user_jobname:
                default_name = user_jobname
            else:
                default_name = ''
                if used_dir:
                    default_name = Path(used_dir).name
                if not default_name:
                    bases = [p.name for p in file_list[:3]]
                    if bases:
                        default_name = ','.join(bases)
                if len(default_name) > 60:
                    default_name = default_name[:57] + '...'
                if not default_name:
                    default_name = 'job'
            with JOB_NAMES_LOCK:
                # Only set if user hasn't already provided one (future-proof if API extends)
                if job_id not in JOB_NAMES or not JOB_NAMES.get(job_id):
                    JOB_NAMES[job_id] = default_name
                    try:
                        with JOBS_LOCK:
                            JOBS.get(job_id, {})['name'] = default_name
                    except Exception:
                        pass
                    _save_job_names(JOB_NAMES)
        except Exception:
            pass
        base = request.host_url.rstrip('/')
        payload = {
            'job_id': job_id,
            'token': token,
            'status_url': f"{base}/job/{job_id}/status",
            'progress_url': f"{base}/job/{job_id}/progress",
            'report_url': f"{base}/job/{job_id}/report",
        }
        return Response(json.dumps(payload), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json', status=500)

@app.route('/api/job/<job_id>/progress')
def api_job_progress(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return Response(json.dumps({'error': 'job not found'}), mimetype='application/json', status=404)
    data = CACHE.get(token) or {}
    out = {
        'job_id': job_id,
        'token': token,
        'status': data.get('status','unknown'),
        'progress': data.get('progress', {}),
        'error': data.get('error')
    }
    return Response(json.dumps(out), mimetype='application/json')

@app.route('/api/job/<job_id>/report/table')
def api_job_report_table(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return Response(json.dumps({'error': 'job not found'}), mimetype='application/json', status=404)
    data = CACHE.get(token) or {}
    rows = data.get('rows', []) or []
    # Use current limit rules similar to report_home (do not persist new limit modifications here)
    lim_raw = (JOBS.get(job_id, {}).get('limit_raw') if job_id in JOBS else '') or ''
    passfail_mode = (lim_raw.strip().lower() in ('', 'none', 'default'))
    try:
        if passfail_mode:
            stats = stats_by_fdv_with_splits(rows, limit=1e9, passfail_mode=True) if rows else []
            limit_template = None
        else:
            try:
                limit_val = float(lim_raw)
            except Exception:
                limit_val = 1e9
                passfail_mode = True
            stats = stats_by_fdv_with_splits(rows, limit=limit_val, passfail_mode=False) if rows else []
            limit_template = limit_val if not passfail_mode else None
    except Exception:
        stats = []
        limit_template = None
    try:
        html = render_template('fdv2_report_table_only.html', token=token, job_id=job_id, stats=stats, limit=limit_template)
    except Exception as e:
        return Response(json.dumps({'error': f'render failed: {e}'}), mimetype='application/json', status=500)
    payload = {
        'job_id': job_id,
        'token': token,
        'status': data.get('status','unknown'),
        'table_html': html,
        'rows_count': len(rows),
        'stats_count': len(stats)
    }
    return Response(json.dumps(payload), mimetype='application/json')

@app.route('/embed/job/<job_id>/progress')
def embed_job_progress(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return Response('<div>Job not found</div>', mimetype='text/html', status=404)
    data = CACHE.get(token) or {}
    base = request.host_url.rstrip('/')
    status_text = data.get('status','unknown')
    # Build JS without f-string to avoid brace escaping complexity
    js = (
        "const jid='JOB_ID';"
        "async function tick(){try{const r=await fetch('BASE_URL/job/JOB_ID/status',{cache:'no-store'});"
        "if(r.ok){const d=await r.json();document.getElementById('st').textContent=d.status;"
        "const p=d.progress||{};const pc=Math.max(0,Math.min(100,Number(p.percent||0)));"
        "document.getElementById('f').style.width=pc.toFixed(1)+'%';"
        "document.getElementById('pct').textContent=pc.toFixed(1)+'% lines='+(p.lines||0);"
        "if(d.status==='done')return;}}catch(e){}setTimeout(tick,1000);}tick();"
    ).replace('JOB_ID', job_id).replace('BASE_URL', base)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Progress {job_id}</title>"
        "<style>body{font:12px Arial;margin:6px}.bar{height:8px;background:#eee;border-radius:4px;overflow:hidden}.bar>div{height:100%;background:#4caf50;width:0%}</style>"
        "</head><body>"
        f"<div>Job <strong>{job_id}</strong> status: <span id='st'>{status_text}</span></div>"
        "<div class='bar'><div id='f'></div></div><div id='pct' style='margin-top:4px;'></div>"
        f"<div style='margin-top:6px;'><a href='{base}/job/{job_id}/report' target='_blank'>Open Report</a> | <a href='{base}/job/{job_id}/status' target='_blank'>JSON</a></div>"
        f"<script>{js}</script>"
        "</body></html>"
    )
    return Response(html, mimetype='text/html')

@app.route('/embed/job/<job_id>/report')
def embed_job_report(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return Response('<div>Job not found</div>', mimetype='text/html', status=404)
    data = CACHE.get(token) or {}
    rows = data.get('rows', []) or []
    lim_raw = (JOBS.get(job_id, {}).get('limit_raw') if job_id in JOBS else '') or ''
    passfail_mode = (lim_raw.strip().lower() in ('', 'none', 'default'))
    try:
        if passfail_mode:
            stats = stats_by_fdv_with_splits(rows, limit=1e9, passfail_mode=True) if rows else []
            limit_template = None
        else:
            try:
                limit_val = float(lim_raw)
            except Exception:
                limit_val = 1e9
                passfail_mode = True
            stats = stats_by_fdv_with_splits(rows, limit=limit_val, passfail_mode=False) if rows else []
            limit_template = limit_val if not passfail_mode else None
    except Exception:
        stats = []
        limit_template = None
    try:
        html = render_template('fdv2_report_table_only.html', token=token, job_id=job_id, stats=stats, limit=limit_template)
    except Exception as e:
        return Response(f"<div>Render error: {e}</div>", mimetype='text/html', status=500)
    return Response(html, mimetype='text/html')

@app.route('/status/<token>/fdvtable')
def report_status_fdvtable(token: str):
    """Return a partial HTML table of current fdvtest stats while parsing is running.

    Optional query params:
      - limit: RBER threshold used to compute PASS/FAIL
    """
    data = CACHE.get(token)
    if not data:
        return Response('<div class="small">Session not found.</div>', mimetype='text/html', status=404)
    lim_raw = (request.args.get('limit') or '').strip()
    passfail_mode = False
    if lim_raw.lower() in ('', 'none', 'default'):
        limit_for_stats = 1e9
        limit_template = None
        passfail_mode = True
    else:
        try:
            limit_for_stats = float(lim_raw)
            limit_template = limit_for_stats
        except Exception:
            limit_for_stats = 1e9
            limit_template = None
            passfail_mode = True
    # Serve cached snapshot if still fresh (<3s) and limit matches
    lr_key = (lim_raw.strip().lower() or 'none')
    with SNAP_LOCK:
        snap = SNAPSHOTS.get(token)
        if snap and (time.time() - float(snap.get('updated', 0))) < 3.0 and snap.get('limit_key') == lr_key:
            html_cached = snap.get('fdvtable_html') or ''
            if html_cached:
                return Response(html_cached, mimetype='text/html')
    # Build from SQLite so table accumulates across all processed files even if in-memory rows were trimmed
    try:
        stats = _stats_from_sqlite(token, limit=limit_for_stats, passfail_mode=passfail_mode)
    except Exception:
        stats = []
    try:
        html = render_template('fdv2_report_table.html', token=token, stats=stats, used_dir=data.get('dir'), limit=limit_template)
        # Cache full build
        with SNAP_LOCK:
            SNAPSHOTS[token] = {'fdvtable_html': html, 'updated': time.time(), 'limit_key': lr_key}
        return Response(html, mimetype='text/html')
    except Exception as e:
        return Response(f"<div class='small'>Error building table: {e}</div>", mimetype='text/html', status=500)

@app.route('/stream/job/<job_id>')
def stream_job(job_id: str):
    token = _resolve_job_token(job_id)
    if not token:
        return Response('not found', status=404)
    def gen():
        last_sent = 0.0
        last_current_file = None
        last_percent = None
        last_file_percent = None
        last_files_done = None
        while True:
            data = CACHE.get(token) or {}
            prog = data.get('progress', {})
            status = data.get('status','unknown')
            now = time.time()
            # Adaptive push: send at least every ~0.5s; send immediately on notable changes
            per = prog.get('percent')
            fper = prog.get('file_percent')
            cfile = prog.get('current_file')
            fdone = prog.get('files_done')
            should_send = False
            if now - last_sent >= 0.5:
                should_send = True
            try:
                if cfile is not None and cfile != last_current_file:
                    should_send = True
                if (per is not None) and (last_percent is None or abs(float(per) - float(last_percent)) >= 0.5):
                    should_send = True
                if (fper is not None) and (last_file_percent is None or abs(float(fper) - float(last_file_percent)) >= 1.0):
                    should_send = True
                if fdone is not None and fdone != last_files_done:
                    should_send = True
            except Exception:
                pass
            if should_send:
                last_sent = now
                last_current_file = cfile
                last_percent = per
                last_file_percent = fper
                last_files_done = fdone
                payload = {
                    'status': status,
                    'progress': {
                        'percent': prog.get('percent'),
                        'lines': prog.get('lines'),
                        'lines_total': prog.get('lines_total'),
                        'expected_overall_lines': prog.get('expected_overall_lines') or prog.get('lines_total'),
                        'files_done': prog.get('files_done'),
                        'files_total': prog.get('files_total'),
                        'current_file': prog.get('current_file'),
                        'file_percent': prog.get('file_percent'),
                        'file_lines_done': prog.get('file_lines_done'),
                        'file_lines_total': prog.get('file_lines_total'),
                        'expected_file_lines': prog.get('expected_file_lines') or prog.get('file_lines_total'),
                        # Include byte counters to improve per-file ETA when line totals are unknown
                        'file_bytes_done': prog.get('file_bytes_done'),
                        'file_bytes_total': prog.get('file_bytes_total'),
                        # Optional server-side ETA if producer sets it
                        'eta_overall_secs': prog.get('eta_overall_secs'),
                    }
                }
                yield f"data: {json.dumps(payload)}\n\n"
            if status in ('done','error'):
                break
            time.sleep(0.2)
    return Response(gen(), mimetype='text/event-stream')

@app.route('/api/jobs')
def api_jobs():
    """Return JSON list of active jobs (recent in-memory only)."""
    jobs_out = []
    now = time.time()
    with JOBS_LOCK:
        for jid, rec in JOBS.items():
            token = rec.get('token')  # type: ignore[arg-type]
            if not token:
                continue
            data = CACHE.get(token) or {}
            prog = data.get('progress', {}) or {}
            status = data.get('status','unknown')
            # Directory entered on landing page, if available
            dir_val = data.get('dir') or ''
            snap = SNAPSHOTS.get(token) if 'SNAPSHOTS' in globals() else None
            updated = None
            if snap:
                try: updated = float(snap.get('updated'))
                except Exception: updated = None
            # Determine if report is ready (done status and rows exist)
            report_ready = (status == 'done' and bool(data.get('rows')))
            created_at = rec.get('created_at') if isinstance(rec.get('created_at'), (int,float)) else None
            ended_at = rec.get('ended_at') if isinstance(rec.get('ended_at'), (int,float)) else None
            def _fmt(ts):
                try:
                    return time.strftime('%H:%M:%S', time.localtime(ts)) if ts else None
                except Exception:
                    return None
            _maybe_reload_job_names()
            with JOB_NAMES_LOCK:
                job_name = JOB_NAMES.get(jid, str(rec.get('name','')))
            jobs_out.append({
                'job_id': jid,
                'token': token,
                'status': status,
                'percent': prog.get('percent'),
                'files_done': prog.get('files_done'),
                'files_total': prog.get('files_total'),
                'lines': prog.get('lines'),
                'updated_secs_ago': (None if updated is None else round(now - updated,2)),
                'created_at': created_at,
                'created_hms': _fmt(created_at) if created_at else None,
                'ended_at': ended_at,
                'ended_hms': _fmt(ended_at) if ended_at else None,
                'duration_secs': (round((ended_at - created_at),2) if (created_at and ended_at) else (round(now-created_at,2) if created_at and status=='running' else None)),
                'report_url': f"/job/{jid}/report",
                'progress_url': f"/job/{jid}/progress",
                'sse_url': f"/stream/job/{jid}",
                'status_url': f"/job/{jid}/status",
                'report_ready': report_ready,
                'name': job_name,
                'dir': dir_val,
            })
    jobs_out.sort(key=lambda j: (j.get('status')!='running', j.get('job_id')))
    return Response(json.dumps({'jobs': jobs_out, 'count': len(jobs_out)}), mimetype='application/json')

@app.route('/api/job/<job_id>/name', methods=['POST'])
def api_job_set_name(job_id: str):
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}
    name = (payload.get('name') or '').strip()
    if len(name) > 120:
        return jsonify({'ok': False, 'error': 'name too long'}), 400
    with JOBS_LOCK:
        if job_id not in JOBS:
            return jsonify({'ok': False, 'error': 'unknown job id'}), 404
    with JOB_NAMES_LOCK:
        JOB_NAMES[job_id] = name
        _save_job_names(JOB_NAMES)
        try:
            p = _job_names_path()
            global JOB_NAMES_MTIME
            JOB_NAMES_MTIME = p.stat().st_mtime if p.exists() else JOB_NAMES_MTIME
        except Exception:
            pass
    with JOBS_LOCK:
        JOBS[job_id]['name'] = name
    return jsonify({'ok': True, 'name': name})

# --- Job action helpers & endpoints (pause/continue/delete) ---
def _resolve_cache_by_job(job_id: str):
    with JOBS_LOCK:
        rec = JOBS.get(job_id)
    if not rec:
        return None, None
    token = rec.get('token') if isinstance(rec, dict) else None
    if not token:
        return None, None
    return token, CACHE.get(token)

@app.route('/api/job/<job_id>/pause', methods=['POST'])
def api_job_pause(job_id: str):
    token, entry = _resolve_cache_by_job(job_id)
    if not token:
        return jsonify({'ok': False, 'error': 'unknown job'}), 404
    if not entry or entry.get('status') not in {'running'}:
        return jsonify({'ok': False, 'error': 'not running'}), 400
    # Flag for worker loops that support pausing; also update UI state immediately
    entry['_pause_flag'] = True
    entry['status'] = 'paused'
    return jsonify({'ok': True, 'status': 'paused'})

@app.route('/api/job/<job_id>/continue', methods=['POST'])
def api_job_continue(job_id: str):
    token, entry = _resolve_cache_by_job(job_id)
    if not token:
        return jsonify({'ok': False, 'error': 'unknown job'}), 404
    if not entry or (entry.get('status') not in {'paused'} and not entry.get('_pause_flag')):
        return jsonify({'ok': False, 'error': 'not paused'}), 400
    entry['_pause_flag'] = False
    entry['status'] = 'running'
    return jsonify({'ok': True, 'status': 'running'})

@app.route('/api/job/<job_id>/delete', methods=['POST'])
def api_job_delete(job_id: str):
    token, entry = _resolve_cache_by_job(job_id)
    if not token:
        return jsonify({'ok': False, 'error': 'unknown job'}), 404
    # Signal stop to any cooperating worker and remove from active set
    if entry is not None:
        entry['_stop_flag'] = True
        # Present as deleted to any subsequent status views
        entry['status'] = 'deleted'
        entry['ended_at'] = entry.get('ended_at') or time.time()
    with JOBS_LOCK:
        if job_id in JOBS:
            del JOBS[job_id]
    return jsonify({'ok': True, 'status': 'deleted'})

@app.route('/api/comments/<token>', methods=['GET'])
def api_comments_get(token: str):
    data = CACHE.get(token)
    if data and 'comments' in data:
        comments = data.get('comments', {}) or {}
    else:
        comments = _load_comments(token)
        if data is not None:
            data['comments'] = comments
    return Response(json.dumps({'token': token, 'comments': comments}), mimetype='application/json')

# Backward-compatible singular alias (user attempted /api/comment/...)
@app.route('/api/comment/<token>', methods=['GET'])
def api_comment_get_alias(token: str):
    return api_comments_get(token)

@app.route('/api/comment/<token>', methods=['POST'])
def api_comment_post_alias(token: str):
    return api_comments_update(token)

@app.route('/api/comments/<token>', methods=['POST'])
def api_comments_update(token: str):
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}
    key = (payload.get('key') or '').strip()
    val = (payload.get('value') or '').strip()
    if not key:
        return Response(json.dumps({'error': 'missing key'}), status=400, mimetype='application/json')
    # Ensure a session container exists (allow comment editing even after restart)
    data = CACHE.get(token)
    if data is None:
        # Attempt to load existing comments from disk to preserve previously saved values
        existing = _load_comments(token)
        data = {'status': 'comments-only', 'comments': existing}
        CACHE[token] = data  # minimal ephemeral session
    comments = data.setdefault('comments', {})
    if val:
        comments[key] = val
    else:
        # Empty value => delete
        comments.pop(key, None)
    # Persist to disk async (best effort)
    try:
        _save_comments(token, comments)
    except Exception:
        pass
    return Response(json.dumps({'ok': True, 'key': key, 'value': comments.get(key,'')}), mimetype='application/json')

@app.route('/api/dispositions/<token>', methods=['GET'])
def api_dispositions_get(token: str):
    data = CACHE.get(token)
    if data and 'dispositions' in data:
        disp = data.get('dispositions', {}) or {}
    else:
        disp = _load_dispositions(token)
        if data is not None:
            data['dispositions'] = disp
    return Response(json.dumps({'token': token, 'dispositions': disp}), mimetype='application/json')

@app.route('/api/dispositions/<token>', methods=['POST'])
def api_dispositions_update(token: str):
    # Allow saving even if original session rows evicted or server restarted.
    # If token not present create minimal ephemeral container seeded from disk.
    if token not in CACHE:
        try:
            existing = _load_dispositions(token)
        except Exception:
            existing = {}
        CACHE[token] = {'status': 'dispositions-only', 'dispositions': existing}
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}
    key = (payload.get('key') or '').strip()
    val = (payload.get('value') or '').strip()
    if not key:
        return Response(json.dumps({'error': 'missing key'}), status=400, mimetype='application/json')
    data = CACHE.get(token)
    disp = data.setdefault('dispositions', {})
    try:
        app.logger.info('disposition save token=%s key=%s len=%d', token, key, len(val))
    except Exception:
        pass
    if val:
        disp[key] = val
    else:
        disp.pop(key, None)
    try:
        _save_dispositions(token, disp)
    except Exception:
        pass
    return Response(json.dumps({'ok': True, 'key': key, 'value': disp.get(key,'')}), mimetype='application/json')

@app.route('/jobs')
def jobs_page():
    """HTML page listing active jobs with auto-refresh."""
    # Build data once; template also has JS to refresh via /api/jobs
    with JOBS_LOCK:
        current_jobs = list(JOBS.keys())
    return render_template('fdv2_jobs.html', job_ids=current_jobs)

@app.route('/persist/report2/<token>')
def report2_persist(token: str):
    """Serve a previously saved persistent HTML snapshot for Report v2 main page."""
    path = _persist_base_dir() / 'report2' / token / 'index.html'
    try:
        if path.is_file():
            return send_file(str(path), mimetype='text/html')
    except Exception:
        pass
    return Response('Snapshot not found. Re-run analysis to regenerate.', status=404)


@app.route('/fdv/<token>/tests', methods=['GET','POST'])
def report_tests(token: str):
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        flash('Session expired. Please re-run the FDV Report.')
        return redirect(url_for('report_home'))
    # Accept multi-select: list via GET ?fdv=A&fdv=B or POST getlist('fdv')
    fdvs: List[str] = []
    if request.method == 'POST':
        fdvs = [s.strip() for s in request.form.getlist('fdv') if (s or '').strip()]
    else:
        fdvs = [s.strip() for s in request.args.getlist('fdv') if (s or '').strip()]
    if not fdvs:
        # Backfill single param 'fdv'
        one = (request.values.get('fdv') or '').strip()
        if one:
            fdvs = [one]
    if not fdvs:
        flash('Missing fdvtest selection.')
        return redirect(url_for('report_home', token=token))
    rows = data.get('rows', [])
    # Determine limit / token mode ('none')
    lim_raw = (request.values.get('limit') or '').strip()
    passfail_mode = False
    if lim_raw.lower() in ('', 'none', 'default'):
        limit_for_stats = 1e9
        limit_template = None
        passfail_mode = True
    else:
        try:
            limit_for_stats = float(lim_raw)
            limit_template = limit_for_stats
        except Exception:
            limit_for_stats = 1e9
            limit_template = None
            passfail_mode = True
    stats = stats_by_testname_selected(rows, fdvs, limit=limit_for_stats, passfail_mode=passfail_mode)
    # Capture split_plane_addr preference from prior page
    try:
        spa = (request.values.get('split_plane_addr') or '').strip().lower()
        split_plane_addr = (spa not in ('', '0', 'false', 'no', 'off'))
    except Exception:
        split_plane_addr = False
    # Build a display table for selected rows (fdv,pr,vcc,tm,temp)
    sel_rows = []
    for s in fdvs:
        fdv, pr, vcc, tm, temp = _parse_fdv_selector(s)
        sel_rows.append({'fdv': fdv, 'pr': pr, 'vcc': vcc, 'tm': tm, 'temp': temp})
    return render_template('fdv2_report_tests.html', token=token, fdvs=fdvs, stats=stats, sel_rows=sel_rows, limit=limit_template, split_plane_addr=split_plane_addr)

@app.route('/fdv/<token>/split_dut', methods=['GET','POST'])
def report_split_dut(token: str):
    """Render per-DUT (FUSEID) aggregation for selected FDV test rows.

    Accepts repeated 'fdv' selectors (fdv|pr|vcc|tm|temp). Uses current limit mode.
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        flash('Session expired. Please re-run the FDV Report.')
        return redirect(url_for('report_home'))
    if request.method == 'POST':
        selectors = [s.strip() for s in request.form.getlist('fdv') if s.strip()]
    else:
        selectors = [s.strip() for s in request.args.getlist('fdv') if s.strip()]
    if not selectors:
        one = (request.values.get('fdv') or '').strip()
        if one:
            selectors = [one]
    if not selectors:
        flash('Missing FDV selection.')
        return redirect(url_for('report_home', token=token))
    rows: List[Dict[str,str]] = data.get('rows', [])
    # Determine limit; blank / 'default' / 'none' => PASS/FAIL-from-log mode (limit=None)
    lim_raw = (request.values.get('limit') or '').strip().lower()
    if lim_raw in ('', 'default', 'none'):
        limit = None
    else:
        try:
            v = float(lim_raw)
            limit = v if v > 0 else None
        except Exception:
            limit = None
    stats = stats_by_dut_selected(rows, selectors, limit=limit)
    # Build info rows for heading
    sel_rows = []
    for s in selectors:
        fdv, pr, vcc, tm, temp = _parse_fdv_selector(s)
        sel_rows.append({'fdv': fdv, 'pr': pr, 'vcc': vcc, 'tm': tm, 'temp': temp})
    return render_template('fdv2_report_split_dut.html', token=token, stats=stats, sel_rows=sel_rows, limit=limit, fdvs=selectors)


@app.route('/fdv/<token>/tests/sample', methods=['GET'])
def report_tests_sample(token: str):
    """Return the first N data rows for selected FDV test(s) (fdv,pr,vcc,tm,temp).

    Query params:
      - fdv: can repeat; value format 'fdv|pr|vcc|tm|temp' (parts optional)
      - limit: default 10
      - format: 'html' (default) or 'json'
    Each row includes testname, DUT, plane_op, plane_addr, blk, WL or PAGE, RBER, pagetype, and line_number.
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        return Response('session expired', status=400)
    rows: List[Dict[str, str]] = data.get('rows', [])
    sel_fdvs = [s.strip() for s in (request.args.getlist('fdv') or []) if s.strip()]
    if not sel_fdvs:
        one = (request.args.get('fdv') or '').strip()
        if one:
            sel_fdvs = [one]
    if not sel_fdvs:
        return Response('missing fdv', status=400)
    # Build records akin to variability but without testname filtering
    parsed = [_parse_fdv_selector(x) for x in sel_fdvs]
    recs: List[Dict] = []
    for idx, r in enumerate(rows):
        rk = _get_split_tuple(r)
        ok = False
        for (sf, spr, svcc, stm, stemp) in parsed:
            if sf and sf != rk[0]:
                continue
            if spr and spr != rk[1]:
                continue
            if svcc and svcc != rk[2]:
                continue
            if stm and stm != rk[3]:
                continue
            if stemp and stemp != rk[4]:
                continue
            ok = True
            break
        if not ok:
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
        wl = _extract_wl_or_page(r, allow_page_fallback=True)
        rber = _get_rber(r)
        if rber is None:
            continue
        if rber <= 0:
            rber = 1e-12
        recs.append({
            'testname': tn,
            'DUT': f"DUT{(r.get('dut_id','') or '').strip() or '?'}",
            'plane': _plane_from_tname_or_default(r),
            'plane_addr': _extract_plane_addr(r),
            'blk': _extract_blk_value(r),
            'WL': wl,
            'RBER': rber,
            'pagetype': (r.get('pagetype','') or '').strip(),
            'line_number': r.get('line_number',''),
            '_idx': r.get('_idx', idx),
        })
    # Slice
    try:
        limit = int((request.args.get('limit') or '10').strip())
    except Exception:
        limit = 10
    head = recs[:max(0, limit)]
    fmt = (request.args.get('format') or 'html').strip().lower()
    if fmt == 'json':
        try:
            body = json.dumps(head)

        except Exception:
            body = json.dumps([])
        return Response(body, mimetype='application/json')
    # Build HTML
    def _fmt_row(r: Dict) -> str:
        wl_txt = '-' if r.get('WL') is None or float(r.get('WL') or -1.0) < 0 else str(int(float(r.get('WL'))))
        blk_txt = '' if r.get('blk') is None else str(r.get('blk'))
        return (
            f"<tr><td>{r.get('testname','')}</td>"
            f"<td>{r.get('DUT','')}</td>"
            f"<td>{r.get('plane','')}</td>"
            f"<td>{r.get('plane_addr','')}</td>"
            f"<td>{blk_txt}</td>"
            f"<td>{wl_txt}</td>"
            f"<td>{float(r.get('RBER',0.0)):.3e}</td>"
            f"<td>{r.get('pagetype','')}</td>"
            f"<td>{r.get('line_number','')}</td>"
            f"<td><a href=\"{url_for('report_rawline', token=token, idx=r.get('_idx', -1))}\" target=\"_blank\">view</a></td>"
            f"</tr>"
        )
    rows_html = ''.join(_fmt_row(r) for r in head)
    table = (
        "<!doctype html><html><head><meta charset='utf-8'><title>FDV sample (first %d)</title>"
        "<style>body{font-family:Arial, sans-serif;margin:16px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:13px;}th{background:#f7f7f7;}</style>"
        "</head><body>" % (len(head))
        + ("<div style='font-size:12px;color:#666;margin-bottom:8px;'>FDV(s): %s</div>" % (', '.join([x.split('|')[0] for x in sel_fdvs])))
        + "<table><thead><tr><th>testname</th><th>DUT</th><th>plane</th><th>plane_addr</th><th>blk</th><th>WL</th><th>RBER</th><th>pagetype</th><th>line #</th><th>raw</th></tr></thead><tbody>"
        + rows_html + "</tbody></table>"
        + "</body></html>"
    )
    return Response(table, mimetype='text/html')


@app.route('/fdv/<token>/tests/variability', methods=['GET'])
def report_tests_variability(token: str):
    """Render variability plots page; images are generated on-demand via a PNG endpoint.

    Query params:
      - fdv: repeated selectors for FDV splits (fdv|pr|vcc|tm|temp)
      - sel: repeated testname entries (testname|pr|fuseid|vcc|tm|temp|plane|op) — plane/op optional
      - split_plane_addr: '1' to indicate per-plane-addr charts (hint only; images decide via &group=)
      - fallback: '1' to enable PAGE fallback when WL missing (default on)
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        flash('Session expired. Please re-run the FDV Report.')
        return redirect(url_for('report_home'))
    rows: List[Dict[str, str]] = data.get('rows', [])
    fdvs = [s.strip() for s in (request.args.getlist('fdv') or []) if s.strip()]
    sels = [s.strip() for s in (request.args.getlist('sel') or []) if s.strip()]
    if not fdvs or not sels:
        flash('Missing selections. Choose FDV rows and testnames first.')
        return redirect(url_for('report_tests', token=token))
    try:
        spa = (request.args.get('split_plane_addr') or '').strip().lower()
        split_plane_addr = (spa not in ('', '0', 'false', 'no', 'off'))
    except Exception:
        split_plane_addr = False
    fallback = True
    try:
        fb = (request.args.get('fallback') or '1').strip().lower()
        fallback = (fb not in ('', '0', 'false', 'no', 'off'))
    except Exception:
        pass
    # Build image URLs per selection; images are generated by a separate endpoint that accepts the same selectors
    img_urls = []
    per_diags = []
    per_outliers = []
    for sel in sels:
        # Label from sel (testname|pr|fuseid|vcc|tm|temp|plane|op)
        parts = sel.split('|')
        label = parts[0] if parts else sel
        # Construct URL with params preserved
        from urllib.parse import urlencode
        q = []
        for f in fdvs:
            q.append(('fdv', f))
        q.append(('sel', sel))
        if fallback:
            q.append(('fallback', '1'))
        url = url_for('report_tests_variability_plot', token=token) + '?' + urlencode(q)
        img_urls.append({'label': label, 'url': url, 'count': 1})
        # Minimal diags/outliers placeholders; actual point-level outliers will be computed in the plot endpoint
        per_diags.append({'matched_rows': None, 'missing_wl': None, 'missing_rber': None, 'used_page_fallback': None})
        per_outliers.append([])
    return render_template('fdv2_report_tests_variability.html', token=token, fdvs=fdvs, img_urls=img_urls, per_diags=per_diags, per_outliers=per_outliers, fallback=fallback)


@app.route('/fdv/<token>/tests/variability/plot', methods=['GET'])
def report_tests_variability_plot(token: str):
    """Return a PNG scatter plot for one selection.

    Params:
      - fdv: repeated split selectors (fdv|pr|vcc|tm|temp)
      - sel: single selection (testname|pr|fuseid|vcc|tm|temp|plane|op)
      - fallback: 1/0 allow PAGE fallback when WL missing (default 1)
      - group: optional 'plane_addr' to color by plane address; otherwise color by pagetype.
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        return Response('session expired', status=400)
    rows: List[Dict[str, str]] = data.get('rows', [])
    fdvs = [s.strip() for s in (request.args.getlist('fdv') or []) if s.strip()]
    sel = (request.args.get('sel') or '').strip()
    if not fdvs or not sel:
        return Response('missing selectors', status=400)
    group = (request.args.get('group') or '').strip().lower()
    try:
        fb = (request.args.get('fallback') or '1').strip().lower()
        fallback = (fb not in ('', '0', 'false', 'no', 'off'))
    except Exception:
        fallback = True
    # Build records for the single selection
    entries = [tuple(sel.split('|') + [''] * 8)[:8]]
    recs = _build_variability_records(rows, fdvs, entries, allow_page_fallback=fallback, allow_missing_wl=True)
    # Build candidates for WL and PAGE axes
    pts_wl = [(float(r['WL']), float(r['RBER']), r) for r in recs if r.get('WL') is not None and float(r['WL']) >= 0 and r.get('RBER') is not None]
    pts_pg = [(float(r['PAGE']), float(r['RBER']), r) for r in recs if (r.get('WL') is None or str(r.get('WL')) == '') and r.get('PAGE') is not None and float(r['PAGE']) >= 0 and r.get('RBER') is not None]
    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import math
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=120)
    # Decide plot mode: WL, PAGE, or RCDF
    mode = 'wl' if len(pts_wl) > 0 else ('page' if len(pts_pg) > 0 else 'rcdf')
    if mode == 'rcdf':
        # RCDF over RBER values
        vals = sorted([float(r.get('RBER', 0.0)) for r in recs if r.get('RBER') is not None and float(r.get('RBER')) > 0.0])
        if not vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_axis_off()
        else:
            n = len(vals)
            xs = vals
            ys = [1.0 - (i + 1) / n for i in range(n)]
            ax.plot(xs, ys, lw=1.5, label='RBER RCDF')
            ax.set_xlabel('RBER')
            ax.set_ylabel('RCDF (1 - CDF)')
            ax.grid(True, ls='--', alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    if mode == 'wl':
        pts = pts_wl
    else:
        pts = pts_pg
    if not pts:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_axis_off()
    else:
        # Group/color
        color_key = 'pagetype'
        if group == 'plane_addr':
            color_key = 'plane_addr'
        groups = {}
        for wl, rber, r in pts:
            key = (r.get(color_key) or '').strip() or 'UNK'
            groups.setdefault(key, []).append((wl, rber, r))
        cmap = plt.cm.get_cmap('tab10')
        keys = sorted(groups.keys())
        for i, k in enumerate(keys):
            g = groups[k]
            xs = [p[0] for p in g]
            ys = [math.log10(max(p[1], 1e-12)) for p in g]
            ax.scatter(xs, ys, s=12, alpha=0.7, label=k, color=cmap(i % 10))
        ax.set_xlabel('WL' if mode == 'wl' else 'PAGE')
        ax.set_ylabel('log10(RBER)')
        ax.grid(True, ls='--', alpha=0.3)
        if len(keys) <= 12:
            ax.legend(loc='best', fontsize=8)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/fdv/<token>/tests/variability/data', methods=['GET'])
def report_tests_variability_data(token: str):
    """Return first N data rows backing the variability view for selected items.

    Query params: fdv (repeat), sel (repeat), limit (default 10), format=json to return JSON
    Note: This endpoint includes rows even when WL is missing so PAGE/RCDF fallback can be inspected.
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        return Response('session expired', status=400)
    rows: List[Dict[str, str]] = data.get('rows', [])
    fdvs = [s.strip() for s in (request.args.getlist('fdv') or []) if s.strip()]
    sels = [s.strip() for s in (request.args.getlist('sel') or []) if s.strip()]
    if not fdvs or not sels:
        return Response('missing selectors', status=400)
    try:
        limit = int((request.args.get('limit') or '10').strip())
    except Exception:
        limit = 10
    try:
        fb = (request.args.get('fallback') or '1').strip().lower()
        fallback = (fb not in ('', '0', 'false', 'no', 'off'))
    except Exception:
        fallback = True
    # Build combined records across all selections
    entries = [tuple(s.split('|') + [''] * 8)[:8] for s in sels]
    recs = _build_variability_records(rows, fdvs, entries, allow_page_fallback=fallback, allow_missing_wl=True)
    head = recs[:max(0, limit)]
    fmt = (request.args.get('format') or 'html').strip().lower()
    if fmt == 'json':
        try:
            body = json.dumps(head)
        except Exception:
            body = json.dumps([])
        return Response(body, mimetype='application/json')
    # HTML table similar to tests/sample but with selection fields
    def _fmt_row(r: Dict) -> str:
        wl_val = r.get('WL')
        wl_txt = '' if (wl_val is None) else str(int(float(wl_val)))
        pg_txt = '' if r.get('PAGE') is None else str(r.get('PAGE'))
        blk_txt = '' if r.get('blk') is None else str(r.get('blk'))
        return (
            f"<tr><td>{r.get('testname','')}</td>"
            f"<td>{r.get('dut','')}</td>"
            f"<td>{r.get('plane','')}</td>"
            f"<td>{r.get('plane_addr','')}</td>"
            f"<td>{blk_txt}</td>"
            f"<td>{wl_txt}</td>"
            f"<td>{pg_txt}</td>"
            f"<td>{float(r.get('RBER',0.0)):.3e}</td>"
            f"<td>{r.get('pagetype','')}</td>"
            f"<td>{r.get('line_number','')}</td>"
            f"<td><a href=\"{url_for('report_rawline', token=token, idx=r.get('_idx', -1))}\" target=\"_blank\">view</a></td>"
            f"</tr>"
        )
    rows_html = ''.join(_fmt_row(r) for r in head)
    table = (
        "<!doctype html><html><head><meta charset='utf-8'><title>FDV variability data (first %d)</title>"
        "<style>body{font-family:Arial, sans-serif;margin:16px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:13px;}th{background:#f7f7f7;}</style>"
        "</head><body>" % (len(head))
        + ("<div style='font-size:12px;color:#666;margin-bottom:8px;'>FDV(s): %s</div>" % (', '.join([x.split('|')[0] for x in fdvs])))
        + "<table><thead><tr><th>testname</th><th>DUT</th><th>plane</th><th>plane_addr</th><th>blk</th><th>WL</th><th>PAGE</th><th>RBER</th><th>pagetype</th><th>line #</th><th>raw</th></tr></thead><tbody>"
        + rows_html + "</tbody></table>"
        + "</body></html>"
    )
    return Response(table, mimetype='text/html')

@app.route('/fdv/<token>/rawline')
def report_rawline(token: str):
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        flash('Session expired. Please re-run the FDV Report.')
        return redirect(url_for('report_home'))
    rows: List[Dict[str,str]] = data.get('rows', [])
    try:
        idx = int(request.args.get('idx', '-1'))
    except Exception:
        idx = -1
    if idx < 0 or idx >= len(rows):
        flash('Invalid raw line index.')
        return redirect(url_for('report_home'))
    r = rows[idx]
    return render_template('fdv2_rawline.html', token=token, row=r)


@app.route('/routes', methods=['GET'])
def list_routes():
    rows = []
    try:
        from markupsafe import escape
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            rows.append(f"<tr><td>{escape(rule.endpoint)}</td><td>{escape(rule.rule)}</td><td>{escape(','.join(sorted(rule.methods or [])))}</td></tr>")
    except Exception:
        pass
    body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Routes</title>"
        "<style>body{font-family:Arial, sans-serif;margin:16px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:13px;}th{background:#f7f7f7;}</style>"
        "</head><body><h3>Registered routes</h3><table><thead><tr><th>endpoint</th><th>rule</th><th>methods</th></tr></thead><tbody>"
        + ''.join(rows) + "</tbody></table></body></html>"
    )
    return Response(body, mimetype='text/html')


@app.route('/fdv/<token>/fails', methods=['GET'])
def report_fails(token: str):
    """Show raw file snippet around failing lines for a selected scope, with next/prev navigation.

    Query params:
      - fdv, pr, vcc, tm, temp (required for fdvtest scope)
      - Optional narrowers: testname, fuseid, plane, op
    - limit (float >0) — numeric threshold mode; omit / blank / 'default' / 'none' => PASS/FAIL-from-log mode
      - context (int lines around target, default 80)
      - idx (0-based index into the failing-lines list, default 0)
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        flash('Session expired. Please re-run the FDV Report.')
        return redirect(url_for('report_home'))
    rows: List[Dict[str, str]] = data.get('rows', [])
    fdv = (request.args.get('fdv') or '').strip()
    pr = (request.args.get('pr') or '').strip()
    vcc = (request.args.get('vcc') or '').strip()
    tm = (request.args.get('tm') or '').strip()
    temp = (request.args.get('temp') or '').strip()
    # Optional override for analyzed/search root directory
    base_dir_q = (
        (request.args.get('dir') or '')
        or (request.args.get('base') or '')
        or (request.args.get('root') or '')
    ).strip().strip(' "\'')
    testname_q = (request.args.get('testname') or '').strip()
    fuseid_q = (request.args.get('fuseid') or '').strip()
    plane_q = (request.args.get('plane') or '').strip().upper()
    op_q = (request.args.get('op') or '').strip().upper()
    file_q = (request.args.get('file') or '').strip()
    # Limit semantics (updated):
    # - blank / 'default' / 'none' / non-positive / invalid => PASS/FAIL-from-log mode (eff_limit=None)
    # - positive numeric => threshold mode
    lim_q = request.args.get('limit')
    eff_limit: float | None = None
    if lim_q is not None:
        _s = lim_q.strip().lower()
        if _s not in ('', 'default', 'none'):
            try:
                v = float(_s)
                if v > 0:
                    eff_limit = v
            except Exception:
                eff_limit = None
    try:
        context = int((request.args.get('context') or '80').strip())
    except Exception:
        context = 80
    try:
        idx_req = int((request.args.get('idx') or '0').strip())
    except Exception:
        idx_req = 0
    # Build a list of likely search roots so we can resolve basenames reliably even without an explicit dir
    try:
        import os as _os
        from pathlib import Path as _Path
        hinted_dirs: set[str] = set()
        for _r in rows:
            _f = (_r.get('fdv_file') or _r.get('fdv') or '').strip()
            if _f and (('/' in _f) or ('\\' in _f)):
                try:
                    hinted_dirs.add(str(_Path(_f).parent))
                except Exception:
                    try:
                        hinted_dirs.add(_os.path.dirname(_f))
                    except Exception:
                        pass
        search_roots: list[str] = []
        for _cand in [base_dir_q, (data.get('dir') or '').strip(), _os.environ.get('FDV_REPORT2_TMPDIR', '').strip(), r'D:\\fdv_tmp', r'C:\\fdv_tmp']:
            if _cand and _cand not in search_roots:
                try:
                    if _os.path.isdir(_cand):
                        search_roots.append(_cand)
                except Exception:
                    pass
        for _d in hinted_dirs:
            try:
                if _d and _d not in search_roots and _os.path.isdir(_d):
                    search_roots.append(_d)
            except Exception:
                pass
    except Exception:
        search_roots = []
    # Collect failing line numbers for the selected split
    fail_lines: List[int] = []
    filename = ''
    orig_filename = ''
    best_row_file_hint = ''
    # Support multiple contributing source files and user selection
    src_q = (request.args.get('src') or '').strip()
    source_files_all: set[str] = set()
    fails_by_src: dict[str, list[int]] = {}
    # Keep per-source metadata for each failing line (fuseid, rber)
    fail_meta_by_src: dict[str, dict[int, dict]] = {}
    def _num_equal(a: str, b: str) -> bool:
        try:
            return abs(float(a) - float(b)) < 1e-9
        except Exception:
            return False
    # Helper: robustly match fdv selector to a row's fdv value (path or name)
    def _fdv_matches(req: str, rowfdv: str) -> bool:
        """Return True when the requested fdv selector matches a row's fdv value.

        Handles these cases robustly:
        - Exact match (path or token)
        - Basename/stem match (case-insensitive)
        - Token extracted from a long log filename (e.g., *_tb_set_utility_TURBO_STATUS.txt)
        - Row values that include pagemap suffixes like _QLC/_TLC/_SSLC/_DSLC while the request path/token does not.
        """
        if not req:
            return True
        if req == rowfdv:
            return True
        try:
            import re as _re
            from pathlib import Path as _Path
            # Normalize row value to basename and stem
            rb = _Path(rowfdv).name if rowfdv else ''
            rbstem = _Path(rb).stem.lower() if rb else ''
            # Normalize request to either basename (if path-like) or raw token
            reqb = _Path(req).name if (("\\" in req) or ("/" in req)) else req
            reqstem = _Path(reqb).stem.lower() if reqb else ''
            # Fast paths
            if reqb and rb and reqb == rb:
                return True
            if reqstem and rbstem and (reqstem == rbstem or rbstem.startswith(reqstem) or reqstem.startswith(rbstem)):
                return True
            if reqstem and rbstem and (reqstem in rbstem or rbstem in reqstem):
                return True
            # Heuristic: drop pagemap suffix from the row fdv (e.g., TURBO_STATUS_QLC -> TURBO_STATUS)
            rbstem_base = _re.sub(r"_(?:qlc|tlc|sslc|dslc|mlc)$", "", rbstem, flags=_re.IGNORECASE) if rbstem else ''
            if rbstem_base:
                # If the base token appears in the long request stem or full request string, consider it a match
                if (rbstem_base in (reqstem or '')):
                    return True
                rq_low = (req or '').lower()
                if rbstem_base and rbstem_base in rq_low:
                    return True
        except Exception:
            pass
        # Final fallback: case-insensitive direct or contains, also with suffix-less row token
        try:
            import re as _re
            R = (rowfdv or '').lower()
            Q = (req or '').lower()
            if Q == R or Q in R or R in Q:
                return True
            Rbase = _re.sub(r"_(?:qlc|tlc|sslc|dslc|mlc)$", "", R)
            if Rbase and (Rbase in Q or Q in Rbase):
                return True
        except Exception:
            pass
        return False

    # Helper: tolerant path-like comparison for src selection (case-insensitive, basename/contains match)
    def _src_like_match(a: str, b: str) -> bool:
        try:
            if not a or not b:
                return False
            import os as _os
            A = (a or '').strip().strip(' "\'')
            B = (b or '').strip().strip(' "\'')
            if A == B:
                return True
            # Normalize slashes and case for contains/basename checks
            An = A.replace('\\','/').lower()
            Bn = B.replace('\\','/').lower()
            if An == Bn:
                return True
            if An in Bn or Bn in An:
                return True
            try:
                Ab = _os.path.basename(A)
            except Exception:
                Ab = A
            try:
                Bb = _os.path.basename(B)
            except Exception:
                Bb = B
            if Ab and Bb and Ab.lower() == Bb.lower():
                return True
            return False
        except Exception:
            return False

    for r in rows:
        k = _get_split_tuple(r)
        if fdv and not _fdv_matches(fdv, k[0]):
            continue
        if pr and pr != k[1]:
            continue
        if vcc and not (vcc == k[2] or _num_equal(vcc, k[2])):
            continue
        if tm and not (tm == k[3] or _num_equal(tm, k[3])):
            continue
        if temp and not (temp == k[4] or _num_equal(temp, k[4])):
            continue
        # Optional additional filters
        if testname_q:
            tn_raw = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
            if (tn_raw or '').strip().lower() != testname_q.strip().lower():
                continue
        if fuseid_q:
            if (_get_fuseid(r) or '').strip() != fuseid_q:
                continue
        if plane_q and plane_q not in ('*','NP'):
            if _plane_from_tname_or_default(r).strip().upper() != plane_q:
                continue
        if op_q and op_q not in ('*','NP'):
            if (r.get('operation') or r.get('op') or '').strip().upper() != op_q:
                continue
        # PASS/FAIL vs threshold by RBER
        pf_raw = (str(r.get('pass_fail') or r.get('PASS_FAIL') or r.get('status') or '')).strip().upper()
        if pf_raw == 'MONITOR':
            continue
        pf_indicated = pf_raw in ('PASS','FAIL')
        if eff_limit is None:
            # Use PASS/FAIL from logs
            if pf_raw != 'FAIL':
                continue
        else:
            # Use numeric threshold only when PASS/FAIL is indicated
            if not pf_indicated:
                continue
            rv = _get_rber(r)
            if rv is None or rv < eff_limit:
                continue
        # Track contributing source file and collect lines per source
        try:
            sf_raw = (r.get('source_file') or '').strip()
            # Prefer only path-like entries for source menu
            def _is_pathlike(s: str) -> bool:
                return bool(s) and (('/' in s) or ('\\' in s))
            sf = sf_raw if _is_pathlike(sf_raw) else ''
            if not sf:
                sf2 = ((r.get('fdv_file') or r.get('fdv') or '') or '').strip()
                sf = sf2 if _is_pathlike(sf2) else ''
            if sf:
                source_files_all.add(sf)
        except Exception:
            sf = ''
        # If a specific source file was requested, skip others
        if src_q:
            rf_full = (r.get('source_file') or '').strip() or (r.get('fdv_file') or r.get('fdv') or '').strip()
            if not _src_like_match(rf_full, src_q):
                continue
        # Keep file and line number; defer final filename decision until after grouping
        if not orig_filename:
            # Keep for header display
            orig_filename = (r.get('fdv_file') or r.get('fdv') or '')
        if not best_row_file_hint:
            try:
                best_row_file_hint = (r.get('source_file') or '').strip() or (r.get('fdv_file') or r.get('fdv') or '').strip()
            except Exception:
                best_row_file_hint = ''
        # Prefer a path-like hint if available
        try:
            _f_hint = (r.get('source_file') or '').strip() or (r.get('fdv_file') or r.get('fdv') or '').strip()
            if _f_hint and (('/' in _f_hint) or ('\\' in _f_hint)):
                # Only override if current hint is empty or token-ish
                if (not best_row_file_hint) or (('/' not in best_row_file_hint) and ('\\' not in best_row_file_hint)):
                    best_row_file_hint = _f_hint
        except Exception:
            pass
        # Robust line number extraction across common keys
        ln = 0
        for lk in ('line_number','lineno','line','line_no','line_num','linenum','_line','_lineno','line_idx','lineindex'):
            if lk in r and r.get(lk) not in (None, ''):
                try:
                    ln = int(str(r.get(lk)).strip())
                    break
                except Exception:
                    try:
                        ln = int(str(r.get(lk)).strip(), 0)
                        break
                    except Exception:
                        ln = 0
        if ln > 0:
            try:
                # Build a robust grouping key: prefer a real path from source_file; else path-like hint; avoid tokenish values
                def _is_pathlike(s: str) -> bool:
                    return bool(s) and (('/' in s) or ('\\' in s))
                # Start with source_file if path-like
                key = ''
                if _is_pathlike(sf_raw):
                    key = sf_raw
                else:
                    # Next, prefer path-like best_row_file_hint
                    key_hint = best_row_file_hint or ''
                    if not _is_pathlike(key_hint):
                        key_hint = orig_filename or ''
                    if _is_pathlike(key_hint):
                        key = key_hint
                    else:
                        # As last resort, try fdv_file/fdv if path-like; else fall back to sf_raw/token
                        sf2 = ((r.get('fdv_file') or r.get('fdv') or '') or '').strip()
                        if _is_pathlike(sf2):
                            key = sf2
                        else:
                            key = sf_raw or sf2 or ''
            except Exception:
                key = sf_raw or ''
            if key:
                fails_by_src.setdefault(key, []).append(ln)
                # Capture metadata for this line if not already set
                try:
                    meta_src = fail_meta_by_src.setdefault(key, {})
                    if ln not in meta_src:
                        meta_src[ln] = {
                            'fuseid': (_get_fuseid(r) or '').strip(),
                            'rber': _get_rber(r),
                        }
                except Exception:
                    pass
            else:
                # no source info; fall back to global list as before
                fail_lines.append(ln)
    # Choose a source for display only when explicitly requested via src=...
    selected_src = ''
    if src_q:
        # prefer tolerant path-like match (exact, basename, or contains)
        try:
            for k in list(fails_by_src.keys()):
                if _src_like_match(k, src_q):
                    selected_src = k
                    break
        except Exception:
            selected_src = src_q
    # Build fail_lines only when a single source is selected (single-source view)
    if selected_src:
        try:
            fail_lines = sorted(set(fails_by_src.get(selected_src, [])))
        except Exception:
            fail_lines = fails_by_src.get(selected_src, []) or fail_lines
    # Metadata map for the chosen source
    selected_meta: dict[int, dict] = {}
    try:
        if selected_src:
            selected_meta = fail_meta_by_src.get(selected_src, {}) or {}
    except Exception:
        selected_meta = {}
    # Establish an initial filename hint based on selected source
    try:
        filename = selected_src or ''
    except Exception:
        filename = filename or ''
    # If caller provided an explicit file path, prefer it unless it looks like an .fdv file (project container, not raw log)
    try:
        if file_q:
            _fq = file_q.strip()
            # Heuristic: ignore .fdv extensions to avoid mislabeling raw file column
            if not _fq.lower().endswith('.fdv'):
                filename = _fq
    except Exception:
        pass
    # If we have a path-like hint from rows and our current filename looks like a token, adopt the hint
    try:
        if (not filename) or (('/' not in filename and '\\' not in filename)):
            if best_row_file_hint and (('/' in best_row_file_hint) or ('\\' in best_row_file_hint)):
                filename = best_row_file_hint
    except Exception:
        pass
    # If nothing set filename yet, and fdv points to a real file path, use it directly
    if not filename and fdv:
        try:
            from pathlib import Path as _Path
            _p = _Path(fdv.strip().strip('"\''))
            if _p.is_file():
                filename = str(_p)
        except Exception:
            pass
    # Fallbacks are handled later during open/read attempt; skip the earlier tight coupling
    # Token-based resolution: if fdv looks like a token (no path), try common patterns under all roots
    if fdv and (('/' not in fdv) and ('\\' not in fdv)):
        try:
            from pathlib import Path as _Path
            import os as _os
            tok = (_Path(fdv).stem or fdv).strip()
            if search_roots and tok:
                candidates2: list[str] = []
                for _root in search_roots:
                    try:
                        R = _Path(_root)
                        # Prioritize exact patterns first
                        for pat in [
                            f"*_tb_set_utility_{tok}.txt",
                            f"*_tb_set_utility*{tok}*.txt",
                            f"*_set_utility_tb_{tok}.txt",
                            f"*_set_utility_tb_*{tok}*.txt",
                            f"*Output*tb_set_utility*{tok}*.txt",
                            f"*FDVLOG*{tok}*.txt",
                            f"*{tok}*.txt",
                            f"*{tok}*.log",
                            f"*FDVLOG*{tok}*.log",
                        ]:
                            try:
                                for p in R.rglob(pat):
                                    if p.is_file():
                                        candidates2.append(str(p))
                            except Exception:
                                continue
                        # Fallback: case-insensitive walk
                        try:
                            tok_l = tok.lower()
                            for root, _dirs, files in _os.walk(str(R)):
                                for name in files:
                                    nlow = name.lower()
                                    if (tok_l in nlow) and (nlow.endswith('.txt') or nlow.endswith('.log')):
                                        try:
                                            candidates2.append(str(_Path(root) / name))
                                        except Exception:
                                            continue
                        except Exception:
                            pass
                    except Exception:
                        continue
                # Also try exact basenames from rows mentioning the token, across all roots
                try:
                    row_basenames: set[str] = set()
                    for rr in rows:
                        rf = (rr.get('fdv_file') or rr.get('fdv') or '').strip()
                        if rf and (tok.lower() in rf.lower()):
                            try:
                                row_basenames.add(_Path(rf).name)
                            except Exception:
                                try:
                                    row_basenames.add(_os.path.basename(rf))
                                except Exception:
                                    pass
                    for bn in row_basenames:
                        for _root in search_roots:
                            try:
                                R = _Path(_root)
                                for p in R.rglob(bn):
                                    if p.is_file():
                                        candidates2.append(str(p))
                            except Exception:
                                continue
                except Exception:
                    pass
                if candidates2:
                    try:
                        candidates2 = list(dict.fromkeys(candidates2))
                    except Exception:
                        pass
                    try:
                        candidates2.sort(key=lambda p: _os.path.getmtime(p), reverse=True)
                    except Exception:
                        pass
                    filename = candidates2[0]
        except Exception:
            pass
    # Sort and clamp index
    fail_lines = sorted(set(fail_lines))
    no_line_numbers = False
    if not fail_lines:
        # There are failing rows but no line numbers available; still open the file at the top
        no_line_numbers = True
        fail_lines = [1]
    if idx_req < 0:
        idx_req = 0
    if idx_req >= len(fail_lines):
        idx_req = len(fail_lines) - 1
    # Read the raw file and build a snippet around the selected failing line
    target_line = fail_lines[idx_req]
    # Track attempted file paths for diagnostics and header display
    attempted_paths: List[str] = []
    # Normalize filename early; if missing, consider fdv query as a direct path
    try:
        _fname_try = filename.strip(' "\'') if filename else ''
    except Exception:
        _fname_try = filename or ''
    if (not _fname_try) and fdv:
        try:
            _fname_try = (fdv or '').strip().strip(' "\'')
        except Exception:
            _fname_try = fdv or ''
    try:
        open_target = _fname_try if _fname_try else filename
        with open(open_target, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            filename = open_target or filename
    except Exception:
        # Record attempted path
        try:
            if open_target:
                attempted_paths.append(str(open_target))
        except Exception:
            pass
        # Try relative to used_dir if absolute path failed
        try:
            base = (base_dir_q or (data.get('dir') or '').strip())
            if base:
                from pathlib import Path as _Path
                fp = _Path(base) / _Path(open_target or filename or '').name
                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
                    filename = str(fp)
            else:
                all_lines = []
        except Exception:
            # Record attempted base-joined path (if constructed)
            try:
                if 'fp' in locals():
                    attempted_paths.append(str(fp))
            except Exception:
                pass
            all_lines = []
    # If we still couldn't read, try additional robust fallbacks:
    n = len(all_lines)
    if n == 0:
        try:
            import os as _os
            from pathlib import Path as _Path
            # For diagnostics, show all roots we’ll search
            roots = search_roots or []
            search_root = '; '.join(roots)
            # Seed diagnostics with the search root so the UI shows something useful
            try:
                if roots:
                    for _sr in roots:
                        attempted_paths.append(f"SEARCH_ROOT: {_sr}")
            except Exception:
                pass
            # Normalize provided filename and get basename
            fn_base = ''
            try:
                _fname_clean = filename.strip(' "\'') if filename else ''
                fn_base = _Path(_fname_clean).name if _fname_clean else ''
            except Exception:
                fn_base = _os.path.basename(filename) if filename else ''

            # 0) Seed candidates from fdv query path directly (if provided)
            candidates: list[str] = []
            try:
                _fdv_req = (fdv or '').strip()
                if _fdv_req:
                    _fdv_clean = _fdv_req.strip(' "\'')
                    if _fdv_clean:
                        _fdv_path = _Path(_fdv_clean)
                        # Try the exact path
                        candidates.append(str(_fdv_path))
                        # Also try joining with each search root if not absolute
                        if roots and not _fdv_path.is_absolute():
                            for _sr in roots:
                                try:
                                    candidates.append(str(_Path(_sr) / _fdv_path.name))
                                except Exception:
                                    continue
            except Exception:
                pass

            # 1) Try any candidate paths from rows that share the same basename
            if fn_base:
                seen: set[str] = set()
                for r in rows:
                    f = (r.get('source_file') or '').strip() or (r.get('fdv_file') or r.get('fdv') or '').strip()
                    if not f:
                        continue
                    try:
                        b = _Path(f).name
                    except Exception:
                        b = _os.path.basename(f)
                    if b == fn_base and f not in seen:
                        candidates.append(f)
                        seen.add(f)
            # 1b) From rows: any source_file/fdv_file that contains the fdv token (case-insensitive)
            try:
                _tok = (fdv or '').strip()
                if _tok:
                    _tok_l = _tok.lower()
                    seen2: set[str] = set(candidates)
                    for r in rows:
                        f = (r.get('source_file') or '').strip() or (r.get('fdv_file') or r.get('fdv') or '').strip()
                        if f and (_tok_l in f.lower()) and (f not in seen2):
                            candidates.append(f)
                            seen2.add(f)
            except Exception:
                pass
            # 2) Search recursively for the basename under all roots
            if roots and fn_base:
                for _root in roots:
                    try:
                        root = _Path(_root)
                        for p in root.rglob(fn_base):
                            fp = str(p)
                            if fp not in candidates:
                                candidates.append(fp)
                    except Exception:
                        continue
            # 2b) If fdv provides a token (like 'AFTER_RESET'), search for files whose name contains it across all roots
            try:
                _fdv_req2 = (fdv or '').strip()
                if _fdv_req2:
                    _fdv_token = _Path(_fdv_req2).stem or _fdv_req2
                    _fdv_token = str(_fdv_token).strip()
                    if roots and _fdv_token:
                        for _root in roots:
                            try:
                                root = _Path(_root)
                                # Generic token search
                                for p in root.rglob(f"*{_fdv_token}*"):
                                    try:
                                        if p.is_file():
                                            fp = str(p)
                                            if fp not in candidates:
                                                candidates.append(fp)
                                    except Exception:
                                        continue
                                # Common FDV naming patterns
                                for pat in [
                                    f"*_tb_set_utility_{_fdv_token}.txt",
                                    f"*_tb_set_utility*{_fdv_token}*.txt",
                                    f"*_set_utility_tb_{_fdv_token}.txt",
                                    f"*_set_utility_tb*{_fdv_token}*.txt",
                                    f"*Output*tb_set_utility*{_fdv_token}*.txt",
                                    f"*FDVLOG*{_fdv_token}*.txt",
                                    f"*{_fdv_token}*.txt",
                                    f"*{_fdv_token}*.log",
                                ]:
                                    try:
                                        for p in root.rglob(pat):
                                            if p.is_file():
                                                fp = str(p)
                                                if fp not in candidates:
                                                    candidates.append(fp)
                                    except Exception:
                                        continue
                            except Exception:
                                continue
            except Exception:
                pass
            # 2c) Also search under any directory hinted by row file paths
            try:
                hinted_dirs: set[str] = set()
                for r in rows:
                    f = (r.get('fdv_file') or r.get('fdv') or '').strip()
                    if not f:
                        continue
                    try:
                        d = str(_Path(f).parent)
                        if d:
                            hinted_dirs.add(d)
                    except Exception:
                        continue
                _fdv_req2 = (fdv or '').strip()
                _fdv_token = _Path(_fdv_req2).stem if _fdv_req2 else ''
                for d in hinted_dirs:
                    try:
                        root = _Path(d)
                        # Search by exact basename first if we have it
                        if fn_base:
                            for p in root.rglob(fn_base):
                                fp = str(p)
                                if fp not in candidates:
                                    candidates.append(fp)
                        # Then search by token
                        if _fdv_token:
                            for p in root.rglob(f"*{_fdv_token}*"):
                                try:
                                    if p.is_file():
                                        fp = str(p)
                                        if fp not in candidates:
                                            candidates.append(fp)
                                except Exception:
                                    continue
                    except Exception:
                        continue
            except Exception:
                pass
            # Choose a better default filename for header display: prefer an existing file or a candidate with a path separator
            if (not filename) and candidates:
                try:
                    from pathlib import Path as _Path
                    chosen = None
                    # 1) Existing file
                    for cand in candidates:
                        try:
                            if _Path(cand).is_file():
                                chosen = cand
                                break
                        except Exception:
                            continue
                    # 2) Path-like strings (contains / or \\)
                    if not chosen:
                        for cand in candidates:
                            if ('/' in cand) or ('\\' in cand):
                                chosen = cand
                                break
                    # 3) Fallback to first
                    if not chosen:
                        chosen = candidates[0]
                    filename = chosen
                except Exception:
                    pass
            # 3) Try to open the first readable candidate
            for cand in candidates:
                try:
                    with open(cand, 'r', encoding='utf-8', errors='replace') as f:
                        all_lines = f.readlines()
                        filename = cand
                        break
                except Exception:
                    try:
                        attempted_paths.append(str(cand))
                    except Exception:
                        pass
                    continue
            n = len(all_lines)
            # If still no candidates were found, record that in attempted_paths so the UI shows it
            if (not candidates) and roots:
                try:
                    tok = (fdv or '').strip()
                    attempted_paths.append(f"NO_MATCHES: token='{tok}' under roots: {', '.join(roots)}")
                except Exception:
                    pass
        except Exception:
            n = 0
    read_failed = (n == 0)
    # Advanced reconstruction: if we failed to read and filename looks like a composite key (contains '|')
    # or a bare token (no path separators), attempt an expanded search using fragments derived from the key
    # and any testname present in the filtered rows. This targets cases like
    #   'QLC_AUTO_READ_CAL|20|2.5|19|25'  -> real file '...tb_set_utility_AUTO_READ_CALIBRATION.txt'
    if read_failed:
        try:
            import os as _os
            from pathlib import Path as _Path
            key_like = filename or fdv or ''
            if key_like and (('|' in key_like) or (('/' not in key_like) and ('\\' not in key_like))):
                # Extract the leftmost token before '|' and strip pagemap prefixes (QLC_, TLC_, etc.)
                core = key_like.split('|')[0]
                pagemap_prefixes = ('QLC_','TLC_','SLC_','SSLC_','DSLC_','MLC_')
                for pre in pagemap_prefixes:
                    if core.upper().startswith(pre):
                        core = core[len(pre):]
                        break
                # Build a set of token fragments (progressively shortened) to widen search
                fragments: set[str] = set()
                if core:
                    fragments.add(core)
                    # Also include progressively truncated forms at underscores
                    parts = core.split('_')
                    if len(parts) > 2:  # keep longest 2-3 segment combinations
                        for L in range(len(parts), 1, -1):
                            frag = '_'.join(parts[:L])
                            if len(frag) >= 6:
                                fragments.add(frag)
                    # Include collapsed form without underscores for broad match
                    try:
                        collapsed = ''.join(parts)
                        if len(collapsed) >= 6:
                            fragments.add(collapsed)
                    except Exception:
                        pass
                # Pull a testname from an example failing row (if any) to refine
                example_testname = ''
                try:
                    # Reconstruct an index->row mapping quickly using line numbers collected
                    # We'll pick the first row matching the selected filters
                    for rr in rows:
                        k = _get_split_tuple(rr)
                        if fdv and not _fdv_matches(fdv, k[0]):
                            continue
                        if pr and pr != k[1]:
                            continue
                        if vcc and not (vcc == k[2] or _num_equal(vcc, k[2])):
                            continue
                        if tm and not (tm == k[3] or _num_equal(tm, k[3])):
                            continue
                        if temp and not (temp == k[4] or _num_equal(temp, k[4])):
                            continue
                        tn_raw = (rr.get('testname','') or rr.get('tname','') or '').strip()
                        if tn_raw:
                            example_testname = tn_raw
                            break
                except Exception:
                    example_testname = ''
                if example_testname:
                    fragments.add(example_testname)
                # Lowercase fragments for matching
                frag_l = {f.lower() for f in fragments if f}
                if frag_l:
                    # Build search roots: existing search_roots plus common log locations and PR-specific subdirs
                    adv_roots: list[str] = []
                    try:
                        for r_ in (search_roots if 'search_roots' in locals() else []):
                            if r_ and r_ not in adv_roots:
                                adv_roots.append(r_)
                    except Exception:
                        pass
                    for guess in [r'D:\\logs', r'D:\\logs\\fdvrun']:
                        if guess not in adv_roots and _os.path.isdir(guess):
                            adv_roots.append(guess)
                    # PR-specific subdirectories under known roots
                    if pr:
                        try:
                            for base in list(adv_roots):
                                pr_dir = _os.path.join(base, f"PR{pr}")
                                if _os.path.isdir(pr_dir) and pr_dir not in adv_roots:
                                    adv_roots.append(pr_dir)
                        except Exception:
                            pass
                    # Deduplicate
                    try:
                        adv_roots = list(dict.fromkeys(adv_roots))
                    except Exception:
                        pass
                    candidates_adv: list[str] = []
                    # Patterns to prefer (ordered)
                    patterns = []
                    for frag in frag_l:
                        patterns.extend([
                            f"*tb_set_utility*{frag}*.txt",
                            f"*{frag}*.txt"
                        ])
                    # Perform search (bounded: stop after reasonable number of hits)
                    try:
                        for root in adv_roots:
                            try:
                                p_root = _Path(root)
                                for pat in patterns:
                                    try:
                                        for p in p_root.rglob(pat):
                                            if p.is_file():
                                                sp = str(p)
                                                candidates_adv.append(sp)
                                                if len(candidates_adv) >= 200:  # safeguard
                                                    break
                                        if len(candidates_adv) >= 200:
                                            break
                                    except Exception:
                                        continue
                                if len(candidates_adv) >= 200:
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass
                    # Rank candidates: prefer those containing longer fragment (e.g., full AUTO_READ_CALIBRATION) and with recent mtime
                    if candidates_adv:
                        try:
                            uniq = []
                            seen = set()
                            for c in candidates_adv:
                                if c not in seen:
                                    uniq.append(c); seen.add(c)
                            candidates_adv = uniq
                        except Exception:
                            pass
                        def _rank_key(path_str: str):
                            try:
                                base = _os.path.basename(path_str).lower()
                                match_score = max((len(f) if f in base else 0) for f in frag_l)
                            except Exception:
                                match_score = 0
                            try:
                                mtime = _os.path.getmtime(path_str)
                            except Exception:
                                mtime = 0
                            return (-match_score, -mtime, len(base))
                        try:
                            candidates_adv.sort(key=_rank_key)
                        except Exception:
                            pass
                        # Pick best candidate we can read
                        for cand in candidates_adv:
                            try:
                                with open(cand, 'r', encoding='utf-8', errors='replace') as f:
                                    all_lines = f.readlines()
                                    filename = cand
                                    read_failed = False
                                    attempted_paths.append(f"ADV_SEARCH_SUCCESS: {cand}")
                                    break
                            except Exception:
                                try:
                                    attempted_paths.append(f"ADV_SEARCH_TRY: {cand}")
                                except Exception:
                                    pass
                        # If still failed, at least record that advanced search ran
                        if read_failed:
                            try:
                                attempted_paths.append("ADV_SEARCH_NO_MATCH: fragments=" + ','.join(sorted(frag_l)) + " roots=" + ';'.join(adv_roots))
                            except Exception:
                                pass
        except Exception:
            pass
    # If still unreadable, continue to render raw view (fail list + note) instead of redirecting
    snippet = []
    if not read_failed:
        start = max(1, target_line - context)
        end = min(n, target_line + context)
        fail_set = set(fail_lines)
        for ln in range(start, end + 1):
            try:
                content = all_lines[ln - 1].rstrip('\n')
            except Exception:
                content = ''
            snippet.append({'ln': ln, 'content': content, 'is_target': (ln in fail_set), 'is_current': (ln == target_line)})
    # Build navigation URLs
    def _build_url(next_idx: int) -> str:
        from urllib.parse import urlencode
        q = {
            # Preserve original fdv filter; carry resolved filename separately via 'file'
            'fdv': fdv,
            'pr': pr, 'vcc': vcc, 'tm': tm, 'temp': temp,
            'limit': ('' if (eff_limit is None and (lim_q is not None)) else f"{eff_limit}"), 'context': f"{context}", 'idx': str(next_idx)
        }
        if testname_q: q['testname'] = testname_q
        if fuseid_q: q['fuseid'] = fuseid_q
        if plane_q: q['plane'] = plane_q
        if op_q: q['op'] = op_q
        if base_dir_q: q['dir'] = base_dir_q
        if src_q: q['src'] = src_q
        if filename:
            q['file'] = filename
        return url_for('report_fails', token=token) + '?' + urlencode({k:v for k,v in q.items() if v is not None})
    back_url = url_for('report_home') + f"?token={token}"
    prev_url = _build_url(max(0, idx_req - 1)) if (idx_req > 0 and not no_line_numbers) else ''
    next_url = _build_url(min(len(fail_lines)-1, idx_req + 1)) if (idx_req < len(fail_lines)-1 and not no_line_numbers) else ''
    note = 'Line numbers were not available; showing file from the top.' if no_line_numbers else ''
    if read_failed:
        note = (note + ' ' if note else '') + 'Could not read the source file; showing failing line numbers only.'
    # If read failed and we have attempted candidates, keep them to show in UI
    try:
        _attempted = attempted_paths if read_failed else []
    except Exception:
        _attempted = []
    # Keep fdv filter stable; do not overwrite fdv with resolved filename
    # Prepare a compact list of all failing lines
    fail_items = []
    total_count = None
    # If no specific source was selected, aggregate all sources into one table
    if (not src_q) and fails_by_src:
        try:
            import os as _os
            from pathlib import Path as _Path
            def _resolve_src_path(_spath: str) -> str:
                try:
                    if _spath and _os.path.isfile(_spath):
                        return _spath
                    bn = ''
                    try:
                        bn = _Path(_spath).name if _spath else ''
                    except Exception:
                        try:
                            bn = _os.path.basename(_spath) if _spath else ''
                        except Exception:
                            bn = ''
                    # Try base_dir then search_roots
                    if bn:
                        base = (base_dir_q or (data.get('dir') or '').strip())
                        if base:
                            cand = str(_Path(base) / bn)
                            try:
                                if _os.path.isfile(cand):
                                    return cand
                            except Exception:
                                pass
                        try:
                            for _root in (search_roots or []):
                                cand = str(_Path(_root) / bn)
                                if _os.path.isfile(cand):
                                    return cand
                        except Exception:
                            pass
                    return _spath
                except Exception:
                    return _spath
            total = 0
            for src_key in sorted(fails_by_src.keys()):
                lines_list = sorted(set(fails_by_src.get(src_key, [])))
                if not lines_list:
                    continue
                resolved = _resolve_src_path(src_key)
                # Read file once
                try:
                    with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                        src_lines = f.readlines()
                except Exception:
                    src_lines = []
                meta_src = fail_meta_by_src.get(src_key, {})
                for ln in lines_list:
                    try:
                        c = src_lines[ln - 1].rstrip('\n') if (ln-1) < len(src_lines) and (ln-1) >= 0 else ''
                    except Exception:
                        c = ''
                    fus = ''
                    rber_val = None
                    # Prefer cached metadata when available
                    try:
                        mi = meta_src.get(ln)
                        if mi:
                            fus = (mi.get('fuseid') or '').strip()
                            rber_val = mi.get('rber')
                    except Exception:
                        fus = fus
                    # Locate the matching parsed row to improve accuracy (content fallback + source file path)
                    rr_match = None
                    try:
                        for rr in rows:
                            k = _get_split_tuple(rr)
                            if fdv and not _fdv_matches(fdv, k[0]):
                                continue
                            if pr and pr != k[1]:
                                continue
                            if vcc and not (vcc == k[2] or _num_equal(vcc, k[2])):
                                continue
                            if tm and not (tm == k[3] or _num_equal(tm, k[3])):
                                continue
                            if temp and not (temp == k[4] or _num_equal(temp, k[4])):
                                continue
                            rf_full = (rr.get('source_file') or '').strip() or (rr.get('fdv_file') or rr.get('fdv') or '').strip()
                            if not _src_like_match(rf_full, src_key):
                                continue
                            ln2 = 0
                            for lk in ('line_number','lineno','line','line_no','line_num','linenum','_line','_lineno','line_idx','lineindex'):
                                if lk in rr and rr.get(lk) not in (None, ''):
                                    try:
                                        ln2 = int(str(rr.get(lk)).strip())
                                        break
                                    except Exception:
                                        try:
                                            ln2 = int(str(rr.get(lk)).strip(), 0)
                                            break
                                        except Exception:
                                            ln2 = 0
                            if ln2 != ln:
                                continue
                            rr_match = rr
                            break
                    except Exception:
                        rr_match = None
                    # If file content was empty, fall back to embedded raw_line from parsed row
                    try:
                        if (not c) and rr_match:
                            c2 = (rr_match.get('raw_line') or '').rstrip('\n')
                            if c2:
                                c = c2
                    except Exception:
                        pass
                    # If metadata missing, fill from matched row
                    if rr_match and (not fus or rber_val is None):
                        try:
                            if not fus:
                                fus = (_get_fuseid(rr_match) or '').strip()
                            if rber_val is None:
                                rber_val = _get_rber(rr_match)
                        except Exception:
                            pass
                    # If the raw line contains an explicit RBER/BER token, prefer that for display
                    try:
                        import re as _re
                        m = _re.search(r"(?:\bRBER\b|\bRAW[_ ]?BER\b|\bBER\b|\bERROR[_ ]?RATE\b)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", c)
                        if m:
                            rber_from_text = float(m.group(1))
                            rber_val = rber_from_text
                    except Exception:
                        pass
                    try:
                        rber_fmt = ("{:.3e}".format(float(rber_val)) if (rber_val is not None) else '')
                    except Exception:
                        rber_fmt = ''
                    # Choose the most accurate file path for the row: prefer the row's actual source_file if available
                    try:
                        rf_row = ''
                        if rr_match:
                            rf_row = (rr_match.get('source_file') or '').strip()
                        # Prefer per-row source_file; fall back to resolved/src_key when unavailable
                        file_full = rf_row or (resolved or src_key or '')
                        # Heuristic: prefer Output_*.txt when available
                        def _pick_output_like(primary: str, fallback1: str, fallback2: str, token_hint: str) -> str:
                            def bn(p: str) -> str:
                                try:
                                    return _os.path.basename(p)
                                except Exception:
                                    return p or ''
                            def starts_output(p: str) -> bool:
                                try:
                                    return bn(p).lower().startswith('output_') and bn(p).lower().endswith('.txt')
                                except Exception:
                                    return False
                            # Quick wins
                            if starts_output(primary):
                                return primary
                            if starts_output(fallback1):
                                return fallback1
                            if starts_output(fallback2):
                                return fallback2
                            # Try to locate a sibling Output file in same dir as primary/fallback1
                            token_l = (token_hint or '').strip().lower()
                            for base in [primary, fallback1]:
                                try:
                                    if not base:
                                        continue
                                    d = _os.path.dirname(base)
                                    if d and _os.path.isdir(d):
                                        try:
                                            cand = []
                                            for name in _os.listdir(d):
                                                low = name.lower()
                                                if low.startswith('output_') and low.endswith('.txt'):
                                                    if (not token_l) or ('tb_set_utility' in low and token_l in low):
                                                        cand.append(_os.path.join(d, name))
                                            if cand:
                                                # pick latest by mtime
                                                try:
                                                    cand.sort(key=lambda p: _os.path.getmtime(p), reverse=True)
                                                except Exception:
                                                    pass
                                                return cand[0]
                                        except Exception:
                                            continue
                                except Exception:
                                    continue
                            # As a broader fallback, search known roots for Output logs matching the token
                            try:
                                roots = list(search_roots) if ('search_roots' in locals() and search_roots) else []
                            except Exception:
                                roots = []
                            if roots:
                                try:
                                    patt = []
                                    if token_l:
                                        patt.extend([
                                            f"*Output*tb_set_utility*{token_l}*.txt",
                                            f"Output_*{token_l}*.txt",
                                        ])
                                    # generic Output fallback
                                    patt.append("Output_*.txt")
                                    hits: list[str] = []
                                    for r_ in roots:
                                        try:
                                            R = _Path(r_)
                                            for pat in patt:
                                                try:
                                                    for pth in R.rglob(pat):
                                                        if pth.is_file():
                                                            hits.append(str(pth))
                                                except Exception:
                                                    continue
                                        except Exception:
                                            continue
                                    if hits:
                                        try:
                                            # dedupe and sort by mtime desc
                                            hits = list(dict.fromkeys(hits))
                                        except Exception:
                                            pass
                                        try:
                                            hits.sort(key=lambda p: _os.path.getmtime(p), reverse=True)
                                        except Exception:
                                            pass
                                        return hits[0]
                                except Exception:
                                    pass
                            return primary or fallback1 or fallback2
                        # Extract a token hint from fdv_file if present
                        token_hint = ''
                        try:
                            if rr_match:
                                _fdv_f = (rr_match.get('fdv_file') or rr_match.get('fdv') or '')
                                import re as _re
                                m = _re.search(r"tb_set_utility_([A-Za-z0-9_\-]+)", _fdv_f, flags=_re.IGNORECASE)
                                if m:
                                    token_hint = m.group(1).lower()
                        except Exception:
                            token_hint = ''
                        # If we have a per-row source file, use it directly (it is the actual processed log)
                        if rf_row:
                            file_full = rf_row
                        else:
                            file_full = _pick_output_like(file_full, resolved or '', src_key or '', token_hint)
                        _fb = _os.path.basename(file_full) if file_full else ''
                    except Exception:
                        file_full = resolved or src_key or ''
                        _fb = file_full
                    fail_items.append({
                        'ln': ln,
                        'file': file_full,
                        'file_base': _fb,
                        'content': c,
                        'url': '',  # no per-line navigation in aggregate view
                        'is_current': False,
                        'fuseid': fus,
                        'rber': rber_val,
                        'rber_fmt': rber_fmt
                    })
                total += len(lines_list)
            total_count = total
            # In aggregate view, disable prev/next navigation
            prev_url = ''
            next_url = ''
            idx_req = 0
        except Exception:
            fail_items = []
    else:
        # Single-source (legacy) behavior
        # We will set per-row file info based on matching row's source_file when possible
        try:
            import os as _os
        except Exception:
            pass
    if not no_line_numbers:
            try:
                for i, ln in enumerate(fail_lines):
                    try:
                        c = all_lines[ln - 1].rstrip('\n')
                    except Exception:
                        c = ''
                    # Lookup metadata (fuseid, rber) for this line
                    fus = ''
                    rber_val = None
                    file_full_single = ''
                    try:
                        mi = selected_meta.get(ln)
                        if mi:
                            fus = (mi.get('fuseid') or '').strip()
                            rber_val = mi.get('rber')
                    except Exception:
                        fus = fus
                    # Fallback: try to locate a matching row for this line
                    if (not fus) and (rber_val is None):
                        try:
                            for rr in rows:
                                # Match basic split and line number
                                k = _get_split_tuple(rr)
                                if fdv and not _fdv_matches(fdv, k[0]):
                                    continue
                                if pr and pr != k[1]:
                                    continue
                                if vcc and not (vcc == k[2] or _num_equal(vcc, k[2])):
                                    continue
                                if tm and not (tm == k[3] or _num_equal(tm, k[3])):
                                    continue
                                if temp and not (temp == k[4] or _num_equal(temp, k[4])):
                                    continue
                                # Source file scope
                                if selected_src:
                                    rf_full = (rr.get('source_file') or '').strip() or (rr.get('fdv_file') or rr.get('fdv') or '').strip()
                                    if not _src_like_match(rf_full, selected_src):
                                        continue
                                # Line number match
                                ln2 = 0
                                for lk in ('line_number','lineno','line','line_no','line_num','linenum','_line','_lineno','line_idx','lineindex'):
                                    if lk in rr and rr.get(lk) not in (None, ''):
                                        try:
                                            ln2 = int(str(rr.get(lk)).strip())
                                            break
                                        except Exception:
                                            try:
                                                ln2 = int(str(rr.get(lk)).strip(), 0)
                                                break
                                            except Exception:
                                                ln2 = 0
                                if ln2 != ln:
                                    continue
                                fus = (_get_fuseid(rr) or '').strip()
                                rber_val = _get_rber(rr)
                                # Capture source log file from the row when available
                                try:
                                    file_full_single = (rr.get('source_file') or '').strip()
                                except Exception:
                                    file_full_single = file_full_single
                                break
                        except Exception:
                            pass
                    # If file content was empty, fall back to embedded raw_line from parsed row
                    try:
                        if (not c):
                            c2 = ''
                            # Locate matching row for fallback content
                            for rr in rows:
                                k = _get_split_tuple(rr)
                                if fdv and not _fdv_matches(fdv, k[0]):
                                    continue
                                if pr and pr != k[1]:
                                    continue
                                if vcc and not (vcc == k[2] or _num_equal(vcc, k[2])):
                                    continue
                                if tm and not (tm == k[3] or _num_equal(tm, k[3])):
                                    continue
                                if temp and not (temp == k[4] or _num_equal(temp, k[4])):
                                    continue
                                if selected_src:
                                    rf_full = (rr.get('source_file') or '').strip() or (rr.get('fdv_file') or rr.get('fdv') or '').strip()
                                    if not _src_like_match(rf_full, selected_src):
                                        continue
                                ln2 = 0
                                for lk in ('line_number','lineno','line','line_no','line_num','linenum','_line','_lineno','line_idx','lineindex'):
                                    if lk in rr and rr.get(lk) not in (None, ''):
                                        try:
                                            ln2 = int(str(rr.get(lk)).strip())
                                            break
                                        except Exception:
                                            try:
                                                ln2 = int(str(rr.get(lk)).strip(), 0)
                                                break
                                            except Exception:
                                                ln2 = 0
                                if ln2 != ln:
                                    continue
                                c2 = (rr.get('raw_line') or '').rstrip('\n')
                                # Capture source file if not already set
                                try:
                                    if not file_full_single:
                                        file_full_single = (rr.get('source_file') or '').strip()
                                except Exception:
                                    pass
                                if c2:
                                    break
                            if c2:
                                c = c2
                    except Exception:
                        pass
                    # If the raw line contains an explicit RBER/BER token, prefer that for display to ensure exact correspondence
                    try:
                        import re as _re
                        m = _re.search(r"(?:\bRBER\b|\bRAW[_ ]?BER\b|\bBER\b|\bERROR[_ ]?RATE\b)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", c)
                        if m:
                            rber_from_text = float(m.group(1))
                            # Use text-extracted RBER for display; keep parsed rber_val as fallback
                            rber_val = rber_from_text
                    except Exception:
                        pass
                    # Format RBER
                    try:
                        rber_fmt = ("{:.3e}".format(float(rber_val)) if (rber_val is not None) else '')
                    except Exception:
                        rber_fmt = ''
                    # Determine display file path for this row (prefer row.source_file; else the opened filename)
                    try:
                        # Prefer Output_*.txt basename when possible in single-source mode
                        def _pick_output_like_single(primary: str, fallback1: str, token_hint: str) -> str:
                            try:
                                import os as __os
                            except Exception:
                                __os = _os
                            def bn(p: str) -> str:
                                try:
                                    return __os.path.basename(p)
                                except Exception:
                                    return p or ''
                            def starts_output(p: str) -> bool:
                                try:
                                    b = bn(p).lower()
                                    return b.startswith('output_') and b.endswith('.txt')
                                except Exception:
                                    return False
                            if starts_output(primary):
                                return primary
                            if starts_output(fallback1):
                                return fallback1
                            # Try sibling search in dir of fallback1
                            token_l = (token_hint or '').strip().lower()
                            try:
                                d = __os.path.dirname(fallback1)
                                if d and __os.path.isdir(d):
                                    cand = []
                                    for name in __os.listdir(d):
                                        low = name.lower()
                                        if low.startswith('output_') and low.endswith('.txt'):
                                            if (not token_l) or ('tb_set_utility' in low and token_l in low):
                                                cand.append(__os.path.join(d, name))
                                    if cand:
                                        try:
                                            cand.sort(key=lambda p: __os.path.getmtime(p), reverse=True)
                                        except Exception:
                                            pass
                                        return cand[0]
                            except Exception:
                                pass
                            # As a broader fallback, search known roots for Output logs matching the token
                            try:
                                roots = list(search_roots) if ('search_roots' in locals() and search_roots) else []
                            except Exception:
                                roots = []
                            if roots:
                                try:
                                    patt = []
                                    if token_l:
                                        patt.extend([
                                            f"*Output*tb_set_utility*{token_l}*.txt",
                                            f"Output_*{token_l}*.txt",
                                        ])
                                    patt.append("Output_*.txt")
                                    hits: list[str] = []
                                    for r_ in roots:
                                        try:
                                            R = _Path(r_)
                                            for pat in patt:
                                                try:
                                                    for pth in R.rglob(pat):
                                                        if pth.is_file():
                                                            hits.append(str(pth))
                                                except Exception:
                                                    continue
                                        except Exception:
                                            continue
                                    if hits:
                                        try:
                                            hits = list(dict.fromkeys(hits))
                                        except Exception:
                                            pass
                                        try:
                                            hits.sort(key=lambda p: __os.path.getmtime(p), reverse=True)
                                        except Exception:
                                            pass
                                        return hits[0]
                                except Exception:
                                    pass
                            return primary or fallback1
                        # token hint from any matching row (if we captured earlier)
                        token_hint2 = ''
                        # Try to extract token from selected_src or filename
                        try:
                            import re as _re
                            m2 = _re.search(r"tb_set_utility_([A-Za-z0-9_\-]+)", (filename or ''), flags=_re.IGNORECASE)
                            if m2:
                                token_hint2 = m2.group(1).lower()
                        except Exception:
                            token_hint2 = ''
                        # If the row captured a source_file, show it directly; else use heuristic to pick Output_*.txt
                        if file_full_single:
                            file_disp = file_full_single
                        else:
                            file_disp = _pick_output_like_single(file_full_single or '', (filename or ''), token_hint2)
                        file_base_disp = _os.path.basename(file_disp) if file_disp else ''
                    except Exception:
                        file_disp = (file_full_single or filename or '')
                        file_base_disp = file_disp
                    fail_items.append({
                        'ln': ln,
                        'file': file_disp,
                        'file_base': file_base_disp,
                        'content': c,
                        'url': _build_url(i),
                        'is_current': (i == idx_req),
                        'fuseid': fus,
                        'rber': rber_val,
                        'rber_fmt': rber_fmt
                    })
            except Exception:
                fail_items = []
    # Ensure we always show something in the filename slot at the top of the page
    try:
        _display_name = filename.strip() if filename else (fdv.strip() if fdv else '(unresolved file)')
    except Exception:
        _display_name = filename or fdv or '(unresolved file)'
    # If display name is token-ish, try to upgrade using row hint or used_dir+basename
    try:
        _tokenish = _display_name and ('/' not in _display_name and '\\' not in _display_name)
        if _tokenish:
            if best_row_file_hint and (('/' in best_row_file_hint) or ('\\' in best_row_file_hint)):
                _display_name = best_row_file_hint
            else:
                import os as _os
                _fdv_req = (fdv or '').strip()
                _used_dir2 = (base_dir_q or (data.get('dir') or '').strip())
                if _fdv_req and _used_dir2:
                    _bn = _os.path.basename(_fdv_req)
                    if _bn:
                        _display_name = str(_os.path.join(_used_dir2, _bn))
    except Exception:
        pass
    # If we couldn't read and only have a token like 'AFTER_RESET', but we collected attempted full paths,
    # prefer showing the first attempted path in the header so users see the real location.
    try:
        _is_tokenish = _display_name and ('/' not in _display_name and '\\' not in _display_name)
        if read_failed and _is_tokenish:
            # If we have attempted path entries, prefer a path-like one
            _cand = None
            try:
                for _p in attempted_paths or []:
                    if ('/' in _p) or ('\\' in _p):
                        _cand = _p
                        break
            except Exception:
                _cand = None
            if not _cand:
                # Synthesize a likely full path using the first search root and fdv basename
                try:
                    import os as _os
                    from pathlib import Path as _Path
                    # Use previously computed search_roots if available; otherwise build a minimal list
                    try:
                        roots = list(search_roots) if 'search_roots' in locals() and search_roots else []
                    except Exception:
                        roots = []
                    if not roots:
                        sr = (base_dir_q or (data.get('dir') or '').strip())
                        if sr:
                            roots.append(sr)
                        envr = _os.environ.get('FDV_REPORT2_TMPDIR','').strip()
                        if envr:
                            roots.append(envr)
                        for guess in [r'D:\\fdv_tmp', r'C:\\fdv_tmp']:
                            roots.append(guess)
                    bn = _os.path.basename((fdv or '').strip())
                    if bn and roots:
                        _cand = str(_Path(roots[0]) / bn)
                except Exception:
                    _cand = None
            if not _cand:
                # Prefer a row-derived basename that contains the token (e.g., full file like 00000_Output_..._AFTER_RESET.txt)
                try:
                    import os as _os
                    from pathlib import Path as _Path
                    tok = (fdv or '').strip()
                    best_bn = ''
                    if tok:
                        toks = tok.lower()
                        names: list[str] = []
                        for rr in rows:
                            rf = (rr.get('fdv_file') or rr.get('fdv') or '').strip()
                            if rf and (toks in rf.lower()):
                                try:
                                    names.append(_Path(rf).name)
                                except Exception:
                                    try:
                                        names.append(_os.path.basename(rf))
                                    except Exception:
                                        continue
                        # Heuristic: prefer names with tb_set_utility or set_utility_tb; otherwise the longest
                        pref = [n for n in names if ('tb_set_utility' in n.lower() or 'set_utility_tb' in n.lower())]
                        if pref:
                            best_bn = sorted(pref, key=lambda s: (-len(s), s.lower()))[0]
                        elif names:
                            best_bn = sorted(names, key=lambda s: (-len(s), s.lower()))[0]
                    if best_bn:
                        try:
                            roots = list(search_roots) if 'search_roots' in locals() and search_roots else []
                        except Exception:
                            roots = []
                        if roots:
                            _cand = str(_Path(roots[0]) / best_bn)
                        else:
                            _cand = best_bn
                except Exception:
                    _cand = None
            if _cand:
                _display_name = _cand
    except Exception:
        pass
    try:
        import os as _os
        # Prefer the original filename for display if present and path-like
        if orig_filename and (('/' in orig_filename) or ('\\' in orig_filename)):
            _display_name = orig_filename
        if _display_name and ('/' in _display_name or '\\' in _display_name):
            _display_base = _os.path.basename(_display_name)
        else:
            _display_base = _display_name
    except Exception:
        _display_base = _display_name
    sel = {'fdv': fdv, 'pr': pr, 'vcc': vcc, 'tm': tm, 'temp': temp}
    # Provide search root info for diagnostics in the header
    search_root = (base_dir_q or (data.get('dir') or '').strip())
    # As a last resort, ensure attempted paths contains minimal diagnostics when read failed
    if read_failed and (not _attempted or len(_attempted) == 0):
        try:
            if search_root:
                attempted_paths.append(f"SEARCH_ROOT: {search_root}")
            attempted_paths.append(f"FDV_PARAM: {fdv}")
            attempted_paths.append("NO_MATCHES: no candidate files found")
            _attempted = attempted_paths
        except Exception:
            pass
    # Build a sorted list for the UI selector
    try:
        available_sources = sorted(source_files_all)
    except Exception:
        available_sources = list(source_files_all)
    return render_template('rawfile.html', filename=_display_name, filename_base=_display_base, orig_filename=orig_filename, snippet=snippet, target_line=target_line, back_url=back_url, idx=idx_req, total=(total_count if (total_count is not None) else len(fail_lines)), prev_url=prev_url, next_url=next_url, note=note, fail_items=fail_items, sel=sel, attempted_paths=_attempted, row_file_hint=best_row_file_hint, read_failed=read_failed, search_root=search_root, src_selected=src_q, src_options=available_sources)


def run_filename_resolver_selftest() -> dict:
    """Core logic for the filename resolver self-test (reusable by route and script)."""
    from pathlib import Path as _Path
    import tempfile as _tempfile
    import time as _time
    # 1) Create a temporary directory under a predictable root when possible
    preferred_root = os.environ.get('FDV_REPORT2_TMPDIR', '') or r'D:\\fdv_tmp'
    try:
        root = _Path(preferred_root)
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        root = _Path(_tempfile.gettempdir())
    tmp_dir = root / f"fdv_selftest_{int(_time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 2) Create a synthetic FDV log filename with a token-like suffix
    token = 'AFTER_RESET'
    fname = tmp_dir / f"00000_Output_tb_set_utility_{token}.txt"
    # 3) Write content with enough lines and a failing line at 42
    lines = []
    for i in range(1, 101):
        if i == 42:
            lines.append(f"{i:05d}: FAIL something RBER=2.5e-2\n")
        else:
            lines.append(f"{i:05d}: info line {i}\n")
    fname.write_text(''.join(lines), encoding='utf-8')
    # 4) Build minimal rows to emulate parsed entries
    pr = 'P1'; vcc = '1.8'; tm = '25'; temp = '25'
    rows = [{
        'fdv_file': str(fname),
        'fdv': str(fname),
        'pr': pr, 'vcc': vcc, 'tm': tm, 'temp': temp,
        'testname': 'SOME_TEST',
        'line_number': 42,
        'rber': 2.5e-2,
    }]
    # 5) Build a tiny resolver that mirrors the key branches from report_fails but stops once it finds an existing file
    def resolve_once(fdv_arg: str, base_dir: str | None) -> dict:
        base_dir_q = (base_dir or '').strip()
        # If direct path exists
        try:
            p = _Path(fdv_arg.strip(' "\'')) if fdv_arg else None
        except Exception:
            p = None
        if p and p.is_file():
            return {'resolved': str(p), 'exists': True, 'attempted': []}
        # Try from rows by exact match or basename
        attempted: list[str] = []
        try:
            import os as _os
            fdv_req = (fdv_arg or '').strip()
            fdv_base = _os.path.basename(fdv_req) if fdv_req else ''
            for r in rows:
                f = (r.get('fdv_file') or r.get('fdv') or '').strip()
                if not f:
                    continue
                fb = _os.path.basename(f)
                if f == fdv_req or fb == fdv_req or fb == fdv_base:
                    if _Path(f).is_file():
                        return {'resolved': f, 'exists': True, 'attempted': attempted}
        except Exception:
            pass
        # Search in base_dir by basename
        if base_dir_q and fdv_arg:
            try:
                b = _Path(fdv_arg).name
            except Exception:
                b = ''
            if b:
                cand = _Path(base_dir_q) / b
                attempted.append(str(cand))
                if cand.is_file():
                    return {'resolved': str(cand), 'exists': True, 'attempted': attempted}
        # Token-based search under base_dir
        try:
            tok = (_Path(fdv_arg).stem if fdv_arg else '').strip()
        except Exception:
            tok = (fdv_arg or '').strip()
        if base_dir_q and tok:
            try:
                for pat in [
                    f"*_tb_set_utility_{tok}.txt",
                    f"*_tb_set_utility*{tok}*.txt",
                    f"*_set_utility_tb_{tok}.txt",
                    f"*_set_utility_tb*{tok}*.txt",
                    f"*Output*tb_set_utility*{tok}*.txt",
                    f"*FDVLOG*{tok}*.txt",
                    f"*{tok}*.txt",
                    f"*{tok}*.log",
                ]:
                    for p2 in _Path(base_dir_q).rglob(pat):
                        attempted.append(str(p2))
                        if p2.is_file():
                            return {'resolved': str(p2), 'exists': True, 'attempted': attempted}
            except Exception:
                pass
        # As a final hint, return the row path (even if not found) and attempts
        hint = rows[0].get('fdv_file') if rows else ''
        return {'resolved': hint or (fdv_arg or ''), 'exists': bool(hint and _Path(hint).is_file()), 'attempted': attempted}

    base_dir = str(tmp_dir)
    cases = {
        'absolute_path': resolve_once(str(fname), base_dir),
        'basename_only': resolve_once(_Path(fname).name, base_dir),
        'token_with_dir': resolve_once(token, base_dir),
        'token_no_dir_but_row_hint': resolve_once(token, ''),
    }
    # Summarize pass/fail for each case
    summary = {k: bool(v.get('exists')) and (_Path(v.get('resolved')).is_file()) for k, v in cases.items()}
    return {'ok': all(summary.values()), 'summary': summary, 'cases': cases, 'tmp_dir': str(tmp_dir)}


@app.route('/selftest/resolve', methods=['GET'])
def selftest_resolve():
    import json as _json
    try:
        res = run_filename_resolver_selftest()
        return Response(_json.dumps(res), mimetype='application/json')
    except Exception as e:
        try:
            return Response(_json.dumps({'ok': False, 'error': str(e)}), mimetype='application/json', status=500)
        except Exception:
            return Response('selftest failed', mimetype='text/plain', status=500)


@app.route('/fdv/<token>/master_csv', methods=['GET'])
def report_master_csv(token: str):
    """Download a master CSV of parsed rows. Optional filtering by fdv selectors.

    Query params:
      - fdv: repeatable selector 'fdv|pr|vcc|tm|temp' (parts optional). If omitted, include all rows.
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        return Response('session expired', status=400)
    rows: List[Dict[str, str]] = data.get('rows', [])
    sels = [s.strip() for s in (request.args.getlist('fdv') or []) if s.strip()]
    parsed = [_parse_fdv_selector(x) for x in sels] if sels else []
    def _match(r: Dict[str, str]) -> bool:
        if not parsed:
            return True
        rk = _get_split_tuple(r)
        for (sf, spr, svcc, stm, stemp) in parsed:
            if sf and sf != rk[0]:
                continue
            if spr and spr != rk[1]:
                continue
            if svcc and svcc != rk[2]:
                continue
            if stm and stm != rk[3]:
                continue
            if stemp and stemp != rk[4]:
                continue
            return True
        return False
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    header = [
        'fdv_file','line_number','testname','tname','fuseid','dut_id','pr','vcc','tm','temp',
        'wl_canonical','page_canonical','plane','plane_addr','plane_addr_canonical','blk','blk_canonical',
        'rber','operation','pagetype','testtime_label','test_start','test_end'
    ]
    writer.writerow(header)
    for r in rows:
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        if not _match(r):
            continue
        tn = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
        writer.writerow([
            (r.get('fdv_file') or r.get('fdv') or ''),
            (r.get('line_number') or ''),
            tn,
            (r.get('tname') or ''),
            _get_fuseid(r),
            (r.get('dut_id') or ''),
            (r.get('pr') or ''),
            (r.get('vcc') or ''),
            (r.get('tm') or ''),
            (r.get('temp') or ''),
            (r.get('wl_canonical') or ''),
            (r.get('page_canonical') or ''),
            _plane_from_tname_or_default(r),
            (r.get('plane_addr') or ''),
            (r.get('plane_addr_canonical') or ''),
            (r.get('blk') or r.get('blk_addr') or ''),
            (r.get('blk_canonical') or ''),
            (r.get('rber') or r.get('RAW_BER') or r.get('raw_ber') or ''),
            (r.get('operation') or r.get('op') or ''),
            (r.get('pagetype') or ''),
            (r.get('testtime_label') or ''),
            (r.get('test_start') or ''),
            (r.get('test_end') or ''),
        ])
    body = buf.getvalue()
    out = io.BytesIO(body.encode('utf-8'))
    out.seek(0)
    fname = 'fdv_master.csv'
    return send_file(out, mimetype='text/csv', as_attachment=True, download_name=fname)


# (removed early __main__ runner to avoid starting the app before all routes are defined)


def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _plane_from_tname_or_default(r: Dict[str, str]) -> str:
    """Return 'SP' or 'MP' if present as a single token in tname; else 'NP'."""
    tname = (r.get('tname','') or '').strip().upper()
    if tname:
        tokens = [tok for tok in re.split(r"[^A-Z0-9]+", tname) if tok]
        for tok in tokens:
            if tok in ('SP','MP'):
                return tok
    # No explicit plane token found; treat as wildcard (empty)
    return ''


# --------- Field alias helpers ---------
def _first_nonempty_str(r: Dict[str, str], keys: List[str], default: str = '') -> str:
    for k in keys:
        if k in r and r.get(k) is not None:
            v = str(r.get(k)).strip()
            if v:
                return v
    return default


def _get_rber(r: Dict[str, str]) -> float | None:
    for k in ('rber','RBER','raw_ber','RAW_BER','ber','BER','bit_error_rate','error_rate'):
        if k in r:
            v = _to_float(r.get(k))
            if v is not None:
                return v
    return None


def _get_fuseid(r: Dict[str, str]) -> str:
    return _first_nonempty_str(r, ['fuseid','fuse_id','fid','chipid','chip_id','device_id'], '')


def _get_split_tuple(r: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    fdv = _first_nonempty_str(r, ['fdv_file','fdv','file','filepath','filename'], '')
    pr = _first_nonempty_str(r, ['pr','PR'], '') or 'XX'
    vcc = _first_nonempty_str(r, ['vcc','VCC','vcc_mv'], '')
    tm = _first_nonempty_str(r, ['tm','TM'], '')
    temp = _first_nonempty_str(r, ['temp','TEMP','temperature'], '')
    return (fdv, pr, vcc, tm, temp)


def _extract_wl_value(r: Dict[str, str]) -> int | None:
    """Return WL as an integer if available.

    Rule per guide_to_fdvlog.txt:
    - If tname contains WL_10 (or WL10 / WL-10 / WL 10), take 10 as WL.
    - Otherwise, check explicit WL-like fields.
    """
    # Prefer tname token per spec
    tname = (r.get('tname', '') or '')
    if tname:
        try:
            tn = tname.upper()
            # Use custom boundaries: underscore is a word char in \\b, so prefer non-alnum guards
            m = re.search(r"(?<![A-Z0-9])WL\\s*[_\\-\\s]?\\s*([0-9]+)(?![A-Z0-9])", tn)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    # Fallback to explicit WL fields if parser already provided one
    wl_keys = [
        # include canonical annotations first if present
        'wl_canonical',
        'wl','WL','Wl','wordline','WORDLINE','word_line','WORD_LINE',
        'wlidx','wl_index','wladdr','wl_addr','wladdress','wl_address',
        'wordline_idx','wordline_index','wl_index_dec','wordline_dec','wl_dec'
    ]
    for k in wl_keys:
        if k in r and r.get(k) not in (None, ''):
            try:
                return int(float(str(r.get(k)).strip()))
            except Exception:
                try:
                    return int(str(r.get(k)).strip(), 0)
                except Exception:
                    continue
    return None


def _extract_page_value(r: Dict[str, str]) -> int | None:
    """Return PAGE as an integer if available.

    Rule per guide_to_fdvlog.txt:
    - If tname contains PG_44 / PAGE_44 / PHYPAGE_44, take 44 as PAGE.
    - Otherwise, check explicit PAGE-like fields.
    """
    # Prefer tname token per spec
    tname = (r.get('tname', '') or '')
    if tname:
        try:
            tn = tname.upper()
            # Custom boundaries to allow underscores after the digits
            m = re.search(r"(?<![A-Z0-9])(?:PG|PAGE|PHYPAGE)\s*[_\-\s]?\s*([0-9]+)(?![A-Z0-9])", tn)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    # Fallback to explicit page fields
    pg_keys = [
        'page_canonical',
        'page','PAGE','Page','page_idx','pageindex','pg','pgidx','page_addr','pageaddr',
        'page_address','pageno','page_no','pagenumber','page_num','pgno','pg_no',
        'pgindex','pg_index','pgaddr','pg_addr','page_address_dec'
    ]
    for k in pg_keys:
        if k in r and r.get(k) not in (None, ''):
            try:
                return int(float(str(r.get(k)).strip()))
            except Exception:
                try:
                    return int(str(r.get(k)).strip(), 0)
                except Exception:
                    continue
    return None


def _extract_wl_or_page(r: Dict[str, str], *, allow_page_fallback: bool = True) -> float | None:
    """Extract WL (preferred) or, if allowed, PAGE from a parsed row using the FDV guide rules.

    - WL from tname token WL_<n> first; then explicit WL fields.
    - If not found and allow_page_fallback=True, PAGE from tname token (PG|PAGE|PHYPAGE)_<n> first; then explicit PAGE fields.
    Returns a float for plotting convenience.
    """
    wl = _extract_wl_value(r)
    if wl is not None:
        return float(wl)
    if not allow_page_fallback:
        return None
    pg = _extract_page_value(r)
    if pg is not None:
        return float(pg)
    return None


def _extract_plane_addr(r: Dict[str, str]) -> str:
    """Extract plane address like 'P0'..'P7' if available.

    Priority:
    - Explicit fields: plane_addr, planeaddress, plane_address
    - Token in tname: P<digits>
    - Derive from block number (blk & 0x7)
    Returns '' if unavailable.
    """
    # Explicit fields (include several aliases; ignore SP/MP tokens)
    for k in ('plane_addr', 'planeaddress', 'plane_address', 'planeaddr', 'plane_id', 'planeid', 'plane_no', 'plane_num', 'planeindex', 'plane_idx'):
        v = (r.get(k, '') or '').strip().upper()
        if v and v not in ('SP', 'MP'):
            m = re.match(r"^P?(\d{1,2})$", v)
            if m:
                try:
                    n = int(m.group(1))
                    if 0 <= n <= 99:
                        return f"P{n}"
                except Exception:
                    pass
    # From tname token
    tn = (r.get('tname', '') or '')
    if tn:
        up = tn.upper()
        m = re.search(r"(?<![A-Z0-9])P(\d+)(?![A-Z0-9])", up)
        if m:
            try:
                n = int(m.group(1))
                if 0 <= n <= 99:
                    return f"P{n}"
            except Exception:
                pass
    # From raw_line token P<digits> if present; also handle PLANE[...] entries
    rl = (r.get('raw_line', '') or '')
    if rl:
        up = rl.upper()
        m = re.search(r"(?<![A-Z0-9])P(\d+)(?![A-Z0-9])", up)
        if m:
            try:
                n = int(m.group(1))
                if 0 <= n <= 99:
                    return f"P{n}"
            except Exception:
                pass
        m3 = re.search(r"(?<![A-Z0-9])PLANE(?:[_\s]?(ADDR|ADDRESS|ID|NO|NUM))?\s*[:=]?\s*(\d{1,2})(?![A-Z0-9])", up)
        if m3:
            try:
                n = int(m3.group(2))
                if 0 <= n <= 99:
                    return f"P{n}"
            except Exception:
                pass
    # Derive from block if numeric (consider many common keys)
    blk_keys = (
        'blk', 'blk_canonical', 'block', 'bl',
        'block_addr', 'blk_addr', 'blockaddress', 'block_address',
        'blkindex', 'blk_index', 'block_idx', 'block_index',
        'blockno', 'block_no', 'blocknum', 'block_num', 'blocknumber', 'block_number',
        'blkno', 'blk_no', 'blknum', 'blk_num',
        'blockid', 'block_id',
        'blkaddr', 'blk_addr_dec', 'block_addr_dec', 'block_address_dec', 'blk_dec', 'block_dec',
        'pbn', 'phy_block', 'phyblk', 'phy_blk', 'phyblock'
    )
    for bk in blk_keys:
        bv = r.get(bk)
        if bv is None:
            continue
        blk = str(bv).strip()
        if not blk:
            continue
        try:
            n = int(float(blk))
        except Exception:
            try:
                n = int(blk, 0)
            except Exception:
                continue
        if n >= 0:
            return f"P{(n & 0x7)}"
    return ''


def _extract_blk_value(r: Dict[str, str]) -> int | None:
    """Extract block address as an integer if present in the row.

    Tries common keys like 'blk', 'block', 'bl', 'block_addr', and parses decimal or base-prefixed strings.
    Returns None if unavailable.
    """
    blk_keys = (
        'blk', 'blk_canonical', 'block', 'bl',
        'block_addr', 'blk_addr', 'blockaddress', 'block_address',
        'blkindex', 'blk_index', 'block_idx', 'block_index',
        # common alternates
        'blockno', 'block_no', 'blocknum', 'block_num', 'blocknumber', 'block_number',
        'blkno', 'blk_no', 'blknum', 'blk_num',
        'blockid', 'block_id',
        'blkaddr', 'blk_addr_dec', 'block_addr_dec', 'block_address_dec', 'blk_dec', 'block_dec',
        # NAND-specific aliases
        'pbn', 'phy_block', 'phyblk', 'phy_blk', 'phyblock'
    )
    # 1) Prefer explicit fields if present
    for k in blk_keys:
        if k in r and r.get(k) not in (None, ''):
            s = str(r.get(k)).strip()
            try:
                return int(float(s))
            except Exception:
                try:
                    return int(s, 0)
                except Exception:
                    continue
    # 2) Derive from tname token e.g., BLK_778 or BLOCK_778
    tn = (r.get('tname', '') or '')
    if tn:
        up = tn.upper()
        m = re.search(r"(?<![A-Z0-9])(BLK|BLOCK)\s*[_\-\s]?\s*([0-9]+)(?![A-Z0-9])", up)
        if m:
            try:
                return int(m.group(2))
            except Exception:
                pass
    # 3) As a last resort, scan raw_line if available
    rl = (r.get('raw_line', '') or '')
    if rl:
        up = rl.upper()
        m = re.search(r"(?<![A-Z0-9])(BLK|BLOCK)\s*[_\-\s]?\s*([0-9]+)(?![A-Z0-9])", up)
        if m:
            try:
                return int(m.group(2))
            except Exception:
                pass
        # Also consider 'BLOCK_NO' or 'BLOCKNUM=123' and 'PBN=123'
        m2 = re.search(r"BLOCK[_\s]?NO\s*[:=]?\s*([0-9]+)", up)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                pass
        m3 = re.search(r"(?<![A-Z0-9])PBN\s*[:=]?\s*([0-9]+)", up)
        if m3:
            try:
                return int(m3.group(1))
            except Exception:
                pass
    return None


## (Deprecated duplicate stats_by_fdv_with_splits removed to unify passfail_mode support)


def stats_by_testname(rows: List[Dict[str, str]], fdv_file: str) -> List[Dict[str, str]]:
    from collections import defaultdict
    vals: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        if (r.get('fdv_file','') or '') != (fdv_file or ''):
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip()
        if not tn:
            raw = (r.get('tname','') or '').strip()
            tn = derive_testname(raw) if raw else ''
        if not tn or ('poweron' in tn.lower() or 'powerup' in tn.lower()):
            continue
        # Skip invalid/missing FUSEID rows
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        # Enforce FUSEID validity
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        pr = (r.get('pr','') or '').strip() or 'XX'
        fid = (r.get('fuseid','') or '').strip()
        vcc = (r.get('vcc','') or '').strip()
        tm = (r.get('tm','') or '').strip()
        temp = (r.get('temp','') or '').strip()
        rv = _to_float(r.get('rber'))
        if rv is None:
            continue
        vals[(tn, pr, fid, vcc, tm, temp)].append(rv)
    def _pr_key(p: str):
        if p == 'XX':
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    for (tn, pr, fid, vcc, tm, temp) in sorted(vals.keys(), key=lambda k: (k[0], _pr_key(k[1]), k[2], k[3], k[4], k[5])):
        v = sorted(vals[(tn, pr, fid, vcc, tm, temp)])
        n = len(v)
        import statistics as _stats
        out.append({
            'testname': tn,
            'pr': pr,
            'fuseid': fid,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'count': str(n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
        })
    return out


def stats_by_testname_multi(rows: List[Dict[str, str]], fdv_files: List[str]) -> List[Dict[str, str]]:
    """Aggregate testname RBER stats limited to selected fdv_files.

    Note: This legacy variant filters only by fdv_file. Prefer stats_by_testname_selected.
    """
    sel = set([f for f in fdv_files if f])
    from collections import defaultdict
    vals: Dict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        fdv = (r.get('fdv_file','') or '')
        if fdv not in sel:
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip()
        if not tn:
            raw = (r.get('tname','') or '').strip()
            tn = derive_testname(raw) if raw else ''
        if not tn or ('poweron' in tn.lower() or 'powerup' in tn.lower()):
            continue
        # Skip invalid/missing FUSEID rows
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        # Enforce FUSEID validity
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        pr = (r.get('pr','') or '').strip() or 'XX'
        fid = (r.get('fuseid','') or '').strip()
        vcc = (r.get('vcc','') or '').strip()
        tm = (r.get('tm','') or '').strip()
        temp = (r.get('temp','') or '').strip()
        rv = _to_float(r.get('rber'))
        if rv is None:
            continue
        vals[(tn, pr, fid, vcc, tm, temp)].append(rv)
    def _pr_key(p: str):
        if p == 'XX':
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    import statistics as _stats
    for (tn, pr, fid, vcc, tm, temp) in sorted(vals.keys(), key=lambda k: (k[0], _pr_key(k[1]), k[2], k[3], k[4], k[5])):
        v = sorted(vals[(tn, pr, fid, vcc, tm, temp)])
        n = len(v)
        out.append({
            'testname': tn,
            'pr': pr,
            'fuseid': fid,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'count': str(n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
        })
    return out


def _parse_fdv_selector(val: str) -> Tuple[str, str, str, str, str]:
    """Parse 'fdv|pr|vcc|tm|temp' or just 'fdv'. Missing parts become empty strings."""
    parts = (val or '').split('|')
    fdv = parts[0] if len(parts) >= 1 else ''
    pr = parts[1] if len(parts) >= 2 else ''
    vcc = parts[2] if len(parts) >= 3 else ''
    tm = parts[3] if len(parts) >= 4 else ''
    temp = parts[4] if len(parts) >= 5 else ''
    return (fdv, pr, vcc, tm, temp)


def stats_by_testname_selected(rows: List[Dict[str, str]], selectors: List[str], *, limit: float | None = None) -> List[Dict[str, str]]:
    """Aggregate testname RBER stats limited to selected fdv rows (fdv,pr,vcc,tm,temp),
    further split by plane_group and operation so users can select precise rows including plane/operation.
    """
    keyset = set(_parse_fdv_selector(s) for s in selectors if s)
    from collections import defaultdict
    # Group by (testname, pr, fuseid, vcc, tm, temp, plane, op)
    vals: Dict[Tuple[str, str, str, str, str, str, str, str], List[float]] = defaultdict(list)
    plane_addr_sets: Dict[Tuple[str, str, str, str, str, str, str, str], set] = defaultdict(set)
    blk_addr_sets: Dict[Tuple[str, str, str, str, str, str, str, str], set] = defaultdict(set)
    for r in rows:
        k = _get_split_tuple(r)
        # Accept match if fdv matches and either exact split matches OR selector omits split parts
        # Build a set of acceptable selectors where empty split fields are wildcards
        matched = False
        for (fdv, pr, vcc, tm, temp) in keyset:
            if not fdv:
                continue
            if fdv != k[0]:
                continue
            if pr and pr != k[1]:
                continue
            if vcc and vcc != k[2]:
                continue
            if tm and tm != k[3]:
                continue
            if temp and temp != k[4]:
                continue
            matched = True
            break
        if not matched:
            continue
        if (r.get('tname','') or '').strip().upper() == 'PR':
            continue
        tn = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
        if not tn or ('poweron' in tn.lower() or 'powerup' in tn.lower()):
            continue
        # Enforce FUSEID validity
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        # Enforce FUSEID validity
        if not _is_valid_fuseid(_get_fuseid(r)):
            continue
        pr = k[1]
        fid = _get_fuseid(r)
        vcc = k[2]
        tm = k[3]
        temp = k[4]
        # Derive plane strictly from tname
        plane = _plane_from_tname_or_default(r)
        op = _first_nonempty_str(r, ['operation','op','readtype'], '').upper()
        rv = _get_rber(r)
        if rv is None:
            continue
        kkey = (tn, pr, fid, vcc, tm, temp, plane, op)
        vals[kkey].append(rv)
        # Prefer previously annotated canonical plane/block if present
        if r.get('plane_addr_canonical'):
            pa = str(r.get('plane_addr_canonical'))
        else:
            pa = _extract_plane_addr(r)
        if pa:
            plane_addr_sets[kkey].add(pa)
        if r.get('blk_canonical'):
            try:
                bv = int(str(r.get('blk_canonical')))
            except Exception:
                bv = _extract_blk_value(r)
        else:
            bv = _extract_blk_value(r)
        if bv is not None:
            blk_addr_sets[kkey].add(bv)
    def _pr_key(p: str):
        if p == 'XX':
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    import statistics as _stats
    for (tn, pr, fid, vcc, tm, temp, plane, op) in sorted(vals.keys(), key=lambda k: (k[0], _pr_key(k[1]), k[2], k[3], k[4], k[5], k[6], k[7])):
        v = sorted(vals[(tn, pr, fid, vcc, tm, temp, plane, op)])
        n = len(v)
        pass_n = sum(1 for x in v if x < limit)
        fail_n = n - pass_n
        pa_set = plane_addr_sets.get((tn, pr, fid, vcc, tm, temp, plane, op), set())
        pa_disp = ''
        if pa_set:
            try:
                pa_disp = ','.join(sorted(pa_set, key=lambda s: int(s[1:]) if (isinstance(s, str) and s.startswith('P') and s[1:].isdigit()) else s))
            except Exception:
                pa_disp = ','.join(sorted(pa_set))
        blk_set = blk_addr_sets.get((tn, pr, fid, vcc, tm, temp, plane, op), set())
        blk_disp = ''
        if blk_set:
            try:
                blk_disp = ','.join(str(x) for x in sorted(blk_set))
            except Exception:
                blk_disp = ','.join(str(x) for x in blk_set)
        out.append({
            'testname': tn,
            'pr': pr,
            'fuseid': fid,
            'vcc': vcc,
            'tm': tm,
            'temp': temp,
            'plane': plane,
            'op': op,
            'plane_op': (plane or '*'),
            'plane_addr': pa_disp,
            'blk_addr': blk_disp,
            'count': str(n),
            'pass': str(pass_n),
            'fail': str(fail_n),
            'pass_n': str(pass_n),
            'fail_n': str(fail_n),
            'min': f"{v[0]:.6g}",
            'max': f"{v[-1]:.6g}",
            'mean': f"{(sum(v)/n):.6g}",
            'stdev': f"{(_stats.stdev(v) if n>=2 else 0.0):.6g}",
            'median': f"{_stats.median(v):.6g}",
        })
    return out


def _build_variability_records(rows: List[Dict[str, str]], selectors: List[str], entries: List[Tuple[str, ...]], *, allow_page_fallback: bool = True, allow_missing_wl: bool = False):
    recs: List[Dict] = []
    # Build parsed selector tuples once
    parsed = [_parse_fdv_selector(x) for x in selectors]
    for s in entries:
        # tuple shape: (testname, pr, fuseid, vcc, tm, temp, plane, op)
        tn = s[0]
        pr = s[1] if len(s) > 1 else ''
        fid = s[2] if len(s) > 2 else ''
        sel_vcc = s[3] if len(s) > 3 else ''
        sel_tm = s[4] if len(s) > 4 else ''
        sel_temp = s[5] if len(s) > 5 else ''
        sel_plane = s[6].upper() if len(s) > 6 and s[6] else ''
        sel_op = s[7].upper() if len(s) > 7 and s[7] else ''

        for idx, r in enumerate(rows):
            rf = (r.get('fdv_file', '') or '')
            # row-level split key
            rk = _get_split_tuple(r)
            # match any selector (empty fields are wildcards)
            ok = False
            for (sf, spr, svcc, stm, stemp) in parsed:
                if sf and sf != rk[0]:
                    continue
                if spr and spr != rk[1]:
                    continue
                if svcc and svcc != rk[2]:
                    continue
                if stm and stm != rk[3]:
                    continue
                if stemp and stemp != rk[4]:
                    continue
                ok = True
                break
            if not ok:
                continue
            if (r.get('tname', '') or '').strip().upper() == 'PR':
                continue
            tnr_raw = (r.get('testname', '') or '').strip()
            tnr = tnr_raw if tnr_raw else derive_testname((r.get('tname', '') or '').strip())
            if (tnr or '').strip().lower() != (tn or '').strip().lower():
                continue
            # If selection's PR/FuseID are empty, treat them as wildcards (testname-only filter)
            _rf, row_pr, row_vcc, row_tm, row_temp = rk
            row_fid = _get_fuseid(r)
            if pr and row_pr != pr:
                continue
            # Soften FuseID: only exclude when row has a different non-empty fid
            if fid:
                if row_fid and row_fid != fid:
                    continue
            if sel_vcc and row_vcc != sel_vcc:
                continue
            if sel_tm and row_tm != sel_tm:
                continue
            if sel_temp and row_temp != sel_temp:
                continue
            # Derive plane strictly from tname; default NP when missing
            row_plane = _plane_from_tname_or_default(r)
            row_op = _first_nonempty_str(r, ['operation', 'op', 'readtype'], '').upper()
            # Treat empty or '*' (and legacy 'NP') as wildcard
            if sel_plane and sel_plane not in ('*', 'NP') and row_plane != sel_plane:
                continue
            if sel_op and sel_op not in ('*', 'NP') and row_op != sel_op:
                continue
            wl = _extract_wl_or_page(r, allow_page_fallback=allow_page_fallback)
            rber = _get_rber(r)
            if rber is None:
                continue
            if rber <= 0:
                rber = 1e-12
            if wl is None:
                # If WL (or PAGE when allowed) is unavailable, include only when explicitly allowed (data view).
                if not allow_missing_wl:
                    continue
                wl = -1.0
            recs.append({
                'testname': tn,
                'WL': wl,
                'RBER': rber,
                'pagetype': (r.get('pagetype', '') or '').strip(),
                'readtype': (r.get('operation', '') or '').strip().upper() or 'READ',
                'dut': f"DUT{(r.get('dut_id', '') or '').strip() or '?'}",
                'plane': row_plane,
                'op': row_op,
                'plane_addr': _extract_plane_addr(r),
                'blk': _extract_blk_value(r),
                '_idx': r.get('_idx', idx),
                'line_number': r.get('line_number', ''),
            })
    return recs


def _mk_app() -> Flask:
    app = Flask(__name__, template_folder=str(_HERE / 'templates'))
    app.secret_key = os.environ.get('FDV_REPORT2_SECRET', 'dev-secret')
    # Configure a reliable temp directory for multipart parsing/uploads.
    # Prefer D:\\fdv_tmp for all temp usage; allow override via FDV_REPORT2_TMPDIR.
    # Fall back gracefully to other candidates only if D: (or override) is unavailable.
    try:
        import tempfile as _tempfile
        from pathlib import Path as _Path
        import shutil as _shutil
        override_tmp = (os.environ.get('FDV_REPORT2_TMPDIR') or r'D:\\fdv_tmp').strip()
        best_path = None
        best_free = -1
        try:
            p = _Path(override_tmp)
            p.mkdir(parents=True, exist_ok=True)
            _ = _shutil.disk_usage(str(p))
            best_path = p
            best_free = int(_[2])
        except Exception:
            best_path = None
        if best_path is None:
            for pth in ['Z:\\fdv_tmp', 'C:\\fdv_tmp']:
                try:
                    q = _Path(pth)
                    q.mkdir(parents=True, exist_ok=True)
                    total, used, free = _shutil.disk_usage(str(q))
                    if free > best_free:
                        best_free = int(free)
                        best_path = q
                except Exception:
                    continue
        if best_path is not None:
            os.environ['FDV_REPORT2_TMPDIR'] = str(best_path)
            os.environ['TMP'] = str(best_path)
            os.environ['TEMP'] = str(best_path)
            try:
                mplcfg = _Path(str(best_path)) / 'mplconfig'
                mplcfg.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault('MPLCONFIGDIR', str(mplcfg))
            except Exception:
                pass
            try:
                _tempfile.tempdir = str(best_path)
            except Exception:
                pass
            try:
                app.config['UPLOAD_FOLDER'] = str(best_path)
            except Exception:
                pass
    except Exception:
        pass
    return app


## (second app init removed)


def _list_files(paths: List[Path]) -> List[Path]:
    """Return all files under the given paths, searching subdirectories recursively."""
    files: List[Path] = []
    for p in paths:
        try:
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for root, _dirs, fnames in os.walk(p):
                    try:
                        root_path = Path(root)
                    except Exception:
                        continue
                    for name in sorted(fnames):
                        fp = root_path / name
                        try:
                            if fp.is_file():
                                files.append(fp)
                        except Exception:
                            continue
        except Exception:
            continue
    return files


def _start_parse_job_legacy(token: str, files: List[Path], used_dir: str | None) -> None:
    """Legacy parser (unused). Kept for reference; main implementation is defined earlier with extended parameters."""
    def job():
        try:
            # Allowed line prefixes; all other lines ignored for performance.
            _ALLOWED_PREFIXES_CS = (
                'Test Start Date',
                'Test End Date',
                'ECHO: FUSEID',
                'FDV OUTPUT',
            )
            _ALLOWED_PREFIXES = tuple(p.upper() for p in _ALLOWED_PREFIXES_CS)
            def _allowed_line(s: str) -> bool:
                ls = s.lstrip()
                up = ls.upper()
                return any(up.startswith(p) for p in _ALLOWED_PREFIXES)
            total_bytes = 0
            sizes: List[int] = []
            # Pre-count lines so UI can show overall and per-file totals
            line_counts: List[int] = []
            total_lines = 0
            for fp in files:
                try:
                    sz = fp.stat().st_size
                except Exception:
                    sz = 0
                # Additional processing can be added here
                sizes.append(sz)
                total_bytes += sz
                lc = 0
                try:
                    with open(fp, 'r', encoding='utf-8', errors='replace') as _lc_f:
                        for lc, _ in enumerate(_lc_f, start=1):
                            pass
                except Exception:
                    lc = 0
                line_counts.append(lc)
                total_lines += lc
            progress = {
                'files_total': len(files),
                'files_done': 0,
                'current_file': '',
                'current_index': 0,
                'percent': 0.0,
                'lines': 0,
                'lines_total': total_lines,
                'file_lines_total': 0,
                'file_lines_done': 0,
                'file_percent': 0.0,
                'file_bytes_total': 0,
                'file_bytes_done': 0,
                'bytes_total': total_bytes,
                'bytes_done': 0,
            }
            CACHE[token] = {
                'status': 'running',
                'progress': progress,
                'rows': [],
                'dir': used_dir,
            }
            all_rows: List[Dict[str, str]] = []
            bytes_done_prev = 0
            lines_done_prev = 0
            for idx, fp in enumerate(files, start=1):
                progress['current_file'] = str(fp)
                progress['current_index'] = idx
                file_size = sizes[idx - 1] if idx - 1 < len(sizes) else 0
                file_lines_total = line_counts[idx - 1] if idx - 1 < len(line_counts) else 0
                progress['file_lines_total'] = file_lines_total
                progress['file_lines_done'] = 0
                last_lineno = {'n': 0}
                progress['file_bytes_total'] = file_size
                progress['file_bytes_done'] = 0
                progress['file_percent'] = 0.0

                def _cb(lineno: int, pct: float) -> None:
                    last_lineno['n'] = lineno
                    # Estimate bytes processed if parser supplies pct relative to file
                    bytes_curr = 0
                    try:
                        bytes_curr = int((pct / 100.0) * file_size)
                    except Exception:
                        bytes_curr = 0
                    total_done = bytes_done_prev + bytes_curr
                    lines_done_now = lines_done_prev + lineno
                    if total_lines > 0:
                        overall_pct = (float(lines_done_now) / float(total_lines) * 100.0)
                    else:
                        overall_pct = (float(total_done) / float(total_bytes) * 100.0) if total_bytes > 0 else 0.0
                    if file_lines_total > 0:
                        file_pct = (float(lineno) / float(file_lines_total) * 100.0)
                    else:
                        file_pct = pct
                    progress.update({
                        'percent': overall_pct,
                        'lines': lines_done_now,
                        'file_lines_done': lineno,
                        'file_percent': file_pct,
                        'file_bytes_done': bytes_curr,
                        'bytes_done': total_done,
                    })
                    # Store back into CACHE for polling clients (tolerate missing entry)
                    try:
                        if token not in CACHE:
                            CACHE[token] = {
                                'status': 'running',
                                'progress': progress,
                                'rows': [],
                                'dir': used_dir,
                            }
                        else:
                            CACHE[token]['progress'] = progress
                    except Exception:
                        pass

                try:
                    if process_file is not None:
                        r, _kept, _markers = process_file(fp, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT, progress=True, progress_cb=_cb)  # type: ignore[arg-type]
                        # Defensive post-filter: remove any rows whose raw line still contains MONITOR or SHMOO
                        try:
                            _filt = []
                            for _row in r:
                                _rl = (_row.get('raw_line') or _row.get('raw') or '')
                                _u = _rl.upper()
                                if 'MONITOR' in _u or 'SHMOO' in _u:
                                    continue
                                _filt.append(_row)
                            r = _filt
                        except Exception:
                            pass
                    else:
                        # Fallback: very slow path without structured parsing
                        r = []
                        _candidate_fp = fp
                        try:
                            enc_guess = _detect_file_encoding(fp)
                            if enc_guess == 'utf-16':
                                try:
                                    print(f"[encoding] Detected possible UTF-16 file: {fp}")
                                except Exception:
                                    pass
                                tmp_conv = _convert_utf16_to_temp_utf8(fp)
                                if tmp_conv and tmp_conv.exists():
                                    _candidate_fp = tmp_conv
                        except Exception:
                            pass
                        try:
                            with open(_candidate_fp, 'r', encoding='utf-8', errors='replace') as f:
                                for i, line in enumerate(f, start=1):
                                    if not _allowed_line(line):
                                        continue
                                    ls = line.lstrip()
                                    if ls.startswith('FDV OUTPUT'):
                                        # Filter MONITOR/SHMOO lines here too
                                        up = ls.upper()
                                        if 'MONITOR' in up or 'SHMOO' in up:
                                            continue
                                        r.append({'raw_line': ls.rstrip('\n'), 'line_number': str(i), 'fdv_file': str(fp)})
                                    # Also capture explicit test start/end lines into rows so timestamps not lost
                                    elif ls.upper().startswith('TEST START DATE') or ls.upper().startswith('TEST END DATE'):
                                        r.append({'raw_line': ls.rstrip('\n'), 'line_number': str(i), 'fdv_file': str(fp)})
                                    if i % 100000 == 0:
                                        _cb(i, 0.0)
                        except Exception as e:
                            try:
                                print(f"[fallback-parse2] Failed reading {fp}: {e}")
                            except Exception:
                                pass
                except Exception:
                    r = []
                # Ensure fdv_file is set for all rows (some parsers may omit)
                try:
                    for rr in r:
                        if not (rr.get('fdv_file') or rr.get('fdv')):
                            rr['fdv_file'] = str(fp)
                except Exception:
                    pass
                # Derive fdv list/run names from filename
                def _extract_run_parts_from_filename(p: Path) -> Tuple[str, str]:
                    name = p.name
                    up = name
                    import re as _re
                    m = _re.search(r"_fdvrun_(.+?)_tb_set_utility_([^\.]+)", up, flags=_re.IGNORECASE)
                    if m:
                        return (m.group(1), m.group(2))
                    # Fallbacks: hyphens or different separators
                    m2 = _re.search(r"fdvrun[_\-]([^\-]+?)[_\-]tb_set_utility[_\-]([^\.]+)", up, flags=_re.IGNORECASE)
                    if m2:
                        return (m2.group(1), m2.group(2))
                    return ('', '')
                run_name, list_name = _extract_run_parts_from_filename(fp)
                # Scan file for Test Start/End markers to compute duration
                start_dt = None
                end_dt = None
                list_name_from_marker = ''
                start_raw_str = ''
                end_raw_str = ''
                try:
                    import re as _re
                    from datetime import datetime as _dt
                    # Patterns (case-insensitive) allowing single-digit month/day/minute/second, and optional list name
                    # Start markers
                    re_start_date_name = _re.compile(r"Test\s+Start\s+Date\s*\(([^)]+)\)\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", _re.IGNORECASE)
                    re_start_date_noname = _re.compile(r"Test\s+Start\s+Date\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", _re.IGNORECASE)
                    re_start_time = _re.compile(r"Test\s+Start\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", _re.IGNORECASE)
                    re_start_both = _re.compile(r"Test\s+Start\s+Date\s*(?:\(([^)]+)\))?\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})\s*Test\s+Start\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", _re.IGNORECASE)
                    # End markers
                    re_end_date_name = _re.compile(r"Test\s+End\s+Date\s*\(([^)]+)\)\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", _re.IGNORECASE)
                    re_end_date_noname = _re.compile(r"Test\s+End\s+Date\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", _re.IGNORECASE)
                    re_end_time = _re.compile(r"Test\s+End\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", _re.IGNORECASE)
                    re_end_both = _re.compile(r"Test\s+End\s+Date\s*(?:\(([^)]+)\))?\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})\s*Test\s+End\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", _re.IGNORECASE)

                    # Collect all candidate (dt, raw) pairs to choose earliest start and latest end
                    start_candidates = []
                    end_candidates = []
                    s_date = s_time = s_ampm = None
                    e_date = e_time = e_ampm = None
                    with open(fp, 'r', encoding='utf-8', errors='replace') as fscan:
                        for line in fscan:
                            if not _allowed_line(line):
                                continue
                            # Combined first (fast path)
                            m_sb = re_start_both.search(line)
                            if m_sb:
                                if m_sb.group(1):
                                    list_name_from_marker = (m_sb.group(1) or '').strip()
                                y, mo, da = int(m_sb.group(2)), int(m_sb.group(3)), int(m_sb.group(4))
                                t = m_sb.group(5).strip()
                                ap = (m_sb.group(6) or '').upper() if m_sb.lastindex and m_sb.lastindex >= 6 else ''
                                # Push combined start candidate immediately
                                start_candidates.append(((y, mo, da), t, ap))
                            m_eb = re_end_both.search(line)
                            if m_eb:
                                if not list_name_from_marker and m_eb.group(1):
                                    list_name_from_marker = (m_eb.group(1) or '').strip()
                                y, mo, da = int(m_eb.group(2)), int(m_eb.group(3)), int(m_eb.group(4))
                                t = m_eb.group(5).strip()
                                ap = (m_eb.group(6) or '').upper() if m_eb.lastindex and m_eb.lastindex >= 6 else ''
                                # Push combined end candidate immediately
                                end_candidates.append(((y, mo, da), t, ap))
                            # Separate patterns — date with optional name
                            m_sdn = re_start_date_name.search(line)
                            if m_sdn:
                                list_name_from_marker = (m_sdn.group(1) or '').strip()
                                s_date = (int(m_sdn.group(2)), int(m_sdn.group(3)), int(m_sdn.group(4)))
                            else:
                                m_sdn2 = re_start_date_noname.search(line)
                                if m_sdn2:
                                    s_date = (int(m_sdn2.group(1)), int(m_sdn2.group(2)), int(m_sdn2.group(3)))
                            m_st = re_start_time.search(line)
                            if m_st:
                                s_time = m_st.group(1).strip()
                                s_ampm = (m_st.group(2) or '').upper()
                            if s_date and s_time:
                                start_candidates.append((s_date, s_time, s_ampm))
                                s_date = s_time = s_ampm = None
                            # End separate patterns — date with optional name
                            m_edn = re_end_date_name.search(line)
                            if m_edn:
                                if not list_name_from_marker:
                                    list_name_from_marker = (m_edn.group(1) or '').strip()
                                e_date = (int(m_edn.group(2)), int(m_edn.group(3)), int(m_edn.group(4)))
                            else:
                                m_edn2 = re_end_date_noname.search(line)
                                if m_edn2:
                                    e_date = (int(m_edn2.group(1)), int(m_edn2.group(2)), int(m_edn2.group(3)))
                            m_et = re_end_time.search(line)
                            if m_et:
                                e_time = m_et.group(1).strip()
                                e_ampm = (m_et.group(2) or '').upper()
                            if e_date and e_time:
                                end_candidates.append((e_date, e_time, e_ampm))
                                e_date = e_time = e_ampm = None
                    def _fmt_raw(parts, times, ampm):
                        try:
                            if not parts or not times:
                                return ''
                            (y, mo, da) = parts
                            t = times
                            ap = f" {ampm}" if ampm else ''
                            return f"{y:04d}_{mo:02d}_{da:02d} {t}{ap}"
                        except Exception:
                            return ''
                    # Choose earliest Start and latest End from candidates
                    def _parse_dt(parts, times, ampm):
                        if not parts or not times:
                            return None
                        (y, mo, da) = parts
                        try:
                            hh, mm, ss = [int(x) for x in times.split(':')[:3]]
                        except Exception:
                            return None
                        # Clamp out-of-range values (some logs show :60)
                        if hh < 0: hh = 0
                        if mm < 0: mm = 0
                        if ss < 0: ss = 0
                        if mm > 59: mm = 59
                        if ss > 59: ss = 59
                        if ampm in ('AM','PM'):
                            if ampm == 'PM' and hh < 12:
                                hh += 12
                            if ampm == 'AM' and hh == 12:
                                hh = 0
                        try:
                            return _dt(y, mo, da, hh, mm, ss)
                        except Exception:
                            return None
                    # Build dt lists
                    start_dt_list = []
                    for (pd, ts, ap) in start_candidates:
                        dt = _parse_dt(pd, ts, ap)
                        if dt is not None:
                            start_dt_list.append((dt, _fmt_raw(pd, ts, ap)))
                    end_dt_list = []
                    for (pd, ts, ap) in end_candidates:
                        dt = _parse_dt(pd, ts, ap)
                        if dt is not None:
                            end_dt_list.append((dt, _fmt_raw(pd, ts, ap)))
                    if start_dt_list:
                        start_dt, start_raw_str = sorted(start_dt_list, key=lambda x: x[0])[0]
                    if end_dt_list:
                        end_dt, end_raw_str = sorted(end_dt_list, key=lambda x: x[0])[-1]
                except Exception:
                    start_dt = start_dt or None
                    end_dt = end_dt or None
                # Build label string: <fdvlistname>::<fdvtest> = seconds
                def _fmt_duration_secs(s_dt, e_dt) -> int:
                    try:
                        if not s_dt or not e_dt:
                            return -1
                        delta = e_dt - s_dt
                        secs = int(delta.total_seconds())
                        if secs < 0:
                            secs = 0
                        return secs
                    except Exception:
                        return -1
                dur_secs = _fmt_duration_secs(start_dt, end_dt)
                # We'll attach a per-row label using the row's fdv_file (fdvtest)
                # Compute the fdvlist display name once here
                testtime_label = ''
                try:
                    # Prefer filename-derived list; fallback to marker-derived list name
                    ln_disp = (list_name or '').strip()
                    if not ln_disp and list_name_from_marker:
                        # strip any numeric prefix like '34_' and leading 'tb_set_utility_'
                        ln_tmp = list_name_from_marker
                        try:
                            ln_tmp = _re.sub(r"^\d+_", "", ln_tmp)
                            ln_tmp = _re.sub(r"(?i)^tb_set_utility_", "", ln_tmp)
                        except Exception:
                            pass
                        ln_disp = ln_tmp.strip()
                    # Only build labels when we have a duration value available (>=0 includes 0s)
                    if ln_disp or (dur_secs is not None and dur_secs >= 0):
                        secs_txt = str(dur_secs if dur_secs is not None and dur_secs >= 0 else '')
                        testtime_label = f"{ln_disp}:: = {secs_txt}".strip()
                except Exception:
                    testtime_label = ''
                # Attach label to each row for this file for downstream grouping
                try:
                    for rr in r:
                        # Derive fdvtest (basename without extension) from rr['fdv_file']
                        fdvtest = (rr.get('fdv_file') or '').strip()
                        if fdvtest:
                            import os as _os
                            fdvtest = _os.path.basename(fdvtest)
                            # strip extension if present
                            if fdvtest.lower().endswith('.fdv'):
                                fdvtest = fdvtest[:-4]
                        # Build and attach label if we have either list name or duration
                        if testtime_label or (fdvtest and (dur_secs is not None and dur_secs >= 0)):
                            try:
                                ln_disp_local = testtime_label.split('::')[0] if testtime_label else (list_name or '').strip()
                            except Exception:
                                ln_disp_local = (list_name or '').strip()
                            secs_txt = str(dur_secs if dur_secs is not None and dur_secs >= 0 else '')
                            rr['testtime_label'] = f"{(ln_disp_local or '').strip()}::{fdvtest} = {secs_txt}".strip()
                            if dur_secs is not None and dur_secs >= 0:
                                rr['testtime_seconds'] = str(dur_secs)
                        # Attach raw start/end strings for visibility
                        if start_raw_str:
                            rr['test_start'] = start_raw_str
                        if end_raw_str:
                            rr['test_end'] = end_raw_str
                        if list_name:
                            rr['fdvlistname'] = list_name
                        # Keep run_name in case it's useful elsewhere, but label uses fdvtest
                        if run_name:
                            rr['fdvtestrun'] = run_name
                except Exception:
                    pass
                all_rows.extend(r)
                # finalize this file's contribution
                lines_done_prev += last_lineno['n']
                bytes_done_prev += file_size
                progress['files_done'] = idx
                progress['bytes_done'] = bytes_done_prev
                progress['lines'] = lines_done_prev
                # Finalize overall percent using lines if available
                if total_lines > 0:
                    progress['percent'] = (float(lines_done_prev) / float(total_lines) * 100.0)
                else:
                    progress['percent'] = (float(bytes_done_prev) / float(total_bytes) * 100.0) if total_bytes > 0 else progress.get('percent', 0.0)
                try:
                    if token in CACHE:
                        CACHE[token]['progress'] = progress
                    else:
                        CACHE[token] = {
                            'status': 'running',
                            'progress': progress,
                            'rows': all_rows,
                            'dir': used_dir,
                        }
                except Exception:
                    pass
            # After parsing, annotate rows with indices for later raw-line lookup
            for i, rr in enumerate(all_rows):
                if '_idx' not in rr:
                    rr['_idx'] = i
                # Apply canonical WL/PAGE extraction per guide for downstream use
                try:
                    wl_v = _extract_wl_value(rr)
                    if wl_v is not None:
                        rr['wl_canonical'] = str(wl_v)
                except Exception:
                    pass
                try:
                    pg_v = _extract_page_value(rr)
                    if pg_v is not None:
                        rr['page_canonical'] = str(pg_v)
                except Exception:
                    pass
                # Apply canonical plane and block extraction
                try:
                    pa = _extract_plane_addr(rr)
                    if pa:
                        rr['plane_addr'] = pa
                        rr['plane_addr_canonical'] = pa
                except Exception:
                    pass
                try:
                    bv = _extract_blk_value(rr)
                    if bv is not None:
                        rr['blk'] = str(bv)
                        rr['blk_addr'] = str(bv)
                        rr['blk_canonical'] = str(bv)
                except Exception:
                    pass
            stats = stats_by_fdv_with_splits(all_rows)
            # Build ordered unique fdv list
            seen = set()
            fdv_order: List[str] = []
            for r in stats:
                f = r.get('fdv_file','') or ''
                if f and f not in seen:
                    seen.add(f)
                    fdv_order.append(f)
            CACHE[token].update({'rows': all_rows, 'stats': stats, 'fdv_order': fdv_order, 'status': 'done'})
            # Mark job end time
            try:
                with JOBS_LOCK:
                    for jid, rec in JOBS.items():
                        if rec.get('token') == token:
                            if 'ended_at' not in rec:
                                rec['ended_at'] = time.time()
                            break
            except Exception:
                pass
        except Exception as e:
            try:
                if token in CACHE:
                    CACHE[token].update({'status': 'error', 'error': str(e)})
                else:
                    CACHE[token] = {'status': 'error', 'error': str(e), 'progress': {}, 'rows': [], 'dir': used_dir}
                # Mark job end time (error)
                try:
                    with JOBS_LOCK:
                        for jid, rec in JOBS.items():
                            if rec.get('token') == token:
                                if 'ended_at' not in rec:
                                    rec['ended_at'] = time.time()
                                break
                except Exception:
                    pass
            except Exception:
                pass

    th = threading.Thread(target=job, name=f"fdv-parse-{token[:6]}", daemon=True)
    th.start()


## Duplicate legacy report_home removed (limit-aware version defined earlier).

# (removed trailing __main__ runner to avoid starting the app automatically)
