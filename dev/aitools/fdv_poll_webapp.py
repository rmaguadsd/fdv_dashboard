#!/usr/bin/env python
"""
Simple web UI to upload a log file and display FDV POLL statistics.
Reuses parsing logic from process_fdv_poll.py and renders two tables:
- Stats by fdv_file
- Stats by (fdv_file, vcc, temp)

Both tables show: min, max, mean, stdev, median (in that order).
"""
from __future__ import annotations
import os
import statistics as stats
import uuid
import io
import tempfile
import colorsys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

from flask import Flask, render_template, request, redirect, url_for, flash, Response, send_file, jsonify
import sys
import threading

# Force a non-GUI Matplotlib backend for server-side rendering to avoid Tkinter thread issues
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# Local import: POLL analyzer for CHAR logs
import process_fdv_poll as pfp
# Ensure local package path is first so `import process_fdv` resolves to this workspace
import process_fdv as pfdv
# Defer importing read_eimpro_plot (and its heavy seaborn/matplotlib dependencies)
# until the /eimpro route is used, to keep app startup fast and robust.
rep = None  # will be imported lazily in the eimpro() route


# ============================================================================
# OPTIMIZATION LAYER: Caching & Performance Improvements
# ============================================================================
from functools import lru_cache
import time

# Regex pattern cache with TTL
_regex_cache = {}
_regex_cache_time = {}
REGEX_CACHE_TTL = 3600  # 1 hour in seconds

def get_compiled_regex(pattern):
    """
    Get a compiled regex pattern, with caching and TTL.
    Avoids recompiling the same pattern multiple times.
    Expected speedup: 30-50% reduction in regex compilation overhead.
    """
    if not pattern:
        return None
    
    current_time = time.time()
    
    # Check cache and TTL
    if pattern in _regex_cache:
        cache_time = _regex_cache_time.get(pattern, 0)
        if current_time - cache_time < REGEX_CACHE_TTL:
            return _regex_cache[pattern]
    
    # Compile and cache
    try:
        compiled = re.compile(pattern)
        _regex_cache[pattern] = compiled
        _regex_cache_time[pattern] = current_time
        return compiled
    except Exception:
        return None

@lru_cache(maxsize=128)
def get_color_palette_cached(n_colors):
    """
    Generate color palette with caching.
    Expected speedup: 10-15% reduction in color generation overhead.
    """
    if n_colors <= 0:
        return []
    
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.85 + (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

# Need to import colorsys at top of file
import colorsys

# ============================================================================


app = Flask(__name__)
app.secret_key = os.environ.get("FDV_POLL_WEBAPP_SECRET", "dev-secret")

# Ensure all temporary files (uploads, generated images/HTML/ZIPs) go to D:\ by default.
# Honor FDV_POLL_TMPDIR if provided; otherwise default to D:\\fdv_tmp, with fallbacks.
def _init_tempdir_once():
    try:
        import tempfile as _tempfile
        from pathlib import Path as _Path
        # Prefer explicit env, else D:\fdv_tmp, else C:\fdv_tmp, else system temp
        override_tmp = (os.environ.get('FDV_POLL_TMPDIR') or r'D:\\fdv_tmp').strip()
        # For Python <3.10 avoid PEP 604 unions; keep simple None assignment
        best_path = None  # type: Optional[_Path]
        if override_tmp:
            p = _Path(override_tmp)
            try:
                p.mkdir(parents=True, exist_ok=True)
                best_path = p
            except Exception:
                best_path = None
        if best_path is None:
            for pth in [r'D:\fdv_tmp', r'C:\fdv_tmp']:
                try:
                    p = _Path(pth)
                    p.mkdir(parents=True, exist_ok=True)
                    best_path = p
                    break
                except Exception:
                    continue
        if best_path is None:
            best_path = _Path(_tempfile.gettempdir())
        # Set environment and tempfile module default dir
        os.environ['FDV_POLL_TMPDIR'] = str(best_path)
        os.environ['TMP'] = str(best_path)
        os.environ['TEMP'] = str(best_path)
        _tempfile.tempdir = str(best_path)
    except Exception:
        # If anything fails, silently continue with system defaults
        pass

_init_tempdir_once()

# In-memory cache for per-request stats (download CSV)
# Structure: { token: { 'vt': List[Dict[str,str]] } }
from typing import Any
CACHE: Dict[str, Dict[str, Any]] = {}

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


# Dedicated base directories for alias snapshots and specs persistence
def _alias_base_dir() -> Path:
    base = (os.environ.get('FDV_POLL_ALIAS_DIR') or r'D:\fdv_poll_alias').strip() or r'D:\fdv_poll_alias'
    try:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        try:
            p = Path(r'C:\fdv_poll_alias')
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return Path(tempfile.gettempdir()) / 'fdv_poll_alias'

def _alias_rows_path(name: str) -> Path:
    """Path to JSON file storing merged raw rows for an alias."""
    return _alias_base_dir() / name / 'rows.json'

def _load_alias_rows(name: str) -> list[dict]:
    p = _alias_rows_path(name)
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding='utf-8') or '[]')
    except Exception:
        pass
    return []

def _save_alias_rows(name: str, rows: list[dict]) -> None:
    base = _alias_base_dir() / name
    try:
        base.mkdir(parents=True, exist_ok=True)
        (_alias_rows_path(name)).write_text(json.dumps(rows), encoding='utf-8')
    except Exception:
        pass

def _row_signature(r: dict) -> str:
    """Best-effort stable signature for a POLL raw row to deduplicate across sessions.
    Priority: basename(fdv_file)+line_number; fallback to sha1 of salient fields.
    """
    try:
        ln = int(str(r.get('line_number') or '0'))
    except Exception:
        ln = 0
    fdv = os.path.basename((r.get('fdv_file') or '').strip())
    if ln > 0 and fdv:
        return f"F:{fdv}|L:{ln}"
    # Fallback: hash key fields
    key_fields = ['fdv_file','vcc','temp','status','plane_group','dut_id','value','pr','blk','page','wl','phypage','source_name']
    blob = '|'.join(str(r.get(k,'')).strip() for k in key_fields)
    return hashlib.sha1(blob.encode('utf-8', errors='ignore')).hexdigest()

def _list_aliases() -> list[str]:
    try:
        base = _alias_base_dir()
        names = [p.name for p in base.iterdir() if p.is_dir()]
        names.sort(key=lambda s: s.lower())
        return names
    except Exception:
        return []


def _specs_store_dir() -> Path:
    base = (os.environ.get('FDV_POLL_SPECS_DIR') or r'D:\fdv_poll_specs').strip() or r'D:\fdv_poll_specs'
    try:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        try:
            p = Path(r'C:\fdv_poll_specs')
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return Path(tempfile.gettempdir()) / 'fdv_poll_specs'


def _specs_file() -> Path:
    return _specs_store_dir() / 'specs.json'


def _load_specs() -> Dict[str, Dict[str, str]]:
    fp = _specs_file()
    try:
        if fp.is_file():
            import json
            data = json.loads(fp.read_text(encoding='utf-8'))
            # Ensure proper dict shape
            if isinstance(data, dict):
                return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_specs(specs: Dict[str, Dict[str, str]]) -> None:
    fp = _specs_file()
    try:
        import json
        tmp = fp.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(fp)
    except Exception:
        # Best-effort persistence
        try:
            import json
            fp.write_text(json.dumps(specs, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass


POLL_MAX_VALUE = 25000.0  # ignore any FDV POLL measurement greater than this

def to_float(row: Dict[str, str]) -> float | None:
    """Return numeric POLL value; ignore sentinel -999 and values > POLL_MAX_VALUE."""
    try:
        v = float(row.get("data_token2_numeric") or row.get("data_token2") or "")
    except Exception:
        return None
    # Ignore invalid measurement sentinel (-999) and oversized values (> threshold)
    if v == -999.0 or v > POLL_MAX_VALUE:
        return None
    return v


def _plane_from_tname_or_default(r: Dict[str, str]) -> str:
    """Return 'SP' or 'MP' if present as a single token in tname; else ''."""
    tname = (r.get('tname','') or '').strip().upper()
    if tname:
        tokens = [tok for tok in re.split(r"[^A-Z0-9]+", tname) if tok]
        for tok in tokens:
            if tok in ('SP','MP'):
                return tok
    return ''


_RE_PAGEMAP = re.compile(r"(?<![A-Z0-9])(MLC|QLC|TLC|SSLC|DSLC)(?![A-Z0-9])", re.IGNORECASE)


def _extract_pagemap_from_row(r: Dict[str, str]) -> str:
    v = (r.get('pagemap') or r.get('product_type') or r.get('product') or r.get('pagemap_type') or '').strip()
    if v:
        return v.upper()
    for s in ((r.get('tname') or ''), (r.get('fdv_file') or ''), (r.get('file') or '')):
        m = _RE_PAGEMAP.search(s or '')
        if m:
            return (m.group(1) or '').upper()
    return ''

def _specname_from_fdv(fdv_name: str) -> str:
    """Return specname from FDV filename (without extension): first token before '_' (uppercased)."""
    try:
        name = (fdv_name or '').strip()
        if not name:
            return ''
        if name.lower().endswith('.fdv'):
            name = name[:-4]
        return name.split('_', 1)[0].strip().upper()
    except Exception:
        return ''

# Helpers aligned with report2 for robust field extraction
def _first_nonempty_str(r: Dict[str, str], keys: List[str], default: str = '') -> str:
    for k in keys:
        try:
            v = r.get(k)
        except Exception:
            v = None
        s = (str(v) if v is not None else '').strip()
        if s:
            return s
    return default

def _get_fuseid(r: Dict[str, str]) -> str:
    # Accept common synonyms used across logs
    return _first_nonempty_str(r, ['fuseid','fuse_id','fid','chipid','chip_id','device_id'], '')

# FUSEID validation and site extraction helpers (aligned with report2)
_RE_FUSEID = re.compile(r"^K\d{6}_(\d+)_(-?\d+)_(-?\d+)$", re.IGNORECASE)
def _is_valid_fuseid(fid: str) -> bool:
    if not fid:
        return False
    m = _RE_FUSEID.match(fid.strip())
    if not m:
        return False
    try:
        return int(m.group(1)) > 0
    except Exception:
        return False
def _extract_site_from_filename(fp: str) -> str:
    try:
        base = Path(fp).name
        up = base.upper()
        m = re.search(r"OUTPUT[_-]SITE(\d+)", up, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r"OUTPUT[_-](\d+)_", up, flags=re.IGNORECASE)
        return m2.group(1) if m2 else ''
    except Exception:
        return ''


def fmt(x: float) -> str:
    return f"{x:.6g}"


def compute_stats(values: List[float]) -> Dict[str, str]:
    if not values:
        return {k: "" for k in ("min", "max", "mean", "stdev", "median")}
    vals = sorted(values)
    n = len(vals)
    _min = vals[0]
    _max = vals[-1]
    _mean = float(sum(vals)) / n
    _stdev = float(stats.stdev(vals)) if n >= 2 else 0.0
    _median = float(stats.median(vals))
    return {
        "min": fmt(_min),
        "max": fmt(_max),
        "mean": fmt(_mean),
        "stdev": fmt(_stdev),
        "median": fmt(_median),
    }


def group_by_fdv(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        fdv = r.get("fdv_file", "") or ""
        v = to_float(r)
        if v is None:
            continue
        groups.setdefault(fdv, []).append(v)
    out = []
    for fdv in sorted(groups.keys(), key=lambda s: (s or "")):
        st = compute_stats(groups[fdv])
        out.append({"fdv_file": fdv, **st})
    return out


def group_by_fdv_vcc_temp(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Legacy: group by (fdv, vcc, temp, status, plane)."""
    return group_by_fdv_with_splits(rows, split_vcc=True, split_temp=True, split_pagetype=False)


def group_by_fdv_with_splits(
    rows: List[Dict[str, str]],
    *,
    split_vcc: bool,
    split_temp: bool,
    split_plane: bool = True,
    split_pagetype: bool = False,
) -> List[Dict[str, str]]:
    """Group rows by (fdv_file, specname, pagemap, status, pr) and optionally split by VCC, TEMP, plane, pagetype.
    This prevents mixing different POLL spec segments or PR bins in one aggregate.
    """
    # key: (fdv, spec, pagemap, pagetype?, pr, status, vcc?, temp?, plane?)
    groups: Dict[Tuple[str, str, str, str, str, str, str, str, str], List[Tuple[float, str, str, str, str, str]]] = {}
    ignored_by_group: Dict[Tuple[str, str, str, str, str, str, str, str, str], List[Tuple[str, str, str]]] = {}
    _re_poll_spec = re.compile(r"POLL_([A-Za-z0-9]+)_")
    for r in rows:
        fdv = r.get("fdv_file", "") or ""
        vcc_full = (r.get("vcc", "") or "").strip()
        temp_full = (r.get("temp", "") or "").strip()
        vcc = vcc_full if split_vcc else ""
        temp = temp_full if split_temp else ""
        polltest = (r.get("poll__test", "") or "").strip()
        tname = (r.get("tname", "") or "").strip()
        status = (r.get("status", "") or "").strip().upper()
        plane_full = (r.get("plane_group", "") or _plane_from_tname_or_default(r) or "").strip().upper()
        plane = plane_full if split_plane else ""
        pr_val = (_first_nonempty_str(r, ['pr','PR'], '') or 'XX')
        fuseid = _get_fuseid(r)
        pagetype = (r.get("pagetype", "") or "").strip()
        pagetype_key = pagetype if split_pagetype else ""
        tm = (r.get("tm", "") or "").strip()
        if (tname or '').strip().upper() == 'PR':
            continue
        if polltest and polltest != fdv:
            continue
        if polltest and ("tbers" in polltest.lower()) and ("tbers" not in fdv.lower()):
            continue
        if tname and ("tbers" in tname.lower() or "erase" in tname.lower()) and ("tbers" not in fdv.lower() and "erase" not in fdv.lower()):
            continue
        vnum = to_float(r)
        if vnum is None:
            continue
        # derive pagemap and spec for this row
        pagemap = _extract_pagemap_from_row(r)
        spec_token = ''
        try:
            if tname.startswith('POLL_'):
                m0 = _re_poll_spec.match(tname)
                if m0:
                    spec_token = (m0.group(1) or '').upper()
            if not spec_token:
                raw_line = (r.get('raw_line') or r.get('raw') or '')
                if raw_line and 'POLL_' in raw_line:
                    m1 = _re_poll_spec.search(raw_line)
                    if m1:
                        spec_token = (m1.group(1) or '').upper()
        except Exception:
            spec_token = ''
        if not spec_token:
            spec_token = _specname_from_fdv(fdv)
        key = (fdv, spec_token, pagemap, pagetype_key, pr_val, status, vcc, temp, plane)
        if not _is_valid_fuseid(fuseid):
            site = _extract_site_from_filename(fdv)
            if pr_val:
                ignored_by_group.setdefault(key, []).append((pr_val, site, (fuseid or '')))  # reuse tuple structure
            continue
        groups.setdefault(key, []).append((vnum, pr_val, fuseid, pagetype, tm, plane))
    persisted_specs = _load_specs()
    def _gkey(fdv: str, spec: str, pagemap: str, pagetype_val: str, pr_val: str, status: str, vcc: str, temp: str, plane: str) -> str:
        return "||".join([
            fdv or '', spec or '', pagemap or '', pagetype_val or '', pr_val or '', (status or '').upper(),
            vcc or '', temp or '', (plane or '').upper()
        ])
    out: List[Dict[str, str]] = []
    for key in sorted(groups.keys(), key=lambda kv: (kv[0] or '', kv[1] or '', kv[2] or '', kv[3] or '', kv[4] or '', kv[5] or '', kv[6] or '', kv[7] or '', kv[8] or '')):
        fdv, spec_token, pagemap, pagetype_key, pr_val, status, vcc, temp, plane = key
        items = groups[key]
        vals = [v for (v, _pr, _fid, _pt, _tm, _pl) in items]
        st = compute_stats(vals)
        # collect metadata sets
        dut_fid_pairs: Dict[str, str] = {}
        pagetypes_set = set()
        tm_set = set()
        valid_ids_set: set[str] = set()
        # We no longer retain DUT id per row (simplified), so fuseid list limited
        for (_v, _pr, _fid, _pt, _tm, _pl) in items:
            if _fid:
                valid_ids_set.add(_fid)
            if _pt:
                pagetypes_set.add(_pt)
            if _tm:
                tm_set.add(_tm)
        # Build comments (reduced)
        def _tm_key(s: str):
            try:
                return (0, float(s))
            except Exception:
                return (1, s)
        tm_comment = ("TM: " + ", ".join(sorted(tm_set, key=_tm_key))) if tm_set else ""
        pt_comment = ("PAGETYPE: " + ", ".join(sorted(pagetypes_set))) if pagetypes_set else ""
        ignored = ignored_by_group.get(key, [])
        if ignored:
            ig_parts = []
            seen_ig = set()
            for (pr_ig, site, fid_str) in ignored:
                lab = f"PR{pr_ig}{('@site' + site) if site else ''}:{fid_str or '(missing)'}"
                if lab not in seen_ig:
                    seen_ig.add(lab)
                    ig_parts.append(lab)
            ig_comment = "IGNORED (invalid FUSEID): " + ", ".join(sorted(ig_parts))
        else:
            ig_comment = ""
        comments = " | ".join([c for c in (tm_comment, pt_comment, ig_comment) if c])
        key_id = _gkey(fdv, spec_token, pagemap, pagetype_key, pr_val, status, vcc, temp, plane)
        spec_persist = persisted_specs.get(key_id, {}) if isinstance(persisted_specs, dict) else {}
        out.append({
            'fdv_file': fdv,
            'specname': spec_token,
            'vcc': vcc,
            'temp': temp,
            'pagemap': pagemap,
            'pagetype': pagetype_key,
            'status': status,
            'plane_group': plane,
            'pr': pr_val,
            'count': str(len(vals)),
            'valid_fuseid_count': (str(len(valid_ids_set)) if valid_ids_set else ''),
            'MinSpec': (spec_persist.get('MinSpec') or ''),
            'MaxSpec': (spec_persist.get('MaxSpec') or ''),
            '_group_key': key_id,
            'comments': comments,
            **st
        })
    return out


def detect_outliers_iqr(values: List[float]) -> Tuple[float, float, List[int]]:
    """Return (low, high, indices) using Tukey IQR rule. indices are positions of outliers."""
    if not values:
        return 0.0, 0.0, []
    vs = sorted(values)
    n = len(vs)
    def percentile(p: float) -> float:
        if n == 1:
            return float(vs[0])
        k = (p/100.0) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        return float(vs[f] + (vs[c] - vs[f]) * (k - f))
    q1 = percentile(25)
    q3 = percentile(75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    idxs = [i for i, v in enumerate(values) if (v < low or v > high)]
    return float(low), float(high), idxs


def _filter_poll_row(r: Dict[str, str]) -> Tuple[bool, str, str, str, str, str, str, float | None]:
    """Apply common FDV POLL filter rules and return tuple:
    (keep, fdv, vcc, temp, status, plane, pagetype, value)
    """
    try:
        v = float(r.get("data_token2_numeric") or r.get("data_token2") or "")
    except Exception:
        # Always return 8 values, pad with None for pagetype and value
        return False, "", "", "", "", "", "", None
    # Ignore invalid measurement sentinel -999 and oversized values
    try:
        if v == -999.0 or v > POLL_MAX_VALUE:
            return False, "", "", "", "", "", "", None
    except Exception:
        pass
    fdv = r.get("fdv_file", "") or ""
    vcc = (r.get("vcc", "") or "").strip()
    temp = (r.get("temp", "") or "").strip()
    polltest = (r.get("poll__test", "") or "").strip()
    tname = (r.get("tname", "") or "").strip()
    status = (r.get("status", "") or "").strip().upper()
    plane = (r.get("plane_group", "") or _plane_from_tname_or_default(r) or "").strip().upper()
    pagetype = (r.get("pagetype", "") or "").strip()
    # Skip PR monitor rows entirely (not measurement)
    if (tname or '').strip().upper() == 'PR':
        return False, fdv, vcc, temp, status, plane, pagetype, None
    if polltest and polltest != fdv:
        return False, fdv, vcc, temp, status, plane, pagetype, None
    if polltest and ("tbers" in polltest.lower()) and ("tbers" not in fdv.lower()):
        return False, fdv, vcc, temp, status, plane, pagetype, None
    if tname and ("tbers" in tname.lower() or "erase" in tname.lower()) and ("tbers" not in fdv.lower() and "erase" not in fdv.lower()):
        return False, fdv, vcc, temp, status, plane, pagetype, None
    return True, fdv, vcc, temp, status, plane, pagetype, v


@app.route("/hist/<token>")
def hist_overview(token: str):
    """Render a simple page with a histogram per fdv_file; clicking a bar or points isn't native,
    so we also list detected outliers with links to raw lines.
    Query params: fdv (optional), vcc (optional), temp (optional) to filter.
    """
    data = CACHE.get(token)
    if not data or "rows" not in data:
        flash("Session expired. Please re-upload the log.")
        return redirect(url_for("index"))
    rows: List[Dict[str, str]] = data["rows"]
    from collections import defaultdict
    # Optional filters
    sel_fdv = (request.args.get("fdv") or "").strip()
    sel_vcc = (request.args.get("vcc") or "").strip()
    sel_temp = (request.args.get("temp") or "").strip()
    sel_status = (request.args.get("status") or "").strip().upper()
    sel_plane = (request.args.get("plane") or "").strip().upper()
    sel_pagetype = (request.args.get("pagetype") or "").strip()
    # Split options (independent): only create separate charts when selected
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
    # Group key: (fdv, vcc?, temp?, plane?, pagetype?)
    groups: Dict[Tuple[str, str, str, str, str], List[Tuple[int, float, str, str, str, str, str, str | None]]] = defaultdict(list)
    for r in rows:
        keep, fdv, vcc, temp, status, plane, pagetype, v = _filter_poll_row(r)
        if not keep or v is None:
            continue
        key = (
            fdv,
            (vcc if split_vcc else ""),
            (temp if split_temp else ""),
            (plane if split_plane else ""),
            (pagetype if split_pagetype else ""),
        )
        # Apply filters (exact match on provided fields)
        if sel_fdv and fdv != sel_fdv:
            continue
        if sel_vcc and vcc != sel_vcc:
            continue
        if sel_temp and temp != sel_temp:
            continue
        if sel_status and status != sel_status:
            continue
        if sel_plane and plane != sel_plane:
            continue
        if sel_pagetype and (pagetype or "") != sel_pagetype:
            continue
        try:
            ln = int(r.get("line_number", "0"))
        except Exception:
            ln = 0
        src_idx = r.get("source_idx") if isinstance(r, dict) else None
        groups[key].append((ln, v, vcc, temp, status, plane, pagetype, (str(src_idx) if src_idx is not None else None)))
    # Generate plots to a temp dir and collect outliers
    out_dir = Path(tempfile.mkdtemp(prefix=f"hist_{token}_"))
    rendered = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (fdv, vcc_key, temp_key, plane_key, pagetype_key), items in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
        vals = [v for _, v, *_ in items]
        if not vals:
            continue
        low, high, _ = detect_outliers_iqr(vals)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(vals, bins=min(40, max(8, int(len(vals) ** 0.5))), color="#4e79a7", alpha=0.85, edgecolor="#1f3551")
        ax.axvline(low, color="#b00020", ls="--", lw=1)
        ax.axvline(high, color="#b00020", ls="--", lw=1)
        parts = []
        # Reflect filters/splits in subtitle
        if sel_status:
            parts.append(f"status={sel_status}")
        if split_plane and plane_key:
            parts.append(f"plane={plane_key}")
        if split_pagetype and pagetype_key:
            parts.append(f"pagetype={pagetype_key}")
        subtitle = (" " + ", ".join(parts)) if parts else ""
        # Use keys when split is on; otherwise show '-' to denote aggregation
        title_vcc = (vcc_key if split_vcc else "-")
        title_temp = (temp_key if split_temp else "-")
        ax.set_title(f"Histogram — {fdv} | VCC={title_vcc or '-'} TEMP={title_temp or '-'}{subtitle} (n={len(vals)})")
        ax.set_xlabel("FDV POLL value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        img_name = f"hist_{uuid.uuid4().hex}.png"
        img_path = out_dir / img_name
        try:
            fig.savefig(img_path, dpi=160)
        finally:
            plt.close(fig)
        # outliers list with links — show only top 10 farthest from IQR bounds
        candidates = []
        for ln, v, vcc, temp, status, plane, pagetype, src_idx in items:
            if v < low or v > high:
                # deviation from the nearest bound
                dev = (low - v) if v < low else (v - high)
                if dev < 0:
                    dev = -dev
                candidates.append((dev, ln, v, vcc, temp, status, plane, pagetype, src_idx))
        candidates.sort(key=lambda t: t[0], reverse=True)
        top = candidates[:10]
        outlier_rows = [{
            "line_number": ln,
            "value": f"{v:.6g}",
            "raw_url": url_for(
                "rawfile",
                token=token,
                line=str(ln),
                src=(str(src_idx) if (src_idx is not None and str(src_idx).isdigit()) else None),
                split_vcc=("1" if split_vcc else "0"),
                split_temp=("1" if split_temp else "0"),
                split_plane=("1" if split_plane else "0"),
                split_pagetype=("1" if split_pagetype else "0"),
            ),
        } for _dev, ln, v, _vcc, _temp, _status, _plane, _pagetype, src_idx in top]
        try:
            img_url = url_for("download_hist_image", token=token, name=img_name)
        except Exception:
            img_url = f"/download/hist/{token}/{img_name}"
        rendered.append({
            "fdv": fdv or "",
            "vcc": (vcc_key or ""),
            "temp": (temp_key or ""),
            "img": str(img_path),
            "img_name": img_name,
            "img_url": img_url,
            "outliers": outlier_rows,
            "status": sel_status,
            "plane": (plane_key or ""),
            "pagetype": (pagetype_key or ""),
        })
    # Save dir and files in cache to serve images
    data.setdefault("hist_dirs", []).append(str(out_dir))
    data.setdefault("hist_files", []).extend([r["img_name"] for r in rendered])
    CACHE[token] = data
    return render_template(
        "histograms.html",
        groups=rendered,
        token=token,
        sel_fdv=sel_fdv,
        sel_vcc=sel_vcc,
        sel_temp=sel_temp,
        sel_status=sel_status,
        sel_plane=sel_plane,
        sel_pagetype=sel_pagetype,
        split_pagetype=split_pagetype,
    )


@app.route("/download/rcdf/<token>/<name>")
def download_rcdf_image(token: str, name: str):
    data = CACHE.get(token)
    if not data or "rcdf_dirs" not in data:
        flash("Session expired.")
        return redirect(url_for("index"))
    for d in reversed(data.get("rcdf_dirs", [])):
        fp = Path(d) / name
        if fp.exists():
            return send_file(str(fp), as_attachment=False)
    flash("Image not found.")
    return redirect(url_for("index"))


@app.route("/download/hist/<token>/<name>")
def download_hist_image(token: str, name: str):
    data = CACHE.get(token)
    if not data or "hist_dirs" not in data:
        flash("Session expired.")
        return redirect(url_for("index"))
    # Search in known hist dirs for the file name
    for d in reversed(data.get("hist_dirs", [])):
        fp = Path(d) / name
        if fp.exists():
            return send_file(str(fp), as_attachment=False)
    flash("Image not found.")
    return redirect(url_for("index"))


@app.route("/rcdf/<token>")
def rcdf_compare(token: str):
    """Render an RCDF (1-CDF) compare plot for selected FDV POLL groups.

    Query params:
      - rows: comma-separated items of the form fdv||vcc||temp||status||plane
      - split_vcc, split_temp, split_plane: '1' to reflect selection in labels
    """
    data = CACHE.get(token)
    if not data or "rows" not in data:
        flash("Session expired. Please re-upload the log.")
        return redirect(url_for("index"))
    rows: List[Dict[str, str]] = data["rows"]
    sel = (request.args.get("rows") or "").strip()
    if not sel:
        flash("No selections provided for RCDF.")
        return redirect(url_for("results_view", token=token))
    items = [s for s in (sel.split(",") if sel else []) if s]
    if not items:
        flash("No valid selections.")
        return redirect(url_for("results_view", token=token))
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")

    # Build value arrays per selection
    series: List[tuple[str, List[float]]] = []
    for it in items:
        parts = it.split("||")
        if len(parts) < 6:
            continue
        s_fdv, s_vcc, s_temp, s_status, s_plane, s_pagetype = [p or "" for p in parts[:6]]
        vals: List[float] = []
        for r in rows:
            keep, fdv, vcc, temp, status, plane, pagetype, v = _filter_poll_row(r)
            if not keep or v is None:
                continue
            if fdv != s_fdv:
                continue
            if s_status and status != (s_status or "").upper():
                continue
            if s_vcc and vcc != s_vcc:
                continue
            if s_temp and temp != s_temp:
                continue
            if s_plane and plane != (s_plane or "").upper():
                continue
            if s_pagetype and (pagetype or "") != s_pagetype:
                continue
            vals.append(float(v))
        if not vals:
            continue
        # Label: fdv plus optional splits
        lab_parts = [s_fdv]
        if split_vcc and s_vcc:
            lab_parts.append(f"VCC={s_vcc}")
        if split_temp and s_temp:
            lab_parts.append(f"TEMP={s_temp}")
        if s_status:
            lab_parts.append(f"status={s_status}")
        if split_plane and s_plane:
            lab_parts.append(f"plane={s_plane}")
        if split_pagetype and s_pagetype:
            lab_parts.append(f"pagetype={s_pagetype}")
        label = " | ".join(lab_parts)
        series.append((label, sorted(vals)))

    # Plot RCDF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not series:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.text(0.5, 0.5, "No data to plot", transform=ax.transAxes, ha="center", va="center")
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for i, (label, vals) in enumerate(series):
            n = len(vals)
            xs = vals
            # 1 - ECDF (RCDF)
            ys = [1.0 - (k + 1) / n for k in range(n)]
            ax.plot(xs, ys, label=label, lw=1.5)
        ax.set_xlabel("FDV POLL value")
        ax.set_ylabel("RCDF (1 - CDF)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

    # Save image to a temp dir and register for download
    out_dir = Path(tempfile.mkdtemp(prefix=f"rcdf_{token}_"))
    img_name = f"rcdf_{uuid.uuid4().hex}.png"
    img_path = out_dir / img_name
    try:
        fig.savefig(img_path, dpi=160)
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
    data.setdefault("rcdf_dirs", []).append(str(out_dir))
    data.setdefault("rcdf_files", []).append(img_name)
    CACHE[token] = data
    try:
        img_url = url_for("download_rcdf_image", token=token, name=img_name)
    except Exception:
        img_url = f"/download/rcdf/{token}/{img_name}"
    return render_template(
        "rcdf_compare.html",
        token=token,
        img_url=img_url,
        sel_fdvs=[p.split("||")[0] for p in items],
        sel_vcc=(request.args.get("sel_vcc") or ""),
        sel_temp=(request.args.get("sel_temp") or ""),
        sel_status=(request.args.get("sel_status") or ""),
        sel_plane=(request.args.get("sel_plane") or ""),
        sel_pagetype=(request.args.get("sel_pagetype") or ""),
    )


@app.route("/raw/<token>/<line>")
def raw_line(token: str, line: str):
    data = CACHE.get(token)
    if not data or "rows" not in data:
        return Response("Session expired", status=410)
    try:
        target = int(line)
    except Exception:
        return Response("Bad line", status=400)
    for r in data["rows"]:
        try:
            if int(r.get("line_number", "-1")) == target:
                # Render an HTML page with the raw line and details
                pagetype = (r.get('pagetype','') or '').strip()
                split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
                split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
                split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
                split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
                back_url = url_for(
                    "hist_overview",
                    token=token,
                    fdv=r.get('fdv_file',''),
                    vcc=(r.get('vcc','') or '').strip(),
                    temp=(r.get('temp','') or '').strip(),
                    status=(r.get('status','') or '').strip(),
                    plane=(r.get('plane_group','') or '').strip(),
                    pagetype=pagetype,
                    split_vcc=("1" if split_vcc else "0"),
                    split_temp=("1" if split_temp else "0"),
                    split_plane=("1" if split_plane else "0"),
                    split_pagetype=("1" if split_pagetype else "0"),
                )
                return render_template("raw.html", row=r, token=token, back_url=back_url)
        except Exception:
            continue
    return Response("Not found", status=404)


@app.route("/rawfile/<token>/<line>")
def rawfile(token: str, line: str):
    data = CACHE.get(token)
    if not data or "rows" not in data:
        flash("Session expired. Please re-upload the log.")
        return redirect(url_for("index"))
    try:
        target = int(line)
    except Exception:
        return Response("Bad line", status=400)
    # Determine which raw file to use (support multiple)
    src_idx_param = request.args.get("src")
    src = None
    src_name = None
    if src_idx_param is not None and str(src_idx_param).isdigit() and "raw_files" in data:
        i = int(src_idx_param)
        try:
            src = Path(str(data.get("raw_files")[i]))
            src_name = (data.get("raw_filenames") or [None])[i] if i < len(data.get("raw_filenames", [])) else None
        except Exception:
            src = None
    if src is None:
        # Fallback to single-file
        rf = data.get("raw_file") or (data.get("raw_files") or [None])[0]
        if rf:
            src = Path(str(rf))
        src_name = data.get("raw_filename") or (data.get("raw_filenames") or [None])[0]
    if src is None:
        return Response("Raw file missing", status=410)
    if not src.exists():
        return Response("Raw file missing", status=410)
    # Read file and extract a window around the target line (±15)
    try:
        text = src.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return Response("Failed to read raw file", status=500)
    n = len(text)
    lo = max(1, target - 15)
    hi = min(n, target + 15)
    snippet = []
    for i in range(lo, hi + 1):
        snippet.append({
            "ln": i,
            "content": text[i-1],
            "is_target": (i == target),
        })
    # Build a back link to filtered histogram if we can find the matching row
    fdv = vcc = temp = status = plane = pagetype = ""
    row_src_idx = None
    for r in data["rows"]:
        try:
            if int(r.get("line_number", "-1")) == target:
                fdv = r.get("fdv_file", "")
                vcc = (r.get("vcc", "") or "").strip()
                temp = (r.get("temp", "") or "").strip()
                status = (r.get("status", "") or "").strip()
                plane = (r.get("plane_group", "") or "").strip()
                pagetype = (r.get("pagetype", "") or "").strip()
                row_src_idx = r.get("source_idx")
                break
        except Exception:
            continue
    # Preserve split flags if provided in query string
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
    back_url = url_for(
        "hist_overview",
        token=token,
        fdv=fdv,
        vcc=vcc,
        temp=temp,
        status=status,
        plane=plane,
        pagetype=pagetype,
        split_vcc=("1" if split_vcc else "0"),
        split_temp=("1" if split_temp else "0"),
        split_plane=("1" if split_plane else "0"),
        split_pagetype=("1" if split_pagetype else "0"),
    )
    return render_template(
        "rawfile.html",
        token=token,
    filename=(src_name or data.get("raw_filename") or str(src.name)),
        target_line=target,
        snippet=snippet,
        back_url=back_url,
    )


# Minimal handler to indicate FDV Report feature was removed
@app.route("/fdv", methods=["GET", "POST"])
def fdv_report():
    return render_template("fdv_report_removed.html")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Support multiple files and directory uploads (webkitdirectory). Both inputs share name 'logfiles'.
        files = request.files.getlist("logfiles")
        files = [f for f in files if f and (f.filename or "").strip()]
        if not files:
            flash("Please choose one or more log files (or a directory).")
            return redirect(url_for("index"))
        # Persist uploads to temp first
        tmp_files: List[Path] = []
        names: List[str] = []
        for f in files:
            suffix = Path(f.filename).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                pth = Path(tf.name)
                try:
                    f.save(pth)
                except Exception:
                    continue
            tmp_files.append(pth)
            names.append(Path(f.filename).name or pth.name)
        if not tmp_files:
            flash("Upload failed. Try again.")
            return redirect(url_for("index"))
        token = uuid.uuid4().hex
        # Pre-count total lines per file for smoother progress like report2
        line_totals: list[int] = []
        total_lines_all = 0
        for pth in tmp_files:
            cnt = 0
            try:
                with open(pth, 'r', encoding='utf-8', errors='replace') as _f:
                    for _ in _f:
                        cnt += 1
            except Exception:
                cnt = 0
            line_totals.append(cnt)
            total_lines_all += cnt
        # Initialize CACHE for progress
        CACHE[token] = {
            'status': 'queued',
            'progress': {
                'files_total': len(tmp_files),
                'files_done': 0,
                'percent': 0.0,
                'current_file': '',
                # line-level progress (global and per current file)
                'lines': 0,
                'lines_total': total_lines_all,
                'file_lines_done': 0,
                'file_lines_total': 0,
                'file_percent': 0.0,
            },
            'rows': [],
            'raw_files': [str(p) for p in tmp_files],
            'raw_filenames': names,
            'source_info': 'Uploaded files: ' + ', '.join(names[:5]) + (f' +{len(names)-5} more' if len(names) > 5 else ''),
        }
        def _worker():
            try:
                progress = CACHE[token].get('progress', {})
                CACHE[token]['status'] = 'running'
                all_rows: List[Dict[str, str]] = []
                for idx, pth in enumerate(tmp_files):
                    try:
                        progress['current_file'] = str(pth)
                        # set current file total lines
                        try:
                            progress['file_lines_total'] = int(line_totals[idx])
                        except Exception:
                            progress['file_lines_total'] = 0
                        progress['file_lines_done'] = 0
                        progress['file_percent'] = 0.0
                        CACHE[token]['progress'] = progress
                    except Exception:
                        pass
                    # Hook progress callback to update UI frequently
                    lines_before = sum(line_totals[:idx]) if line_totals and idx < len(line_totals) else 0
                    def _cb(lineno: int, pct: float) -> None:
                        try:
                            progress['file_lines_done'] = int(lineno)
                            progress['file_percent'] = float(max(0.0, min(100.0, pct)))
                            # global lines = finished files + current file progress
                            progress['lines'] = int(lines_before + lineno)
                            lt = int(progress.get('lines_total') or 0)
                            if lt > 0:
                                progress['percent'] = float(max(0.0, min(100.0, (progress['lines'] / lt) * 100.0)))
                            CACHE[token]['progress'] = progress
                        except Exception:
                            pass
                    try:
                        # Smaller interval for smoother updates (every 50k lines)
                        rows, _kept, _markers = pfp.process_file(pth, pfp.PREFIX_DEFAULT, pfp.IGNORE_VALUE_DEFAULT, progress=50000, progress_cb=_cb)
                    except Exception:
                        rows = []
                    # Annotate with source index and human name
                    src_name = names[idx] if idx < len(names) else Path(pth).name
                    for r in rows:
                        r['source_idx'] = str(idx)
                        r['source_name'] = src_name
                    all_rows.extend(rows)
                    # Update progress
                    try:
                        progress['files_done'] = idx + 1
                        # Snap current file to done
                        try:
                            progress['file_lines_done'] = int(line_totals[idx])
                            progress['file_percent'] = 100.0 if int(line_totals[idx]) > 0 else float(progress.get('file_percent') or 0.0)
                        except Exception:
                            pass
                        # Global lines and percent
                        progress['lines'] = int(sum(line_totals[:idx+1]))
                        lt = int(progress.get('lines_total') or 0)
                        if lt > 0:
                            progress['percent'] = float(max(0.0, min(100.0, (progress['lines'] / lt) * 100.0)))
                        else:
                            # fallback to files-based percent
                            ft = float(progress.get('files_total') or len(tmp_files) or 1)
                            progress['percent'] = float(min(100.0, max(0.0, (progress['files_done'] / ft) * 100.0)))
                        CACHE[token]['progress'] = progress
                    except Exception:
                        pass
                    # Update a lightweight table HTML for progress page
                    try:
                        vt_now = group_by_fdv_with_splits(all_rows, split_vcc=False, split_temp=False, split_plane=False)
                        # Simple inline table with fdv and count so far
                        parts = [
                            '<table style="border-collapse:collapse;">\n',
                            '<thead><tr><th style="border:1px solid #ddd;padding:4px 6px;">fdv_file</th><th style="border:1px solid #ddd;padding:4px 6px;">count</th></tr></thead>\n',
                            '<tbody>\n'
                        ]
                        for r in vt_now[:50]:
                            fdv = (r.get('fdv_file') or '').replace('&','&amp;').replace('<','&lt;')
                            cnt = r.get('count') or ''
                            parts.append(f'<tr><td style="border:1px solid #ddd;padding:4px 6px;">{fdv}</td><td style="border:1px solid #ddd;padding:4px 6px;">{cnt}</td></tr>\n')
                        parts.append('</tbody></table>')
                        CACHE[token]['fdvtable_html'] = ''.join(parts)
                    except Exception:
                        pass
                # Store rows and mark done
                CACHE[token]['rows'] = all_rows
                # Precompute default stats for convenience
                try:
                    CACHE[token]['vt'] = group_by_fdv_with_splits(all_rows, split_vcc=False, split_temp=False, split_plane=False)
                except Exception:
                    pass
                CACHE[token]['status'] = 'done'
            except Exception as e:
                try:
                    CACHE[token]['status'] = 'error'
                    CACHE[token]['error'] = str(e)
                except Exception:
                    pass
        threading.Thread(target=_worker, daemon=True).start()
        return render_template('poll_progress.html', token=token)
    return render_template("index.html")


@app.route("/download/<token>/<kind>")
def download_csv(token: str, kind: str):
    data = CACHE.get(token)
    if not data or kind not in ("vt", "raw"):
        flash("Download expired or invalid.")
        return redirect(url_for("index"))
    # Respect current split flags if provided
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
    base_rows: List[Dict[str, str]] = data.get("rows", [])
    if kind == "vt":
        # Recompute from rows to reflect flags
        rows = group_by_fdv_with_splits(
            base_rows,
            split_vcc=split_vcc,
            split_temp=split_temp,
            split_plane=split_plane,
            split_pagetype=split_pagetype,
        )
        headers = [
            "fdv_file", "specname", "vcc", "temp", "pagemap", "pagetype", "status", "plane_group", "pr", "count",
            "valid_fuseid_count", "MinSpec", "MaxSpec", "min", "max", "mean", "stdev", "median", "comments"
        ]
        filename = (
            "stats_by_fdv_"
            f"{'vcc_' if split_vcc else ''}"
            f"{'temp_' if split_temp else ''}"
            f"{'plane_' if split_plane else ''}"
            f"{'pagetype_' if split_pagetype else ''}"
            "pagemap_status_plane.csv"
        )
        buf = io.StringIO()
        buf.write(",".join(headers) + "\n")
        for r in rows:
            buf.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
        csv_bytes = buf.getvalue().encode("utf-8", errors="replace")
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )
    else:
        # Build master CSV with the raw rows used to compute stats.
        # Apply the same row-level filters as stats and exclude invalid FUSEIDs.
        import csv as _csv
        headers = [
            "line_number", "source_name", "fdv_file", "vcc", "temp", "status", "plane_group",
            "dut_id", "value", "pr", "fuseid", "pagemap", "pagetype", "blk", "page", "wl", "phypage",
            "tm", "poll__test", "tname"
        ]
        filename = "fdv_poll_master_rows_for_stats.csv"
        buf = io.StringIO()
        w = _csv.writer(buf, lineterminator="\n")
        w.writerow(headers)
        for r in base_rows:
            keep, fdv, vcc, temp, status, plane, v = _filter_poll_row(r)
            if not keep or v is None:
                continue
            # Exclude invalid FUSEID rows to match stats input
            fuseid = _get_fuseid(r)
            if not _is_valid_fuseid(fuseid):
                continue
            row = [
                r.get("line_number", ""),
                r.get("source_name", ""),
                fdv,
                vcc,
                temp,
                status,
                plane,
                r.get("dut_id", ""),
                (f"{float(v):.6g}" if v is not None else ""),
                (r.get("pr", "") or ""),
                fuseid or "",
                _extract_pagemap_from_row(r) or "",
                (r.get("pagetype", "") or ""),
                (r.get("blk", "") or ""),
                (r.get("page", "") or ""),
                (r.get("wl", "") or ""),
                (r.get("phypage", "") or ""),
                (r.get("tm", "") or ""),
                (r.get("poll__test", "") or ""),
                (r.get("tname", "") or ""),
            ]
            w.writerow(row)
        csv_bytes = buf.getvalue().encode("utf-8", errors="replace")
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )


@app.route("/eimpro", methods=["GET", "POST"])
def eimpro():
    """Upload a CSV and generate EIMPRO plots using read_eimpro_plot.py utilities."""
    global rep
    # Lazy import here to avoid heavy dependency load at app startup
    if rep is None:
        try:
            import read_eimpro_plot as _rep
            globals()['rep'] = _rep
        except Exception as e:
            flash(f"EIMPRO utilities are unavailable: {e}")
            if request.method == "POST":
                return redirect(url_for("eimpro"))
            return render_template("eimpro.html")
    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file or file.filename == "":
            flash("Please choose a CSV file to upload.")
            return redirect(url_for("eimpro"))
        # Save uploaded CSV to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".csv") as tf:
            tmp_path = Path(tf.name)
            file.save(tmp_path)
        # Load data with robust loader
        try:
            df = rep.load_data(str(tmp_path))
        except Exception as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            flash(f"Failed to read CSV: {e}")
            return redirect(url_for("eimpro"))
        finally:
            # keep tmp_path for provenance only; safe to remove now
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Create an output temp directory for generated images
        out_dir = Path(tempfile.mkdtemp(prefix="eimpro_"))
        base = Path(file.filename).stem or "eimpro"
        generated: list[Path] = []

        # Variability plot (WL vs RBER by pagetype, split by readtype/dut)
        try:
            title = "RBER vs WL by pagetype - split by readtype and dut"
            if getattr(rep, "_HAVE_SEABORN", False):
                fig = rep.plot_with_seaborn(df, title)
            else:
                fig = rep.plot_with_matplotlib(df, title)
            var_path = out_dir / f"{base}_variability.png"
            try:
                fig.savefig(var_path, dpi=200)
                generated.append(var_path)
            finally:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close(fig)
                except Exception:
                    pass
        except Exception as e:
            flash(f"Variability plot failed: {e}")

        # Summary
        try:
            p = out_dir / f"{base}_summary.png"
            rep.generate_summary_png(df, str(p))
            generated.append(p)
        except Exception as e:
            flash(f"Summary generation failed: {e}")

        # RCDF by pagetype
        try:
            p = out_dir / f"{base}_rcdf.png"
            rep.generate_rcdf_pagetype_only_png(df, str(p))
            generated.append(p)
        except Exception as e:
            flash(f"RCDF (pagetype) failed: {e}")

        # RCDF by pagetype and deck
        try:
            p = out_dir / f"{base}_rcdf_pagetype_deck.png"
            rep.generate_rcdf_pagetype_deck_png(df, str(p))
            generated.append(p)
        except Exception as e:
            flash(f"RCDF (pagetype/deck) failed: {e}")

        # RCDF per DUT with readtypes side-by-side
        try:
            rep.generate_rcdf_pagetype_deck_per_dut_pngs(df, str(out_dir), base)
            # collect files matching pattern
            for p in out_dir.glob(f"{base}_rcdf_pagetype_deck_dut_*.png"):
                generated.append(p)
        except Exception as e:
            flash(f"Per-DUT RCDFs failed: {e}")

        # Correlation between readtypes (requires blk)
        try:
            p = out_dir / f"{base}_correlation.png"
            rep.generate_readtype_correlation_png(df, str(p))
            generated.append(p)
        except Exception as e:
            # Non-fatal; often missing a blk column
            flash(f"Correlation skipped/failed: {e}")

        # Register files in cache under a new token
        token = uuid.uuid4().hex
        files = sorted({str(p) for p in generated if p.exists()})
        CACHE[token] = {"eimpro_dir": str(out_dir), "eimpro_files": [str(Path(f).name) for f in files]}
        return render_template(
            "eimpro_results.html",
            files=CACHE[token]["eimpro_files"],
            token=token,
        )
    # GET
    return render_template("eimpro.html")


@app.route("/download/eimpro/<token>/<name>")
def download_eimpro_file(token: str, name: str):
    info = CACHE.get(token)
    if not info or "eimpro_dir" not in info or "eimpro_files" not in info:
        flash("Download expired or invalid.")
        return redirect(url_for("eimpro"))
    # Prevent path traversal by restricting to known basenames
    if name not in info["eimpro_files"]:
        flash("File not found.")
        return redirect(url_for("eimpro"))
    fp = Path(info["eimpro_dir"]) / name
    if not fp.exists():
        flash("File missing on server.")
        return redirect(url_for("eimpro"))
    # Inline display so it can be embedded as <img>
    return send_file(str(fp), as_attachment=False)


@app.route("/download/eimprozip/<token>")
def download_eimpro_zip(token: str):
    info = CACHE.get(token)
    if not info or "eimpro_dir" not in info or "eimpro_files" not in info:
        flash("Download expired or invalid.")
        return redirect(url_for("eimpro"))
    out_dir = Path(info["eimpro_dir"])
    files = [out_dir / name for name in info["eimpro_files"]]
    # Build a zip in-memory
    import zipfile, io as _io
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if fp.exists():
                zf.write(str(fp), arcname=str(fp.name))
    buf.seek(0)
    return Response(
        buf.read(),
        mimetype="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=eimpro_plots_{token}.zip"
        },
    )


# Note: FDV OUTPUT (fdvrun) reporting routes removed per request.


@app.route("/results/<token>")
def results_view(token: str):
    data = CACHE.get(token)
    if not data or ("vt" not in data and "rows" not in data):
        flash("Session expired. Please re-upload the log.")
        return redirect(url_for("index"))
    rows = data.get("rows", [])
    # Determine current split flags (default off)
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    split_pagetype = (request.args.get("split_pagetype") or "0").strip() not in ("0", "false", "off")
    stats_vt = group_by_fdv_with_splits(
        rows,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
        split_pagetype=split_pagetype,
    )
    persist_url = url_for('poll_persist', token=token)
    # Extract data source info for banner
    source_info = data.get('source_info', 'Unknown source')
    html = render_template(
        "results.html",
        stats_vt=stats_vt,
        rows_count=len(rows),
        source_info=source_info,
        download_vt=url_for(
            "download_csv",
            token=token,
            kind="vt",
            split_vcc=("1" if split_vcc else "0"),
            split_temp=("1" if split_temp else "0"),
            split_plane=("1" if split_plane else "0"),
            split_pagetype=("1" if split_pagetype else "0"),
        ),
        token=token,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
        split_pagetype=split_pagetype,
        persist_url=persist_url,
        aliases=_list_aliases(),
    )
    try:
        _persist_write('poll', token, html)
    except Exception:
        pass
    return html


# Progress/status endpoints for gradual display
@app.route('/progress/<token>')
def poll_progress(token: str):
    # If already done, go straight to results
    info = CACHE.get(token)
    if info and info.get('status') == 'done':
        return redirect(url_for('results_view', token=token))
    return render_template('poll_progress.html', token=token)


@app.route('/poll/status/<token>')
def poll_status(token: str):
    info = CACHE.get(token) or {}
    st = info.get('status') or 'unknown'
    pr = info.get('progress') or {}
    import json
    return Response(json.dumps({'status': st, 'progress': pr, 'error': info.get('error') or ''}), mimetype='application/json')


@app.route('/poll/status/fdvtable/<token>')
def poll_status_fdvtable(token: str):
    info = CACHE.get(token) or {}
    html = info.get('fdvtable_html') or '<em>Collecting…</em>'
    return Response(html, mimetype='text/html')


@app.route('/persist/poll/<token>')
def poll_persist(token: str):
    """Serve a previously saved persistent HTML snapshot for POLL results."""
    path = _persist_base_dir() / 'poll' / token / 'index.html'
    try:
        if path.is_file():
            return send_file(str(path), mimetype='text/html')
    except Exception:
        pass
    return Response('Snapshot not found. Re-run analysis to regenerate.', status=404)


@app.route('/alias/save', methods=['POST'])
def alias_save():
    alias = (request.form.get('alias') or '').strip()
    token = (request.form.get('token') or '').strip()
    if not alias or not token:
        return Response('alias and token required', status=400)
    # Render current results page and persist under alias dir
    data = CACHE.get(token)
    if not data or ('rows' not in data):
        return Response('Invalid token or expired session', status=410)
    split_vcc = (request.form.get('split_vcc') or '0').strip() not in ('0','false','off')
    split_temp = (request.form.get('split_temp') or '0').strip() not in ('0','false','off')
    split_plane = (request.form.get('split_plane') or '0').strip() not in ('0','false','off')
    split_pagetype = (request.form.get('split_pagetype') or '0').strip() not in ('0','false','off')
    rows = data.get('rows', [])
    stats_vt = group_by_fdv_with_splits(
        rows,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
        split_pagetype=split_pagetype,
    )
    html = render_template(
        'results.html',
        stats_vt=stats_vt,
        rows_count=len(rows),
        download_vt=url_for(
            'download_csv',
            token=token,
            kind='vt',
            split_vcc=('1' if split_vcc else '0'),
            split_temp=('1' if split_temp else '0'),
            split_plane=('1' if split_plane else '0'),
            split_pagetype=('1' if split_pagetype else '0'),
        ),
        token=token,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
        split_pagetype=split_pagetype,
        persist_url=url_for('alias_view', name=alias, _external=True),
        aliases=_list_aliases(),
    )
    base = _alias_base_dir() / alias
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / 'index.html').write_text(html, encoding='utf-8')
        # also persist raw rows to support future updates with dedup
        _save_alias_rows(alias, rows)
    except Exception as e:
        return Response(f'Failed to save alias: {e}', status=500)
    return redirect(url_for('alias_view', name=alias))


@app.route('/alias/update', methods=['POST'])
def alias_update():
    """Merge current session rows into an existing alias and regenerate its snapshot, avoiding duplicates."""
    alias = (request.form.get('alias') or '').strip()
    token = (request.form.get('token') or '').strip()
    if not alias or not token:
        return Response('alias and token required', status=400)
    data = CACHE.get(token) or {}
    new_rows = list(data.get('rows') or [])
    if not new_rows:
        return Response('No rows in current session', status=400)
    # Load existing rows for alias and merge with dedup
    merged = []
    sigs = set()
    for r in (_load_alias_rows(alias) or []):
        merged.append(r)
        try:
            sigs.add(_row_signature(r))
        except Exception:
            pass
    added = 0
    for r in new_rows:
        try:
            s = _row_signature(r)
        except Exception:
            s = None
        if s and s in sigs:
            continue
        if s:
            sigs.add(s)
        merged.append(r)
        added += 1
    _save_alias_rows(alias, merged)
    # Re-render results with merged rows and overwrite alias HTML
    split_vcc = (request.form.get('split_vcc') or '0').strip() not in ('0','false','off')
    split_temp = (request.form.get('split_temp') or '0').strip() not in ('0','false','off')
    split_plane = (request.form.get('split_plane') or '0').strip() not in ('0','false','off')
    split_pagetype = (request.form.get('split_pagetype') or '0').strip() not in ('0','false','off')
    stats_vt = group_by_fdv_with_splits(
        merged,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
        split_pagetype=split_pagetype,
    )
    html = render_template(
        'results.html',
        stats_vt=stats_vt,
        rows_count=len(merged),
        download_vt=url_for(
            'download_csv',
            token=token,
            kind='vt',
            split_vcc=('1' if split_vcc else '0'),
            split_temp=('1' if split_temp else '0'),
            split_plane=('1' if split_plane else '0'),
            split_pagetype=('1' if split_pagetype else '0'),
        ),
        token=token,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
        split_pagetype=split_pagetype,
        persist_url=url_for('alias_view', name=alias, _external=True),
        aliases=_list_aliases(),
    )
    base = _alias_base_dir() / alias
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / 'index.html').write_text(html, encoding='utf-8')
    except Exception as e:
        return Response(f'Failed to update alias: {e}', status=500)
    # Redirect to the alias view permalink
    return redirect(url_for('alias_view', name=alias))


@app.route('/results/<token>/update', methods=['POST'])
def update_report(token: str):
    """Append additional uploaded logs to the existing session and refresh results."""
    data = CACHE.get(token)
    if not data or ('rows' not in data):
        flash('Session expired. Please re-upload the log.')
        return redirect(url_for('index'))
    files = request.files.getlist('logfiles')
    files = [f for f in files if f and (f.filename or '').strip()]
    if not files:
        flash('Choose one or more log files to append.')
        return redirect(url_for('results_view', token=token))
    # Save to temp and process each
    tmp_files: list[Path] = []
    names: list[str] = []
    for f in files:
        suffix = Path(f.filename).suffix or '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            pth = Path(tf.name)
            try:
                f.save(pth)
            except Exception:
                continue
        tmp_files.append(pth)
        names.append(Path(f.filename).name or pth.name)
    if not tmp_files:
        flash('Append failed.')
        return redirect(url_for('results_view', token=token))
    # Process and extend rows
    all_rows = list(data.get('rows') or [])
    for idx, pth in enumerate(tmp_files):
        try:
            rows, _kept, _markers = pfp.process_file(pth, pfp.PREFIX_DEFAULT, pfp.IGNORE_VALUE_DEFAULT, progress=50000, progress_cb=None)
        except Exception:
            rows = []
        src_name = names[idx] if idx < len(names) else Path(pth).name
        for r in rows:
            r['source_idx'] = str(len(data.get('raw_files') or []))
            r['source_name'] = src_name
        all_rows.extend(rows)
        try:
            pth.unlink(missing_ok=True)
        except Exception:
            pass
    # Dedup on the combined set
    deduped = []
    sigs = set()
    for r in all_rows:
        try:
            s = _row_signature(r)
        except Exception:
            s = None
        if s and s in sigs:
            continue
        if s:
            sigs.add(s)
        deduped.append(r)
    CACHE[token]['rows'] = deduped
    # Re-render
    return redirect(url_for('results_view', token=token))


@app.route('/alias/<name>')
def alias_view(name: str):
    path = _alias_base_dir() / name / 'index.html'
    try:
        if path.is_file():
            return send_file(str(path), mimetype='text/html')
    except Exception:
        pass
    return Response('Alias not found', status=404)


@app.route('/specs/update', methods=['POST'])
def specs_update():
    # Expect form fields: key (composite group key), MinSpec, MaxSpec
    key = (request.form.get('key') or '').strip()
    if not key:
        return Response('key required', status=400)
    # Only act on fields that are actually present to avoid unintended deletions
    present_min = ('MinSpec' in request.form)
    present_max = ('MaxSpec' in request.form)
    minv = (request.form.get('MinSpec') if present_min else None)
    maxv = (request.form.get('MaxSpec') if present_max else None)
    if minv is not None:
        minv = minv.strip()
    if maxv is not None:
        maxv = maxv.strip()
    specs = _load_specs()
    entry = specs.get(key, {})
    if present_min:
        if (minv or '') != '':
            entry['MinSpec'] = minv
        else:
            entry.pop('MinSpec', None)
    if present_max:
        if (maxv or '') != '':
            entry['MaxSpec'] = maxv
        else:
            entry.pop('MaxSpec', None)
    if entry:
        specs[key] = entry
    else:
        specs.pop(key, None)
    _save_specs(specs)
    # Return JSON
    import json
    return Response(json.dumps({'ok': True, 'key': key, 'MinSpec': entry.get('MinSpec',''), 'MaxSpec': entry.get('MaxSpec','')}), mimetype='application/json')


@app.route('/rawdata/<token>')
def rawdata_view(token: str):
    """Show raw data table for one or more selected FDV POLL items and provide XY plotting interface.
    Query params:
      - selections: comma-separated items of the form fdv||vcc||temp||status||plane||pagetype
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        flash('Session expired. Please re-upload the log.')
        return redirect(url_for('index'))
    
    rows: List[Dict[str, str]] = data.get('rows', [])
    
    # Parse selections parameter
    selections_str = (request.args.get('selections') or '').strip()
    selections = [s.strip() for s in selections_str.split(',') if s.strip()] if selections_str else []
    
    if not selections:
        flash('No selections provided. Please select at least one row from results.')
        return redirect(url_for('results_view', token=token))
    
    # Filter rows based on selections - STRICTLY enforce only exact matches
    # A row is included only if it matches the criteria of at least one selection
    filtered_rows: List[Dict[str, str]] = []
    for r in rows:
        keep, fdv, vcc, temp, status, plane, pagetype, v = _filter_poll_row(r)
        if not keep or v is None:
            continue
        
        # Check if this row matches ANY of the selections with STRICT criteria
        matches_selection = False
        for sel in selections:
            parts = sel.split('||')
            if len(parts) < 6:
                continue
            sel_fdv, sel_vcc, sel_temp, sel_status, sel_plane, sel_pagetype = [p or "" for p in parts[:6]]
            
            # For strict matching: a row matches only if it has the EXACT same values
            # for all non-empty dimensions in the selection criteria
            row_matches = True
            
            # Check FDV - if selection specifies FDV, row must match exactly
            if sel_fdv and fdv != sel_fdv:
                row_matches = False
            
            # Check VCC - if selection specifies VCC, row must match exactly
            if sel_vcc and vcc != sel_vcc:
                row_matches = False
            
            # Check TEMP - if selection specifies TEMP, row must match exactly
            if sel_temp and temp != sel_temp:
                row_matches = False
            
            # Check STATUS - if selection specifies STATUS, row must match exactly
            if sel_status and status != sel_status:
                row_matches = False
            
            # Check PLANE - if selection specifies PLANE, row must match exactly
            if sel_plane and plane != sel_plane:
                row_matches = False
            
            # Check PAGETYPE - if selection specifies PAGETYPE, row must match exactly
            if sel_pagetype and (pagetype or "") != sel_pagetype:
                row_matches = False
            
            # If this row matches all specified criteria for this selection, include it
            if row_matches:
                matches_selection = True
                break
        
        if matches_selection:
            filtered_rows.append(r)
    
    # Build a table of the raw data with REQUIRED fields as per criteria
    table_data = []
    for r in filtered_rows:
        keep, fdv, vcc, temp, status, plane, pagetype, v = _filter_poll_row(r)
        
        # Extract required fields
        pagemap = _extract_pagemap_from_row(r)
        specname = _specname_from_fdv(fdv) if fdv else ''
        pr = r.get('pr', '')
        
        # Only include rows that have ALL required criteria fields
        # (fdv file, specname, pagemap, status, pr are always required)
        if not all([fdv, specname, pagemap, status, pr]):
            continue  # Skip rows missing required fields
        
        table_data.append({
            'line': r.get('line_number', ''),
            'fdv': fdv,
            'specname': specname,
            'pagemap': pagemap,
            'status': status,
            'pr': pr,
            'vcc': vcc,
            'temp': temp,
            'plane': plane,
            'pagetype': pagetype,
            'value': f'{v:.6g}' if v is not None else '',
            'dut_id': r.get('dut_id', ''),
            'wl': r.get('wl', ''),
            'blk': r.get('blk', ''),
            'page': r.get('page', ''),
            'phypage': r.get('phypage', ''),
            'tm': r.get('tm', ''),
            'tname': r.get('tname', ''),
        })
    
    # Collect unique values for X/Y axis dropdowns
    available_fields = ['value', 'wl', 'blk', 'page', 'phypage', 'tm', 'pr']
    
    # Format selections for display
    selection_info = f"{len(selections)} item(s) selected"
    
    return render_template(
        'rawdata.html',
        token=token,
        selections=selections,
        selection_info=selection_info,
        table_data=table_data,
        available_fields=available_fields,
        row_count=len(table_data),
    )


@app.route('/rawdata/<token>/plot', methods=['POST'])
def rawdata_plot(token: str):
    r"""Generate XY scatter plot for selected fields from raw data across multiple selections.
    POST params:
      - selections: comma-separated filter strings (fdv||vcc||temp||status||plane||pagetype)
      - x_field: field name to plot on X axis
      - y_field: field name to plot on Y axis
      - split_by_selection: 'true' to create separate subplots per selection, else all on one plot
      - color_by_field: optional field to color-code points by (vcc, temp, status, plane, pagetype, pr, wl, blk, tm, fdv)
      - x_custom: optional custom extraction from tname for x-axis (e.g., "POLL_(\w+)_")
      - y_custom: optional custom extraction from tname for y-axis (e.g., "POLL_(\w+)_")
    """
    data = CACHE.get(token)
    if not data or 'rows' not in data:
        return jsonify({'error': 'Session expired'}), 410
    
    rows: List[Dict[str, str]] = data.get('rows', [])
    
    # Parse excluded rows
    excluded_rows = set()
    try:
        import json as json_module
        excluded_str = (request.form.get('excluded_rows') or '').strip()
        if excluded_str:
            excluded_rows = set(json_module.loads(excluded_str))
    except Exception:
        pass
    
    # Parse tname exclusion pattern
    excluded_rows_tname_pattern = (request.form.get('excluded_rows_tname_pattern') or '').strip()
    tname_regex = None
    if excluded_rows_tname_pattern:
        try:
            import re as regex_module_exclude
            tname_regex = regex_module_exclude.compile(excluded_rows_tname_pattern)
        except Exception as e:
            return jsonify({'error': f'Invalid tname exclusion regex pattern: {str(e)}'}), 400
    
    # Parse selections
    selections_str = (request.form.get('selections') or '').strip()
    selections = [s.strip() for s in selections_str.split(',') if s.strip()] if selections_str else []
    
    x_field = (request.form.get('x_field') or '').strip()
    y_field = (request.form.get('y_field') or '').strip()
    split_by_selection = (request.form.get('split_by_selection') or 'false').strip().lower() == 'true'
    color_by_field = (request.form.get('color_by_field') or '').strip()
    legend_base_field = (request.form.get('legend_base_field') or '').strip()
    x_custom = (request.form.get('x_custom') or '').strip()
    y_custom = (request.form.get('y_custom') or '').strip()
    color_custom = (request.form.get('color_custom') or '').strip()
    legend_custom = (request.form.get('legend_custom') or '').strip()
    custom_x_title = (request.form.get('custom_x_title') or '').strip()
    custom_y_title = (request.form.get('custom_y_title') or '').strip()
    
    # Parse custom legend names
    custom_legends = {}
    try:
        import json as json_module
        legends_str = (request.form.get('custom_legends') or '').strip()
        if legends_str:
            custom_legends = json_module.loads(legends_str)
    except Exception:
        pass
    
    if not x_field or not y_field:
        return jsonify({'error': 'X and Y fields required'}), 400
    
    if not selections:
        return jsonify({'error': 'No selections provided'}), 400
    
    # Helper function to extract value from tname using regex
    import re as regex_module
    def extract_from_tname(tname_val, pattern, compiled_regex=None):
        """Extract value from tname using regex pattern (with caching)"""
        if not tname_val or not pattern:
            return None
        try:
            # Use pre-compiled regex if provided, else use cached compilation
            regex_obj = compiled_regex or get_compiled_regex(pattern)
            if regex_obj:
                match = regex_obj.search(tname_val)
                if match:
                    if match.groups():
                        return match.group(1)
                    else:
                        return match.group(0)
        except Exception:
            pass
        return None
    
    # PRE-COMPILE ALL REGEX PATTERNS for performance (avoid recompilation in loop)
    x_custom_compiled = get_compiled_regex(x_custom) if x_custom else None
    y_custom_compiled = get_compiled_regex(y_custom) if y_custom else None
    color_custom_compiled = get_compiled_regex(color_custom) if color_custom else None
    legend_custom_compiled = get_compiled_regex(legend_custom) if legend_custom else None
    
    # Organize data by selection if split_by_selection is true
    data_by_selection = {}  # selection -> list of (x, y, color_val) tuples
    
    for r in rows:
        keep, fdv, vcc, temp, status, plane, pagetype, v = _filter_poll_row(r)
        if not keep or v is None:
            continue
        
        # Skip rows that have been excluded by the user
        line_num = str(r.get('line_number', ''))
        if line_num in excluded_rows:
            continue
        
        # Skip rows matching tname exclusion pattern
        if tname_regex:
            row_tname = r.get('tname', '')
            if row_tname and tname_regex.search(row_tname):
                continue
        
        # Check if this row matches any selection with STRICT criteria
        matched_sel = None
        for sel in selections:
            parts = sel.split('||')
            if len(parts) < 6:
                continue
            sel_fdv, sel_vcc, sel_temp, sel_status, sel_plane, sel_pagetype = [p or "" for p in parts[:6]]
            
            # For strict matching: check ALL non-empty dimensions must match exactly
            if sel_fdv and fdv != sel_fdv:
                continue
            if sel_vcc and vcc != sel_vcc:
                continue
            if sel_temp and temp != sel_temp:
                continue
            if sel_status and status != sel_status:
                continue
            if sel_plane and plane != sel_plane:
                continue
            if sel_pagetype and (pagetype or "") != sel_pagetype:
                continue
            
            # All non-empty dimensions matched strictly
            matched_sel = sel
            break
        
        if not matched_sel:
            continue
        
        # ENFORCE REQUIRED CRITERIA: rows must have all mandatory fields
        # fdv file, specname, pagemap, status, pr are always required
        pagemap = _extract_pagemap_from_row(r)
        specname = _specname_from_fdv(fdv) if fdv else ''
        pr = r.get('pr', '')
        
        # Skip rows missing any required field
        if not all([fdv, specname, pagemap, status, pr]):
            continue
        
        # Extract X value
        if x_field == 'value':
            x = v
        elif x_field == 'tname_custom' and x_custom:
            x_str = extract_from_tname(r.get('tname', ''), x_custom, x_custom_compiled)
            if x_str is None:
                continue
            try:
                x = float(x_str)
            except Exception:
                continue
        else:
            x_str = (r.get(x_field, '') or '').strip()
            try:
                x = float(x_str)
            except Exception:
                continue
        
        # Extract Y value
        if y_field == 'value':
            y = v
        elif y_field == 'tname_custom' and y_custom:
            y_str = extract_from_tname(r.get('tname', ''), y_custom, y_custom_compiled)
            if y_str is None:
                continue
            try:
                y = float(y_str)
            except Exception:
                continue
        else:
            y_str = (r.get(y_field, '') or '').strip()
            try:
                y = float(y_str)
            except Exception:
                continue
        
        # Extract color dimension if provided
        color_val = ""
        if color_by_field:
            if color_by_field == 'value':
                color_val = f"{v:.4g}"
            elif color_by_field == 'fdv':
                color_val = fdv
            elif color_by_field == 'tname_custom' and color_custom:
                color_str = extract_from_tname(r.get('tname', ''), color_custom, color_custom_compiled)
                color_val = color_str if color_str else ""
            else:
                color_val = (r.get(color_by_field, '') or '').strip()
        
        # Extract legend base dimension if provided
        legend_val = ""
        if legend_base_field:
            if legend_base_field == 'value':
                legend_val = f"{v:.4g}"
            elif legend_base_field == 'fdv':
                legend_val = fdv
            elif legend_base_field == 'dut':
                legend_val = (r.get('dut', '') or '').strip()
            elif legend_base_field == 'tname_custom' and legend_custom:
                legend_str = extract_from_tname(r.get('tname', ''), legend_custom, legend_custom_compiled)
                legend_val = legend_str if legend_str else ""
            else:
                legend_val = (r.get(legend_base_field, '') or '').strip()
        
        # Store data by selection (include legend_val for grouping)
        sel_key = matched_sel
        if sel_key not in data_by_selection:
            data_by_selection[sel_key] = []
        data_by_selection[sel_key].append((x, y, color_val, legend_val))
    
    if not data_by_selection or sum(len(v) for v in data_by_selection.values()) == 0:
        return jsonify({'error': 'No data points to plot'}), 400
    
    # Generate interactive plot using Plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return jsonify({'error': 'Plotly not installed. Install with: pip install plotly'}), 500
    
    import colorsys
    
    # Prepare data for plotting
    all_color_vals = set()
    if color_by_field:
        for points in data_by_selection.values():
            for point in points:
                x, y, color_val, legend_val = point if len(point) == 4 else (point[0], point[1], point[2], '')
                if color_val:
                    all_color_vals.add(color_val)
    unique_color_vals = sorted(list(all_color_vals))
    color_palette = get_color_palette_cached(len(unique_color_vals))
    color_map = {val: color_palette[i] for i, val in enumerate(unique_color_vals)}
    
    # Create subplots if split_by_selection is true
    if split_by_selection:
        num_selections = len(data_by_selection)
        ncols = 1  # One chart per row (vertical stacking)
        nrows = num_selections
        subplot_titles = []
        for sel_key in sorted(data_by_selection.keys()):
            sel_label = sel_key.split('||')[0] if '||' in sel_key else sel_key
            # Use custom legend name if provided
            display_label = custom_legends.get(sel_key, sel_label)
            subplot_titles.append(display_label)
        
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=subplot_titles,
            specs=[[{'secondary_y': False} for _ in range(ncols)] for _ in range(nrows)]
        )
        
        plot_idx = 0
        for sel_key in sorted(data_by_selection.keys()):
            points = data_by_selection[sel_key]
            row = (plot_idx // ncols) + 1
            col = (plot_idx % ncols) + 1
            
            # Group by color_by_field if provided
            if color_by_field:
                grouped = {}
                for point in points:
                    x, y, color_val, legend_val = point if len(point) == 4 else (point[0], point[1], point[2], '')
                    if color_val not in grouped:
                        grouped[color_val] = ([], [])
                    grouped[color_val][0].append(x)
                    grouped[color_val][1].append(y)
                
                for color_val, (xs, ys) in grouped.items():
                    hover_text = [f'<b>{color_by_field}:</b> {color_val}<br><b>{x_field}:</b> {x:.4g}<br><b>{y_field}:</b> {y:.4g}' 
                                  for x, y in zip(xs, ys)]
                    fig.add_trace(
                        go.Scatter(x=xs, y=ys, mode='markers',
                                  marker=dict(size=8, symbol='x', color=color_map.get(color_val, '#4e79a7'),
                                            line=dict(color='#1f3551', width=1.5)),
                                  name=f'{color_by_field}={color_val}',
                                  hovertext=hover_text,
                                  hoverinfo='text',
                                  showlegend=(plot_idx == 0)),  # Only show legend for first subplot
                        row=row, col=col
                    )
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                hover_text = [f'<b>{x_field}:</b> {x:.4g}<br><b>{y_field}:</b> {y:.4g}' 
                              for x, y in zip(xs, ys)]
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode='markers',
                              marker=dict(size=8, symbol='x', color='#4e79a7',
                                        line=dict(color='#1f3551', width=1.5)),
                              name='data',
                              hovertext=hover_text,
                              hoverinfo='text',
                              showlegend=False),
                    row=row, col=col
                )
            
            # Use custom titles if provided
            x_label = custom_x_title if custom_x_title else (x_field if x_field != 'tname_custom' else f'tname: {x_custom}')
            y_label = custom_y_title if custom_y_title else (y_field if y_field != 'tname_custom' else f'tname: {y_custom}')
            fig.update_xaxes(title_text=x_label, row=row, col=col)
            fig.update_yaxes(title_text=y_label, row=row, col=col)
            plot_idx += 1
        
        fig.update_layout(height=500*nrows, width=800, hovermode='closest')
    else:
        # Single plot with all selections
        fig = go.Figure()
        
        # Alphanumeric markers (A-Z, 0-9)
        # We'll use simple circle markers and display the character as text overlay
        marker_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
                       '6', '7', '8', '9']
        
        # Different marker symbols for visual differentiation
        marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
                         'triangle-left', 'triangle-right', 'star', 'pentagon', 'hexagon', 'octagon']
        
        marker_idx = 0
        
        for sel_key in sorted(data_by_selection.keys()):
            points = data_by_selection[sel_key]
            sel_label = sel_key.split('||')[0] if '||' in sel_key else sel_key
            
            # Use custom legend name if provided
            display_label = custom_legends.get(sel_key, sel_label)
            
            # If legend_base_field is set, group by that field instead
            if legend_base_field:
                grouped_legend = {}
                for point in points:
                    x, y, color_val, legend_val = point if len(point) == 4 else (point[0], point[1], point[2], '')
                    if legend_val not in grouped_legend:
                        grouped_legend[legend_val] = ([], [], [])
                    grouped_legend[legend_val][0].append(x)
                    grouped_legend[legend_val][1].append(y)
                    grouped_legend[legend_val][2].append(color_val)
                
                # Generate distinct colors for each legend group
                legend_colors = get_color_palette_cached(len(grouped_legend))
                
                for idx, (legend_val, (xs, ys, color_vals)) in enumerate(sorted(grouped_legend.items())):
                    hover_text = [f'<b>Selection:</b> {sel_label}<br><b>{legend_base_field}:</b> {legend_val}<br><b>{x_field}:</b> {x:.4g}<br><b>{y_field}:</b> {y:.4g}' 
                                  for x, y in zip(xs, ys)]
                    marker_char = marker_chars[marker_idx % len(marker_chars)]
                    marker_symbol = marker_symbols[idx % len(marker_symbols)]
                    marker_idx += 1
                    fig.add_trace(
                        go.Scatter(x=xs, y=ys, mode='text',
                                  text=marker_char,
                                  textposition='middle center',
                                  textfont=dict(size=14, color=legend_colors[idx]),
                                  name=f'{legend_val}',
                                  hovertext=hover_text,
                                  hoverinfo='text',
                                  showlegend=True)
                    )
            elif color_by_field:
                grouped = {}
                for point in points:
                    x, y, color_val, legend_val = point if len(point) == 4 else (point[0], point[1], point[2], '')
                    if color_val not in grouped:
                        grouped[color_val] = ([], [])
                    grouped[color_val][0].append(x)
                    grouped[color_val][1].append(y)
                
                for color_val, (xs, ys) in grouped.items():
                    hover_text = [f'<b>Selection:</b> {sel_label}<br><b>{color_by_field}:</b> {color_val}<br><b>{x_field}:</b> {x:.4g}<br><b>{y_field}:</b> {y:.4g}' 
                                  for x, y in zip(xs, ys)]
                    marker_char = marker_chars[marker_idx % len(marker_chars)]
                    marker_idx += 1
                    fig.add_trace(
                        go.Scatter(x=xs, y=ys, mode='text',
                                  text=marker_char,
                                  textposition='middle center',
                                  textfont=dict(size=14, color=color_map.get(color_val, '#4e79a7')),
                                  name=f'{display_label} - {color_by_field}={color_val}',
                                  hovertext=hover_text,
                                  hoverinfo='text')
                    )
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                hover_text = [f'<b>Selection:</b> {sel_label}<br><b>{x_field}:</b> {x:.4g}<br><b>{y_field}:</b> {y:.4g}' 
                              for x, y in zip(xs, ys)]
                marker_char = marker_chars[marker_idx % len(marker_chars)]
                marker_color = get_color_palette_cached(1)[0] if marker_idx == 0 else get_color_palette_cached(marker_idx % 10)[marker_idx % 10]
                marker_idx += 1
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode='text',
                              text=marker_char,
                              textposition='middle center',
                              textfont=dict(size=14, color=marker_color),
                              name=display_label,
                              hovertext=hover_text,
                              hoverinfo='text')
                )
        
        # Use custom titles if provided, otherwise use defaults
        x_label = custom_x_title if custom_x_title else (x_field if x_field != 'tname_custom' else f'tname: {x_custom}')
        y_label = custom_y_title if custom_y_title else (y_field if y_field != 'tname_custom' else f'tname: {y_custom}')
        title = f'XY Plot: {y_label} vs {x_label}'
        if legend_base_field:
            title += f' (legend by {legend_base_field})'
        elif color_by_field:
            title += f' (colored by {color_by_field})'
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode='closest',
            height=1000,
            width=800,
            template='plotly_white'
        )
    
    # Get the Plotly JSON representation
    import json as json_module
    plot_json_str = fig.to_json()
    plot_json = json_module.loads(plot_json_str)  # Parse to object for JSON serialization
    
    # Save plot as HTML
    out_dir = Path(tempfile.mkdtemp(prefix=f'xyplot_{token}_'))
    html_name = f'xyplot_{uuid.uuid4().hex}.html'
    html_path = out_dir / html_name
    
    try:
        fig.write_html(str(html_path), config={'responsive': True, 'displayModeBar': True})
    except Exception as e:
        return jsonify({'error': f'Failed to save plot: {str(e)}'}), 500
    
    # Register in cache
    data.setdefault('xyplot_dirs', []).append(str(out_dir))
    data.setdefault('xyplot_files', []).append(html_name)
    CACHE[token] = data
    
    try:
        html_url = url_for('download_xyplot_image', token=token, name=html_name)
    except Exception:
        html_url = f'/download/xyplot/{token}/{html_name}'
    
    return jsonify({
        'img_url': html_url,
        'plot_json': plot_json,
        'point_count': sum(len(p) for p in data_by_selection.values()),
        'x_field': x_field,
        'y_field': y_field,
        'split_by_selection': split_by_selection,
        'color_by_field': color_by_field,
        'selections_count': len(selections),
        'is_interactive': True,
    })


@app.route('/download/xyplot/<token>/<name>')
def download_xyplot_image(token: str, name: str):
    """Download XY plot image."""
    data = CACHE.get(token)
    if not data or 'xyplot_dirs' not in data:
        flash('Session expired.')
        return redirect(url_for('index'))
    for d in reversed(data.get('xyplot_dirs', [])):
        fp = Path(d) / name
        if fp.exists():
            return send_file(str(fp), as_attachment=False)
    flash('Image not found.')
    return redirect(url_for('index'))



if __name__ == "__main__":
    # For local testing; in production consider a WSGI server
    dbg_env = os.environ.get("FDV_POLL_DEBUG", "1").strip().lower()
    debug = dbg_env not in ("0", "false", "no", "off")
    host = os.environ.get("FDV_POLL_HOST", "0.0.0.0").strip() or "0.0.0.0"
    try:
        port = int(os.environ.get("FDV_POLL_PORT", "5055"))
    except Exception:
        port = 5055
    # Avoid flapping reloader that can cause ephemeral connection refused; keep single process.
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
