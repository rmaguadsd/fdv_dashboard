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
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
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
        best_path: _Path | None = None
        if override_tmp:
            p = _Path(override_tmp)
            try:
                p.mkdir(parents=True, exist_ok=True)
                best_path = p
            except Exception:
                best_path = None
        if best_path is None:
            for pth in [r'D:\\fdv_tmp', r'C:\\fdv_tmp']:
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
CACHE: dict[str, dict] = {}

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
    return group_by_fdv_with_splits(rows, split_vcc=True, split_temp=True)


def group_by_fdv_with_splits(rows: List[Dict[str, str]], *, split_vcc: bool, split_temp: bool, split_plane: bool = True) -> List[Dict[str, str]]:
    """Group rows by fdv and status, with optional split on VCC, TEMP, and plane operation.
    When a dimension isn't split, its key is set to an empty string and the output column will be blank.
    """
    groups: Dict[Tuple[str, str, str, str, str], List[Tuple[float, str, str, str, str, str, str]]] = {}
    # Track invalid FuseIDs per group as (dut, site, fuseid_string)
    ignored_by_group: Dict[Tuple[str, str, str, str, str], List[Tuple[str, str, str]]] = {}
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
        dut = (r.get("dut_id", "") or "").strip()
        pr = (_first_nonempty_str(r, ['pr','PR'], '') or 'XX')
        fuseid = _get_fuseid(r)
        pagetype = (r.get("pagetype", "") or "").strip()
        tm = (r.get("tm", "") or "").strip()
        # Skip PR monitor rows similar to report2
        if (tname or '').strip().upper() == 'PR':
            continue
        # If poll__test present, require it to match fdv_file
        if polltest and polltest != fdv:
            continue
        # Exclude poll_erase_tbers (or any poll__test containing 'tbers') data if fdv_file is not a *tbers
        if polltest and ("tbers" in polltest.lower()) and ("tbers" not in fdv.lower()):
            continue
        # Exclude when tname mentions tbers/erase but fdv_file is not an erase test name
        if tname and ("tbers" in tname.lower() or "erase" in tname.lower()) and ("tbers" not in fdv.lower() and "erase" not in fdv.lower()):
            continue
        v = to_float(r)
        if v is None:
            continue
        # Accumulate into group for this row
        pagemap = _extract_pagemap_from_row(r)
        key = (fdv, vcc, temp, status, plane)
        # Record invalid FUSEID DUT@site info and skip adding this row to stats
        if not _is_valid_fuseid(fuseid):
            site = _extract_site_from_filename(fdv)
            if dut:
                ignored_by_group.setdefault(key, []).append((dut, site, (fuseid or '')))
            continue
        groups.setdefault(key, []).append((v, pr, dut, fuseid, pagetype, tm, pagemap))
    # Load persisted specs (MinSpec/MaxSpec) indexed by composite key
    persisted_specs = _load_specs()
    def _gkey(fdv: str, vcc: str, temp: str, status: str, plane: str) -> str:
        return "||".join([fdv or '', vcc or '', temp or '', (status or '').upper(), (plane or '').upper()])
    out = []
    for key in sorted(groups.keys(), key=lambda kv: (kv[0] or "", kv[1] or "", kv[2] or "", kv[3] or "", kv[4] or "")):
        fdv, vcc, temp, status, plane = key
        items = groups[key]
        vals = [v for (v, _pr, _dut, _fid, _pt, _tm, _pm) in items]
        st = compute_stats(vals)
        specname = _specname_from_fdv(fdv)
        # Aggregate PR values and comments (FUSEID per DUT) for this group
        pr_set: List[str] = []
        seen_pr: set[str] = set()
        dut_fid_pairs: Dict[str, str] = {}
        pagetypes_set = set()
        tm_set = set()
        pm_counts: Dict[str, int] = {}
        valid_ids_set: set[str] = set()
        for (_v, _pr, _dut, _fid, _pt, _tm, _pm) in items:
            if _pr and _pr not in seen_pr:
                seen_pr.add(_pr)
                pr_set.append(_pr)
            if _dut and _fid and _dut not in dut_fid_pairs:
                dut_fid_pairs[_dut] = _fid
            if _fid:
                valid_ids_set.add(_fid)
            if _pt:
                pagetypes_set.add(_pt)
            if _tm:
                tm_set.add(_tm)
            if _pm:
                pm_counts[_pm] = pm_counts.get(_pm, 0) + 1
        # Numeric-aware PR sort with 'XX' last
        def _pr_key(p: str):
            if p == 'XX':
                return (1, float('inf'))
            try:
                return (0, int(p))
            except Exception:
                return (0, p)
        pr_join = ", ".join(sorted(pr_set, key=_pr_key)) if pr_set else ""
        # Build VALID FuseID list as DUT@site:FUSEID
        site_label = _extract_site_from_filename(fdv)
        valid_items = []
        for d in sorted(dut_fid_pairs.keys(), key=lambda s: int(s) if s.isdigit() else 9999):
            fid = dut_fid_pairs[d]
            lbl = f"DUT{d}{('@' + site_label) if site_label else ''}:{fid}"
            valid_items.append(lbl)
        base_comment = ("VALID: " + ", ".join(valid_items)) if valid_items else ""
        # Sort TM numerically when possible, else lexicographically
        def _tm_key(s: str):
            try:
                return (0, float(s))
            except Exception:
                return (1, s)
        tm_comment = ("TM: " + ", ".join(sorted(tm_set, key=_tm_key))) if tm_set else ""
        pt_comment = ("PAGETYPE: " + ", ".join(sorted(pagetypes_set))) if pagetypes_set else ""
        # Append ignored DUT@site info if any
        ignored = ignored_by_group.get(key, [])
        if ignored:
            seen_ig = set()
            ig_parts = []
            for (d, s, fid_str) in ignored:
                fid_show = (fid_str if fid_str else "(missing)")
                lab = f"DUT{d}{('@site' + s) if s else ''}:{fid_show}"
                if lab not in seen_ig:
                    seen_ig.add(lab)
                    ig_parts.append(lab)
            ig_comment = "IGNORED (invalid FUSEID): " + ", ".join(sorted(ig_parts))
        else:
            ig_comment = ""
        comments = " | ".join([c for c in (base_comment, tm_comment, pt_comment, ig_comment) if c])
        # Choose most common non-empty pagemap
        pagemap_best = ''
        if pm_counts:
            pagemap_best = sorted(pm_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        key_id = _gkey(fdv, vcc, temp, status, plane)
        spec = persisted_specs.get(key_id, {}) if isinstance(persisted_specs, dict) else {}
        out.append({
            "fdv_file": fdv,
            "specname": specname,
            "vcc": vcc,
            "temp": temp,
            "pagemap": pagemap_best,
            "status": status,
            "plane_group": plane,
            "pr": pr_join,
            "count": str(len(vals)),
            "valid_fuseid_count": (str(len(valid_ids_set)) if valid_ids_set else ''),
            "MinSpec": (spec.get('MinSpec') or ''),
            "MaxSpec": (spec.get('MaxSpec') or ''),
            "_group_key": key_id,
            "comments": comments,
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


def _filter_poll_row(r: Dict[str, str]) -> Tuple[bool, str, str, str, str, str, float | None]:
    """Apply common FDV POLL filter rules and return tuple:
    (keep, fdv, vcc, temp, status, plane, value)
    """
    try:
        v = float(r.get("data_token2_numeric") or r.get("data_token2") or "")
    except Exception:
        return False, "", "", "", "", "", None
    # Ignore invalid measurement sentinel -999 and oversized values
    try:
        if v == -999.0 or v > POLL_MAX_VALUE:
            return False, "", "", "", "", "", None
    except Exception:
        pass
    fdv = r.get("fdv_file", "") or ""
    vcc = (r.get("vcc", "") or "").strip()
    temp = (r.get("temp", "") or "").strip()
    polltest = (r.get("poll__test", "") or "").strip()
    tname = (r.get("tname", "") or "").strip()
    status = (r.get("status", "") or "").strip().upper()
    plane = (r.get("plane_group", "") or _plane_from_tname_or_default(r) or "").strip().upper()
    # Skip PR monitor rows entirely (not measurement)
    if (tname or '').strip().upper() == 'PR':
        return False, fdv, vcc, temp, status, plane, None
    if polltest and polltest != fdv:
        return False, fdv, vcc, temp, status, plane, None
    if polltest and ("tbers" in polltest.lower()) and ("tbers" not in fdv.lower()):
        return False, fdv, vcc, temp, status, plane, None
    if tname and ("tbers" in tname.lower() or "erase" in tname.lower()) and ("tbers" not in fdv.lower() and "erase" not in fdv.lower()):
        return False, fdv, vcc, temp, status, plane, None
    return True, fdv, vcc, temp, status, plane, v


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
    # Split options (independent): only create separate charts when selected
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    # Group key: (fdv, vcc?, temp?, plane?)
    groups: Dict[Tuple[str, str, str, str], List[Tuple[int, float, str, str, str, str, str | None]]] = defaultdict(list)
    for r in rows:
        keep, fdv, vcc, temp, status, plane, v = _filter_poll_row(r)
        if not keep or v is None:
            continue
        key = (
            fdv,
            (vcc if split_vcc else ""),
            (temp if split_temp else ""),
            (plane if split_plane else ""),
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
        try:
            ln = int(r.get("line_number", "0"))
        except Exception:
            ln = 0
        src_idx = r.get("source_idx") if isinstance(r, dict) else None
        groups[key].append((ln, v, vcc, temp, status, plane, (str(src_idx) if src_idx is not None else None)))
    # Generate plots to a temp dir and collect outliers
    out_dir = Path(tempfile.mkdtemp(prefix=f"hist_{token}_"))
    rendered = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (fdv, vcc_key, temp_key, plane_key), items in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1], kv[0][2], kv[0][3])):
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
        for ln, v, vcc, temp, status, plane, src_idx in items:
            if v < low or v > high:
                # deviation from the nearest bound
                dev = (low - v) if v < low else (v - high)
                if dev < 0:
                    dev = -dev
                candidates.append((dev, ln, v, vcc, temp, status, plane, src_idx))
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
            ),
        } for _dev, ln, v, _vcc, _temp, _status, _plane, src_idx in top]
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

    # Build value arrays per selection
    series: List[tuple[str, List[float]]] = []
    for it in items:
        parts = it.split("||")
        if len(parts) < 5:
            continue
        s_fdv, s_vcc, s_temp, s_status, s_plane = [p or "" for p in parts[:5]]
        vals: List[float] = []
        for r in rows:
            keep, fdv, vcc, temp, status, plane, v = _filter_poll_row(r)
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
                back_url = url_for(
                    "hist_overview",
                    token=token,
                    fdv=r.get('fdv_file',''),
                    vcc=(r.get('vcc','') or '').strip(),
                    temp=(r.get('temp','') or '').strip(),
                    status=(r.get('status','') or '').strip(),
                    plane=(r.get('plane_group','') or '').strip(),
                    split=1,
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
    fdv = vcc = temp = status = plane = ""
    row_src_idx = None
    for r in data["rows"]:
        try:
            if int(r.get("line_number", "-1")) == target:
                fdv = r.get("fdv_file", "")
                vcc = (r.get("vcc", "") or "").strip()
                temp = (r.get("temp", "") or "").strip()
                status = (r.get("status", "") or "").strip()
                plane = (r.get("plane_group", "") or "").strip()
                row_src_idx = r.get("source_idx")
                break
        except Exception:
            continue
    # Preserve split flags if provided in query string
    split_vcc = (request.args.get("split_vcc") or "0").strip() not in ("0", "false", "off")
    split_temp = (request.args.get("split_temp") or "0").strip() not in ("0", "false", "off")
    split_plane = (request.args.get("split_plane") or "0").strip() not in ("0", "false", "off")
    back_url = url_for(
        "hist_overview",
        token=token,
        fdv=fdv,
        vcc=vcc,
        temp=temp,
        status=status,
        plane=plane,
        split_vcc=("1" if split_vcc else "0"),
        split_temp=("1" if split_temp else "0"),
        split_plane=("1" if split_plane else "0"),
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
    base_rows: List[Dict[str, str]] = data.get("rows", [])
    if kind == "vt":
        # Recompute from rows to reflect flags
        rows = group_by_fdv_with_splits(base_rows, split_vcc=split_vcc, split_temp=split_temp, split_plane=split_plane)
        headers = [
            "fdv_file", "specname", "vcc", "temp", "pagemap", "status", "plane_group", "pr", "count",
            "valid_fuseid_count", "MinSpec", "MaxSpec", "min", "max", "mean", "stdev", "median", "comments"
        ]
        filename = f"stats_by_fdv_{'vcc_' if split_vcc else ''}{'temp_' if split_temp else ''}{'plane_' if split_plane else ''}pagemap_status_plane.csv"
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
    stats_vt = group_by_fdv_with_splits(rows, split_vcc=split_vcc, split_temp=split_temp, split_plane=split_plane)
    persist_url = url_for('poll_persist', token=token)
    html = render_template(
        "results.html",
        stats_vt=stats_vt,
        rows_count=len(rows),
        download_vt=url_for("download_csv", token=token, kind="vt", split_vcc=("1" if split_vcc else "0"), split_temp=("1" if split_temp else "0"), split_plane=("1" if split_plane else "0")),
        token=token,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
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
    rows = data.get('rows', [])
    stats_vt = group_by_fdv_with_splits(rows, split_vcc=split_vcc, split_temp=split_temp, split_plane=split_plane)
    html = render_template(
        'results.html',
        stats_vt=stats_vt,
        rows_count=len(rows),
        download_vt=url_for('download_csv', token=token, kind='vt', split_vcc=('1' if split_vcc else '0'), split_temp=('1' if split_temp else '0'), split_plane=('1' if split_plane else '0')),
        token=token,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
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
    stats_vt = group_by_fdv_with_splits(merged, split_vcc=split_vcc, split_temp=split_temp, split_plane=split_plane)
    html = render_template(
        'results.html',
        stats_vt=stats_vt,
        rows_count=len(merged),
        download_vt=url_for('download_csv', token=token, kind='vt', split_vcc=('1' if split_vcc else '0'), split_temp=('1' if split_temp else '0'), split_plane=('1' if split_plane else '0')),
        token=token,
        split_vcc=split_vcc,
        split_temp=split_temp,
        split_plane=split_plane,
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
