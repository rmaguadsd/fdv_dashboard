#!/usr/bin/env python
"""
Restored from backup28 fdv_report2_webapp_run.py
"""
from __future__ import annotations

# Core imports (minimal set needed early; later functions may import lazily as needed)
import os
import re
import json
import time
import math
import threading
import traceback
from typing import Dict, List, Tuple, Any

from flask import (
    Flask, request, render_template, jsonify, send_from_directory,
    abort, redirect, url_for, make_response
)

# Create Flask app early so later route decorators bind correctly.
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global caches / locks (lightweight initialization; detailed population occurs later)
CACHE: Dict[str, Any] = {}
CACHE_LOCK = threading.Lock()

# Configuration toggles (extend as needed)
ENABLE_DEBUG_LOGGING = bool(int(os.environ.get('FDV_DEBUG', '0')))

def _dbg(msg: str):  # tiny helper to avoid scattering if ENABLE_DEBUG_LOGGING off
    if ENABLE_DEBUG_LOGGING:
        try:
            print(f"[FDV-RUN-DBG] {msg}", flush=True)
        except Exception:
            pass

# ---- Early helper definitions (must precede later references) ----

def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


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


def derive_testname(tname: str) -> str:
    return (tname or '').strip()


def _extract_wl_value(r: Dict[str, str]) -> int | None:
    tname = (r.get('tname', '') or '')
    if tname:
        try:
            tn = tname.upper()
            m = re.search(r"(?<![A-Z0-9])WL\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", tn)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    wl_keys = ['wl_canonical','wl','WL','Wl','wordline','WORDLINE','word_line','WORD_LINE','wlidx','wl_index','wladdr','wl_addr','wladdress','wl_address','wordline_idx','wordline_index','wl_index_dec','wordline_dec','wl_dec']
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
    tname = (r.get('tname', '') or '')
    if tname:
        try:
            tn = tname.upper()
            m = re.search(r"(?<![A-Z0-9])(?:PG|PAGE|PHYPAGE)\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", tn)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    pg_keys = ['page_canonical','page','PAGE','Page','page_idx','pageindex','pg','pgidx','page_addr','pageaddr','page_address','pageno','page_no','pagenumber','page_num','pgno','pg_no','pgindex','pg_index','pgaddr','pg_addr','page_address_dec']
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


def _plane_from_tname_or_default(r: Dict[str, str]) -> str:
    tname = (r.get('tname','') or '').strip().upper()
    if tname:
        tokens = [tok for tok in re.split(r"[^A-Z0-9]+", tname) if tok]
        for tok in tokens:
            if tok in ('SP','MP'):
                return tok
    return ''


def _extract_wl_from_tname_only(r: Dict[str, str]) -> int | None:
    """Extract WL strictly from tname tokens (WL_<n> or variants)."""
    tname = (r.get('tname', '') or '')
    if not tname:
        return None
    try:
        tn = tname.upper()
        m = re.search(r"(?<![A-Z0-9])WL\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", tn)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def _extract_page_from_tname_only(r: Dict[str, str]) -> int | None:
    """Extract PAGE strictly from tname tokens (PG|PAGE|PHYPAGE_<n>)."""
    tname = (r.get('tname', '') or '')
    if not tname:
        return None
    try:
        tn = tname.upper()
        m = re.search(r"(?<![A-Z0-9])(?:PG|PAGE|PHYPAGE)\s*[_:\-\s]?\s*([0-9]+)(?![A-Z0-9])", tn)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def _extract_plane_addr_from_tname_only(r: Dict[str, str]) -> str:
    """Extract plane address strictly from tname as 'P<digits>'."""
    tn = (r.get('tname', '') or '')
    if not tn:
        return ''
    try:
        up = tn.upper()
        m = re.search(r"(?<![A-Z0-9])P(\d+)(?![A-Z0-9])", up)
        if m:
            n = int(m.group(1))
            if 0 <= n <= 99:
                return f"P{n}"
    except Exception:
        return ''
    return ''


def _extract_blk_from_tname_only(r: Dict[str, str]) -> int | None:
    """Extract first BLK number strictly from tname tokens like BLK_12 or BLOCK_12_13."""
    tn = (r.get('tname', '') or '')
    if not tn:
        return None
    try:
        up = tn.upper()
        # Find BLK or BLOCK followed by one or more numbers separated by non-alnum/underscores
        m = re.search(r"(?<![A-Z0-9])(BLK|BLOCK)\s*[_:\-\s]?\s*([0-9]+)(?:[_:\-\s][0-9]+)*", up)
        if m:
            return int(m.group(2))
    except Exception:
        return None
    return None


def _extract_pagetype_from_tname_only(r: Dict[str, str]) -> str:
    """Extract PAGETYPE strictly from tname tokens (PGTYPE|PAGETYPE_<token>)."""
    tn = (r.get('tname', '') or '')
    if not tn:
        return ''
    try:
        up = tn.upper()
        m = re.search(r"(?<![A-Z0-9])(?:PGTYPE|PAGETYPE)\s*[_:\-\s]?\s*([A-Z0-9]+)(?![A-Z0-9])", up)
        if m:
            return m.group(1)
    except Exception:
        return ''
    return ''


# --- PASS/FAIL (token-based) support ---------------------------------------------------------
def _classify_pass_fail(r: Dict[str, str]) -> str | None:
    """Return 'PASS', 'FAIL', or None based on row content.

    Priority:
      1. Explicit fields: pf_token / pf / passfail / result / status (case-insensitive)
         Accept exact PASS / FAIL (first FAIL wins if ambiguous).
      2. raw_line heuristic: search for whole-word FAIL or PASS (FAIL has precedence).
      3. tname or testname rarely encode PASS/FAIL; ignored to avoid false positives.

    Note: We intentionally require whole-word boundaries to avoid matching e.g. 'FAILURE'.
    """
    try:
        # Explicit fields
        for k in ('pf_token','pf','passfail','result','status'):
            if k in r and r.get(k) not in (None, ''):
                v = str(r.get(k)).strip().upper()
                if v in ('PASS','FAIL'):
                    return v
        rl = (r.get('raw_line') or '').upper()
        if rl:
            # Prefer FAIL if both appear
            if re.search(r"\bFAIL\b", rl):
                return 'FAIL'
            if re.search(r"\bPASS\b", rl):
                return 'PASS'
    except Exception:
        return None
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
