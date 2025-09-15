"""
FDV OUTPUT log parser and reporting utilities.

Parses entries of the form:
FDV OUTPUT [<fdv_path>/<fdv_filename>.FDV::<tname>,<conditions>]: DUT<n>,<...fields...>

We extract:
- line_number, dut_id
- fdv_file (basename without .FDV)
- tname
- conditions: VCC, TEMP, TM, etc.
- status (C0/E0/80/E1/F0) via explicit status= or inferred from tname
- plane_group (SP/MP/XP) via explicit PLANE= or inferred from tokens/tname; XP when unspecified
- Pagetype, product, blk, page, wl derived from tname
- pr (probe rev) per DUT by scanning FDV OUTPUT PR lines (convert last hex byte to decimal; fallback 'XX')
- fuseid per DUT via FUSEID: lines, mapped by occurrence order

Notes: This module is intentionally similar to process_fdv_poll but tailored to FDV OUTPUT
rather than FDV POLL numeric series. We preserve a consistent API so tooling can be shared.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Set
import statistics as _stats
from datetime import datetime as _dt

PREFIX_DEFAULT = "FDV OUTPUT"
IGNORE_VALUE_DEFAULT = None  # not used but kept for API compat


def short_fdv_name(p: str) -> str:
    if not p:
        return ""
    name = os.path.basename(p.replace("\\", "/"))
    if name.lower().endswith(".fdv"):
        name = name[:-4]
    return name


def _parse_conditions(cond: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in cond.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().upper()] = v.strip()
    return out


def derive_testname(tname: str) -> str:
    """Derive the base/root testname from tname using guide_to_fdvlog rules.

    Heuristic: take the prefix of tname before any "parameter" tokens appear, where
    parameters include (case-insensitive):
      - BLK_<n>, (PG|PAGE|PHYPAGE)_<n>, WL_<n>, SB_<n>, BL_<n>
      - PAGETYPE/PGTYPE/PAGE[_-]?TYPE/PT[_-]?<LP|UP|XP|TP|SSLC|DSLC|SLC|QLC|TLC>
    - product/pagemap tokens: MLC, QLC, TLC, SSLC, DSLC
      - plane: SP, MP
      - deck: UD, LD, MD
      - status: C0, E0, 80, E1, F0
      - SHMOO, STEP

    Example: EIMPRO_ECC_BLK_778_PG_44_PGTYPE_LP_WL_2 -> EIMPRO_ECC
    """
    if not tname:
        return ""
    s = tname.strip().strip("<>")
    if not s:
        return ""

    # Build a set of regexes to find the earliest parameter marker in the raw string
    # Use custom boundaries that consider underscores as separators as well
    SEP_BEFORE = r"(?:(?<=^)|(?<=[^A-Za-z0-9]))"
    SEP_AFTER = r"(?:(?=$)|(?=[^A-Za-z0-9]))"
    param_patterns = [
        SEP_BEFORE + r"BLK[_:\-]?\d+" + SEP_AFTER,
        SEP_BEFORE + r"(?:PG|PAGE|PHYPAGE)[_:\-]?\d+" + SEP_AFTER,
        SEP_BEFORE + r"WL[_:\-]?\d+" + SEP_AFTER,
        SEP_BEFORE + r"SB[_:\-]?\d+" + SEP_AFTER,
        SEP_BEFORE + r"BL[_:\-]?\d+" + SEP_AFTER,
        # Note: PT values include LP/UP/XP/TP/SSLC/DSLC; 'SLC' is not used per guide tokens
        SEP_BEFORE + r"(?:PAGETYPE|PGTYPE|PAGE[_:\-]?TYPE|PT)[_:\-]?(?:LP|UP|XP|TP|SSLC|DSLC|QLC|TLC|MLC)" + SEP_AFTER,
        SEP_BEFORE + r"(?:MLC|QLC|TLC|SSLC|DSLC)" + SEP_AFTER,
        SEP_BEFORE + r"(?:SP|MP)" + SEP_AFTER,
        SEP_BEFORE + r"(?:UD|LD|MD)" + SEP_AFTER,
        SEP_BEFORE + r"(?:C0|E0|80|E1|F0)" + SEP_AFTER,
        SEP_BEFORE + r"SHMOO" + SEP_AFTER,
        SEP_BEFORE + r"STEP" + SEP_AFTER,
    ]

    earliest = None
    for pat in param_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            idx = m.start()
            if earliest is None or idx < earliest:
                earliest = idx

    base_str = s if earliest is None else s[:earliest]
    # Trim trailing separators and non-alnum chars
    base_str = re.sub(r"[^A-Za-z0-9]+$", "", base_str)

    # If still empty, fall back to token-based heuristic: take leading non-numeric tokens until a stop token
    if not base_str:
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9]+", s) if tok]
        stop_words = {
            "QLC", "TLC", "SSLC", "DSLC",
            "SP", "MP", "UD", "LD", "MD", "SHMOO", "STEP", "C0", "E0", "80", "E1", "F0",
        }
        base: list[str] = []
        for tok in tokens:
            up = tok.upper()
            if up.isdigit() or up in stop_words or re.match(r"^(?:BLK|WL|SB|BL)\d+$", up) or re.match(r"^(?:PG|PAGE|PHYPAGE)\d+$", up) or up.startswith("PT_"):
                break
            # Stop if token is a param prefix
            if up in {"BLK", "PG", "PAGE", "PHYPAGE", "WL", "SB", "BL", "PAGETYPE", "PGTYPE", "PAGE_TYPE", "DECK"}:
                break
            base.append(tok)
            if len(base) >= 5:
                break
        base_str = "_".join(base)

    # Final cleanup: collapse multiple underscores and strip
    base_str = re.sub(r"_+", "_", base_str).strip("_")
    return base_str or s


def _is_valid_fuseid(fid: str) -> bool:
    """Validate FUSEID format: K<6 digits>_<int>_<int>_<int>
    - The first integer must be > 0
    - The last two integers may be negative
    """
    if not fid:
        return False
    m = re.match(r"^K\d{6}_([+-]?\d+)_([+-]?\d+)_([+-]?\d+)$", fid.strip())
    if not m:
        return False
    try:
        first = int(m.group(1))
    except Exception:
        return False
    return first > 0


def _prescan_invalid_duts(in_path: Path) -> Tuple[Set[str], List[str]]:
    """First pass: collect FUSEIDs by occurrence order and mark invalid DUT indices.

    Returns (invalid_duts, fuseid_order), where invalid_duts is a set like {"1","2"}
    and fuseid_order maps DUT index (1-based) to the FUSEID token captured.
    """
    invalid: Set[str] = set()
    fuseid_order: List[str] = []
    # Capture broader FUSEID token (trim after whitespace/pipe/comma/semicolon)
    _fuseid_line = re.compile(r"\bFUSEID:\s*([^\r\n]+)", re.IGNORECASE)
    try:
        with open(in_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _fuseid_line.search(line)
                if not m:
                    continue
                raw = (m.group(1) or "").strip()
                token = re.split(r"[\s,;|]", raw, maxsplit=1)[0].strip().strip("[]()")
                if token:
                    fuseid_order.append(token)
    except Exception:
        return invalid, fuseid_order
    # Map by occurrence order: index 0 -> DUT1, index 1 -> DUT2, etc.
    for i, fid in enumerate(fuseid_order):
        dut = str(i + 1)
        if not _is_valid_fuseid(fid):
            invalid.add(dut)
    return invalid, fuseid_order


def process_file(
    in_path: Path,
    starts_with: str = PREFIX_DEFAULT,
    ignore_value=IGNORE_VALUE_DEFAULT,
    max_lines: Optional[int] = None,
    max_rows: Optional[int] = None,
    *,
    progress: bool | int = False,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> Tuple[List[Dict[str, str]], List[str], List[Dict[str, str]]]:
    prefix_re = re.compile(r"^\s*" + re.escape(starts_with) + r"\b")
    rows: List[Dict[str, str]] = []
    kept_lines: List[str] = []
    last_fdv: str = ""
    fdv_idx = -1
    markers: List[Dict[str, str]] = []

    # Inline detection of Test Start/End markers (avoid extra file pass in callers)
    # Patterns allow optional list name and AM/PM; tolerate single-digit components
    _re_start_date_name = re.compile(r"Test\s+Start\s+Date\s*\(([^)]+)\)\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", re.IGNORECASE)
    _re_start_date_noname = re.compile(r"Test\s+Start\s+Date\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", re.IGNORECASE)
    _re_start_time = re.compile(r"Test\s+Start\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", re.IGNORECASE)
    _re_start_both = re.compile(r"Test\s+Start\s+Date\s*(?:\(([^)]+)\))?\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})\s*Test\s+Start\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", re.IGNORECASE)
    _re_end_date_name = re.compile(r"Test\s+End\s+Date\s*\(([^)]+)\)\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", re.IGNORECASE)
    _re_end_date_noname = re.compile(r"Test\s+End\s+Date\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})", re.IGNORECASE)
    _re_end_time = re.compile(r"Test\s+End\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", re.IGNORECASE)
    _re_end_both = re.compile(r"Test\s+End\s+Date\s*(?:\(([^)]+)\))?\s*:\s*(\d{4})_(\d{1,2})_(\d{1,2})\s*Test\s+End\s+Time\s*:?\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})(?:\s*(AM|PM))?", re.IGNORECASE)
    _start_best = {'dt': None, 'raw': ''}
    _end_best = {'dt': None, 'raw': ''}
    _list_name_marker = ''
    _s_date = _s_time = _s_ampm = None
    _e_date = _e_time = _e_ampm = None
    def _parse_dt(_parts, _times, _ampm):
        if not _parts or not _times:
            return None
        (y, mo, da) = _parts
        try:
            hh, mm, ss = [int(x) for x in str(_times).split(':')[:3]]
        except Exception:
            return None
        if hh < 0: hh = 0
        if mm < 0: mm = 0
        if ss < 0: ss = 0
        if mm > 59: mm = 59
        if ss > 59: ss = 59
        ap = (str(_ampm or '')).upper()
        if ap in ('AM','PM'):
            if ap == 'PM' and hh < 12:
                hh += 12
            if ap == 'AM' and hh == 12:
                hh = 0
        try:
            return _dt(int(y), int(mo), int(da), hh, mm, ss)
        except Exception:
            return None
    def _fmt_raw(_parts, _times, _ampm):
        try:
            (y, mo, da) = _parts
            ap = f" {_ampm}" if _ampm else ''
            return f"{int(y):04d}_{int(mo):02d}_{int(da):02d} {_times}{ap}"
        except Exception:
            return ''

    fdv_re = re.compile(r"([A-Za-z0-9_./\\:-]+\.fdv)\b", re.IGNORECASE)
    tname_re = re.compile(r"\.fdv::\s*([^,\]]+)", re.IGNORECASE)
    conds_re = re.compile(r"\.fdv::[^\]]*?,([^\]]+)\]", re.IGNORECASE)
    dut_re = re.compile(r"\bDUT(\d+)\b", re.IGNORECASE)
    # After the ']:' delimiter, fields are CSV-like: DUTn,PASS/FAIL,bytes,fail_bytes,byber,fail_bits,rber,...
    tail_split_re = re.compile(r"\]:\s*(.*)$")

    # PR and FUSEID accumulation
    pr_line_re = re.compile(r"::\s*PR\b", re.IGNORECASE)
    # Capture FUSEID value more broadly (rest of line), then trim common delimiters.
    # This avoids truncation when IDs contain hyphens/underscores or other non-space characters.
    fuseid_re = re.compile(r"\bFUSEID:\s*([^\r\n]+)", re.IGNORECASE)
    pr_by_dut: Dict[str, str] = {}
    fuseid_order: List[str] = []
    # Pre-scan to determine which DUTs should be ignored due to invalid FUSEID and capture FUSEID tokens by order
    invalid_duts, prescan_fuseids = _prescan_invalid_duts(in_path)
    try:
        # Emit a prescan marker so callers can surface ignored DUTs with FUSEIDs in UIs
        markers.append({
            'type': 'fuseid_prescan',
            'source_file': str(in_path),
            'dut_fuseids': [{'dut': str(i+1), 'fuseid': prescan_fuseids[i]} for i in range(len(prescan_fuseids))],
            'invalid_duts': sorted(list(invalid_duts)),
        })
    except Exception:
        pass

    rows_count = 0  # optional cap; None means unlimited
    file_size = 0
    try:
        file_size = os.path.getsize(in_path)
    except Exception:
        file_size = 0
    env_prog = os.environ.get("FDV_PROGRESS", "0").strip().lower()
    enable_progress = bool(progress) or env_prog in ("1", "true", "yes", "on") or progress_cb is not None
    lines_interval = int(progress) if isinstance(progress, int) and progress else 100000
    last_report_lines = 0
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            # Speed-up: only consider lines beginning with one of the allowed prefixes
            # after trimming leading whitespace. This retains FUSEID lines even if they
            # are echoed as "ECHO: FUSEID:".
            _ls = line.lstrip()
            _up = _ls.upper()
            # Always emit progress based on byte position to avoid "stuck" UI on long stretches
            # of non-matching lines.
            if enable_progress and (lineno - last_report_lines >= lines_interval or lineno == 1):
                pct = 0.0
                if file_size > 0:
                    try:
                        pct = (float(f.tell()) / float(file_size)) * 100.0
                    except Exception:
                        pct = 0.0
                if progress_cb:
                    progress_cb(lineno, pct)
                else:
                    print(f"Parsing… {lineno} lines (~{pct:.1f}%)")
                last_report_lines = lineno
            # Allow FDV OUTPUT, FDV POLL, FUSEID echoes, and test time headers for start/end date
            if not (
                _up.startswith("FDV OUTPUT")
                or _up.startswith("FDV POLL")
                or _up.startswith("ECHO: FUSEID:")
                or _up.startswith("TEST START DATE")
                or _up.startswith("TEST END DATE")
            ):
                continue
            # Capture combined start/end first (fast path)
            try:
                m_sb = _re_start_both.search(line)
                if m_sb:
                    if m_sb.group(1):
                        _list_name_marker = (m_sb.group(1) or '').strip()
                    y, mo, da = m_sb.group(2), m_sb.group(3), m_sb.group(4)
                    t, ap = m_sb.group(5), (m_sb.group(6) or '')
                    dt = _parse_dt((y, mo, da), t, ap)
                    if dt is not None and (_start_best['dt'] is None or dt < _start_best['dt']):
                        _start_best = {'dt': dt, 'raw': _fmt_raw((int(y), int(mo), int(da)), t, ap)}
                m_eb = _re_end_both.search(line)
                if m_eb:
                    if not _list_name_marker and m_eb.group(1):
                        _list_name_marker = (m_eb.group(1) or '').strip()
                    y, mo, da = m_eb.group(2), m_eb.group(3), m_eb.group(4)
                    t, ap = m_eb.group(5), (m_eb.group(6) or '')
                    dt = _parse_dt((y, mo, da), t, ap)
                    if dt is not None and (_end_best['dt'] is None or dt > _end_best['dt']):
                        _end_best = {'dt': dt, 'raw': _fmt_raw((int(y), int(mo), int(da)), t, ap)}
                # Separate start date/time
                m_sdn = _re_start_date_name.search(line) or _re_start_date_noname.search(line)
                if m_sdn:
                    try:
                        # If named variant, list name may be present in group 1
                        if m_sdn.re is _re_start_date_name and m_sdn.lastindex and m_sdn.lastindex >= 4:
                            _list_name_marker = (m_sdn.group(1) or '').strip()
                            _s_date = (int(m_sdn.group(2)), int(m_sdn.group(3)), int(m_sdn.group(4)))
                        else:
                            _s_date = (int(m_sdn.group(1)), int(m_sdn.group(2)), int(m_sdn.group(3)))
                    except Exception:
                        _s_date = None
                m_st = _re_start_time.search(line)
                if m_st:
                    _s_time = m_st.group(1).strip()
                    _s_ampm = (m_st.group(2) or '').upper()
                if _s_date and _s_time:
                    dt = _parse_dt(_s_date, _s_time, _s_ampm)
                    if dt is not None and (_start_best['dt'] is None or dt < _start_best['dt']):
                        _start_best = {'dt': dt, 'raw': _fmt_raw(_s_date, _s_time, _s_ampm)}
                    _s_date = _s_time = _s_ampm = None
                # Separate end date/time
                m_edn = _re_end_date_name.search(line) or _re_end_date_noname.search(line)
                if m_edn:
                    try:
                        if m_edn.re is _re_end_date_name and m_edn.lastindex and m_edn.lastindex >= 4:
                            if not _list_name_marker:
                                _list_name_marker = (m_edn.group(1) or '').strip()
                            _e_date = (int(m_edn.group(2)), int(m_edn.group(3)), int(m_edn.group(4)))
                        else:
                            _e_date = (int(m_edn.group(1)), int(m_edn.group(2)), int(m_edn.group(3)))
                    except Exception:
                        _e_date = None
                m_et = _re_end_time.search(line)
                if m_et:
                    _e_time = m_et.group(1).strip()
                    _e_ampm = (m_et.group(2) or '').upper()
                if _e_date and _e_time:
                    dt = _parse_dt(_e_date, _e_time, _e_ampm)
                    if dt is not None and (_end_best['dt'] is None or dt > _end_best['dt']):
                        _end_best = {'dt': dt, 'raw': _fmt_raw(_e_date, _e_time, _e_ampm)}
                    _e_date = _e_time = _e_ampm = None
            except Exception:
                # Ignore time marker parsing issues
                pass
            # (progress already updated above to reflect file scan advancement)
            if max_lines is not None and lineno > max_lines:
                break
            # Track latest FDV filename and marker
            mfdv = fdv_re.search(line)
            if mfdv:
                last_fdv = mfdv.group(1)
                fdv_idx += 1
                markers.append({
                    "fdv_idx": str(fdv_idx),
                    "fdv_file": short_fdv_name(last_fdv),
                    "marker_line": str(lineno),
                })

            # Capture FUSEID by DUT order
            mfid = fuseid_re.search(line)
            if mfid:
                _fid_raw = mfid.group(1).strip()
                # If extra tokens trail the ID, keep the first segment before whitespace or pipe/comma/semicolon
                _fid_token = re.split(r"[\s,;|]", _fid_raw, maxsplit=1)[0].strip()
                # Trim common surrounding brackets if present
                _fid_token = _fid_token.strip("[]()")
                fuseid_order.append(_fid_token)

            # Capture PR per DUT
            if pr_line_re.search(line):
                mdut = dut_re.search(line)
                if mdut:
                    dnum = mdut.group(1)
                    mhex = re.search(r":\s*([0-9A-Fa-f]{2})[^0-9A-Fa-f]*$", line)
                    pr_by_dut[dnum] = str(int(mhex.group(1), 16)) if mhex else "XX"

            # Only FDV OUTPUT lines proceed to detailed FDV OUTPUT parsing
            if not prefix_re.search(line):
                continue

            # Requirement: skip any FDV OUTPUT line that contains the word MONITOR anywhere
            # This is broader than checking only the PASS/FAIL field and ensures such rows are ignored upstream.
            try:
                if re.search(r"\bMONITOR\b", _up):
                    continue
            except Exception:
                # If regex/search fails for any reason, fall through to normal parsing
                pass

            # Extract tname and conditions
            mtname = tname_re.search(line)
            tname = (mtname.group(1).strip() if mtname else "")
            mconds = conds_re.search(line)
            conds = _parse_conditions(mconds.group(1)) if mconds else {}

            # Extract DUT id and parse tail fields
            mdut = dut_re.search(line)
            if not mdut:
                continue
            dutnum = mdut.group(1)
            # Skip all rows for DUTs marked invalid per FUSEID prescan
            if dutnum in invalid_duts:
                continue
            mtail = tail_split_re.search(line)
            tail = mtail.group(1).strip() if mtail else ""
            # Split by commas but tolerate trailing commas
            tail_fields = [p.strip() for p in tail.split(",") if p.strip() != ""]
            pass_fail = tail_fields[1] if len(tail_fields) >= 2 else ""
            bytes_total = tail_fields[2] if len(tail_fields) >= 3 else ""
            fail_bytes = tail_fields[3] if len(tail_fields) >= 4 else ""
            byber = tail_fields[4] if len(tail_fields) >= 5 else ""
            fail_bits = tail_fields[5] if len(tail_fields) >= 6 else ""
            rber = tail_fields[6] if len(tail_fields) >= 7 else ""

            # Parse useful conditions
            vcc = conds.get("VCC", "")
            temp = conds.get("TEMP", "")
            tm = conds.get("TM", "")

            # Ignore any FDV OUTPUT rows marked as MONITOR
            try:
                if (pass_fail or "").strip().upper() == "MONITOR":
                    continue
            except Exception:
                # If pass_fail is unexpected, fall through
                pass

            # Status via token or inferred from tname
            status = ""
            # scan tokens in tname first
            for tok in re.split(r"[^A-Za-z0-9]+", tname):
                tu = tok.upper()
                if tu in ("C0", "E0", "80", "E1", "F0"):
                    status = tu
                    break

            # Plane via conditions or tname tokens; normalize to SP/MP/XP
            plane_group = conds.get("PLANE", "").upper()
            if not plane_group:
                for tok in re.split(r"[^A-Za-z0-9]+", tname):
                    tu = tok.upper()
                    if tu in ("SP", "MP"):
                        plane_group = tu
                        break
            # Default or sanitize
            if not plane_group or plane_group not in ("SP", "MP"):
                plane_group = "XP"

            # product/pagemap, pagetype, blk, page, wl
            # Per guide: pagemap is determined if tname contains any of MLC, QLC, TLC, SSLC, DSLC
            # Extraction priority: tname tokens -> PT when SSLC/DSLC -> fdv filename -> fdv path
            product = ""
            _tup = tname.upper()
            for cand in ("MLC", "QLC", "TLC", "SSLC", "DSLC"):
                # Respect token/separator boundaries to avoid matching substrings of larger words
                if re.search(rf"(?<![A-Z0-9]){cand}(?![A-Z0-9])", _tup, re.IGNORECASE):
                    product = cand
                    break
            pagetype = ""
            mpt = re.search(r"(?:PAGETYPE|PGTYPE|PAGE[_:\-]?TYPE|PT)[_:\-]?([A-Za-z0-9\-]+)", tname, re.IGNORECASE)
            if mpt:
                pagetype = re.split(r"[^A-Za-z0-9]+", mpt.group(1))[0].upper()
                # If pagemap not yet set and PT suggests SSLC/DSLC, infer pagemap accordingly
                if not product and pagetype in ("SSLC", "DSLC"):
                    product = pagetype
            # If still not found, try from fdv_file then from full path captured earlier
            if not product:
                fdv_up = short_fdv_name(last_fdv).upper()
                for cand in ("MLC", "QLC", "TLC", "SSLC", "DSLC"):
                    if re.search(rf"(?<![A-Z0-9]){cand}(?![A-Z0-9])", fdv_up, re.IGNORECASE):
                        product = cand
                        break
            if not product and last_fdv:
                path_up = last_fdv.upper()
                for cand in ("MLC", "QLC", "TLC", "SSLC", "DSLC"):
                    if re.search(rf"(?<![A-Z0-9]){cand}(?![A-Z0-9])", path_up, re.IGNORECASE):
                        product = cand
                        break
            mblk = re.search(r"(?<![A-Za-z0-9])BLK\s*[_:\-\s]?\s*([0-9]+)(?![A-Za-z0-9])", tname, re.IGNORECASE)
            blk = mblk.group(1) if mblk else ""
            mpg = re.search(r"(?<![A-Za-z0-9])(?:PG|PAGE|PHYPAGE)\s*[_:\-\s]?\s*([0-9]+)(?![A-Za-z0-9])", tname, re.IGNORECASE)
            page = mpg.group(1) if mpg else ""
            mwl = re.search(r"(?<![A-Za-z0-9])WL\s*[_:\-\s]?\s*([0-9]+)(?![A-Za-z0-9])", tname, re.IGNORECASE)
            wl = mwl.group(1) if mwl else ""
            # PHYPAGE: 13 LSB of page address when page is available (per guide)
            phypage = ""
            try:
                if page:
                    ph = int(page) & 0x1FFF
                    phypage = str(ph)
            except Exception:
                phypage = ""

            # Operation keyword from tname
            operation = ""
            tl = tname.lower()
            for op in ("erase", "program", "read", "eimpro", "tbers", "tprog", "tcbsy"):
                if op in tl:
                    operation = op.upper()
                    break

            # Ensure tname has a sensible value; if missing, fallback to fdv base name
            if not tname:
                tname = short_fdv_name(last_fdv)
            # Derive base testname
            testname = derive_testname(tname)

            # Map fuseid by DUT order: index 0 -> DUT1, etc.
            _n = len(fuseid_order := fuseid_order)
            fuseids_by_dut = {str(i+1): fuseid_order[i] for i in range(_n)}
            fuseid_val = fuseids_by_dut.get(dutnum, f"DUT{dutnum}_9999999_999_99_99")
            pr_val = pr_by_dut.get(dutnum, "XX")

            rows.append({
                "line_number": str(lineno),
                "dut_id": dutnum,
                "fdv_file": short_fdv_name(last_fdv),
                "tname": tname,
                "testname": testname,
                "operation": operation,
                "product": product,
                # also expose as 'pagemap' for UI consumption
                "pagemap": product,
                "pagetype": pagetype,
                "blk": blk,
                "page": page,
                "wl": wl,
                "phypage": phypage,
                "vcc": vcc,
                "temp": temp,
                "tm": tm,
                "status": status,
                "plane_group": plane_group,
                "fdv_idx": str(fdv_idx),
                "fuseid": fuseid_val,
                "pr": pr_val,
                "pass_fail": pass_fail,
                "bytes_total": bytes_total,
                "fail_bytes": fail_bytes,
                "byber": byber,
                "fail_bits": fail_bits,
                "rber": rber,
                "raw_line": line.rstrip("\r\n"),
            })
            kept_lines.append(line.rstrip("\r\n"))
            rows_count += 1
            if max_rows is not None and rows_count >= max_rows:
                break

    # Finalize test time markers (if any)
    try:
        start_iso = _start_best['dt'].isoformat() if _start_best['dt'] is not None else ''
        end_iso = _end_best['dt'].isoformat() if _end_best['dt'] is not None else ''
        if start_iso or end_iso or _list_name_marker:
            markers.append({
                'type': 'test_time',
                'start_raw': _start_best['raw'],
                'end_raw': _end_best['raw'],
                'start_iso': start_iso,
                'end_iso': end_iso,
                'list_name': _list_name_marker,
            })
    except Exception:
        pass
    # ------------------------------------------------------------------
    # NEW: propagate test time info onto every FDV OUTPUT row (for downstream aggregation)
    # This mirrors older working behavior (see backup36 reference) so that
    # stats_by_fdv_with_splits can always see start/end without needing markers.
    try:  # pragma: no cover (integration path)
        _tm_marker = None
        for _m in markers:
            if isinstance(_m, dict) and _m.get('type') == 'test_time':
                _tm_marker = _m
                break
        if _tm_marker:
            _s_raw = (_tm_marker.get('start_raw') or '').strip()
            _e_raw = (_tm_marker.get('end_raw') or '').strip()
            _ln_marker = (_tm_marker.get('list_name') or '').strip()
            # Duration (seconds) if both ISO timestamps available
            _dur_secs = ''
            try:
                _s_iso = (_tm_marker.get('start_iso') or '').strip()
                _e_iso = (_tm_marker.get('end_iso') or '').strip()
                if _s_iso and _e_iso:
                    from datetime import datetime as _dt_calc
                    _ds = _dt_calc.fromisoformat(_s_iso)
                    _de = _dt_calc.fromisoformat(_e_iso)
                    _dsec = int((_de - _ds).total_seconds())
                    if _dsec < 0:
                        _dsec = 0
                    _dur_secs = str(_dsec)
            except Exception:
                _dur_secs = ''
            # Clean list name similar to webapp logic (strip numeric prefix, tb_set_utility_)
            if _ln_marker:
                try:
                    import re as _re_ln
                    _ln_marker = _re_ln.sub(r"^\d+_", '', _ln_marker)
                    _ln_marker = _re_ln.sub(r"(?i)^tb_set_utility_", '', _ln_marker)
                except Exception:
                    pass
            # Determine fdvtest from last_fdv (base filename without extension)
            try:
                import os as _os_path
                _fdvtest = short_fdv_name(last_fdv)
                if _fdvtest.lower().endswith('.fdv'):
                    _fdvtest = _fdvtest[:-4]
            except Exception:
                _fdvtest = ''
            _label = ''
            if _ln_marker or _fdvtest or _dur_secs:
                _label = f"{_ln_marker}::{_fdvtest} = {_dur_secs}".strip()
            # Attach to each row if not already present
            for _r in rows:
                if _s_raw and not _r.get('test_start'):
                    _r['test_start'] = _s_raw
                if _e_raw and not _r.get('test_end'):
                    _r['test_end'] = _e_raw
                if _dur_secs and not _r.get('testtime_seconds'):
                    _r['testtime_seconds'] = _dur_secs
                if _label and not _r.get('testtime_label'):
                    _r['testtime_label'] = _label
    except Exception:
        pass
    return rows, kept_lines, markers


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "line_number", "dut_id",
        "fdv_file", "tname", "operation", "product", "pagemap", "pagetype", "blk", "page", "wl", "phypage",
        "status", "plane_group", "vcc", "temp", "tm", "fdv_idx", "fuseid", "pr",
        "pass_fail", "bytes_total", "fail_bytes", "byber", "fail_bits", "rber",
        "raw_line",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_text(lines: List[str], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fo:
        for s in lines:
            fo.write(s + "\n")


def write_stats_by_fdv_occurrence(rows: List[Dict[str, str]], markers: List[Dict[str, str]], out_csv: Path) -> None:
    # For FDV OUTPUT we can still summarize counts per occurrence if needed
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Count rows per fdv_idx
    from collections import defaultdict
    counts: Dict[int, int] = defaultdict(int)
    for r in rows:
        try:
            idx = int(r.get("fdv_idx", "-1"))
        except Exception:
            continue
        if idx >= 0:
            counts[idx] += 1
    cols = ["occurrence", "marker_line", "fdv_file", "rows"]
    # Build index -> meta mapping
    idx_meta: Dict[int, Dict[str, str]] = {}
    for m in markers:
        try:
            idx_meta[int(m.get("fdv_idx", "-1"))] = m
        except Exception:
            pass
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for idx in sorted(counts.keys()):
            meta = idx_meta.get(idx, {})
            w.writerow({
                "occurrence": idx,
                "marker_line": meta.get("marker_line", ""),
                "fdv_file": meta.get("fdv_file", ""),
                "rows": counts[idx],
            })


def write_stats_by_fdv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    # Count rows per fdv_file as a proxy summary
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.get("fdv_file", "") or ""] += 1
    cols = ["fdv_file", "rows"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for fdv in sorted(counts.keys(), key=lambda s: (s or "")):
            w.writerow({"fdv_file": fdv, "rows": counts[fdv]})


def write_stats_by_fdv_vcc_temp(rows: List[Dict[str, str]], out_csv: Path) -> None:
    # Count rows per (fdv, vcc, temp)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    groups: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in rows:
        key = (
            r.get("fdv_file", "") or "",
            (r.get("vcc", "") or "").strip(),
            (r.get("temp", "") or "").strip(),
        )
        groups[key] += 1
    cols = ["fdv_file", "vcc", "temp", "rows"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for (fdv, vcc, temp), cnt in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "", kv[0][2] or "")):
            w.writerow({
                "fdv_file": fdv,
                "vcc": vcc,
                "temp": temp,
                "rows": cnt,
            })


# --- RBER aggregation helpers ---

def _to_float_safe(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def compute_stats(values: List[float]) -> Dict[str, str]:
    if not values:
        return {k: "" for k in ("count", "min", "max", "mean", "stdev", "median")}
    vals = sorted(values)
    n = len(vals)
    _min = vals[0]
    _max = vals[-1]
    _mean = float(sum(vals)) / n
    _stdev = float(_stats.stdev(vals)) if n >= 2 else 0.0
    _median = float(_stats.median(vals))
    return {
        "count": str(n),
        "min": f"{_min:.6g}",
        "max": f"{_max:.6g}",
        "mean": f"{_mean:.6g}",
        "stdev": f"{_stdev:.6g}",
        "median": f"{_median:.6g}",
    }


def rber_stats_by_fdv(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Group FDV OUTPUT rows by fdv_file and compute stats on RBER."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        # Skip PR monitor rows
        if (r.get("tname", "") or "").strip().upper() == "PR":
            continue
        fdv = r.get("fdv_file", "") or ""
        # Exclude *poweron* fdv tests
        if "poweron" in fdv.lower():
            continue
        rv = _to_float_safe(r.get("rber"))
        if rv is None:
            continue
        groups.setdefault(fdv, []).append(rv)
    out: List[Dict[str, str]] = []
    for fdv in sorted(groups.keys(), key=lambda s: (s or "")):
        st = compute_stats(groups[fdv])
        out.append({"fdv_file": fdv, **st})
    return out


def rber_stats_from_dir(dir_path: Path) -> List[Dict[str, str]]:
    """Scan all files directly under dir_path, aggregate FDV OUTPUT rows and compute RBER stats per fdv_file (with PR)."""
    all_rows = read_dir_rows(dir_path)
    return rber_stats_by_fdv_with_pr(all_rows)


def write_rber_stats_csv(rows_stats: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["fdv_file", "pr", "count", "min", "max", "mean", "stdev", "median"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        import csv as _csv
        w = _csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for r in rows_stats:
            w.writerow({k: r.get(k, "") for k in cols})


def read_dir_rows(dir_path: Path) -> List[Dict[str, str]]:
    """Read all files directly under dir_path and return concatenated FDV OUTPUT rows."""
    all_rows: List[Dict[str, str]] = []
    for p in Path(dir_path).iterdir():
        if not p.is_file():
            continue
        try:
            rows, _kept, _markers = process_file(p, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT)
            all_rows.extend(rows)
        except Exception:
            continue
    return all_rows


def rber_stats_by_fdv_with_pr(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Group by (fdv_file, pr) and compute RBER stats. Each output row has a single PR value."""
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        # Skip PR monitor rows
        if (r.get("tname", "") or "").strip().upper() == "PR":
            continue
        fdv = r.get("fdv_file", "") or ""
        # Exclude *poweron* fdv tests
        if "poweron" in fdv.lower():
            continue
        prv = (r.get("pr", "") or "").strip() or "XX"
        rv = _to_float_safe(r.get("rber"))
        if rv is None:
            continue
        groups[(fdv, prv)].append(rv)
    out: List[Dict[str, str]] = []
    for (fdv, prv) in sorted(groups.keys(), key=lambda kv: (kv[0] or "", kv[1] == "XX", kv[1])):
        st = compute_stats(groups[(fdv, prv)])
        out.append({"fdv_file": fdv, "pr": prv, **st})
    return out


def rber_stats_by_tname(rows: List[Dict[str, str]], fdv_file: str) -> List[Dict[str, str]]:
    """Within a given fdv_file, group by (testname, PR, FID) and compute RBER stats.

    - testname is derived per guide (operation-centric prefix)
    - PR is taken from per-DUT parsed value (fallback 'XX')
    - FID is the per-DUT fuseid (fallback placeholder)
    - PR monitor rows (tname == 'PR') are excluded
    """
    # Exclude *poweron* fdv tests entirely
    if fdv_file and "poweron" in fdv_file.lower():
        return []
    cand = [r for r in rows if (r.get("fdv_file", "") or "") == (fdv_file or "")]
    from collections import defaultdict
    vals_by_key: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in cand:
        if (r.get("tname", "") or "").strip().upper() == "PR":
            continue
        tn = (r.get("testname", "") or "").strip()
        if not tn:
            raw_tn = (r.get("tname", "") or "").strip()
            tn = derive_testname(raw_tn) if raw_tn else ""
        if not tn:
            tn = "UNKNOWN"
        # Exclude any test whose name (derived or raw) contains 'poweron'
        if "poweron" in tn.lower() or "poweron" in (r.get("tname", "").lower() if r.get("tname") else ""):
            continue
        pr = (r.get("pr", "") or "").strip() or "XX"
        fid = (r.get("fuseid", "") or "").strip() or ""
        rv = _to_float_safe(r.get("rber"))
        if rv is None:
            continue
        vals_by_key[(tn, pr, fid)].append(rv)
    def _pr_key(p: str):
        # Place 'XX' at end; numeric sort when possible
        if p == "XX":
            return (1, 9999)
        try:
            return (0, int(p))
        except Exception:
            return (0, p)
    out: List[Dict[str, str]] = []
    for (tn, pr, fid) in sorted(vals_by_key.keys(), key=lambda k: ((k[0] or ""), _pr_key(k[1]), (k[2] or ""))):
        st = compute_stats(vals_by_key[(tn, pr, fid)])
        out.append({"testname": tn, "pr": pr, "fuseid": fid, **st})
    return out
