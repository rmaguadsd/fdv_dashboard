#!/usr/bin/env python
"""
Process a log file, extracting lines that start with "FDV POLL" and capturing the
second token to the right of a "DUT<number>" occurrence. Lines where that second
token is numerically equal to -999 are ignored (filtered out). In addition, any
line that contains the whole word "MONITOR" (case-insensitive) will be ignored.

Additionally, track the most recent FDV filename (.fdv) mentioned in the log and
compute per-FDV statistics (count, mean, stdev, min, median, p90, max) for the
captured numeric data.

Outputs:
- A CSV summarizing matches with derived fields:
    line_number, dut_id, data_token2, data_token2_numeric, fdv_file, tname,
    operation, product, pagetype, blk, page, wl, poll__test, status, plane_group,
    vcc, temp, fdv_idx, raw_line
- A filtered text file containing only the original matching lines that pass the filter
- A per-FDV statistics CSV (fdv_file, count, mean, stdev, min, p50, p90, max)

Usage (PowerShell):
    python aitools/process_fdv_poll.py -i "C:/path/to/Output_site...txt"

Options:
    --starts-with   Override the line prefix to match (default: 'FDV POLL')
    --ignore-value  Numeric value to filter out (default: -999)
    -o / --output   Output base path (without extension). Defaults to input path stem.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Set
import statistics as stats

PREFIX_DEFAULT = "FDV POLL"
IGNORE_VALUE_DEFAULT = -999

def extract_after_dut_number(line: str) -> Tuple[str | None, str | None, List[str]]:
    """Return (dut_number, second_token_after_dut, tokens_after).
    Looks for 'DUT<number>' and captures tokens to its right.
    Tokens are sequences of [A-Za-z0-9_.+-] separated by whitespace or punctuation.
    """
    m = re.search(r"\bDUT(\d+)\b", line)
    if not m:
        return None, None, []
    dutnum = m.group(1)
    tail = line[m.end():]
    tail = tail.lstrip(" =:\t,;")
    tokens = re.findall(r"[A-Za-z0-9_.+-]+", tail)
    t2 = tokens[1] if len(tokens) >= 2 else None
    return dutnum, t2, tokens


def maybe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def short_fdv_name(p: str) -> str:
    """Return the filename without path and without the .fdv extension (case-insensitive)."""
    if not p:
        return ""
    # Normalize slashes then take basename
    name = os.path.basename(p.replace("\\", "/"))
    if name.lower().endswith(".fdv"):
        name = name[: -4]
    return name


def process_file(
    in_path: Path,
    starts_with: str,
    ignore_value: float,
    *,
    progress: bool | int = False,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> Tuple[List[Dict[str, str]], List[str], List[Dict[str, str]]]:
    # Pre-scan for FUSEIDs to identify invalid DUTs by occurrence order
    def _is_valid_fuseid(fid: str) -> bool:
        m = re.match(r"^K\d{6}_([+-]?\d+)_([+-]?\d+)_([+-]?\d+)$", (fid or '').strip())
        if not m:
            return False
        try:
            return int(m.group(1)) > 0
        except Exception:
            return False
    def _prescan_invalid_duts(path: Path) -> Set[str]:
        invalid: Set[str] = set()
        order: List[str] = []
        rx = re.compile(r"\bFUSEID\s*:\s*([^\r\n]+)", re.IGNORECASE)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    m = rx.search(line)
                    if not m:
                        continue
                    raw = (m.group(1) or '').strip()
                    token = re.split(r"[\s,;|]", raw, maxsplit=1)[0].strip().strip('[]()')
                    if token:
                        order.append(token)
        except Exception:
            return invalid
        for i, fid in enumerate(order):
            if not _is_valid_fuseid(fid):
                invalid.add(str(i+1))
        return invalid

    invalid_duts: Set[str] = _prescan_invalid_duts(in_path)
    prefix_re = re.compile(r"^\s*" + re.escape(starts_with) + r"\b")
    rows: List[Dict[str, str]] = []
    kept_lines: List[str] = []
    last_fdv: str = ""
    fdv_re = re.compile(r"([A-Za-z0-9_./\\:-]+\.fdv)\b", re.IGNORECASE)
    # Track each occurrence of an .fdv mention (marker) with an increasing index.
    markers: List[Dict[str, str]] = []
    try:
        # Also append a prescan marker so UIs can show ignored DUTs and their FUSEIDs
        # Re-run a tiny scan here to capture fuseid order for display without duplicating logic
        order: List[str] = []
        rx = re.compile(r"\bFUSEID\s*:\s*([^\r\n]+)", re.IGNORECASE)
        with open(in_path, 'r', encoding='utf-8', errors='replace') as fscan:
            for line in fscan:
                m = rx.search(line)
                if not m:
                    continue
                raw = (m.group(1) or '').strip()
                token = re.split(r"[\s,;|]", raw, maxsplit=1)[0].strip().strip('[]()')
                if token:
                    order.append(token)
        markers.append({
            'type': 'fuseid_prescan',
            'source_file': str(in_path),
            'dut_fuseids': [{'dut': str(i+1), 'fuseid': order[i]} for i in range(len(order))],
            'invalid_duts': sorted(list(invalid_duts)),
        })
    except Exception:
        pass
    fdv_idx = -1
    # Capture VCC and TEMP values if present on the line
    vcc_re = re.compile(r"\bVCC\s*=\s*([0-9.+\-Ee]+)")
    temp_re = re.compile(r"\bTEMP(?:ERATURE)?\s*=\s*([0-9.+\-Ee]+)")
    # Capture timing mode (TM)
    tm_re = re.compile(r"\bTM\s*=\s*([0-9.+\-Ee]+)")
    # Capture poll__test if present: poll__test=NAME
    polltest_re = re.compile(r"\bpoll__test\s*=\s*([A-Za-z0-9_.+\-]+)", re.IGNORECASE)
    # Capture status (C0/E0) if present, either as status= or as token in tail
    status_re = re.compile(r"\bstatus\s*=\s*([CcEe]0)\b")
    # Capture plane group (SP/MP) if present: plane=SP/MP
    plane_re = re.compile(r"\bplane\s*=\s*(SP|MP)\b", re.IGNORECASE)
    # Capture tname/testname right after ".FDV::" up to next comma
    tname_re = re.compile(r"\.fdv::\s*([^,]+)", re.IGNORECASE)
    # Capture FUSEID lines and map by DUT order; capture PR (proberev) per DUT from PR lines
    # Capture broader FUSEID values (allow hyphen/plus) until whitespace or delimiter
    fuseid_re = re.compile(r"\bFUSEID\s*:\s*([A-Za-z0-9_+\-]+)", re.IGNORECASE)
    pr_line_re = re.compile(r"::\s*PR\b", re.IGNORECASE)
    fuseids_by_dut: Dict[str, str] = {}
    fuseid_order: List[str] = []  # order of FUSEIDs as seen; index 0 -> DUT1, etc.
    pr_by_dut: Dict[str, str] = {}  # decimal string, or 'XX'

    file_size = 0
    try:
        file_size = os.path.getsize(in_path)
    except Exception:
        file_size = 0
    # Normalize progress config
    env_prog = os.environ.get("FDV_PROGRESS", "0").strip().lower()
    enable_progress = bool(progress) or env_prog in ("1", "true", "yes", "on") or progress_cb is not None
    # progress interval: if an int was provided, treat as lines interval; else default to 100000 lines
    lines_interval = int(progress) if isinstance(progress, int) and progress else 100000
    last_report_lines = 0
    last_report_pct = -1.0
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            # Speed-up: only consider relevant lines:
            #   - FDV POLL data lines
            #   - echoed FUSEID lines
            #   - test time headers ('Test Start Date', 'Test End Date')
            _ls = line.lstrip()
            _up = _ls.upper()
            # Always emit progress based on byte position to avoid apparent stalls
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
                    print(f"Parsing… {lineno} lines (~{pct:.1f}%)", file=sys.stderr)
                last_report_lines = lineno
            # Include PR monitor lines anywhere in the log so we can extract per-DUT PR values
            _is_pr_line = bool(pr_line_re.search(line))
            if not (
                _up.startswith("FDV POLL")
                or _up.startswith("ECHO: FUSEID:")
                or _up.startswith("TEST START DATE")
                or _up.startswith("TEST END DATE")
                or _is_pr_line
            ):
                continue
            # Track latest FDV filename seen anywhere in the log
            mfdv = fdv_re.search(line)
            if mfdv:
                last_fdv = mfdv.group(1)
                fdv_idx += 1
                markers.append({
                    "fdv_idx": str(fdv_idx),
                    "fdv_file": short_fdv_name(last_fdv),
                    "marker_line": str(lineno),
                })
            # Capture FUSEID lines; build mapping by DUT order (1-based)
            mfid = fuseid_re.search(line)
            if mfid:
                fid = mfid.group(1).strip()
                fuseid_order.append(fid)
                # refresh mapping by observed order so far
                _n = len(fuseid_order := fuseid_order)
                fuseids_by_dut = {str(i+1): fuseid_order[i] for i in range(_n)}
            # Capture PR (proberev) from PR lines; convert last hex token to decimal
            if _is_pr_line:
                mdut = re.search(r"\bDUT(\d+)\b", line)
                if mdut:
                    dnum = mdut.group(1)
                    # Take the last two-hex-digit group at end of line
                    mhex = re.search(r":\s*([0-9A-Fa-f]{2})[^0-9A-Fa-f]*$", line)
                    if mhex:
                        try:
                            pr_dec = str(int(mhex.group(1), 16))
                        except Exception:
                            pr_dec = "XX"
                    else:
                        pr_dec = "XX"
                    pr_by_dut[dnum] = pr_dec
            if not prefix_re.search(line):
                continue
            # Requirement: skip any line that contains the word MONITOR (whole word, case-insensitive)
            # This mirrors behavior in FDV OUTPUT parsing to exclude monitor rows entirely
            if re.search(r"\bMONITOR\b", _up):
                continue
            dutnum, data2, tokens = extract_after_dut_number(line)
            # If we didn't find DUT<number> and the needed token, skip
            if dutnum is None or data2 is None:
                continue
            # Skip any lines from DUTs with invalid FUSEID (pre-scanned)
            if dutnum in invalid_duts:
                continue
            val = maybe_float(data2)
            if val is not None and val == ignore_value:
                # Ignore this line per requirement
                continue
            # Extract conditions
            vcc = ""
            temp = ""
            tm = ""
            poll_test = ""
            status = ""
            plane_group = ""
            tname = ""
            mv = vcc_re.search(line)
            if mv:
                vcc = mv.group(1)
            mt = temp_re.search(line)
            if mt:
                temp = mt.group(1)
            m_tm = tm_re.search(line)
            if m_tm:
                tm = m_tm.group(1)
            mp = polltest_re.search(line)
            if mp:
                poll_test = short_fdv_name(mp.group(1))
            # Extract tname/testname as early as possible so we can derive fields from it
            mtname = tname_re.search(line)
            if mtname:
                tname = mtname.group(1).strip()
            # Prefer status token found within tname (e.g., ..._C0_SP_READ_...) per guide
            if tname:
                for tt in str(tname).replace('-', '_').split('_'):
                    tu = tt.strip().upper()
                    if tu in ("C0", "E0", "F0", "80", "E1", "E2", "E3", "E4"):
                        status = tu
                        break
            # If not found in tname, try explicit 'status=' field
            if not status:
                ms = status_re.search(line)
                if ms:
                    status = ms.group(1).upper()
            # Fallback: scan loose tokens for C0/E0
            if not status:
                for t in tokens:
                    tt = str(t).strip().upper()
                    if tt in ("C0", "E0"):
                        status = tt
                        break
            mpl = plane_re.search(line)
            if mpl:
                plane_group = mpl.group(1).upper()
            if not plane_group:
                for t in tokens:
                    tt = str(t).strip().upper()
                    if tt in ("SP", "MP"):
                        plane_group = tt
                        break
            # If still missing, try to derive plane group from tname tokens
            if not plane_group and tname:
                for tt in str(tname).replace('-', '_').split('_'):
                    tu = tt.strip().upper()
                    if tu in ("SP", "MP"):
                        plane_group = tu
                        break
            if not plane_group:
                # Heuristic: infer from fdv_file name suffixes
                try:
                    name = short_fdv_name(poll_test or last_fdv)
                    lname = name.lower()
                    if "_sp" in lname or lname.endswith("sp"):
                        plane_group = "SP"
                    elif "_mp" in lname or lname.endswith("mp"):
                        plane_group = "MP"
                except Exception:
                    pass
            # Last resort fallbacks: infer status by operation keywords, default plane to SP
            if not status:
                tl = (tname or "").lower()
                if any(k in tl for k in ("read", "eimpro")):
                    status = "80"
                elif any(k in tl for k in ("erase", "tbers", "program", "tprog", "tcbsy")):
                    status = "E0"
            # Normalize plane group to SP/MP/XP; default XP when unspecified/unknown
            if not plane_group or plane_group not in ("SP", "MP"):
                plane_group = "XP"
            # Derive attributes from tname
            product = ""
            pagetype = ""
            blk = ""
            page = ""
            wl = ""
            operation = ""
            if tname:
                # product/pagemap
                for cand in ("QLC", "TLC", "SSLC", "DSLC", "SLC"):
                    if cand in tname.upper():
                        product = cand
                        break
                # pagetype like PAGETYPE_LP
                try:
                    import re as _re
                    # Match PAGETYPE_XX or PGTYPE_XX anywhere in tname, even if preceded by an underscore
                    mpt = _re.search(r"(?:PAGETYPE|PGTYPE|PAGE[_:\-]?TYPE|PT)[_:\-]?([A-Za-z0-9\-]+)", tname, _re.IGNORECASE)
                    if mpt:
                        # Take the first alphanumeric token from the captured group (handles 'LP-QLC')
                        pagetype = _re.split(r"[^A-Za-z0-9]+", mpt.group(1))[0].upper()
                    mblk = _re.search(r"(?<![A-Za-z0-9])BLK\s*[_:\-\s]?\s*([0-9]+)(?![A-Za-z0-9])", tname, _re.IGNORECASE)
                    if mblk:
                        blk = mblk.group(1)
                    mpg = _re.search(r"(?<![A-Za-z0-9])(?:PG|PAGE|PHYPAGE)\s*[_:\-\s]?\s*([0-9]+)(?![A-Za-z0-9])", tname, _re.IGNORECASE)
                    if mpg:
                        page = mpg.group(1)
                    mwl = _re.search(r"(?<![A-Za-z0-9])WL\s*[_:\-\s]?\s*([0-9]+)(?![A-Za-z0-9])", tname, _re.IGNORECASE)
                    if mwl:
                        wl = mwl.group(1)
                except Exception:
                    pass
                # infer operation keyword presence
                tl = tname.lower()
                for op in ("erase", "program", "read", "eimpro", "tbers", "tprog", "tcbsy"):
                    if op in tl:
                        operation = op.upper()
                        break
            # Look up FUSEID and PR for this DUT
            fuseid_val = fuseids_by_dut.get(dutnum, f"DUT{dutnum}_9999999_999_99_99")
            pr_val = pr_by_dut.get(dutnum, "XX")
            # Derive PHYPAGE as 13 LSB of PAGE if available
            phypage = ""
            try:
                if page:
                    ph = int(page) & 0x1FFF
                    phypage = str(ph)
            except Exception:
                phypage = ""
            rows.append({
                "line_number": str(lineno),
                "dut_id": dutnum,
                "data_token2": data2,
                "data_token2_numeric": ("" if val is None else ("{:.6f}".format(val).rstrip("0").rstrip("."))),
                "fdv_file": short_fdv_name(last_fdv),
                "tname": tname,
                "operation": operation,
                "product": product,
                # also expose as pagemap for UI
                "pagemap": product,
                "pagetype": pagetype,
                "blk": blk,
                "page": page,
                "wl": wl,
                "phypage": phypage,
                "vcc": vcc,
                "temp": temp,
                "tm": tm,
                "poll__test": poll_test,
                "status": status,
                "plane_group": plane_group,
                "fdv_idx": str(fdv_idx),
                "fuseid": fuseid_val,
                "pr": pr_val,
                "raw_line": line.rstrip("\r\n"),
            })
            kept_lines.append(line.rstrip("\r\n"))
    return rows, kept_lines, markers


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
    "line_number", "dut_id", "data_token2", "data_token2_numeric",
        "fdv_file", "tname", "operation", "product", "pagemap", "pagetype", "blk", "page", "wl", "phypage",
    "poll__test", "status", "plane_group", "vcc", "temp", "tm", "fdv_idx", "fuseid", "pr", "raw_line"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_text(lines: List[str], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fo:
        for s in lines:
            fo.write(s + "\n")


def write_stats_by_fdv_occurrence(rows: List[Dict[str, str]], markers: List[Dict[str, str]], out_csv: Path) -> None:
    """Compute stats per occurrence of an .fdv mention line.
    Each time an .fdv filename appears in the log, an occurrence index is incremented.
    All subsequent FDV POLL values are associated with the latest occurrence index
    until the next .fdv mention.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Build index -> meta mapping
    idx_meta: Dict[int, Dict[str, str]] = {}
    for m in markers:
        try:
            idx_meta[int(m.get("fdv_idx", "-1"))] = m
        except Exception:
            continue
    # Group numeric values by fdv_idx
    groups: Dict[int, List[float]] = {}
    for r in rows:
        try:
            idx = int(r.get("fdv_idx", "-1"))
        except Exception:
            continue
        if idx < 0:
            continue
        try:
            v = float(r.get("data_token2_numeric") or r.get("data_token2") or "")
        except Exception:
            continue
        groups.setdefault(idx, []).append(v)
    cols = ["occurrence", "marker_line", "fdv_file", "count", "mean", "stdev", "min", "max"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for idx in sorted(groups.keys()):
            vals = groups[idx]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            count = len(vals_sorted)
            mean = float(sum(vals_sorted)) / count
            stdev = float(stats.stdev(vals_sorted)) if count >= 2 else 0.0
            vmin = vals_sorted[0]
            vmax = vals_sorted[-1]
            meta = idx_meta.get(idx, {})
            w.writerow({
                "occurrence": idx,
                "marker_line": meta.get("marker_line", ""),
                "fdv_file": meta.get("fdv_file", ""),
                "count": count,
                "mean": f"{mean:.6g}",
                "stdev": f"{stdev:.6g}",
                "min": f"{vmin:.6g}",
                "max": f"{vmax:.6g}",
            })


def percentile(sorted_vals: List[float], pct: float) -> float:
    """Compute percentile (0-100) using nearest-rank on a sorted list."""
    if not sorted_vals:
        return float('nan')
    k = max(1, int(round(pct / 100.0 * len(sorted_vals))))
    return float(sorted_vals[k-1])


def write_stats_by_fdv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Group values by fdv_file
    groups: Dict[str, List[float]] = {}
    for r in rows:
        fdv = r.get("fdv_file", "") or ""
        try:
            v = float(r.get("data_token2_numeric") or r.get("data_token2") or "")
        except Exception:
            continue
        groups.setdefault(fdv, []).append(v)
    cols = ["fdv_file", "count", "mean", "stdev", "min", "p50", "p90", "max"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for fdv, vals in sorted(groups.items(), key=lambda kv: (kv[0] or "")):
            if not vals:
                continue
            vals_sorted = sorted(vals)
            count = len(vals_sorted)
            mean = float(sum(vals_sorted)) / count
            stdev = float(stats.stdev(vals_sorted)) if count >= 2 else 0.0
            vmin = vals_sorted[0]
            vmax = vals_sorted[-1]
            p50 = percentile(vals_sorted, 50)
            p90 = percentile(vals_sorted, 90)
            w.writerow({
                "fdv_file": fdv,
                "count": count,
                "mean": f"{mean:.6g}",
                "stdev": f"{stdev:.6g}",
                "min": f"{vmin:.6g}",
                "p50": f"{p50:.6g}",
                "p90": f"{p90:.6g}",
                "max": f"{vmax:.6g}",
            })


def write_stats_by_fdv_vcc_temp(rows: List[Dict[str, str]], out_csv: Path) -> None:
    """Group by (fdv_file, vcc, temp) and compute stats on data_token2.
    Output columns: fdv_file, vcc, temp, min, max, mean, stdev, median (in this order).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    for r in rows:
        fdv = r.get("fdv_file", "") or ""
        vcc = (r.get("vcc", "") or "").strip()
        temp = (r.get("temp", "") or "").strip()
        polltest = (r.get("poll__test", "") or "").strip()
        tname = (r.get("tname", "") or "").strip()
        # If poll__test present, require it to match fdv_file
        if polltest and polltest != fdv:
            continue
        # Exclude TBERS when fdv_file is not TBERS
        if polltest and ("tbers" in polltest.lower()) and ("tbers" not in fdv.lower()):
            continue
        # Exclude when tname mentions tbers/erase but fdv_file is not an erase test name
        fdv_l = fdv.lower()
        if tname and ("tbers" in tname.lower() or "erase" in tname.lower()) and ("tbers" not in fdv_l and "erase" not in fdv_l):
            continue
        try:
            v = float(r.get("data_token2_numeric") or r.get("data_token2") or "")
        except Exception:
            continue
        groups.setdefault((fdv, vcc, temp), []).append(v)
    cols = ["fdv_file", "vcc", "temp", "min", "max", "mean", "stdev", "median"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for (fdv, vcc, temp), vals in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "", kv[0][2] or "")):
            if not vals:
                continue
            vals_sorted = sorted(vals)
            count = len(vals_sorted)
            mean = float(sum(vals_sorted)) / count
            stdev = float(stats.stdev(vals_sorted)) if count >= 2 else 0.0
            vmin = vals_sorted[0]
            vmax = vals_sorted[-1]
            median = float(stats.median(vals_sorted))
            w.writerow({
                "fdv_file": fdv,
                "vcc": vcc,
                "temp": temp,
                "min": f"{vmin:.6g}",
                "max": f"{vmax:.6g}",
                "mean": f"{mean:.6g}",
                "stdev": f"{stdev:.6g}",
                "median": f"{median:.6g}",
            })


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Filter and extract 'FDV POLL' lines and the second token after DUT")
    ap.add_argument("-i", "--input", required=True, help="Path to the input log file")
    ap.add_argument("--starts-with", default=PREFIX_DEFAULT, help="Line prefix to match (default: 'FDV POLL')")
    ap.add_argument("--ignore-value", type=float, default=IGNORE_VALUE_DEFAULT, help="Numeric value to filter out (default: -999)")
    ap.add_argument("-o", "--output", default=None, help="Output base path (without extension). Defaults to <input> without extension")
    ap.add_argument("--progress", action="store_true", help="Print parsing progress (lines and percent by bytes)")
    ap.add_argument("--progress-interval", type=int, default=100000, help="Progress update interval in lines (default: 100000)")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    base = Path(args.output) if args.output else in_path.with_suffix("")
    out_csv = Path(str(base) + "_fdv_poll.csv")
    out_txt = Path(str(base) + "_fdv_poll_filtered.txt")
    out_stats = Path(str(base) + "_fdv_poll_stats_by_fdv.csv")
    out_stats_vt = Path(str(base) + "_fdv_poll_stats_by_fdv_vcc_temp.csv")
    out_stats_occ = Path(str(base) + "_fdv_poll_stats_by_fdv_line.csv")

    rows, kept, markers = process_file(
        in_path,
        args.starts_with,
        args.ignore_value,
        progress=(args.progress_interval if args.progress else False),
    )

    write_csv(rows, out_csv)
    write_text(kept, out_txt)
    write_stats_by_fdv(rows, out_stats)
    write_stats_by_fdv_occurrence(rows, markers, out_stats_occ)
    write_stats_by_fdv_vcc_temp(rows, out_stats_vt)

    print(f"saved: {out_csv} ({len(rows)} rows)")
    print(f"saved: {out_txt} ({len(kept)} lines)")
    print(f"saved: {out_stats}")
    print(f"saved: {out_stats_occ}")
    print(f"saved: {out_stats_vt}")
    if args.progress:
        print(f"Parsing complete. Total lines processed: {len(kept) + (len(rows) - len(kept)) if kept else len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
