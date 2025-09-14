#!/usr/bin/env python
"""
Extract a table from a raw log by selecting only lines containing a marker (default: 'FDV OUTPUT').
Writes a CSV with columns: line_number, timestamp (best-effort), level, known keys (dut, blk, page/WL, pagetype,
readtype, deck, parttype, rber), any other parsed key=value pairs, and the trailing message.

Usage (PowerShell):
  python aitools/fdv_output_table.py -i C:/path/to/raw_log.txt -o C:/path/to/raw_log_fdv_output.csv

Notes:
- Pattern match is case-insensitive.
- Timestamp extraction is best-effort across common formats; if none is found, field is left blank.
- Key=value parsing supports both '=' and ':' separators; values end at ',', ';', or whitespace.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Tuple, Optional

# Regex patterns
PATTERN_DEFAULT = r"/FDV OUTPUT/"
# Common timestamp formats (first match wins)
TS_PATTERNS = [
    # 2025-08-11 14:20:33 or 2025/08/11 14:20:33
    re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}\b"),
    # 2025-08-11T14:20:33.123Z or with offset
    re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
    # [14:20:33] or 14:20:33
    re.compile(r"\b\[?\d{2}:\d{2}:\d{2}\]?\b"),
]
# Key-value pairs like key=value or key: value; stop value at comma/semicolon/space unless quoted
KV_RE = re.compile(r"(?P<k>[A-Za-z0-9_.-]+)\s*(?:=|:)\s*(?P<v>\"[^\"]*\"|'[^']*'|[^,;\s]+)")

# Known keys we prefer to order
PREFERRED_KEYS = [
    "dut", "die", "unit", "blk", "block", "wl", "page", "pagetype", "ptype",
    "readtype", "rtype", "deck", "parttype", "rber",
]

# Canonical target columns for qlc-compatible output
QLC_REQUIRED = ["WL", "RBER", "pagetype", "readtype", "dut"]
QLC_OPTIONAL = ["parttype", "deck", "blk", "page"]
ALIASES_QKC = {
    "WL": ["wl", "wordline", "word_line", "page"],
    "RBER": ["rber", "raw_ber", "rawber", "ber", "error_rate", "ber_raw"],
    "pagetype": ["pagetype", "ptype", "page_type", "pagemaptype", "maptype", "type"],
    "readtype": ["readtype", "rtype", "read_type", "readmode", "read_mode"],
    "dut": ["dut", "device", "unit", "chip", "die", "die_id", "dut_id"],
    "parttype": ["parttype", "part_type", "ptype_group"],
    "deck": ["deck"],
    "blk": ["blk", "block", "bl"],
    "page": ["page", "wl"],
}


def extract_timestamp(line: str) -> str:
    for rx in TS_PATTERNS:
        m = rx.search(line)
        if m:
            return m.group(0).strip("[]")
    return ""


def detect_level(line: str) -> str:
    # Heuristic: look for typical levels
    up = line.upper()
    for lvl in ("ERROR", "ERR", "WARN", "WARNING", "INFO", "DEBUG"):
        if lvl in up:
            return lvl
    return ""


def parse_key_values(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in KV_RE.finditer(text):
        k = m.group("k").strip()
        v = m.group("v").strip().strip('"').strip("'")
        if k and k not in out:
            out[k] = v
    return out


def compile_marker_regex(pattern: str) -> re.Pattern:
    """Compile a user-supplied regex, accepting Perl-like /pattern/flags syntax.
    Supported flags: i (ignorecase), m (multiline), s (dotall), x (verbose).
    If not in /.../flags form, compile with IGNORECASE by default for backwards compatibility.
    """
    try:
        if pattern and pattern.startswith('/') and '/' in pattern[1:]:
            last = pattern.rfind('/')
            body = pattern[1:last]
            flags_str = pattern[last+1:]
            flag_map = {
                'i': re.IGNORECASE,
                'm': re.MULTILINE,
                's': re.DOTALL,
                'x': re.VERBOSE,
            }
            flags = 0
            ok = True
            for ch in flags_str:
                if not ch:
                    continue
                if ch in flag_map:
                    flags |= flag_map[ch]
                else:
                    ok = False
                    break
            if ok:
                return re.compile(body, flags)
        # Fallback: plain compile with IGNORECASE for continuity
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        # As a last resort, compile a literal search on the given pattern (ignorecase)
        esc = re.escape(pattern)
        return re.compile(esc, re.IGNORECASE)


def split_after_marker(line: str, marker_re: re.Pattern) -> Tuple[str, str]:
    m = marker_re.search(line)
    if not m:
        return line, ""
    # Trailing content after the marker token
    after = line[m.end():].lstrip(" :-\t")
    return line[:m.start()], after


def collect_rows(path: str, pattern: str, require_bytes: Optional[int]) -> Tuple[List[Dict[str, str]], List[str]]:
    marker_re = compile_marker_regex(pattern)
    rx_bytes = None
    if require_bytes and int(require_bytes) > 0:
        n = int(require_bytes)
        rx_bytes = re.compile(rf"\b({n})\b\s*bytes(?:\s+of\s+data)?\b", re.IGNORECASE)
    rows: List[Dict[str, str]] = []
    all_keys: set[str] = set()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            if not marker_re.search(line):
                continue
            # If a specific byte-count is required, enforce it
            bytes_val = ""
            if rx_bytes is not None:
                mbytes = rx_bytes.search(line)
                if not mbytes:
                    continue
                bytes_val = mbytes.group(1)
            prefix, after = split_after_marker(line, marker_re)
            ts = extract_timestamp(prefix) or extract_timestamp(line)
            lvl = detect_level(prefix)
            kv = parse_key_values(after)
            # Normalize a few common key aliases
            if "block" in kv and "blk" not in kv:
                kv["blk"] = kv["block"]
            if "wl" not in kv and "page" in kv:
                kv["wl"] = kv["page"]
            row: Dict[str, str] = {
                "line_number": str(idx),
                "timestamp": ts,
                "level": lvl,
                "bytes": bytes_val,
                "message": after.strip(),
            }
            row.update(kv)
            rows.append(row)
            all_keys.update(kv.keys())

    # Build ordered columns: fixed, preferred that exist (dedup), other keys alpha, message last
    fixed = ["line_number", "timestamp", "level", "bytes"]
    pref = [k for k in PREFERRED_KEYS if k in all_keys]
    other = sorted([k for k in all_keys if k not in set(pref)])
    columns = fixed + pref + other + ["message"]
    return rows, columns


def write_csv(rows: List[Dict[str, str]], columns: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def kv_to_canonical(kv: Dict[str, str]) -> Dict[str, str]:
    """Map a generic kv dict into canonical qlc-like columns using alias rules."""
    out: Dict[str, str] = {}
    # Build case-insensitive lookup
    lower_map = {k.lower(): k for k in kv.keys()}
    def get_first(keys: List[str]) -> Optional[str]:
        for cand in keys:
            if cand.lower() in lower_map:
                return kv[lower_map[cand.lower()]]
        return None
    # Resolve required and optional
    for canon, aliases in ALIASES_QKC.items():
        val = get_first(aliases)
        if val is not None:
            out[canon] = val
    # If WL missing but page present, copy
    if "WL" not in out and "page" in out:
        out["WL"] = out["page"]
    # Numeric coercions for WL, RBER, blk, page
    def to_float(s: str) -> Optional[float]:
        try:
            return float(str(s).replace(",", ""))
        except Exception:
            return None
    def to_intlike(s: str) -> Optional[int]:
        try:
            return int(float(str(s).replace(",", "")))
        except Exception:
            return None
    if "WL" in out:
        v = to_float(out["WL"]) or to_intlike(out["WL"])
        if v is not None:
            out["WL"] = str(v)
    if "RBER" in out:
        v = to_float(out["RBER"]) or to_float(out["RBER"])
        if v is not None:
            out["RBER"] = f"{v:.6e}"
    if "blk" in out:
        v = to_intlike(out["blk"]) or to_float(out["blk"]) 
        if v is not None:
            out["blk"] = str(int(v))
    if "page" in out:
        v = to_intlike(out["page"]) or to_float(out["page"]) 
        if v is not None:
            out["page"] = str(int(v))
    # Normalize deck labels to LD/MD/UP if possible
    if "deck" in out:
        s = str(out["deck"]).strip().upper()
        if s.startswith("L"): s = "LD"
        elif s.startswith("M"): s = "MD"
        elif s.startswith("U"): s = "UP"
        out["deck"] = s
    # Ensure strings for categorical fields
    for c in ("pagetype", "readtype", "dut", "parttype"):
        if c in out:
            out[c] = str(out[c])
    return out


def rows_to_qlc(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Transform generic extracted rows into qlc-compatible rows and columns."""
    out_rows: List[Dict[str, str]] = []
    # Build rows with canonical fields only
    for r in rows:
        kv = {k: v for k, v in r.items() if k not in ("line_number", "timestamp", "level", "bytes", "message")}
        can = kv_to_canonical(kv)
        # Must have required fields
        if not all(k in can for k in QLC_REQUIRED):
            continue
        # Filter invalid/non-positive RBER
        try:
            if float(can.get("RBER", "nan")) <= 0:
                continue
        except Exception:
            continue
        out_rows.append(can)
    # Columns: required, then available optional in preferred order
    cols = list(QLC_REQUIRED)
    for c in QLC_OPTIONAL:
        if any(c in r for r in out_rows):
            cols.append(c)
    return out_rows, cols


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract FDV OUTPUT lines into CSV: notes-pipeline (default), qlc-compatible, or table")
    p.add_argument(
        "-i", "--input",
        default=r"C:\\Users\\rmaguad\\Documents\\Work\\logs\\read_eimpro\\raw_log.txt",
        help="Path to raw log (default points to Work/logs/read_eimpro/raw_log.txt)"
    )
    p.add_argument(
        "-o", "--output",
        default=None,
    help="Output CSV path (default: alongside input; suffix depends on mode)"
    )
    p.add_argument(
        "--pattern",
        default=PATTERN_DEFAULT,
        help=(
            "Regex pattern to match log lines. Accepts Perl-like '/pattern/flags' (flags: i,m,s,x), "
            "or a bare pattern compiled case-insensitive by default. Default: 'FDV\\s+OUTPUT'"
        )
    )
    p.add_argument(
        "--require-bytes",
        type=int,
        default=18592,
        help="If >0, keep only lines containing '<N> bytes [of data]'; default: 18592. Use 0 to disable."
    )
    p.add_argument(
        "--mode",
        choices=["qlc", "table", "notes"],
        default="notes",
        help=(
            "Output mode: 'notes' (default) to reproduce notes.txt; 'qlc' for qlc.csv-compatible columns; "
            "or 'table' for a rich table."
        )
    )
    p.add_argument(
        "--op-filter",
        default="/(PAGE_READ|EIMPRO_READ)/",
        help="Only for --mode notes: additional regex to keep operation lines (default: '/(PAGE_READ|EIMPRO_READ)/')"
    )
    args = p.parse_args(argv)

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    rows, columns = collect_rows(in_path, args.pattern, args.require_bytes)
    if args.mode == "qlc":
        qrows, qcols = rows_to_qlc(rows)
        if not args.output:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_dir = os.path.dirname(in_path)
            out_path = os.path.join(out_dir, f"{base}_qlc.csv")
        else:
            out_path = args.output
        if not qrows:
            print("warning: no qlc-compatible rows found; writing header-only CSV")
        write_csv(qrows, qcols, out_path)
        print(f"saved: {out_path} ({len(qrows)} rows)")
    elif args.mode == "table":
        if not args.output:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_dir = os.path.dirname(in_path)
            out_path = os.path.join(out_dir, f"{base}_fdv_output.csv")
        else:
            out_path = args.output
        if not rows:
            print("warning: no matching lines found; writing header-only CSV")
        write_csv(rows, columns, out_path)
        print(f"saved: {out_path} ({len(rows)} rows)")
    else:
        # notes mode: replicate pipeline from notes.txt
        marker_re = compile_marker_regex(args.pattern)
        op_re = compile_marker_regex(args.op_filter)
        rx_bytes = None
        if args.require_bytes and int(args.require_bytes) > 0:
            n = int(args.require_bytes)
            rx_bytes = re.compile(rf"\b({n})\b\s*bytes(?:\s+of\s+data)?\b", re.IGNORECASE)
        if not args.output:
            out_dir = os.path.dirname(in_path)
            out_path = os.path.join(out_dir, "read_v_eimpro.csv")
        else:
            out_path = args.output
        count = 0
        with open(in_path, "r", encoding="utf-8", errors="replace") as fi, \
             open(out_path, "w", encoding="utf-8", newline="") as fo:
            for line in fi:
                # grep 'FDV OUTPUT'
                if not marker_re.search(line):
                    continue
                # egrep -e 'PAGE_READ|EIMPRO_READ'
                if not op_re.search(line):
                    continue
                # optional bytes filter
                if rx_bytes is not None and not rx_bytes.search(line):
                    continue
                s = line.rstrip("\r\n")
                # sed 's/^.*_\([A-Z]\+\)\.FDV::/\1,/g'
                s = re.sub(r"^.*_([A-Z]+)\.FDV::", r"\1,", s)
                # sed 's/SSYNC.*: \|,FAILCOUNT.*\|_ECC//g'
                s = re.sub(r"SSYNC.*?:\s|,FAILCOUNT.*|_ECC", "", s)
                # sed 's/_/,/g'
                s = s.replace("_", ",")
                # sed 's/,[0-9.]\+$//g'
                s = re.sub(r",[0-9.]+$", "", s)
                fo.write(s + "\n")
                count += 1
        print(f"saved: {out_path} ({count} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
