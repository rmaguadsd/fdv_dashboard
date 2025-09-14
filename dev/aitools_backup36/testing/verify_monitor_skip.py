from __future__ import annotations
import json
import sys
import re
from pathlib import Path

try:
    from aitools.process_fdv.core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT
except Exception as e:
    print(json.dumps({"ok": False, "error": f"import failed: {e}"}))
    sys.exit(1)

def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: verify_monitor_skip.py <path>"}))
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(json.dumps({"ok": False, "error": f"file not found: {p}"}))
        return 2

    # Parse rows via structured parser
    rows, kept, markers = process_file(p, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT)

    # Count FDV OUTPUT lines and how many contain MONITOR
    fdv_out = 0
    fdv_out_mon = 0
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                ls = line.lstrip()
                up = ls.upper()
                if up.startswith("FDV OUTPUT"):
                    fdv_out += 1
                    if re.search(r"\bMONITOR\b", up):
                        fdv_out_mon += 1
    except Exception:
        pass

    # Sanity: ensure emitted rows themselves don't include MONITOR (as a whole word)
    _mon_word = re.compile(r"\bMONITOR\b", re.IGNORECASE)
    any_mon_in_rows = any(
        (str(r.get("pass_fail", "")).strip().upper() == "MONITOR")
        or bool(_mon_word.search(r.get("raw_line", "")))
        for r in rows
    )

    out = {
        "ok": True,
        "file": str(p),
        "rows_emitted": len(rows),
        "fdv_output_lines": fdv_out,
        "fdv_output_monitor_lines": fdv_out_mon,
        "any_monitor_in_rows": any_mon_in_rows,
    }
    print(json.dumps(out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
