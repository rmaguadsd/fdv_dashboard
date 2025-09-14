from __future__ import annotations
import argparse
from pathlib import Path
from .core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT, write_csv, write_text, write_stats_by_fdv, write_stats_by_fdv_occurrence, write_stats_by_fdv_vcc_temp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Parse FDV OUTPUT logs and write CSV/text summaries")
    ap.add_argument("input", help="Path to input log file")
    ap.add_argument("--prefix", default=PREFIX_DEFAULT, help=f"Line prefix to match (default: {PREFIX_DEFAULT!r})")
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap on number of lines to scan")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on extracted rows")
    ap.add_argument("--progress", action="store_true", help="Print parsing progress (lines and percent by bytes)")
    ap.add_argument("--progress-interval", type=int, default=100000, help="Progress update interval in lines (default: 100000)")
    args = ap.parse_args(argv)

    p = Path(args.input)
    rows, kept, markers = process_file(
        p,
        starts_with=args.prefix,
        ignore_value=IGNORE_VALUE_DEFAULT,
        max_lines=args.max_lines,
        max_rows=args.max_rows,
        progress=(args.progress_interval if args.progress else False),
    )
    base = p.with_suffix("")
    write_csv(rows, base.with_name(base.name + "_fdv_output.csv"))
    write_text(kept, base.with_name(base.name + "_fdv_output_filtered.txt"))
    write_stats_by_fdv(rows, base.with_name(base.name + "_fdv_output_stats_by_fdv.csv"))
    write_stats_by_fdv_occurrence(rows, markers, base.with_name(base.name + "_fdv_output_stats_by_fdv_line.csv"))
    write_stats_by_fdv_vcc_temp(rows, base.with_name(base.name + "_fdv_output_stats_by_fdv_vcc_temp.csv"))
    print(f"Parsed {len(rows)} rows from {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
