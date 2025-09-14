"""
process_fdv package

Lightweight facade for parsing and reporting FDV logs.
This package re-exports the stable API from the existing implementation
to provide a clean import path for apps (e.g., fdv_poll_webapp).

Public API:
- process_file(in_path: Path, starts_with: str, ignore_value: float)
- write_csv(rows, out_csv)
- write_text(lines, out_txt)
- write_stats_by_fdv_occurrence(rows, markers, out_csv)
- write_stats_by_fdv(rows, out_csv)
- write_stats_by_fdv_vcc_temp(rows, out_csv)
- PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT
"""
from __future__ import annotations
import importlib

_mod = None
try:  # Prefer the dedicated FDV OUTPUT parser
    _mod = importlib.import_module(__name__ + ".core")
except Exception:  # pragma: no cover
    try:
        _mod = importlib.import_module(__name__ + ".compat")
    except Exception as _e:  # as a last resort raise a clear error
        raise ImportError(f"process_fdv: failed to import core/compat: {_e}")

# Re-export selected API
process_file = getattr(_mod, "process_file")
write_csv = getattr(_mod, "write_csv")
write_text = getattr(_mod, "write_text")
write_stats_by_fdv_occurrence = getattr(_mod, "write_stats_by_fdv_occurrence")
write_stats_by_fdv = getattr(_mod, "write_stats_by_fdv")
write_stats_by_fdv_vcc_temp = getattr(_mod, "write_stats_by_fdv_vcc_temp")
# RBER helpers (FDV OUTPUT)
rber_stats_by_fdv = getattr(_mod, "rber_stats_by_fdv", None)
rber_stats_by_fdv_with_pr = getattr(_mod, "rber_stats_by_fdv_with_pr", None)
rber_stats_from_dir = getattr(_mod, "rber_stats_from_dir", None)
write_rber_stats_csv = getattr(_mod, "write_rber_stats_csv", None)
read_dir_rows = getattr(_mod, "read_dir_rows", None)
rber_stats_by_tname = getattr(_mod, "rber_stats_by_tname", None)
PREFIX_DEFAULT = getattr(_mod, "PREFIX_DEFAULT", "FDV OUTPUT")
IGNORE_VALUE_DEFAULT = getattr(_mod, "IGNORE_VALUE_DEFAULT", None)

# Provide a fallback implementation if core/compat lacks rber_stats_by_fdv_with_pr
if rber_stats_by_fdv_with_pr is None:
    try:
        import statistics as _stats
        from typing import Dict, List, Tuple
        def _compute_stats(values: List[float]) -> Dict[str, str]:
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
        def rber_stats_by_fdv_with_pr(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:  # type: ignore[no-redef]
            from collections import defaultdict
            groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            def _to_f(x):
                try:
                    return float(x)
                except Exception:
                    return None
            for r in rows:
                if (r.get("tname", "") or "").strip().upper() == "PR":
                    continue
                fdv = r.get("fdv_file", "") or ""
                if "poweron" in fdv.lower():
                    continue
                pr = (r.get("pr", "") or "").strip() or "XX"
                rv = _to_f(r.get("rber"))
                if rv is None:
                    continue
                groups[(fdv, pr)].append(rv)
            out: List[Dict[str, str]] = []
            for (fdv, pr) in sorted(groups.keys(), key=lambda kv: (kv[0] or "", kv[1] == "XX", kv[1])):
                st = _compute_stats(groups[(fdv, pr)])
                out.append({"fdv_file": fdv, "pr": pr, **st})
            return out
    except Exception:
        pass

__all__ = [
    "process_file",
    "write_csv",
    "write_text",
    "write_stats_by_fdv_occurrence",
    "write_stats_by_fdv",
    "write_stats_by_fdv_vcc_temp",
    "rber_stats_by_fdv",
    "rber_stats_by_fdv_with_pr",
    "rber_stats_from_dir",
    "write_rber_stats_csv",
    "read_dir_rows",
    "rber_stats_by_tname",
    "PREFIX_DEFAULT",
    "IGNORE_VALUE_DEFAULT",
]
