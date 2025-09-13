"""
Compatibility layer that forwards to the existing process_fdv_poll module.
This lets us standardize imports as `from process_fdv import ...` while
reusing the proven implementation.
"""
from __future__ import annotations
from pathlib import Path

# Import the existing implementation located alongside this package's parent
# i.e. aitools/process_fdv_poll.py
try:
    # Relative import based on package location
    from .. import process_fdv_poll as _impl
except Exception as e:  # pragma: no cover - import fallback
    # If relative import fails (e.g., run outside package context), try absolute
    import process_fdv_poll as _impl  # type: ignore

# Re-export public API
process_file = _impl.process_file
write_csv = _impl.write_csv
write_text = _impl.write_text
write_stats_by_fdv_occurrence = _impl.write_stats_by_fdv_occurrence
write_stats_by_fdv = _impl.write_stats_by_fdv
write_stats_by_fdv_vcc_temp = _impl.write_stats_by_fdv_vcc_temp
PREFIX_DEFAULT = getattr(_impl, "PREFIX_DEFAULT", "FDV POLL")
IGNORE_VALUE_DEFAULT = getattr(_impl, "IGNORE_VALUE_DEFAULT", -999)

__all__ = [
    "process_file",
    "write_csv",
    "write_text",
    "write_stats_by_fdv_occurrence",
    "write_stats_by_fdv",
    "write_stats_by_fdv_vcc_temp",
    "PREFIX_DEFAULT",
    "IGNORE_VALUE_DEFAULT",
]
