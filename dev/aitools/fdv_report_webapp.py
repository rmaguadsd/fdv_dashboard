#!/usr/bin/env python
"""
FDV Report v1 (fdv_report_webapp.py) is obsolete.

Please use FDV Report v2 instead:
  - Run: fdv_report2_webapp.py (respects FDV_REPORT2_* env vars)
  - Features: PASS/FAIL columns with configurable RBER limit, plane-op handling, progress, plots.
"""
from __future__ import annotations
import sys

MSG = (
    "FDV Report v1 has been removed.\n"
    "Launch v2 with: python <path>/fdv_report2_webapp.py\n"
    "(or set FDV_REPORT2_HOST/FDV_REPORT2_PORT and run it).\n"
)

def main() -> int:
    sys.stdout.write(MSG)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
