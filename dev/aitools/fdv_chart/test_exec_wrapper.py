#!/usr/bin/env python3
import sys
sys.path.insert(0, r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart')

try:
    print("About to import fdv_chart...")
    exec(open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py').read())
except Exception as e:
    print("Exception:", e)
    import traceback
    traceback.print_exc()
