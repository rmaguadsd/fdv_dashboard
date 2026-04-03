#!/usr/bin/env python3
import sys
import traceback

# Enable all warnings and errors
import warnings
warnings.simplefilter("always")

try:
    print("Importing fdv_chart...", flush=True)
    sys.path.insert(0, r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart')
    import fdv_chart
    
    print("Starting server...", flush=True)
    fdv_chart.run_server(5058)
    
except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
