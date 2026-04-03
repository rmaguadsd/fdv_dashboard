#!/usr/bin/env python3
import subprocess
import sys

result = subprocess.run([
    r'C:\Python312\python.exe',
    '-u',
    r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py'
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print("Return code:", result.returncode)
