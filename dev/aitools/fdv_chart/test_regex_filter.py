#!/usr/bin/env python3
"""
Quick test: call parse_log_file directly with the user's exact inputs
to verify the regex filter logic independently of the web layer.
"""
import sys, os, time

# Add the fdv_chart dir so we can import the function
sys.path.insert(0, os.path.dirname(__file__))

# re-implement parse_log_file here to isolate from web layer
import re

def parse_log_file(file_path, regex_pattern=None, include_mode=True):
    headers = ['Line#', 'DUT', 'Test Name', 'Status', 'Value',
               'VCC', 'VCCQ', 'TEMP', 'WL', 'BLK', 'Notes']
    rows = []
    compiled_regex = None
    if regex_pattern:
        compiled_regex = re.compile(regex_pattern)

    max_lines = 500000
    line_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            if line_count > max_lines:
                break
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
            if compiled_regex:
                matches = compiled_regex.search(line_stripped)
                if include_mode and not matches:
                    continue
                if not include_mode and matches:
                    continue
            parts = [p.strip() for p in line_stripped.split('|')]
            if len(parts) < 2 or all(not p for p in parts):
                parts = line_stripped.split()
            while len(parts) < len(headers):
                parts.append('')
            row_data = parts[:len(headers)]
            row_data[0] = str(line_num)
            rows.append(row_data)
    return headers, rows

# Test
FILE = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX = r"^FDV OUT.*WL.*SB.*BL.*"
MODE  = True   # include matching

print(f"File : {FILE}")
print(f"Regex: {REGEX}")
print(f"Mode : {'include' if MODE else 'exclude'}")
print()

t0 = time.time()
headers, rows = parse_log_file(FILE, REGEX, MODE)
elapsed = time.time() - t0

print(f"Headers: {headers}")
print(f"Total matching rows: {len(rows)}")
print(f"Elapsed: {elapsed:.2f}s")
if rows:
    print(f"\nFirst 3 rows:")
    for r in rows[:3]:
        print(r)
else:
    print("\n*** NO ROWS RETURNED – regex filter may be broken ***")
