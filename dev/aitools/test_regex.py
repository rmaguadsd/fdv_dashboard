#!/usr/bin/env python3
"""Test regex matching behavior"""
import re

test_lines = [
    "FDV OUTPUT [test] DUT1,value1",
    "Some other FDV OUTPUT line DUT2,value2",
    "PREFIX FDV OUTPUT something DUT3,value3",
    "FDV POLL [test] DUT1,value1",
]

patterns = [
    "^FDV OUT",  # Should match only lines starting with FDV OUT
    "^FDV",      # Should match lines starting with FDV
    "FDV OUT",   # Should match anywhere
]

for pattern in patterns:
    print(f"\nPattern: {pattern}")
    regex = re.compile(pattern)
    for line in test_lines:
        match = bool(regex.search(line))
        print(f"  {match}: {line[:50]}")
