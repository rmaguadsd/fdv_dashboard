#!/usr/bin/env python3
import tempfile
from pathlib import Path

print("tempfile module imported")
temp_dir = tempfile.gettempdir()
print("Temp dir:", temp_dir)

log_file = str(Path(temp_dir) / 'test.log')
print("Log file path:", log_file)

try:
    with open(log_file, 'w') as f:
        f.write("TEST\n")
    print("Write succeeded")
except Exception as e:
    print("Write failed:", e)
