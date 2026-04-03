#!/usr/bin/env python3
print("TEST 1", flush=True)

import tempfile
print("TEST 2", flush=True)

from pathlib import Path
print("TEST 3", flush=True)

temp_dir = tempfile.gettempdir()
print("TEST 4", flush=True)

log_file = str(Path(temp_dir) / 'test.log')
print("TEST 5", flush=True)

print("DONE", flush=True)
