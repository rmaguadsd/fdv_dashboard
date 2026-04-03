#!/usr/bin/env python3
import tempfile
from pathlib import Path

LOG_FILE = str(Path(tempfile.gettempdir()) / 'fdv_chart_server.log')

with open(LOG_FILE, 'w') as f:
    f.write("MODULE_LOADED\n")
    f.flush()

print("Module level code executed")
