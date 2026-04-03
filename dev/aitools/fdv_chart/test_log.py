#!/usr/bin/env python3
import tempfile
from pathlib import Path

LOG_FILE = str(Path(tempfile.gettempdir()) / 'test_log.log')

def log_msg(msg):
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception as e:
        print("Log error:", e)

log_msg("TEST MESSAGE")
print("Done")
