#!/usr/bin/env python3
print("START")

try:
    import tempfile
    print("tempfile imported")
    
    result = tempfile.gettempdir()
    print("gettempdir() returned:", result)
    
except Exception as e:
    print("Exception:", e)
    import traceback
    traceback.print_exc()

print("END")
