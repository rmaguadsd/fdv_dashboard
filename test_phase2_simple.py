#!/usr/bin/env python3
# REV9 Phase 2 - Simple Test
import urllib.request
import json
import time
from pathlib import Path

BASE = "http://localhost:5059"
LOG_DIR = r"D:\FDV\logs\A2\DOE\PPSR"

print("="*80)
print("REV9 PHASE 2 TEST SUITE")
print("="*80)

# Test 1: Server connectivity
print("\n[1] Testing server connectivity...")
try:
    with urllib.request.urlopen(f"{BASE}/", timeout=5) as r:
        print("    [PASS] Server responding on port 5059")
except Exception as e:
    print(f"    [FAIL] {e}")
    exit(1)

# Test 2: Get test file
print("\n[2] Finding test file...")
try:
    files = sorted(Path(LOG_DIR).glob("*.txt"), key=lambda f: f.stat().st_size)
    test_file = files[-1]
    size_mb = test_file.stat().st_size / (1024**2)
    size_gb = size_mb / 1024
    print(f"    Using: {test_file.name}")
    print(f"    Size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
except Exception as e:
    print(f"    [FAIL] {e}")
    exit(1)

# Test 3: Dynamic timeout - parse the file
print("\n[3] Testing dynamic timeout on 1.4GB file...")
print("    NOTE: This will take 15-20 minutes!")
print("    Starting parse job...")
try:
    with open(test_file, 'rb') as f:
        file_content = f.read()
    
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = bytearray()
    body.extend(f'--{boundary}\r\n'.encode())
    body.extend(b'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n')
    body.extend(b'Content-Type: text/plain\r\n\r\n')
    body.extend(file_content)
    body.extend(f'\r\n--{boundary}\r\n'.encode())
    body.extend(b'Content-Disposition: form-data; name="regex"\r\n\r\n')
    body.extend(b'FDV OUT.*::READ_RBER_PAGE.*')
    body.extend(f'\r\n--{boundary}--\r\n'.encode())
    
    req = urllib.request.Request(
        f"{BASE}/parse",
        data=bytes(body),
        method="POST"
    )
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    start = time.time()
    print("    Sending request (timeout set to 25 minutes)...")
    
    with urllib.request.urlopen(req, timeout=1500) as resp:
        result = json.loads(resp.read().decode())
        elapsed = time.time() - start
        
        csv_id = result.get('csv_id', 'UNKNOWN')
        matches = result.get('match_count', 0)
        
        print(f"    [PASS] Parse completed successfully!")
        print(f"    CSV ID: {csv_id}")
        print(f"    Matches found: {matches}")
        print(f"    Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        
except urllib.error.HTTPError as e:
    elapsed = time.time() - start
    print(f"    [FAIL] HTTP Error {e.code}")
    print(f"    Elapsed: {elapsed:.1f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"    [FAIL] {e}")
    print(f"    Elapsed: {elapsed:.1f}s")
    exit(1)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
