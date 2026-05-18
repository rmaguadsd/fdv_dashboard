#!/usr/bin/env python3
"""
REV9 Phase 2 - Quick API Test (no external dependencies)
Tests core endpoints directly
"""
import urllib.request
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:5059"
LOG_DIR = r"D:\FDV\logs\A2\DOE\PPSR"

def test_root():
    """Test server is responding"""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/") as resp:
            if resp.status == 200:
                print("[OK] Server is running")
                return True
    except Exception as e:
        print(f"[FAILED] Server check failed: {e}")
        return False

def get_test_file():
    """Get largest available test file"""
    try:
        files = sorted(Path(LOG_DIR).glob("*.txt"), key=lambda f: f.stat().st_size)
        if files:
            largest = files[-1]
            size_mb = largest.stat().st_size / (1024**2)
            print(f"  Test file: {largest.name}")
            print(f"  File size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")
            return largest
    except Exception as e:
        print(f"[FAILED] Error finding test file: {e}")
    return None

def test_parse_api(test_file):
    """Test /parse endpoint with actual file"""
    try:
        print(f"\n[TEST] Starting parse job...")
        print(f"  Regex: FDV OUT.*::READ_RBER_PAGE.*")
        print(f"  Expected timeout: ~20+ minutes for 1.4GB")
        
        start_time = time.time()
        
        # Read file and prepare multipart request
        with open(test_file, 'rb') as f:
            file_content = f.read()
        
        # Build multipart form data manually
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        body = bytearray()
        
        # Add file part
        body.extend(f'--{boundary}\r\n'.encode())
        body.extend(b'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n')
        body.extend(b'Content-Type: text/plain\r\n\r\n')
        body.extend(file_content)
        body.extend(f'\r\n--{boundary}\r\n'.encode())
        
        # Add regex part
        body.extend(b'Content-Disposition: form-data; name="regex"\r\n\r\n')
        body.extend(b'FDV OUT.*::READ_RBER_PAGE.*')
        body.extend(f'\r\n--{boundary}--\r\n'.encode())
        
        # Make request
        req = urllib.request.Request(
            f"{BASE_URL}/parse",
            data=bytes(body),
            method="POST"
        )
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        
        with urllib.request.urlopen(req, timeout=1800) as resp:
            response_data = json.loads(resp.read().decode())
            elapsed = time.time() - start_time
            
            if resp.status == 200:
                csv_id = response_data.get('csv_id', 'UNKNOWN')
                match_count = response_data.get('match_count', 0)
                print(f"[OK] Parse job completed")
                print(f"  CSV ID: {csv_id}")
                print(f"  Matches: {match_count}")
                print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
                return csv_id
            else:
                print(f"[FAILED] Parse failed with status {resp.status}")
                return None
                
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        print(f"[FAILED] HTTP Error {e.code}: {e.reason}")
        print(f"  Elapsed: {elapsed:.1f}s")
        if e.code == 408:
            print("  (This may indicate timeout - server took too long)")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[FAILED] Parse request failed: {e}")
        print(f"  Elapsed: {elapsed:.1f}s")
        return None

def test_csv_download(csv_id):
    """Test /download_csv endpoint"""
    if not csv_id:
        print("\n[SKIP] CSV download (no CSV ID)")
        return
    
    try:
        print(f"\n[TEST] CSV download")
        start_time = time.time()
        
        with urllib.request.urlopen(f"{BASE_URL}/download_csv/{csv_id}", timeout=300) as resp:
            content = resp.read(1000)  # Just read first 1KB to test
            elapsed = time.time() - start_time
            
            if resp.status == 200:
                print(f"✓ CSV download endpoint works")
                print(f"  Response time: {elapsed:.3f}s")
                print(f"  Content type: {resp.headers.get('Content-Type', 'unknown')}")
                return True
            else:
                print(f"✗ CSV download failed with status {resp.status}")
                return False
    except Exception as e:
        print(f"✗ CSV download failed: {e}")
        return False

def test_pagination(csv_id):
    """Test /rows endpoint"""
    if not csv_id:
        print("\n[SKIP] Pagination (no CSV ID)")
        return
    
    try:
        print(f"\n[TEST] Pagination")
        
        test_cases = [
            (0, 1000, "offset=0, limit=1000"),
            (10000, 1000, "offset=10000, limit=1000"),
        ]
        
        for offset, limit, desc in test_cases:
            start_time = time.time()
            
            url = f"{BASE_URL}/rows?csv_id={csv_id}&offset={offset}&limit={limit}"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                elapsed = time.time() - start_time
                
                if resp.status == 200:
                    row_count = len(data.get('rows', []))
                    total = data.get('total', 0)
                    print(f"✓ {desc}: {row_count} rows in {elapsed*1000:.1f}ms (total: {total})")
                else:
                    print(f"✗ {desc}: HTTP {resp.status}")
                    
    except Exception as e:
        print(f"✗ Pagination test failed: {e}")

def test_job_status(csv_id):
    """Test /job_status endpoint"""
    if not csv_id:
        print("\n[SKIP] Job status (no CSV ID)")
        return
    
    try:
        print(f"\n[TEST] Job status")
        
        # Convert csv_id to job_id
        job_id = csv_id.replace('csv_', 'job_')
        
        with urllib.request.urlopen(f"{BASE_URL}/job_status/{job_id}", timeout=10) as resp:
            data = json.loads(resp.read().decode())
            
            if resp.status == 200:
                state = data.get('state', 'unknown')
                elapsed = data.get('elapsed_seconds', 0)
                print(f"✓ Job status available")
                print(f"  State: {state}")
                print(f"  Elapsed: {elapsed}s")
            else:
                print(f"✗ Job status failed with status {resp.status}")
                
    except Exception as e:
        print(f"✗ Job status test failed: {e}")

def main():
    print("""
==============================================================================
              REV9 PHASE 2 TESTING - LIVE ENDPOINT VERIFICATION             
                                                                            
  Tests: Dynamic Timeouts, CSV Download, Pagination, Job Status            
==============================================================================
    """)
    
    # Test 1: Server health
    print("[TEST] Server connectivity")
    if not test_root():
        print("\nFATAL: Server not responding")
        return 1
    
    # Test 2: Get test file
    print("\n[SETUP] Finding test file")
    test_file = get_test_file()
    if not test_file:
        print("\nFATAL: No test files found")
        return 1
    
    # Test 3: Parse (tests dynamic timeout)
    print("\n[PHASE 2 TEST 1] Dynamic Timeout")
    print("=" * 80)
    csv_id = test_parse_api(test_file)
    
    # Test 4-6: Other endpoints
    print("\n[PHASE 2 TEST 2] CSV Download")
    print("=" * 80)
    test_csv_download(csv_id)
    
    print("\n[PHASE 2 TEST 3] Pagination")
    print("=" * 80)
    test_pagination(csv_id)
    
    print("\n[PHASE 2 TEST 4] Job Status")
    print("=" * 80)
    test_job_status(csv_id)
    
    print("\n" + "=" * 80)
    print("TEST SESSION COMPLETE")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test aborted by user")
        sys.exit(1)
