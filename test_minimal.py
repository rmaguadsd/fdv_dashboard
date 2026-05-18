#!/usr/bin/env python3
"""
Minimal REV9 Phase 2 Test - No server shutdown
"""
import requests
import time
import os
import sys

BASE_URL = "http://localhost:5059"

def test_connectivity():
    """Test 1: Server connectivity"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✓ [TEST 1] Server connectivity: PASS")
            return True
        else:
            print(f"✗ [TEST 1] Server connectivity: FAIL (status {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ [TEST 1] Server connectivity: FAIL ({e})")
        return False

def test_models():
    """Test 2: Models endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        data = response.json()
        if "success" in data and data["success"]:
            print(f"✓ [TEST 2] Models endpoint: PASS ({len(data.get('models', []))} models)")
            return True
        else:
            print(f"✗ [TEST 2] Models endpoint: FAIL")
            return False
    except Exception as e:
        print(f"✗ [TEST 2] Models endpoint: FAIL ({e})")
        return False

def test_find_file():
    """Test 3: Find test file"""
    try:
        log_dir = r"D:\FDV\logs\A2\DOE\PPSR"
        if not os.path.isdir(log_dir):
            print(f"✗ [TEST 3] Find test file: FAIL (directory not found: {log_dir})")
            return False
        
        files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
        if not files:
            print(f"✗ [TEST 3] Find test file: FAIL (no .txt files found)")
            return False
        
        # Get largest file
        test_file = max([(f, os.path.getsize(os.path.join(log_dir, f))) for f in files], key=lambda x: x[1])
        size_mb = test_file[1] / (1024*1024)
        size_gb = size_mb / 1024
        
        print(f"✓ [TEST 3] Found test file: {test_file[0]} ({size_gb:.2f} GB)")
        return True, os.path.join(log_dir, test_file[0])
    except Exception as e:
        print(f"✗ [TEST 3] Find test file: FAIL ({e})")
        return False

def test_parse_simple():
    """Test 4: Simple parse (small regex, limited results)"""
    try:
        # Find test file
        log_dir = r"D:\FDV\logs\A2\DOE\PPSR"
        files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
        test_file = max([(f, os.path.getsize(os.path.join(log_dir, f))) for f in files], key=lambda x: x[1])
        test_file_path = os.path.join(log_dir, test_file[0])
        
        print(f"  Parsing: {test_file[0]} ({test_file[1]/(1024**3):.2f} GB)")
        print(f"  This will take ~15-25 minutes...")
        
        # Send parse request
        with open(test_file_path, 'rb') as f:
            files_data = {'file': f}
            data = {'regex': 'FDV OUT.*::READ_RBER_PAGE.*'}
            
            response = requests.post(
                f"{BASE_URL}/parse",
                files=files_data,
                data=data,
                timeout=1800  # 30 minute timeout
            )
        
        result = response.json()
        
        if result.get('success'):
            print(f"✓ [TEST 4] Parse completed!")
            print(f"    CSV ID: {result.get('csv_id', 'N/A')}")
            print(f"    Matches: {result.get('match_count', 'N/A')}")
            print(f"    Time: {result.get('time_seconds', 'N/A')}s")
            return True, result.get('csv_id')
        else:
            print(f"✗ [TEST 4] Parse failed: {result.get('message', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ [TEST 4] Parse: FAIL ({e})")
        import traceback
        traceback.print_exc()
        return False

def test_pagination(csv_id):
    """Test 5: Pagination"""
    try:
        response = requests.get(
            f"{BASE_URL}/rows",
            params={'csv_id': csv_id, 'offset': 0, 'limit': 100},
            timeout=10
        )
        
        result = response.json()
        if result.get('success'):
            rows = result.get('rows', [])
            print(f"✓ [TEST 5] Pagination: PASS ({len(rows)} rows fetched)")
            return True
        else:
            print(f"✗ [TEST 5] Pagination: FAIL")
            return False
            
    except Exception as e:
        print(f"✗ [TEST 5] Pagination: FAIL ({e})")
        return False

def test_csv_download(csv_id):
    """Test 6: CSV download"""
    try:
        response = requests.get(
            f"{BASE_URL}/download_csv/{csv_id}",
            timeout=300
        )
        
        if response.status_code == 200:
            size_mb = len(response.content) / (1024*1024)
            print(f"✓ [TEST 6] CSV download: PASS ({size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ [TEST 6] CSV download: FAIL (status {response.status_code})")
            return False
            
    except Exception as e:
        print(f"✗ [TEST 6] CSV download: FAIL ({e})")
        return False

def test_job_status(csv_id):
    """Test 7: Job status"""
    try:
        job_id = csv_id.replace('csv_', 'job_')
        response = requests.get(
            f"{BASE_URL}/job_status/{job_id}",
            timeout=10
        )
        
        result = response.json()
        if result.get('success'):
            print(f"✓ [TEST 7] Job status: PASS")
            print(f"    State: {result.get('state', 'N/A')}")
            print(f"    Elapsed: {result.get('elapsed_seconds', 'N/A')}s")
            return True
        else:
            print(f"✗ [TEST 7] Job status: FAIL")
            return False
            
    except Exception as e:
        print(f"✗ [TEST 7] Job status: FAIL ({e})")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("REV9 PHASE 2 - MINIMAL TEST")
    print("=" * 50)
    
    # Test 1: Connectivity
    if not test_connectivity():
        print("\nServer not responding. Start it with:")
        print("  python3 dev/aitools/fdv_chart_rev9/fdv_chart.py 5059")
        sys.exit(1)
    
    # Test 2: Models
    test_models()
    
    # Test 3: Find file
    result = test_find_file()
    if not result:
        sys.exit(1)
    
    # Test 4: Parse
    result = test_parse_simple()
    if not result:
        sys.exit(1)
    
    csv_id = result[1]
    
    # Wait for parse to complete
    print("\nWaiting for parse job to complete...")
    
    # Test 5: Pagination
    time.sleep(2)  # Give server time
    test_pagination(csv_id)
    
    # Test 6: CSV download
    test_csv_download(csv_id)
    
    # Test 7: Job status
    test_job_status(csv_id)
    
    print("\n" + "=" * 50)
    print("TESTS COMPLETE")
    print("=" * 50)
