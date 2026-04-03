#!/usr/bin/env python3
"""
Manual test of FDV Chart Parser
"""
import requests
import json
import time

time.sleep(1)

test_file = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
regex_filter = r"(?i)^FDV OUT.*WL.*SB.*BL.*"
mode = "include"

print("=" * 60)
print("Testing FDV Chart Parser")
print("=" * 60)
print(f"\n1. File: {test_file}")
print(f"2. Regex: {regex_filter}")
print(f"3. Mode: {mode}")
print()

try:
    # Open and upload file
    print("Step 1: Opening file...", end="", flush=True)
    f = open(test_file, 'rb')
    print(" OK")
    
    print("Step 2: Creating form data...", end="", flush=True)
    files = {'file': f}
    data = {'regex': regex_filter, 'mode': mode}
    print(" OK")
    
    print("Step 3: Sending POST request...", end="", flush=True)
    response = requests.post('http://localhost:5058/parse', files=files, data=data, timeout=300)
    f.close()
    print(f" OK (HTTP {response.status_code})")
    
    if response.status_code != 200:
        print(f"ERROR: Server returned {response.status_code}")
        print(response.text)
        exit(1)
    
    print("Step 4: Parsing JSON response...", end="", flush=True)
    result = response.json()
    print(" OK")
    print()
    print(f"Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Total Rows: {result.get('total_rows', result.get('row_count', 'N/A'))}")
    print(f"  Display Rows: {len(result.get('rows', []))}")
    print(f"  Headers: {len(result.get('headers', []))} columns")
    print(f"  CSV ID: {result.get('csv_id', 'N/A')}")
    
    if result.get('success'):
        print()
        print("✓ PARSE SUCCESS")
        
        if result.get('rows'):
            print()
            print("First 2 rows:")
            for i, row in enumerate(result['rows'][:2]):
                print(f"  Row {i+1}: {row[:3]}...")
        
        # Try download
        csv_id = result.get('csv_id')
        if csv_id:
            print()
            print(f"Downloading CSV ({csv_id})...", end="", flush=True)
            csv_response = requests.get(f'http://localhost:5058/download/{csv_id}')
            print(f" (HTTP {csv_response.status_code}, {len(csv_response.content)} bytes)")
            
            if csv_response.status_code == 200:
                print("✓ CSV DOWNLOAD SUCCESS")
                # Show first few lines
                csv_lines = csv_response.text.split('\n')[:3]
                print(f"  Header: {csv_lines[0]}")
                if len(csv_lines) > 1:
                    print(f"  Data row 1: {csv_lines[1][:80]}...")
            else:
                print(f"✗ CSV DOWNLOAD FAILED: {csv_response.status_code}")
    else:
        print(f"✗ PARSE FAILED: {result.get('error')}")
        
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
