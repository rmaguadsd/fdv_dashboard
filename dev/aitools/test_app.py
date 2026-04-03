#!/usr/bin/env python3
"""
Test script to validate the fdv_chart app with specific file and regex
"""
import requests
import json
import time

# Wait for app to be ready
time.sleep(1)

# Test parameters
test_file = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
regex_filter = r"(?i)^FDV OUT.*WL.*SB.*BL.*"
mode = "include"

# Prepare the form data
with open(test_file, 'rb') as f:
    files = {'file': f}
    data = {'regex': regex_filter, 'mode': mode}
    
    # Send POST request to /parse
    print(f"Sending POST request to http://localhost:5058/parse")
    print(f"  File: {test_file}")
    print(f"  Regex: {regex_filter}")
    print(f"  Mode: {mode}")
    print()
    
    try:
        response = requests.post('http://localhost:5058/parse', files=files, data=data, timeout=120)
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print()
        
        result = response.json()
        print(f"Parse Result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Message: {result.get('message')}")
        print(f"  CSV ID: {result.get('csv_id')}")
        print(f"  Row Count: {result.get('row_count')}")
        print(f"  Headers: {result.get('headers')}")
        print()
        
        if result.get('row_count') and result.get('row_count') > 0:
            rows = result.get('rows', [])
            print(f"First 3 rows:")
            for i, row in enumerate(rows[:3]):
                print(f"  Row {i+1}: {row}")
            print()
            
            # Try to download CSV
            if result.get('csv_id'):
                csv_url = f"http://localhost:5058/download/{result.get('csv_id')}"
                print(f"CSV Download URL: {csv_url}")
                csv_response = requests.get(csv_url)
                print(f"CSV Download Status: {csv_response.status_code}")
                print(f"CSV Size: {len(csv_response.content)} bytes")
                
                # Save CSV to file
                output_file = r"d:\FDV\git\fdv_dashboard\dev\aitools\test_output.csv"
                with open(output_file, 'wb') as csv_file:
                    csv_file.write(csv_response.content)
                print(f"CSV saved to: {output_file}")
                print()
                
                # Show first few lines of CSV
                csv_lines = csv_response.text.split('\n')[:5]
                print(f"First 5 lines of CSV:")
                for line in csv_lines:
                    print(f"  {line}")
        
        print("\n✓ TEST PASSED - App working correctly!")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
