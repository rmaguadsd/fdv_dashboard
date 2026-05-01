#!/usr/bin/env python3
"""Test the /chat_stream endpoint to diagnose text panel deactivation"""
import json
import urllib.request
import urllib.error
import sys

body = json.dumps({
    'csv_id': 'test',
    'message': 'what is 2+2',
    'context': 'test context',
    'model': 'llama3'
}).encode('utf-8')

req = urllib.request.Request(
    'http://localhost:5059/chat_stream',
    data=body,
    headers={'Content-Type': 'application/json'}
)

try:
    print("Sending request to /chat_stream endpoint...")
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"Status: {r.status}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        print(f"Headers: {dict(r.headers)}")
        print("\n--- Response data ---")
        
        count = 0
        for line in r:
            line_str = line.decode('utf-8').strip()
            if line_str:
                print(f"Line {count}: {line_str}")
                count += 1
                if count >= 20:
                    print("...")
                    break
                    
except urllib.error.URLError as e:
    print(f"URL Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    sys.exit(1)
