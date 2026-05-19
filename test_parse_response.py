#!/usr/bin/env python3
import requests
import json

# Test what the server actually returns for /parse_multi
url = 'http://localhost:5059/parse_multi'

# Create a simple test file
test_file_path = '/tmp/test_log.txt'
with open(test_file_path, 'w') as f:
    f.write("FDV OUT::TEST_EVENT::1\n")
    f.write("FDV OUT::TEST_EVENT::2\n")

# Prepare the multipart request
files = {'file': open(test_file_path, 'rb')}
data = {
    'regex_include': 'FDV OUT.*::TEST_EVENT::',
    'regex_exclude': ''
}

try:
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body (first 500 chars):")
    print(response.text[:500])
    print(f"\nFull JSON:")
    try:
        resp_json = response.json()
        print(json.dumps(resp_json, indent=2))
    except:
        print("Could not parse as JSON")
except Exception as e:
    print(f"Error: {e}")
