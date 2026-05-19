import requests
import json

# Test loading a session
url = 'http://localhost:5059/store/register_session'
payload = {
    'dir': r'D:\FDV\recipes',
    'file': 'babysteps2.fdv_session'
}

try:
    print(f'Sending request to {url}')
    print(f'Payload: {payload}')
    response = requests.post(url, json=payload)
    print(f'Status code: {response.status_code}')
    print(f'Response: {response.text[:500]}')
    if response.ok:
        data = response.json()
        print(f'Success: {data.get("success")}')
        if data.get("success"):
            print(f'CSV ID: {data.get("csv_id")}')
            print(f'Total rows: {data.get("total_rows")}')
            print(f'Headers: {len(data.get("headers", []))} items')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
