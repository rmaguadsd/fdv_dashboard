#!/usr/bin/env python3
"""
Download the HTML from server and save it to examine differences
"""
import urllib.request

url = "http://localhost:5059"
output_file = r"d:\FDV\git\fdv_dashboard\dev\aitools\server_html_actual.html"

print(f"[TEST] Fetching {url}...")
with urllib.request.urlopen(url, timeout=5) as response:
    html = response.read()
    with open(output_file, 'wb') as f:
        f.write(html)
    print(f"✅ Saved {len(html)} bytes to {output_file}")
    
    # Show relevant section
    html_str = html.decode('utf-8')
    idx = html_str.find('font-item-select')
    if idx > 0:
        print("\nActual HTML around font-item-select:")
        print(html_str[idx:idx+500])
