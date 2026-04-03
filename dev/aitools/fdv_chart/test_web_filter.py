#!/usr/bin/env python3
"""
End-to-end test of the fdv_chart web endpoint.
Simulates exactly what the browser does: multipart/form-data POST to /parse.
"""
import http.client
import os
import uuid
import json
import time

HOST = "127.0.0.1"
PORT = 5058

FILE_PATH = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX = r"^FDV OUT.*WL.*SB.*BL.*"
MODE = "include"

# --- Build multipart/form-data body exactly like a browser would ---
boundary = "----WebKitFormBoundary" + uuid.uuid4().hex[:16]

# Read file (limit to first 200 lines to keep the test fast)
with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as fh:
    all_lines = fh.readlines()
    subset = all_lines[:2000]   # enough to include some FDV OUTPUT lines
file_bytes = "".join(subset).encode("utf-8")
file_name = os.path.basename(FILE_PATH)

parts = []

# file part
parts.append(
    "--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
    "Content-Type: application/octet-stream\r\n"
    "\r\n".format(boundary=boundary, fname=file_name)
)
parts.append(None)  # placeholder for file bytes

# regex part
parts.append(
    "\r\n--{boundary}\r\n"
    'Content-Disposition: form-data; name="regex"\r\n'
    "\r\n"
    "{regex}".format(boundary=boundary, regex=REGEX)
)

# mode part
parts.append(
    "\r\n--{boundary}\r\n"
    'Content-Disposition: form-data; name="mode"\r\n'
    "\r\n"
    "{mode}".format(boundary=boundary, mode=MODE)
)

# closing boundary
parts.append(
    "\r\n--{boundary}--\r\n".format(boundary=boundary)
)

# Assemble body
body = b""
for p in parts:
    if p is None:
        body += file_bytes
    else:
        body += p.encode("utf-8")

content_type = "multipart/form-data; boundary=" + boundary

print("=== TEST: POST /parse ===")
print("Content-Type:", content_type)
print("Body length:", len(body))
print("Boundary:", boundary)
print("Regex:", REGEX)
print("Mode:", MODE)
print("File lines sent:", len(subset))
print()

# Send request
t0 = time.time()
try:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("POST", "/parse", body=body, headers={
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    elapsed = time.time() - t0
    print("HTTP status:", resp.status)
    data = resp.read().decode("utf-8", errors="ignore")

    try:
        result = json.loads(data)
        print("success:", result.get("success"))
        if result.get("success"):
            print("total_rows:", result.get("total_rows"))
            print("headers:", result.get("headers"))
            rows = result.get("rows", [])
            print("rows returned for display:", len(rows))
            if rows:
                print("First row:", rows[0])
            else:
                print("*** NO ROWS IN RESPONSE ***")
        else:
            print("error:", result.get("error"))
    except json.JSONDecodeError:
        print("Raw response (not JSON):", data[:500])
    print("Elapsed:", round(elapsed, 2), "s")
except ConnectionRefusedError:
    print("ERROR: Could not connect to {}:{}".format(HOST, PORT))
    print("Make sure the fdv_chart server is running first.")
except Exception as e:
    print("ERROR:", e)
