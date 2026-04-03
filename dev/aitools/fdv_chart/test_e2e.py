#!/usr/bin/env python3
"""Quick self-contained test: start server, POST, print result, stop."""
import subprocess, time, http.client, json, uuid, os, sys, signal

SERVER_SCRIPT = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py"
FILE_PATH = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX = r"^FDV OUT.*WL.*SB.*BL.*"
MODE = "include"
HOST = "127.0.0.1"
PORT = 5058

# Check if server already running
server_proc = None
try:
    c = http.client.HTTPConnection(HOST, PORT, timeout=2)
    c.request("GET", "/")
    c.getresponse()
    print("Server already running on port", PORT)
except Exception:
    print("Starting server...")
    server_proc = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    time.sleep(2)
    print("Server PID:", server_proc.pid)

# Build multipart body
boundary = "----TestBoundary" + uuid.uuid4().hex[:8]

# Read first 2000 lines of the file
with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as fh:
    subset = fh.readlines()[:2000]
file_bytes = "".join(subset).encode("utf-8")
fname = os.path.basename(FILE_PATH)

body = b""
# File part
body += ("--" + boundary + "\r\n").encode()
body += ('Content-Disposition: form-data; name="file"; filename="' + fname + '"\r\n').encode()
body += b"Content-Type: application/octet-stream\r\n"
body += b"\r\n"
body += file_bytes
body += b"\r\n"
# Regex part
body += ("--" + boundary + "\r\n").encode()
body += b'Content-Disposition: form-data; name="regex"\r\n'
body += b"\r\n"
body += REGEX.encode("utf-8")
body += b"\r\n"
# Mode part
body += ("--" + boundary + "\r\n").encode()
body += b'Content-Disposition: form-data; name="mode"\r\n'
body += b"\r\n"
body += MODE.encode("utf-8")
body += b"\r\n"
# Close
body += ("--" + boundary + "--\r\n").encode()

ct = "multipart/form-data; boundary=" + boundary

print("Sending POST /parse ...")
print("  Body size:", len(body))
print("  Regex:", REGEX)
print("  Mode:", MODE)
print("  File lines:", len(subset))

try:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    conn.request("POST", "/parse", body=body, headers={
        "Content-Type": ct,
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="ignore")
    print("HTTP status:", resp.status)
    try:
        result = json.loads(raw)
        print("success:", result.get("success"))
        if result.get("success"):
            print("total_rows:", result.get("total_rows"))
            rows = result.get("rows", [])
            print("display rows:", len(rows))
            if rows:
                print("First row:", rows[0])
        else:
            print("ERROR:", result.get("error"))
    except json.JSONDecodeError:
        print("Non-JSON response:", raw[:500])
except Exception as e:
    print("Request failed:", e)

# Cleanup
if server_proc:
    server_proc.terminate()
    server_proc.wait(timeout=5)
    print("Server stopped.")
