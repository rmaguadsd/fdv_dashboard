#!/usr/bin/env python3
"""
Self-contained test: spins up fdv_chart server in subprocess,
sends a real multipart POST with the exact user inputs, prints results.
Writes everything to run_test_output.txt so we can read it back.
"""
import subprocess, time, http.client, json, uuid, os, sys

OUT = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\run_test_output.txt"
SERVER = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py"
FILE_PATH = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX = r"^FDV OUT.*WL.*SB.*BL.*"
MODE = "include"
HOST, PORT = "127.0.0.1", 5058

log = []
def say(msg):
    print(msg)
    log.append(str(msg))

say("=== FDV Chart Regex Filter Test ===")
say(f"File : {FILE_PATH}")
say(f"Regex: {REGEX}")
say(f"Mode : {MODE}")
say("")

# --- Step 1: verify regex independently ---
import re
compiled = re.compile(REGEX)
match_count = 0
with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as fh:
    for i, line in enumerate(fh, 1):
        s = line.strip()
        if s and compiled.search(s):
            match_count += 1
            if match_count == 1:
                say(f"[Direct regex] First match at line {i}: {s[:100]}")
say(f"[Direct regex] Total lines matching pattern: {match_count}")
say("")

# --- Step 2: start server ---
say("Starting server...")
proc = subprocess.Popen(
    [sys.executable, SERVER],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    cwd=os.path.dirname(SERVER),
)
# Wait up to 5s for server to be ready
ready = False
for _ in range(10):
    time.sleep(0.5)
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=2)
        c.request("GET", "/")
        resp = c.getresponse()
        resp.read()
        ready = True
        say(f"Server ready (GET / => {resp.status})")
        break
    except Exception:
        pass

if not ready:
    say("ERROR: Server did not start in time")
    proc.terminate()
    open(OUT, "w").write("\n".join(log))
    sys.exit(1)

# --- Step 3: build multipart body (first 5000 lines for speed) ---
with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as fh:
    lines_subset = fh.readlines()  # FULL file
file_bytes = "".join(lines_subset).encode("utf-8")
fname = os.path.basename(FILE_PATH)
boundary = "----FDVTestBoundary" + uuid.uuid4().hex[:8]

def mp_field(name, value_bytes):
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n'
        f"\r\n"
    ).encode() + value_bytes + b"\r\n"

body = b""
# file field
body += (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
    f"Content-Type: text/plain\r\n"
    f"\r\n"
).encode()
body += file_bytes + b"\r\n"
# regex field
body += mp_field("regex", REGEX.encode("utf-8"))
# mode field
body += mp_field("mode", MODE.encode("utf-8"))
# closing
body += f"--{boundary}--\r\n".encode()

ct = f"multipart/form-data; boundary={boundary}"
say(f"POST /parse  body={len(body)} bytes  lines_in_file={len(lines_subset)}")

# --- Step 4: send POST ---
try:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    conn.request("POST", "/parse", body=body, headers={
        "Content-Type": ct,
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="ignore")
    say(f"HTTP {resp.status}")

    try:
        result = json.loads(raw)
        say(f"success    : {result.get('success')}")
        if result.get("success"):
            say(f"total_rows : {result.get('total_rows')}")
            say(f"display    : {len(result.get('rows', []))}")
            rows = result.get("rows", [])
            if rows:
                say(f"headers    : {result.get('headers')}")
                say(f"first row  : {rows[0]}")
            else:
                say("*** ZERO ROWS RETURNED — filter is broken ***")
        else:
            say(f"ERROR from server: {result.get('error')}")
    except json.JSONDecodeError:
        say("Non-JSON response: " + raw[:500])

except Exception as e:
    say(f"Request exception: {e}")

# --- Step 5: teardown ---
proc.terminate()
try:
    proc.wait(timeout=5)
except Exception:
    pass
say("Server stopped.")
say("")
say("=== DONE ===")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(log))
say(f"Output written to {OUT}")
