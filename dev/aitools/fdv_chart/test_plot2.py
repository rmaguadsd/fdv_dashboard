#!/usr/bin/env python3
"""Test all /plot_data scenarios with correct URL encoding."""
import subprocess, time, http.client, json, uuid, os, sys, urllib.parse

SERVER = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py"
FILE   = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX  = r"^FDV OUT.*WL.*SB.*BL.*"
HOST, PORT = "127.0.0.1", 5058

# Start server
proc = subprocess.Popen([sys.executable, SERVER], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
for _ in range(10):
    time.sleep(0.5)
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=2)
        c.request("GET", "/"); c.getresponse()
        print("Server ready"); break
    except: pass

# Parse first 5000 lines
bnd = "TestBnd" + uuid.uuid4().hex[:8]
with open(FILE, "r", encoding="utf-8", errors="ignore") as fh:
    chunk = "".join(fh.readlines()[:5000]).encode("utf-8")
fname = os.path.basename(FILE)
body = b""
body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\nContent-Type: text/plain\r\n\r\n").encode() + chunk + b"\r\n"
body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"regex\"\r\n\r\n").encode() + REGEX.encode() + b"\r\n"
body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\ninclude\r\n").encode()
body += f"--{bnd}--\r\n".encode()
conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
conn.request("POST", "/parse", body=body, headers={
    "Content-Type": f"multipart/form-data; boundary={bnd}",
    "Content-Length": str(len(body))})
pr = json.loads(conn.getresponse().read())
assert pr["success"], pr.get("error")
csv_id = pr["csv_id"]
print(f"Parsed: {pr['total_rows']} rows  csv_id={csv_id}")
print(f"Headers: {pr['headers']}")

# Spot-check first row
row0 = pr["rows"][0]
h = pr["headers"]
print(f"\nFirst row values:")
for col in ["WL","BLK","BYBER","RBER","tname","Line#","DUT","Result"]:
    if col in h:
        print(f"  {col} = {repr(row0[h.index(col)])}")

print()

# Test plot_data endpoint
def plot(x_col, y_col, x_re="", y_re="", color="", label=""):
    params = urllib.parse.urlencode({
        "csv_id": csv_id, "x_col": x_col, "y_col": y_col,
        "x_regex": x_re, "y_regex": y_re,
        "color_col": color, "max_pts": 2000
    })
    c2 = http.client.HTTPConnection(HOST, PORT, timeout=10)
    c2.request("GET", "/plot_data?" + params)
    pd = json.loads(c2.getresponse().read())
    pts  = sum(len(g["points"]) for g in pd.get("groups", []))
    skip = pd.get("n_skipped", 0)
    ng   = len(pd.get("groups", []))
    samp = pd["groups"][0]["points"][:2] if pd.get("groups") else []
    status = "OK" if pd.get("success") else "FAIL"
    err = f"  ERROR: {pd.get('error')}" if not pd.get("success") else ""
    print(f"[{status}] {label}")
    print(f"       pts={pts}  skipped={skip}  groups={ng}  sample={samp}{err}")

plot("WL",    "RBER",  "",            "",  "DUT",    "WL(num) vs RBER — color by DUT")
plot("BLK",   "BYBER", r"(\d+)",      "",  "",       "BLK first-num via regex vs BYBER")
plot("tname", "RBER",  r"WL_(\d+)",   "",  "Result", "WL extracted from tname vs RBER — color by Result")
plot("Line#", "RBER",  "",            "",  "",       "Line# vs RBER")
plot("SB",    "RBER",  "",            "",  "DUT",    "SB vs RBER — color by DUT")
plot("BL",    "RBER",  "",            "",  "",       "BL vs RBER")
plot("WL",    "BYBER", "",            "",  "",       "WL vs BYBER")

proc.terminate(); proc.wait(5)
print("\n=== ALL TESTS DONE ===")
