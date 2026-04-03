#!/usr/bin/env python3
"""Test the /plot_data endpoint: parse file, then request WL vs RBER scatter data."""
import subprocess, time, http.client, json, uuid, os, sys

SERVER  = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py"
FILE    = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX   = r"^FDV OUT.*WL.*SB.*BL.*"
MODE    = "include"
HOST, PORT = "127.0.0.1", 5058

def mp_body(file_bytes, fname, regex, mode):
    boundary = "----TestBnd" + uuid.uuid4().hex[:8]
    def field(name, val):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                ).encode() + val + b"\r\n"
    body = b""
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{fname}\"\r\nContent-Type: text/plain\r\n\r\n").encode()
    body += file_bytes + b"\r\n"
    body += field("regex", regex.encode())
    body += field("mode",  mode.encode())
    body += f"--{boundary}--\r\n".encode()
    return body, boundary

# Start server
proc = subprocess.Popen([sys.executable, SERVER], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
for _ in range(10):
    time.sleep(0.5)
    try:
        http.client.HTTPConnection(HOST,PORT,timeout=2).request("GET","/")
        print("Server ready"); break
    except: pass

# 1) Parse (first 5000 lines for speed)
with open(FILE,"r",encoding="utf-8",errors="ignore") as fh:
    chunk = "".join(fh.readlines()[:5000]).encode("utf-8")
body, bnd = mp_body(chunk, os.path.basename(FILE), REGEX, MODE)
conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
conn.request("POST","/parse",body=body,headers={
    "Content-Type": f"multipart/form-data; boundary=----TestBnd{bnd.split('TestBnd')[1]}",
    "Content-Length": str(len(body))})
r = conn.getresponse(); pr = json.loads(r.read())
print(f"Parse: success={pr['success']} total_rows={pr.get('total_rows')} csv_id={pr.get('csv_id')}")
assert pr['success'], pr.get('error')
csv_id = pr['csv_id']
headers = pr['headers']
print("Headers:", headers)

# 2) Plot WL vs RBER (direct numeric columns)
tests = [
    dict(x_col='WL', y_col='RBER', x_regex='', y_regex='', color_col='DUT', label='WL vs RBER'),
    dict(x_col='BLK', y_col='BYBER', x_regex=r'(\d+)', y_regex='', color_col='', label='BLK(1st num) vs BYBER'),
    dict(x_col='tname', y_col='RBER', x_regex=r'WL_(\d+)', y_regex='', color_col='Result', label='WL extracted from tname vs RBER'),
    dict(x_col='Line#', y_col='RBER', x_regex='', y_regex='', color_col='', label='Line# vs RBER'),
]

for t in tests:
    url = (f"/plot_data?csv_id={csv_id}&x_col={t['x_col']}&y_col={t['y_col']}"
           f"&x_regex={t['x_regex']}&y_regex={t['y_regex']}"
           f"&color_col={t['color_col']}&max_pts=2000")
    conn2 = http.client.HTTPConnection(HOST,PORT,timeout=10)
    conn2.request("GET", url)
    pd = json.loads(conn2.getresponse().read())
    ok = pd.get('success')
    n  = pd.get('n_plotted',0)
    sk = pd.get('n_skipped',0)
    ng = len(pd.get('groups',[]))
    sample = pd['groups'][0]['points'][:2] if pd.get('groups') else []
    print(f"\n[{t['label']}]")
    print(f"  success={ok}  plotted={n}  skipped={sk}  groups={ng}")
    print(f"  sample points: {sample}")
    if not ok: print("  ERROR:", pd.get('error'))

proc.terminate(); proc.wait(5)
print("\n=== ALL PLOT TESTS DONE ===")
