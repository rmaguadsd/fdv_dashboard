import sys, re, json, time, threading, http.client, uuid
sys.path.insert(0, r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart")

LOG   = r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt"
REGEX = r"^FDV OUT.*WL.*SB.*BL.*"
PORT  = 5059

# ── 1. Direct parse ──────────────────────────────────────────────────────────
print("=== 1. Direct parse ===")
from fdv_chart import parse_log_file, RequestHandler
from http.server import HTTPServer

headers, rows = parse_log_file(LOG, REGEX, include_mode=True)
wi = headers.index("WL"); ri = headers.index("RBER"); di = headers.index("DUT")
print("[1] total rows   :", len(rows))
print("[1] sample WL    :", rows[0][wi])
print("[1] sample RBER  :", rows[0][ri])
print("[1] sample DUT   :", rows[0][di])
wl_n = [int(r[wi]) for r in rows[:500] if re.match(r"^\d+$", r[wi])]
rb_n = [float(r[ri]) for r in rows[:500] if re.match(r"^[\d.eE+\-]+$", r[ri])]
print("[1] WL  numeric", len(wl_n), "/500  range", min(wl_n), "-", max(wl_n))
print("[1] RBER numeric", len(rb_n), "/500  range", round(min(rb_n),8), "-", round(max(rb_n),6))
assert len(rows) > 100000, "Too few rows!"
assert len(wl_n) > 0, "No numeric WL!"
assert len(rb_n) > 0, "No numeric RBER!"

# ── 2. Start embedded HTTP server on test port ────────────────────────────────
print("\n=== 2. HTTP server ===")
srv = HTTPServer(("127.0.0.1", PORT), RequestHandler)
t = threading.Thread(target=srv.serve_forever, daemon=True)
t.start()
time.sleep(0.8)

c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
c.request("GET", "/")
r = c.getresponse()
body = r.read().decode()
print("[2] GET / status=", r.status)
print("[2] plot-panel in HTML:", "plot-panel" in body)
print("[2] chart.js CDN in HTML:", "chart.js" in body.lower())
print("[2] /plot_data endpoint in HTML:", "/plot_data" in body)
assert r.status == 200
assert "plot-panel" in body
assert "chart.js" in body.lower()

# ── 3. POST /parse ────────────────────────────────────────────────────────────
print("\n=== 3. POST /parse ===")
boundary = "BND" + uuid.uuid4().hex
with open(LOG, "rb") as f:
    fb = f.read()

def mp_field(name, value_bytes, filename=None):
    cd = f'Content-Disposition: form-data; name="{name}"'
    if filename:
        cd += f'; filename="{filename}"'
        ct = "Content-Type: text/plain\r\n"
    else:
        ct = ""
    return f"--{boundary}\r\n{cd}\r\n{ct}\r\n".encode() + value_bytes + b"\r\n"

parts = (
    mp_field("file", fb, filename="test.txt") +
    mp_field("regex", REGEX.encode()) +
    mp_field("mode", b"include") +
    f"--{boundary}--\r\n".encode()
)

c2 = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
c2.request("POST", "/parse", body=parts,
           headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                    "Content-Length": str(len(parts))})
r2 = c2.getresponse()
rb2 = json.loads(r2.read())
print("[3] status        :", r2.status)
print("[3] success       :", rb2.get("success"))
print("[3] total_rows    :", rb2.get("total_rows"))
print("[3] headers count :", len(rb2.get("headers", [])))
assert r2.status == 200, f"Expected 200 got {r2.status}"
assert rb2.get("success"), f"Parse failed: {rb2.get('error')}"
assert rb2.get("total_rows", 0) > 100000, "Too few rows in response"

csv_id = rb2["csv_id"]
h2 = rb2["headers"]
s2 = rb2["rows"]
wi2 = h2.index("WL") if "WL" in h2 else -1
ri2 = h2.index("RBER") if "RBER" in h2 else -1
print("[3] WL col idx    :", wi2, " sample:", s2[0][wi2] if s2 else "?")
print("[3] RBER col idx  :", ri2, " sample:", s2[0][ri2] if s2 else "?")
assert wi2 >= 0, "WL not in headers"
assert ri2 >= 0, "RBER not in headers"
assert s2 and s2[0][wi2], "WL value empty in first row"
assert s2 and s2[0][ri2], "RBER value empty in first row"

# ── 4. GET /plot_data (x=WL, y=RBER) ────────────────────────────────────────
print("\n=== 4. GET /plot_data x=WL y=RBER ===")
from urllib.parse import urlencode
params = urlencode({
    "csv_id": csv_id, "x_col": "WL", "y_col": "RBER",
    "x_regex": "", "y_regex": "", "color_col": "DUT", "max_pts": "5000"
})
c3 = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
c3.request("GET", "/plot_data?" + params)
r3 = c3.getresponse()
pd3 = json.loads(r3.read())
pts = pd3.get("points", [])
print("[4] status        :", r3.status)
print("[4] success       :", pd3.get("success"))
print("[4] points        :", len(pts))
print("[4] skipped       :", pd3.get("skipped"))
if pts:
    print("[4] sample points :", pts[:3])
assert r3.status == 200
assert pd3.get("success"), f"plot_data failed: {pd3.get('error')}"
assert len(pts) > 0, "No plot points returned"
assert "x" in pts[0] and "y" in pts[0], "Points missing x/y"
assert isinstance(pts[0]["x"], (int, float)), "x not numeric"
assert isinstance(pts[0]["y"], (int, float)), "y not numeric"
assert pd3.get("skipped", 999999) < len(pts), "Too many skipped"

srv.shutdown()
print("\n=============================")
print("ALL SELF-CHECKS PASSED - OK")
print("=============================")
