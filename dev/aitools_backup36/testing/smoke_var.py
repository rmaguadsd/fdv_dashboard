import os, re, time, sys
import requests as R

BASE = 'http://127.0.0.1:5067'
FILE = r'C:\\Users\\rmaguad\\Documents\\Work\\logs\\read_eimpro\\today_fdvrun_demo\\Output_site114_8_12_2025_14_12_40.txt'

print('POST file...')
files = {'dirfiles': (os.path.basename(FILE), open(FILE,'rb'), 'text/plain')}
r = R.post(BASE+'/', files=files, timeout=120)
print('POST / status', r.status_code)
text = r.text
# Progress page doesn't include token in a URL; it's in a JS const line
m = re.search(r'const\s+token\s*=\s*\"([0-9a-f]{32})\"', text)
if not m:
    print('No token found in progress page. First 500 chars:')
    print(text[:500])
    sys.exit(2)

token = m.group(1)
print('Token', token)

for i in range(120):
    st = R.get(f'{BASE}/status/{token}', timeout=30)
    j = st.json()
    print('status', j.get('status'), j.get('progress',{}).get('percent'))
    if j.get('status') == 'done':
        break
    time.sleep(1)

home = R.get(f'{BASE}/?token={token}', timeout=60)
print('HOME status', home.status_code, 'len', len(home.text))

m = re.search(r'name=\"fdv\"[^>]*value=\"([^\"]+)\"', home.text)
if not m:
    print('No fdv checkbox found. First 500 chars:')
    print(home.text[:500])
    sys.exit(3)
fdv_val = m.group(1)
print('fdv selected', fdv_val)

tests = R.post(f'{BASE}/fdv/{token}/tests', data={'fdv': fdv_val}, timeout=60)
print('TESTS status', tests.status_code, 'len', len(tests.text))

m2 = re.search(r'<input[^>]*name=\"sel\"[^>]*value=\"([^\"]+)\"', tests.text)
if not m2:
    print('No sel checkbox found. First 500 chars:')
    print(tests.text[:500])
    sys.exit(4)
sel_val = m2.group(1)
print('sel', sel_val)

var = R.post(f'{BASE}/fdv/{token}/tests/variability', data={'fdv': fdv_val, 'sel': sel_val}, timeout=60)
print('VAR page', var.status_code, 'len', len(var.text))

m3 = re.search(r'<img[^>]*src=\"([^\"]+)\"', var.text)
if not m3:
    print('No image url found. Trying direct image endpoint...')
    from urllib.parse import urlencode
    q = urlencode([('fdv', fdv_val), ('sel', sel_val)])
    durl = f"{BASE}/fdv/{token}/tests/variability_image?{q}"
    print('direct url', durl)
    img = R.get(durl, timeout=120)
    print('IMG status', img.status_code, 'ct', img.headers.get('Content-Type'))
    if img.status_code != 200:
        print('IMG body first 300:', img.text[:300])
    else:
        print('Image bytes', len(img.content))
    sys.exit(5)
img_url = m3.group(1)
if img_url.startswith('/'):
    img_url = BASE + img_url
print('img_url', img_url)

img = R.get(img_url, timeout=60)
print('IMG status', img.status_code, 'ct', img.headers.get('Content-Type'))
if img.status_code != 200:
    print('IMG body first 200:', img.text[:200])
