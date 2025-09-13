import os, re, sys
from pathlib import Path
HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from fdv_report2_webapp import _build_variability_records, _parse_fdv_selector, _get_split_tuple, _extract_wl_or_page, _get_rber

# Input
FILE = r'C:\\Users\\rmaguad\\Documents\\Work\\logs\\read_eimpro\\today_fdvrun_demo\\Output_site114_8_12_2025_14_12_40.txt'
SEL_FDV = 'BASIC_PROGRAM_PAGE_READ_EIMPRO_DSLC|19|2.5|12|25'
ENTRY = 'EIMPRO_READ|19|K450917_753_-13_3|2.5|12|25'

print('Parsing file:', FILE)

rows = []
try:
    from process_fdv.core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT
    rows, _kept, _markers = process_file(Path(FILE), PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT)
except Exception as e:
    print('Failed to parse with process_fdv:', e)
    sys.exit(2)

print('Total rows:', len(rows))

# Inspect keys
if rows:
    print('Sample keys:', sorted(list(rows[0].keys()))[:40])

sel_fdvs = [SEL_FDV]
entry_parts = ENTRY.split('|')
entries = [tuple(entry_parts)]

recs = _build_variability_records(rows, sel_fdvs, entries)
print('Built records:', len(recs))

# Step-by-step diagnostics
fdv_ok = 0
pr_ok = 0
vcc_ok = 0
tm_ok = 0
temp_ok = 0
not_pr_monitor = 0
tname_ok = 0
fid_ok = 0
rber_ok = 0
wl_ok = 0

fdv_sel = _parse_fdv_selector(SEL_FDV)
print('Selector tuple:', fdv_sel)

# Decompose entry
tn = entry_parts[0]; pr=entry_parts[1]; fid=entry_parts[2]; svcc=entry_parts[3]; stm=entry_parts[4]; stemp=entry_parts[5]

for r in rows:
    k = _get_split_tuple(r)
    # fdv
    if fdv_sel[0] and k[0] != fdv_sel[0]:
        continue
    fdv_ok += 1
    # pr
    if fdv_sel[1] and k[1] != fdv_sel[1]:
        continue
    pr_ok += 1
    # vcc
    if fdv_sel[2] and k[2] != fdv_sel[2]:
        continue
    vcc_ok += 1
    # tm
    if fdv_sel[3] and k[3] != fdv_sel[3]:
        continue
    tm_ok += 1
    # temp
    if fdv_sel[4] and k[4] != fdv_sel[4]:
        continue
    temp_ok += 1
    # skip PR monitors
    if (r.get('tname','') or '').strip().upper() == 'PR':
        continue
    not_pr_monitor += 1
    # testname match (case-insensitive), using available fields
    tnr_raw = (r.get('testname','') or '').strip()
    if not tnr_raw:
        raw = (r.get('tname','') or '').strip()
        from process_fdv.core import derive_testname
        tnr_raw = derive_testname(raw) if raw else ''
    if (tnr_raw or '').strip().lower() != tn.strip().lower():
        continue
    tname_ok += 1
    # fid
    row_fid = (r.get('fuseid') or r.get('fuse_id') or r.get('fid') or r.get('chipid') or r.get('chip_id') or r.get('device_id') or '').strip()
    if fid and row_fid != fid:
        continue
    fid_ok += 1
    rv = _get_rber(r)
    if rv is None:
        continue
    rber_ok += 1
    wl = _extract_wl_or_page(r)
    if wl is None:
        continue
    wl_ok += 1

print('Counts by stage: fdv_ok', fdv_ok, 'pr_ok', pr_ok, 'vcc_ok', vcc_ok, 'tm_ok', tm_ok, 'temp_ok', temp_ok,
      'not_pr_monitor', not_pr_monitor, 'tname_ok', tname_ok, 'fid_ok', fid_ok, 'rber_ok', rber_ok, 'wl_ok', wl_ok)

# Also display unique split field values seen under the selector
vcc_vals = set()
tm_vals = set()
temp_vals = set()
for r in rows:
    k = _get_split_tuple(r)
    if fdv_sel[0] and k[0] != fdv_sel[0]:
        continue
    if fdv_sel[1] and k[1] != fdv_sel[1]:
        continue
    vcc_vals.add(k[2]); tm_vals.add(k[3]); temp_vals.add(k[4])
print('Unique VCC:', sorted(v for v in vcc_vals if v)[:10])
print('Unique TM:', sorted(v for v in tm_vals if v)[:10])
print('Unique TEMP:', sorted(v for v in temp_vals if v)[:10])

# If no recs, dump a few rows that passed fdv/pr/vcc/tm/temp and tname
if tname_ok == 0:
    print('Dumping 3 sample rows after fdv/pr/vcc/tm/temp filters:')
    cnt=0
    for r in rows:
        k = _get_split_tuple(r)
        if fdv_sel[0] and k[0] != fdv_sel[0]:
            continue
        if fdv_sel[1] and k[1] != fdv_sel[1]:
            continue
        if fdv_sel[2] and k[2] != fdv_sel[2]:
            continue
        if fdv_sel[3] and k[3] != fdv_sel[3]:
            continue
        if fdv_sel[4] and k[4] != fdv_sel[4]:
            continue
        print({kk: r.get(kk) for kk in ['tname','testname','fuseid','vcc','tm','temp','rber','wl','page'] if kk in r})
        cnt+=1
        if cnt>=3:
            break
