#!/usr/bin/env python3
"""
Patch fdv_chart.py: replace parse_log_file with a proper FDV OUTPUT parser
per guide_to_fdvlog.txt. Also replaces parse_fdv_output_line if present.

Run this script whenever fdv_chart.py needs to be regenerated.
"""

CHART = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py"

# ─── Read current file ───────────────────────────────────────────────────
content = open(CHART, "r", encoding="utf-8").read()
print(f"File: {len(content)} chars")

# ─── New FDV parser functions ────────────────────────────────────────────
NEW_PARSE = """
def parse_fdv_output_line(line_stripped):
    import re
    m = re.match(r'^FDV OUTPUT \\[(.+?)\\]:\\s*(.+)$', line_stripped)
    if not m:
        return None
    bracket, result_section = m.group(1), m.group(2).strip()
    bm = re.match(r'^(.+?\\.FDV)::([^,]+),(.+)$', bracket, re.IGNORECASE)
    if not bm:
        return None
    fdv_file = bm.group(1).replace('\\\\', '/').split('/')[-1]
    tname    = bm.group(2).strip()
    cond_str = bm.group(3).strip()
    cond = {}
    for tok in cond_str.split(','):
        if '=' in tok:
            k, v = tok.split('=', 1)
            cond[k.strip().upper()] = v.strip()
    def ep(s, *keys):
        for key in keys:
            hm = re.search(rf'(?:^|_){key}[_:](\\d+(?:[_\\d]*\\d)?)', s, re.IGNORECASE)
            if hm:
                raw = hm.group(1)
                # collect all consecutive numeric tokens for multi-BLK
                nums = re.findall(r'\\d+', raw)
                return ','.join(nums)
        return ''
    blk      = ep(tname, 'BLK')
    page     = ep(tname, 'PAGE', 'PG')
    wl       = ep(tname, 'WL')
    sb       = ep(tname, 'SB')
    bl       = ep(tname, 'BL')
    step     = ep(tname, 'STEP')
    pt_m = re.search(r'(?:PAGETYPE|PGTYPE)[_:]([A-Za-z0-9]+)', tname, re.IGNORECASE)
    pagetype = pt_m.group(1) if pt_m else ''
    pagemap  = ''
    for pm in ('QLC','TLC','DSLC','SSLC','MLC','SLC'):
        if re.search(rf'(?:^|_){pm}(?:_|$)', tname, re.IGNORECASE) or \
           re.search(rf'(?:^|_){pm}(?:_|$)', fdv_file, re.IGNORECASE):
            pagemap = pm; break
    deck = ''
    for dk in ('UD','LD','MD'):
        if re.search(rf'(?:^|_){dk}(?:_|$)', tname, re.IGNORECASE):
            deck = dk; break
    tpk = r'(?:BLK|PAGE|PG|PAGETYPE|PGTYPE|WL|SB|BL|STEP|UD|LD|MD|QLC|TLC|MLC|SSLC|DSLC|SLC|LUN)'
    tn_m = re.match(rf'^(.*?)(?:_(?={tpk}[_:\\d])|$)', tname, re.IGNORECASE)
    testname = (tn_m.group(1).strip('_') if tn_m else '').strip('_') or tname
    rp = result_section.split(',')
    def rg(i): return rp[i].strip() if i < len(rp) else ''
    return dict(
        fdv_file=fdv_file, tname=tname, testname=testname,
        dut=rg(0), result=rg(1),
        blk=blk, page=page, pagetype=pagetype, pagemap=pagemap,
        wl=wl, sb=sb, bl=bl, step=step, deck=deck,
        vcc=cond.get('VCC',''), vccq=cond.get('VCCQ',''),
        temp=cond.get('TEMP',''), tac=cond.get('TAC',''), tm=cond.get('TM',''),
        nbytes=rg(2), nfailbytes=rg(3), byber=rg(4),
        nfailbits=rg(5), rber=rg(6), rber_limit=rg(7),
        faildata=','.join(rp[8:]).strip() if len(rp) > 8 else '',
    )


def parse_log_file(file_path, regex_pattern=None, include_mode=True):
    import re
    headers = [
        'Line#','DUT','Result','tname','testname',
        'BLK','PAGE','PAGETYPE','PAGEMAP',
        'WL','SB','BL','STEP','DECK',
        'VCC','VCCQ','TEMP','TAC','TM',
        'nBytes','nFailBytes','BYBER','nFailBits','RBER','RBER_Limit',
        'FDV_File','FailData',
    ]
    rows = []
    compiled_regex = None
    if regex_pattern:
        try:
            compiled_regex = re.compile(regex_pattern)
        except re.error as e:
            raise Exception("Invalid regex: " + str(e))
    max_lines = 500000
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                if line_count > max_lines:
                    rows.append(['...']*len(headers)); break
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                if compiled_regex:
                    m = compiled_regex.search(line_stripped)
                    if include_mode and not m: continue
                    if not include_mode and m: continue
                p = parse_fdv_output_line(line_stripped)
                if p:
                    row = [
                        str(line_num), p['dut'], p['result'], p['tname'], p['testname'],
                        p['blk'], p['page'], p['pagetype'], p['pagemap'],
                        p['wl'], p['sb'], p['bl'], p['step'], p['deck'],
                        p['vcc'], p['vccq'], p['temp'], p['tac'], p['tm'],
                        p['nbytes'], p['nfailbytes'], p['byber'],
                        p['nfailbits'], p['rber'], p['rber_limit'],
                        p['fdv_file'], p['faildata'],
                    ]
                else:
                    row = [str(line_num),'','',line_stripped]+['']*(len(headers)-4)
                rows.append(row)
    except Exception as e:
        raise Exception("Error parsing: " + str(e))
    return headers, rows
"""

# ─── Replace old parse_log_file (and parse_fdv_output_line if present) ──
import re as _re

# Remove existing parse_fdv_output_line if present
if 'def parse_fdv_output_line' in content:
    content = _re.sub(
        r'\ndef parse_fdv_output_line\(.*?\n(?=\ndef )',
        '\n', content, flags=_re.DOTALL)
    print("Removed old parse_fdv_output_line")

# Replace parse_log_file up to next top-level def/class
# Use lambda to avoid re interpreting backslashes in replacement string
content = _re.sub(
    r'\ndef parse_log_file\(.*?\n(?=\ndef |\nclass )',
    lambda m: '\n' + NEW_PARSE + '\n',
    content, flags=_re.DOTALL)

open(CHART, "w", encoding="utf-8").write(content)
verify = open(CHART, "r", encoding="utf-8").read()
print("parse_fdv_output_line present:", "def parse_fdv_output_line" in verify)
print("parse_log_file present:", "def parse_log_file" in verify)
print("get_html present:", "def get_html" in verify)
print("RequestHandler present:", "class RequestHandler" in verify)
lines_out = verify.count("\n")
print(f"Output: {len(verify)} chars, {lines_out} lines")
print("DONE")
print("Patch confirmed:", "skip leading empty element" in verify)
