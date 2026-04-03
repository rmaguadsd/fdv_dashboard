#!/usr/bin/env python3
"""
Self-check: parse representative FDV OUTPUT lines and verify all fields
match what the guide_to_fdvlog.txt specifies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Load functions directly from fdv_chart.py
exec(compile(open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py').read(),
             'fdv_chart.py', 'exec'), globals())

PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if str(got) == str(expected):
        print(f"  PASS  {label}: {repr(got)}")
        PASS += 1
    else:
        print(f"  FAIL  {label}: got={repr(got)}  expected={repr(expected)}")
        FAIL += 1

# ─── Test 1: from actual log file ───────────────────────────────────────
line1 = (
    r"FDV OUTPUT [D:\NAND\150S\N59A\FDV\STAGING\RMAGUAD\PLC_BASIC_ERASE_PROGRAM_PAGE_READ"
    r"/BASIC_PROGRAM_PAGE_READ_PLC.FDV"
    r"::PROGRAM_STATUS_BLK_680_497_962_51_812_421_PAGE_3654_PAGETYPE_LP_WL_203_SB_0_BL_4_PASS1_STEP_49"
    r",SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.419000,SPECOFFSET=0.119,TM=15,VCC=2.5,VCCQ=1.2,TEMP=25]"
    r": DUT1,FAIL,8,8,1,8,0.13,0,|0:***:e0:e0:e1|7:e0:***:e0:e1,"
)
print("\n=== Test 1: PLC log line ===")
p = parse_fdv_output_line(line1)
assert p is not None, "PARSE FAILED - returned None"
check("dut",       p['dut'],       "DUT1")
check("result",    p['result'],    "FAIL")
check("testname",  p['testname'],  "PROGRAM_STATUS")
check("blk",       p['blk'],       "680,497,962,51,812,421")
check("page",      p['page'],      "3654")
check("pagetype",  p['pagetype'],  "LP")
check("wl",        p['wl'],        "203")
check("sb",        p['sb'],        "0")
check("bl",        p['bl'],        "4")
check("step",      p['step'],      "49")
check("vcc",       p['vcc'],       "2.5")
check("vccq",      p['vccq'],      "1.2")
check("temp",      p['temp'],      "25")
check("nbytes",    p['nbytes'],    "8")
check("nfailbytes",p['nfailbytes'],"8")
check("rber",      p['rber'],      "0.13")
check("fdv_file",  p['fdv_file'],  "BASIC_PROGRAM_PAGE_READ_PLC.FDV")

# ─── Test 2: guide example QLC line ─────────────────────────────────────
line2 = (
    r"FDV OUTPUT [D:\NAND\150S\FDV\STAGING\RMAGUAD/BASIC_ERASE_PROGRAM_PAGE_READ_EIMPRO"
    r"/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV"
    r"::READ_ECC_BLK_778_PG_37_PGTYPE_LP_WL_2_SB_1_BL_3"
    r",SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.725000,SPECOFFSET=0.125,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]"
    r": DUT1,PASS,18592,16,0.00086,16,0.00011,0.008,FAILCOUNT_ONLY,"
)
print("\n=== Test 2: guide QLC example ===")
p = parse_fdv_output_line(line2)
assert p is not None, "PARSE FAILED"
check("dut",       p['dut'],       "DUT1")
check("result",    p['result'],    "PASS")
check("testname",  p['testname'],  "READ_ECC")
check("blk",       p['blk'],       "778")
check("page",      p['page'],      "37")
check("pagetype",  p['pagetype'],  "LP")
check("pagemap",   p['pagemap'],   "QLC")
check("wl",        p['wl'],        "2")
check("sb",        p['sb'],        "1")
check("bl",        p['bl'],        "3")
check("nbytes",    p['nbytes'],    "18592")
check("nfailbytes",p['nfailbytes'],"16")
check("byber",     p['byber'],     "0.00086")
check("rber_limit",p['rber_limit'],"0.008")
check("faildata",  p['faildata'],  "FAILCOUNT_ONLY,")

# ─── Test 3: colon-delimited from guide ─────────────────────────────────
line3 = (
    r"FDV OUTPUT [D:\NAND\150S\FDV\FEATURE/FBM/FBM.FDV"
    r"::FBM_SSLC_READ_PAGE_MP_BLK:89_PG:49206_SSLC_LUN:0_SEQ_54"
    r",SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.718000,SPECOFFSET=0.118,TM=19,VCC=2.35,VCCQ=1.2,TEMP=25]"
    r": DUT2,PASS,18592,0,0,0,0,0.02,FAILCOUNT_ONLY,"
)
print("\n=== Test 3: colon-delimited BLK/PG ===")
p = parse_fdv_output_line(line3)
assert p is not None, "PARSE FAILED"
check("dut",       p['dut'],       "DUT2")
check("result",    p['result'],    "PASS")
check("blk",       p['blk'],       "89")
check("page",      p['page'],      "49206")
check("pagemap",   p['pagemap'],   "SSLC")
check("fdv_file",  p['fdv_file'],  "FBM.FDV")

# ─── Test 4: full file regex filter via parse_log_file ───────────────────
print("\n=== Test 4: parse_log_file with regex on real file ===")
headers, rows = parse_log_file(
    r"D:\FDV\logs\A1\PLC\Output_site114_4_01_2026_20_01_45.txt",
    regex_pattern=r"^FDV OUT.*WL.*SB.*BL.*",
    include_mode=True
)
print(f"  headers ({len(headers)}):", headers[:6], "...")
print(f"  total rows: {len(rows)}")
check("row count",    len(rows),         184992)
check("first line#",  rows[0][0],        "1698")
check("first DUT",    rows[0][1],        "DUT1")
check("first result", rows[0][2],        "FAIL")
check("first WL",     rows[0][headers.index('WL')],  "203")
check("first SB",     rows[0][headers.index('SB')],  "0")
check("first BL",     rows[0][headers.index('BL')],  "4")

# ─── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TOTAL: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED - see above")
