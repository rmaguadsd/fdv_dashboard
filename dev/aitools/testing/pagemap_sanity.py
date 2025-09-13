import json, os
from pathlib import Path
import sys
sys.path.insert(0, r'd:\dev\aitools')
from process_fdv.core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT

def parse_line(s):
    p = Path('d:/dev/aitools/testing/tmp_fdv_line.log')
    p.write_text(s, encoding='utf-8')
    rows, kept, markers = process_file(p, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT)
    return rows

samples = [
    "FDV OUTPUT [D:/NAND/FDV/FBM/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV::READ_ECC_BLK_778_PG_37_PGTYPE_LP_WL_2_SB_1_BL_3,SSYNC=TRUE,TM=12,VCC=2.5,TEMP=25]: DUT1,PASS,18592,16,0.00086,16,0.00011,0.008,FAILCOUNT_ONLY,\n",
    "FDV OUTPUT [D:/NAND/FDV/FEATURE/FBM/FBM.FDV::FBM_SSLC_READ_PAGE_MP_BLK:89_PG:49206_SSLC_LUN:0_SEQ_54,SSYNC=TRUE,TM=19,VCC=2.35,TEMP=25]: DUT2,PASS,18592,0,0,0,0,0.02,FAILCOUNT_ONLY,\n",
    "FDV OUTPUT [D:/NAND/FDV/FEATURE/preamble/N59A_QLC_POWERUP_VPPON.FDV::READ_BLK:1_PG:2_PAGETYPE:SSLC_WL:3,SSYNC=TRUE,TM=12,VCC=2.5,TEMP=25]: DUT2,PASS,1,0,0,0,0,0.0,\n",
    "FDV OUTPUT [D:/NAND/FDV/FEATURE/MLC_SOMETHING.FDV::READ_ECC_BLK_1_PG_2_PT_UP_WL_3,SSYNC=TRUE,TM=12,VCC=2.5,TEMP=25]: DUT1,PASS,1,0,0,0,0,0.0,\n",
]

for s in samples:
    rows = parse_line(s)
    for r in rows:
        print(json.dumps({k:r.get(k,'') for k in ('fdv_file','tname','pagetype','product','pagemap')}, sort_keys=True))
