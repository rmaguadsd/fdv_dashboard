"""Self-test for VECTOR semantics.

Assertions:
1. VECTOR == number of raw lines (before filtering) whose line starts with 'FDV OUTPUT' (case-insensitive, leading spaces ignored) per fdv test key.
2. VECTOR >= PASS+FAIL (since pass/fail are a subset of FDV OUTPUT lines that are not MONITOR/SHMOO and have valid FUSEID + numeric RBER).
3. PASS+FAIL counts are unaffected by MONITOR/SHMOO lines.

This script fabricates synthetic rows covering:
- Valid PASS/FAIL lines
- MONITOR and SHMOO lines (should count toward VECTOR if they start with FDV OUTPUT, but not toward pass/fail)
- Invalid fuseid rows (should not contribute to pass/fail but still to VECTOR)
- Mixed keys

Run:
    python -m aitools.testing.selftest_vector_semantics
"""
from __future__ import annotations
import sys
from typing import List, Dict

# Import the function under test
try:
    from aitools.fdv_report2_webapp import stats_by_fdv_with_splits  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"Import failed: {e}")
    sys.exit(1)

# Helper to build FDV OUTPUT raw lines

def make_line(dut: int, verdict: str, extra: str = '', fuseid: str | None = None, rber: float | None = 1.23) -> Dict[str,str]:
    # Provide a valid fuseid unless overridden: K123456_<dut>_1_0
    if fuseid is None:
        fuseid = f"K123456_{dut}_1_0"
    parts = [f"DUT{dut}", verdict]
    if extra:
        parts.append(extra)
    core = ','.join(parts)
    rl = f"FDV OUTPUT [{dut}]: {core}"
    row: Dict[str,str] = {
        'raw_line': rl,
        'fdv_file': 'testA.fdv',
        'pr': 'P1',
        'vcc': '3.3',
        'tm': '25C',
        'temp': '25',
        'dut_id': str(dut),
        'fuseid': fuseid,
        'line_number': '1',
    }
    if rber is not None:
        row['rber'] = f"{rber}"
    return row

rows: List[Dict[str,str]] = []
# 3 PASS, 2 FAIL
rows += [make_line(1,'PASS'), make_line(2,'PASS'), make_line(3,'PASS')]
rows += [make_line(1,'FAIL'), make_line(2,'FAIL')]
# 2 MONITOR (should not add to pass/fail)
rows += [make_line(4,'MONITOR'), make_line(5,'MONITOR')]
# 1 SHMOO example (represented as verdict token SHMOO) -> treat same as MONITOR logic in existing code
rows += [make_line(6,'SHMOO')]
# 2 invalid fuseid rows (bad fuseid so excluded from pass/fail but counted in VECTOR)
rows += [make_line(7,'PASS', fuseid='INVALID123'), make_line(8,'FAIL', fuseid='BADXYZ')]
# 1 row without numeric rber (removed rber) still counts for VECTOR
rows.append(make_line(9,'PASS'))
del rows[-1]['rber']

# Another key (different PR) with only monitor lines -> expect synthetic / placeholder row with vector count
rows += [
    { **make_line(1,'MONITOR'), 'pr':'P2' },
    { **make_line(2,'MONITOR'), 'pr':'P2' },
]

stats = stats_by_fdv_with_splits(rows, limit=0.0, passfail_mode=False)

# Build raw expectations per key
from collections import defaultdict
expected_vector: Dict[tuple,str] = defaultdict(int)  # type: ignore[type-arg]
pass_fail_counts: Dict[tuple, Dict[str,int]] = defaultdict(lambda: {'pass':0,'fail':0})

for r in rows:
    key = (r.get('fdv_file'), r.get('pr'), r.get('vcc'), r.get('tm'), r.get('temp'))
    rl = (r.get('raw_line') or '').lstrip().upper()
    if rl.startswith('FDV OUTPUT'):
        expected_vector[key] += 1
    verdict = rl.split(',')[1] if ',' in rl else ''
    if verdict in ('PASS','FAIL') and r.get('rber') and 'INVALID' not in (r.get('fuseid') or '').upper() and 'BAD' not in (r.get('fuseid') or '').upper():
        if verdict == 'PASS':
            pass_fail_counts[key]['pass'] += 1
        elif verdict == 'FAIL':
            pass_fail_counts[key]['fail'] += 1

errors = []
for s in stats:
    key = (s['fdv_file'], s['pr'], s['vcc'], s['tm'], s['temp'])
    vec = int(s.get('vector') or '0') if s.get('vector') else 0
    cnt = int(s.get('count') or '0') if s.get('count') else 0
    exp_vec = expected_vector[key]
    pf_sum = pass_fail_counts[key]['pass'] + pass_fail_counts[key]['fail']
    if vec != exp_vec:
        errors.append(f"Key {key}: vector={vec} expected={exp_vec}")
    if vec < pf_sum:
        errors.append(f"Key {key}: vector={vec} < pass+fail={pf_sum}")
    if cnt != pf_sum:
        errors.append(f"Key {key}: count={cnt} expected_pass_fail_sum={pf_sum}")

if errors:
    print('SELFTEST VECTOR SEMANTICS: FAIL')
    for e in errors:
        print(' -', e)
    sys.exit(1)
else:
    print('SELFTEST VECTOR SEMANTICS: PASS')
    for s in stats:
        key = (s['fdv_file'], s['pr'], s['vcc'], s['tm'], s['temp'])
        print(f"  {key}: vector={s.get('vector')} count={s.get('count')} pass={s.get('pass_n')} fail={s.get('fail_n')}")
