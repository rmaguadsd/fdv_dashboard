import json
import sys

file = r'D:\FDV\recipes\n59a_a2_pr36_rel005_25c_ppsr_hote.fdv_session'
try:
    print(f'Loading {file}...')
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    headers = data.get('headers', [])
    rows = data.get('rows', [])
    
    print(f'Headers count: {len(headers)}')
    print(f'Headers: {headers}')
    print(f'Row count: {len(rows)}')
    
    if rows:
        print(f'First row length: {len(rows[0])}')
        print(f'First row: {rows[0][:5]}...')  # First 5 values
        
        # Check for length mismatches
        row_lengths = set()
        for i, row in enumerate(rows):
            row_lengths.add(len(row))
            if i > 1000:  # Sample first 1000 rows
                break
        
        print(f'Row length variations (first 1000): {sorted(row_lengths)}')
        
        if len(row_lengths) > 1:
            print('ERROR: Rows have inconsistent lengths!')
            for length in sorted(row_lengths):
                matching_rows = [i for i, row in enumerate(rows[:1000]) if len(row) == length]
                print(f'  Length {length}: {len(matching_rows)} rows (e.g., index {matching_rows[0]})')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
