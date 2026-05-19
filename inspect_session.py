import json

file = r'D:\FDV\recipes\babysteps2.fdv_session'
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

headers = data.get('headers', [])
rows = data.get('rows', [])

print(f'Headers count: {len(headers)}')
print(f'Headers: {headers}')
print(f'Row count: {len(rows)}')
if rows:
    print(f'First row length: {len(rows[0])}')
    print(f'First row values: {rows[0]}')
