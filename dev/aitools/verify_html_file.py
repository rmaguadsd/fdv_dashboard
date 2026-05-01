#!/usr/bin/env python3
with open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart_rev6\fdv_chart.html', 'r', encoding='utf-8') as f:
    content = f.read()
    # Find both possible strings
    idx_choose = content.find('-- Choose --')
    idx_point_sel = content.find('value="point" selected')
    print(f'File size: {len(content)}')
    print(f'Has "-- Choose --": {idx_choose >= 0} (at {idx_choose})')
    print(f'Has "value=\"point\" selected": {idx_point_sel >= 0} (at {idx_point_sel})')
    print()
    print('Context around font-item-select:')
    idx_select = content.find('font-item-select')
    if idx_select > 0:
        print(content[idx_select:idx_select+350])
