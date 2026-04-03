#!/usr/bin/env python3
"""
Patch fdv_chart.py in-place:
1. Move plot section ABOVE the table
2. Expand table height to calc(100vh - 40px)  
3. Auto-draw plot after parse
4. Increase server display cap from 1000 to 5000
5. Auto-select WL/RBER defaults (already present, verify)
"""

CHART = r"d:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.py"

content = open(CHART, 'r', encoding='utf-8').read()
orig_len = len(content)

# ─── 1. Table max-height: 380px → full viewport height ────────────────────
old = '#table-container{overflow:auto;max-height:380px;border:1px solid #ddd;border-radius:4px;display:none;margin:12px 0}'
new = '#table-container{overflow:auto;max-height:calc(100vh - 40px);border:1px solid #ddd;border-radius:4px;display:none;margin:8px 0}'
assert old in content, f"MISSING: {old[:60]}"
content = content.replace(old, new, 1)
print("1. table max-height → 100vh")

# ─── 2. Move plot section ABOVE table ──────────────────────────────────────
# Current order: table-container → dl-row → plot-section
# Target order:  plot-section → table-container → dl-row
TABLE_BLOCK = '''<div id="table-container">
  <table id="result-table"></table>
</div>

<div style="margin:6px 0;display:none" id="dl-row">
  <button class="sec" onclick="downloadCSV()">&#8595; Download CSV</button>
</div>

<!-- ===== XY PLOT SECTION ===== -->'''

PLOT_START_MARKER = '<!-- ===== XY PLOT SECTION ===== -->'
DOWNLOAD_BTN_MARKER = '''<button id="download-btn" onclick="downloadCSV()" style="padding:10px 20px;cursor:pointer;display:none">Download CSV</button>'''

# Find the table block and plot-section block, swap them
idx_plot_comment = content.find('<!-- ===== XY PLOT SECTION ===== -->')
idx_table_div = content.find('<div id="table-container">')

if idx_plot_comment > 0 and idx_table_div > 0 and idx_table_div < idx_plot_comment:
    # Find end of plot section (the </div> that closes plot-section + download-btn)
    idx_dl_row = content.find('<div style="margin:6px 0;display:none" id="dl-row">')
    idx_dl_end = content.find('</div>', idx_dl_row) + len('</div>')

    # Find end of entire XY plot section block
    # It ends with the download-btn button line 
    idx_dlbtn = content.find('<button id="download-btn"')
    idx_dlbtn_end = content.find('\n', idx_dlbtn) + 1

    # Extract the three blocks:
    # Block A: table-container + dl-row
    block_a = content[idx_table_div:idx_dl_end].rstrip()
    # Block B: comment + plot section div
    block_b_start = content.find('\n\n<!-- ===== XY PLOT SECTION', idx_dl_end - 5)
    block_b = content[block_b_start:idx_dlbtn_end].rstrip()

    # New arrangement: block_b (plot) then block_a (table+dl)
    old_section = block_a + '\n\n' + block_b
    new_section = '\n\n<!-- ===== XY PLOT SECTION ===== -->\n' + \
                  content[content.find('\n<div id="plot-section">', idx_dl_end)+1:idx_dlbtn_end].rstrip() + \
                  '\n\n' + block_a

    if old_section in content:
        content = content.replace(old_section, new_section, 1)
        print("2. plot section moved above table")
    else:
        print("2. SKIP: structural swap not applied (sections not in expected order)")
else:
    print("2. SKIP: plot already above table or markers not found")

# ─── 3. Auto-draw plot after parse ─────────────────────────────────────────
old3 = "    // populate plot column selectors\n    populatePlotSelectors(result.headers);\n    document.getElementById('plot-section').style.display='block';\n\n  }catch"
new3 = "    // populate plot column selectors\n    populatePlotSelectors(result.headers);\n    document.getElementById('plot-section').style.display='block';\n    // auto-draw with smart defaults\n    setTimeout(drawPlot, 250);\n\n  }catch"
if old3 in content:
    content = content.replace(old3, new3, 1)
    print("3. auto-draw setTimeout added")
else:
    print("3. SKIP: auto-draw already present or pattern not found")

# ─── 4. Server display cap 1000 → 5000 rows ────────────────────────────────
old4 = "display_rows = rows[:1000] if len(rows) > 1000 else rows"
new4 = "display_rows = rows[:5000] if len(rows) > 5000 else rows"
if old4 in content:
    content = content.replace(old4, new4, 1)
    print("4. display cap 1000→5000")
else:
    print("4. SKIP: display cap already updated or not found")

# ─── 5. Table info bar showing total vs displayed ─────────────────────────
old5 = "    showMessage('Parsed '+result.total_rows+' rows (showing first '+result.rows.length+' in table)','ok');\n    showInfo('<b>Total rows:</b> '+result.total_rows+"
new5 = "    const tot=result.total_rows, shown=result.rows.length;\n    showMessage('Parsed '+tot.toLocaleString()+' rows','ok');\n    showInfo('<b>Total rows:</b> '+tot.toLocaleString()+"
if old5 in content:
    content = content.replace(old5, new5, 1)
    print("5. info bar updated")
else:
    print("5. SKIP: info bar already updated")

# ─── Write back ────────────────────────────────────────────────────────────
open(CHART, 'w', encoding='utf-8').write(content)
print(f"\nFile: {orig_len} → {len(content)} chars")

# ─── Verify syntax ─────────────────────────────────────────────────────────
import py_compile, sys
try:
    py_compile.compile(CHART, doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR:", e)
    sys.exit(1)

# ─── Spot-check key strings ───────────────────────────────────────────────
lines = open(CHART).readlines()
print(f"Lines: {len(lines)}")
checks = {
    'table 100vh':          any('100vh' in l for l in lines),
    'auto-draw setTimeout': any('setTimeout(drawPlot' in l for l in lines),
    'display cap 5000':     any('5000' in l and 'display' in l for l in lines),
    'WL default':           any("'WL'" in l for l in lines),
    'RBER default':         any("'RBER'" in l for l in lines),
    'multipart i==0 fix':   any('i == 0' in l for l in lines),
    'plot-section':         any('plot-section' in l for l in lines),
}
all_ok = True
for k,v in checks.items():
    print(f"  {'OK  ' if v else 'MISS'} {k}")
    if not v: all_ok=False
if all_ok:
    print("\nAll checks passed!")
else:
    print("\nWARNING: some checks failed")
