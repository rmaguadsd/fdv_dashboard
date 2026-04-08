"""
patch_multidim.py
Apply multi-dimension color-by / split-chart grouping to fdv_chart.html.

Changes:
  1. CSS  — .gdim-wrap / .gdim-row / .gdim-rx / .gdim-add / .gdim-del
  2. HTML — Color-by: replace label+select+regex with multi-dim builder span
  3. HTML — Split-by (cumproba): replace label+select+regex with multi-dim builder span
  4. HTML — Split-chart: add "+" button next to the existing select (keep select intact)
  5. JS   — _colorDims / _splitDims / _scDims globals
  6. JS   — _compoundKey(), _gdimAdd(), _gdimDel(), _gdimRead(),
            _gdimChanged(), _gdimSyncLegacy(), _gdimRebuildSelects(),
            _gdimSetDims(), _gdimInit()
  7. JS   — colorKey() in drawScatterLine uses _colorDims when set
  8. JS   — groupKey() in drawCumProba uses _splitDims when set
  9. JS   — _splitChartKeys updated to accept multi-dim via _scDims
 10. JS   — populatePlotSelectors calls _gdimRebuildSelects
 11. JS   — DOMContentLoaded calls _gdimInit
"""

TARGET = r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.html'

with open(TARGET, encoding='utf-8') as f:
    src = f.read()

errors = []
ok     = []

def patch(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        errors.append(f'NOT FOUND: {label}')
        return
    if count == 1 and n > 1:
        errors.append(f'AMBIGUOUS ({n} matches): {label}')
        return
    src = src.replace(old, new, count)
    ok.append(f'OK: {label}')

# ─────────────────────────────────────────────────────────────────
# 1. CSS
# ─────────────────────────────────────────────────────────────────
patch('CSS gdim rules',
old='''        #summary-btn:disabled { opacity: 0.55; cursor: not-allowed; }''',
new='''        #summary-btn:disabled { opacity: 0.55; cursor: not-allowed; }
        /* ── Multi-dimension group builder ── */
        .gdim-wrap { display:flex; flex-direction:column; gap:2px; }
        .gdim-row  { display:flex; align-items:center; gap:3px; }
        .gdim-row select {
            padding:2px 3px; font-size:0.83em; border:1px solid #ccc;
            border-radius:3px; max-width:130px; min-width:60px;
        }
        .gdim-rx {
            padding:2px 3px; font-size:0.83em; width:88px;
            border:1px solid #ccc; border-radius:3px; font-family:monospace;
        }
        .gdim-rx::placeholder { color:#bbb; font-style:italic; }
        .gdim-add { padding:1px 6px; font-size:0.78em; background:#28a745;
                    color:#fff; border:none; border-radius:3px; cursor:pointer; }
        .gdim-add:hover { background:#1e7e34; }
        .gdim-del { padding:1px 5px; font-size:0.78em; background:#dc3545;
                    color:#fff; border:none; border-radius:3px; cursor:pointer; }
        .gdim-del:hover { background:#a71d2a; }''')

# ─────────────────────────────────────────────────────────────────
# 2. HTML — Color-by
# ─────────────────────────────────────────────────────────────────
patch('HTML color-by builder',
old='''        <label id="color-col-label" style="margin-left:8px">Color&nbsp;by:
            <select id="color-col"><option value="">-- none --</option></select></label>
        <label id="color-rx-label" style="margin-left:4px">Color&nbsp;regex:
            <input type="text" id="color-regex" placeholder="e.g. DUT(\\d+)" autocomplete="off" style="width:110px"></label>''',
new='''        <span id="color-col-label" style="margin-left:8px;display:inline-flex;align-items:flex-start;gap:4px">
            <span style="font-size:0.82em;color:#495057;white-space:nowrap;padding-top:3px">Color&nbsp;by:</span>
            <span id="color-dims-wrap" class="gdim-wrap"><!-- rows injected by _gdimInit --></span>
            <button class="gdim-add" onclick="_gdimAdd('color')" title="Add another color-by dimension">+</button>
        </span>
        <!-- legacy ids kept for recipe snap compat -->
        <select id="color-col" style="display:none"></select>
        <input  id="color-regex" style="display:none">
        <span   id="color-rx-label" style="display:none"></span>''')

# ─────────────────────────────────────────────────────────────────
# 3. HTML — Split-by (cumproba)
# ─────────────────────────────────────────────────────────────────
patch('HTML split-by builder',
old='''        <label id="split-label" style="margin-left:8px;display:none">Split&nbsp;by:
            <select id="split-col"><option value="">-- none --</option></select></label>
        <label id="split-rx-label" style="margin-left:4px;display:none">Split&nbsp;regex:
            <input type="text" id="split-regex" placeholder="e.g. DUT(\\d+)" autocomplete="off" style="width:110px"></label>''',
new='''        <span id="split-label" style="margin-left:8px;display:none;align-items:flex-start;gap:4px">
            <span style="font-size:0.82em;color:#495057;white-space:nowrap;padding-top:3px">Split&nbsp;by:</span>
            <span id="split-dims-wrap" class="gdim-wrap"><!-- rows injected by _gdimInit --></span>
            <button class="gdim-add" onclick="_gdimAdd('split')" title="Add another split-by dimension">+</button>
        </span>
        <select id="split-col"  style="display:none"></select>
        <input  id="split-regex" style="display:none">
        <span   id="split-rx-label" style="display:none"></span>''')

# ─────────────────────────────────────────────────────────────────
# 4. HTML — Split-chart: add "+" button next to existing select
# ─────────────────────────────────────────────────────────────────
patch('HTML split-chart + button',
old='''            <label style="font-size:0.82em;color:#17a2b8">By:
                <select id="split-chart-col" onchange="_onSplitColChange()" style="border-color:#17a2b8">
                    <option value="">-- none (single chart) --</option>
                </select></label>
            <label style="font-size:0.82em;color:#17a2b8">Regex:
                <input type="text" id="split-chart-rx" placeholder="e.g. DUT(\\d+)" autocomplete="off"
                       style="width:110px;border-color:#17a2b8" onchange="_onSplitColChange()"
                       title="Optional regex applied to each cell&#10;• One capture group → use that group as tile key  e.g. DUT(\\d+)&#10;• No capture group → use full match as tile key&#10;• No regex → split cell on commas  e.g. A,B → two tiles&#10;• Regex matching multiple times in one cell → row goes into ALL matched tiles"></label>''',
new='''            <label style="font-size:0.82em;color:#17a2b8">By:
                <select id="split-chart-col" onchange="_onSplitColChange()" style="border-color:#17a2b8">
                    <option value="">-- none (single chart) --</option>
                </select></label>
            <label style="font-size:0.82em;color:#17a2b8">Regex:
                <input type="text" id="split-chart-rx" placeholder="e.g. DUT(\\d+)" autocomplete="off"
                       style="width:110px;border-color:#17a2b8" onchange="_onSplitColChange()"
                       title="Optional regex applied to each cell&#10;• One capture group → use that group as tile key  e.g. DUT(\\d+)&#10;• No capture group → use full match as tile key&#10;• No regex → split cell on commas  e.g. A,B → two tiles&#10;• Regex matching multiple times in one cell → row goes into ALL matched tiles"></label>
            <!-- Extra split-chart dims -->
            <span id="sc-dims-wrap" class="gdim-wrap" style="display:inline-flex;flex-direction:row;flex-wrap:wrap;gap:3px;align-items:center">
            </span>
            <button class="gdim-add" onclick="_gdimAdd('sc')" title="Add an extra split-chart dimension (combined with the By column above)">+</button>''')

# ─────────────────────────────────────────────────────────────────
# 5–10. JS block — insert before _extractGroupKey
# ─────────────────────────────────────────────────────────────────
patch('JS multidim engine',
old='''/* Extracts a single color/group key from a raw string value using a regex.
   Like _splitChartKeys but returns ONE compound key string (not an array).
   Multiple capture groups are joined with ", " — e.g. PGTYPE_(..)_.*_BL_(\\d)
   on "BLAH_PGTYPE_UP_BLAH_BL_3" → "UP, 3".
   No regex: returns raw value.
   No capture groups: returns full match. */''',
new='''/* ================================================================
   MULTI-DIMENSION GROUP BUILDER
   _colorDims  → used by colorKey()  in scatter / line
   _splitDims  → used by groupKey()  in cumproba / histogram
   _scDims     → extra dims added to split-chart-col

   Each dim: { col: colName, colIdx: int, rx: regexStr }
   Compound key = dim0_result " | " dim1_result …
================================================================ */
var _colorDims = [];   /* color-by dims */
var _splitDims = [];   /* split-by (cumproba) dims */
var _scDims    = [];   /* extra split-chart dims */

/** Build compound key from one allRows[] row + dims array */
function _compoundKey(row, dims) {
    var parts = [];
    for (var d = 0; d < dims.length; d++) {
        var dim = dims[d];
        if (dim.colIdx < 0) continue;
        var raw = row[dim.colIdx] != null ? String(row[dim.colIdx]) : '';
        parts.push(_extractGroupKey(raw, dim.rx));
    }
    return parts.length ? parts.join(' \u2502 ') : null;
}

/** Read dim rows from a builder wrap into an array of {col,colIdx,rx} */
function _gdimRead(type) {
    var wrap = document.getElementById(type + '-dims-wrap');
    if (!wrap) return [];
    var dims = [];
    wrap.querySelectorAll('.gdim-row').forEach(function(row) {
        var sel = row.querySelector('select');
        var inp = row.querySelector('.gdim-rx');
        var col = sel ? sel.value : '';
        var rx  = inp ? inp.value.trim() : '';
        dims.push({ col: col, colIdx: col ? (currentHeaders.indexOf(col)) : -1, rx: rx });
    });
    return dims;
}

/** Add one dim row to a builder */
function _gdimAdd(type) {
    var wrap = document.getElementById(type + '-dims-wrap');
    if (!wrap) return;
    var opts = '<option value="">\u2014 col \u2014</option>';
    currentHeaders.forEach(function(h) {
        opts += '<option value="' + escHtml(h) + '">' + escHtml(h) + '</option>';
    });
    var row = document.createElement('div');
    row.className = 'gdim-row';
    row.innerHTML = '<select onchange="_gdimChanged(\'' + type + '\')">' + opts + '</select>'
        + '<input type="text" class="gdim-rx" placeholder="regex\u2026"'
        + ' title="Optional regex — capture group = extracted key, no group = full match"'
        + ' oninput="_gdimChanged(\'' + type + '\')">'
        + '<button class="gdim-del" onclick="_gdimDel(this,\'' + type + '\')"'
        + ' title="Remove this dimension">\u00d7</button>';
    wrap.appendChild(row);
    _gdimChanged(type);
}

/** Remove a dim row */
function _gdimDel(btn, type) {
    btn.parentNode.parentNode.removeChild(btn.parentNode);
    _gdimChanged(type);
}

/** Sync dim arrays + legacy hidden inputs */
function _gdimChanged(type) {
    if      (type === 'color') _colorDims = _gdimRead('color');
    else if (type === 'sc')    _scDims    = _gdimRead('sc');
    else                       _splitDims = _gdimRead('split');
    _gdimSyncLegacy(type);
}

function _gdimSyncLegacy(type) {
    var map = {
        color: { col: 'color-col', rx: 'color-regex',  dims: '_colorDims' },
        split: { col: 'split-col', rx: 'split-regex',  dims: '_splitDims' },
        sc:    { col: 'split-chart-col', rx: 'split-chart-rx', dims: '_scDims' }
    };
    var m = map[type]; if (!m) return;
    var dims = (type==='color') ? _colorDims : (type==='sc') ? _scDims : _splitDims;
    /* Only update the legacy hidden inputs — the sc select is real so skip for sc */
    if (type !== 'sc') {
        var cEl = document.getElementById(m.col);
        var rEl = document.getElementById(m.rx);
        if (cEl) cEl.value = dims.length ? (dims[0].col || '') : '';
        if (rEl) rEl.value = dims.length ? (dims[0].rx  || '') : '';
    }
}

/** Rebuild all selects when headers change */
function _gdimRebuildSelects(headers) {
    ['color-dims-wrap', 'split-dims-wrap', 'sc-dims-wrap'].forEach(function(wid) {
        var wrap = document.getElementById(wid);
        if (!wrap) return;
        wrap.querySelectorAll('.gdim-row select').forEach(function(sel) {
            var prev = sel.value;
            var opts = '<option value="">\u2014 col \u2014</option>';
            headers.forEach(function(h) {
                opts += '<option value="' + escHtml(h) + '"'
                      + (h === prev ? ' selected' : '') + '>' + escHtml(h) + '</option>';
            });
            sel.innerHTML = opts;
            sel.value = prev;
        });
    });
    /* Re-read dims after options are rebuilt */
    _colorDims = _gdimRead('color');
    _splitDims = _gdimRead('split');
    _scDims    = _gdimRead('sc');
}

/** Restore a builder from a saved dims array */
function _gdimSetDims(type, dims) {
    var wrap = document.getElementById(type + '-dims-wrap');
    if (!wrap) return;
    wrap.innerHTML = '';
    dims.forEach(function(dim) {
        _gdimAdd(type);
        var rows = wrap.querySelectorAll('.gdim-row');
        var last = rows[rows.length - 1];
        var sel  = last.querySelector('select');
        var inp  = last.querySelector('.gdim-rx');
        if (sel && dim.col) sel.value = dim.col;
        if (inp) inp.value = dim.rx || '';
    });
    _gdimChanged(type);
}

/** Create the initial single-row builders */
function _gdimInit() {
    _gdimAdd('color');
    _gdimAdd('split');
    /* sc-dims-wrap is additive — starts empty, user clicks + to add */
}

/* Extracts a single color/group key from a raw string value using a regex.
   Like _splitChartKeys but returns ONE compound key string (not an array).
   Multiple capture groups are joined with ", " — e.g. PGTYPE_(..)_.*_BL_(\\d)
   on "BLAH_PGTYPE_UP_BLAH_BL_3" → "UP, 3".
   No regex: returns raw value.
   No capture groups: returns full match. */''')

# ─────────────────────────────────────────────────────────────────
# 7. colorKey in drawScatterLine uses _colorDims when set
# ─────────────────────────────────────────────────────────────────
patch('colorKey uses _colorDims',
old='''    var scatterColIdx = colCol ? currentHeaders.indexOf(colCol) : -1;
    function colorKey(pt) {
        if (!colCol) return '(all)';
        var raw = (scatterColIdx >= 0 && pt._ri != null && allRows[pt._ri] != null)
            ? (allRows[pt._ri][scatterColIdx] != null ? String(allRows[pt._ri][scatterColIdx]) : '')
            : (pt.group != null ? String(pt.group) : '');
        return _extractGroupKey(raw || '(blank)', colorRx);
    }''',
new='''    var scatterColIdx = colCol ? currentHeaders.indexOf(colCol) : -1;
    function colorKey(pt) {
        /* Multi-dim: use _colorDims when at least one dim has a column selected */
        var hasColorDims = _colorDims.length > 0 && _colorDims.some(function(d){ return d.colIdx >= 0; });
        if (hasColorDims && pt._ri != null && allRows[pt._ri] != null) {
            return _compoundKey(allRows[pt._ri], _colorDims) || '(all)';
        }
        /* Legacy single-dim fallback */
        if (!colCol) return '(all)';
        var raw = (scatterColIdx >= 0 && pt._ri != null && allRows[pt._ri] != null)
            ? (allRows[pt._ri][scatterColIdx] != null ? String(allRows[pt._ri][scatterColIdx]) : '')
            : (pt.group != null ? String(pt.group) : '');
        return _extractGroupKey(raw || '(blank)', colorRx);
    }''')

# ─────────────────────────────────────────────────────────────────
# 8. groupKey in drawCumProba / drawHistogram uses _splitDims
# ─────────────────────────────────────────────────────────────────
patch('groupKey uses _splitDims',
old='''    var splitColIdx = splitCol ? currentHeaders.indexOf(splitCol) : -1;
    function groupKey(pt) {
        if (!splitCol) return '(all)';
        var raw = (splitColIdx >= 0 && pt._ri != null && allRows[pt._ri] != null)
            ? (allRows[pt._ri][splitColIdx] != null ? String(allRows[pt._ri][splitColIdx]) : '')
            : (pt.group != null ? String(pt.group) : '');
        return _extractGroupKey(raw || '(blank)', splitRx);
    }''',
new='''    var splitColIdx = splitCol ? currentHeaders.indexOf(splitCol) : -1;
    function groupKey(pt) {
        /* Multi-dim: use _splitDims when at least one dim has a column selected */
        var hasSplitDims = _splitDims.length > 0 && _splitDims.some(function(d){ return d.colIdx >= 0; });
        if (hasSplitDims && pt._ri != null && allRows[pt._ri] != null) {
            return _compoundKey(allRows[pt._ri], _splitDims) || '(all)';
        }
        /* Legacy fallback */
        if (!splitCol) return '(all)';
        var raw = (splitColIdx >= 0 && pt._ri != null && allRows[pt._ri] != null)
            ? (allRows[pt._ri][splitColIdx] != null ? String(allRows[pt._ri][splitColIdx]) : '')
            : (pt.group != null ? String(pt.group) : '');
        return _extractGroupKey(raw || '(blank)', splitRx);
    }''')

# ─────────────────────────────────────────────────────────────────
# 9. _splitChartKeys: prepend extra _scDims key to each tile key
# ─────────────────────────────────────────────────────────────────
patch('_splitChartKeys uses _scDims',
old='''function _splitChartKeys(row, colIdx, rxStr) {
    if (colIdx < 0) return ['(blank)'];
    var raw = row[colIdx] != null ? String(row[colIdx]) : '(blank)';''',
new='''function _splitChartKeys(row, colIdx, rxStr) {
    /* If extra _scDims are configured, prepend their compound key to every tile key */
    var scDimPrefix = null;
    if (_scDims.length > 0 && _scDims.some(function(d){ return d.colIdx >= 0; })) {
        scDimPrefix = _compoundKey(row, _scDims) || '';
    }
    /* If ONLY extra dims are present (no primary split-chart-col selected) return dim key */
    if (colIdx < 0) {
        if (scDimPrefix !== null) return [scDimPrefix || '(blank)'];
        return ['(blank)'];
    }
    var raw = row[colIdx] != null ? String(row[colIdx]) : '(blank)';''')

# also patch the return statements to prepend scDimPrefix
patch('_splitChartKeys prefix returns',
old='''    if (!rxStr) {
        /* No regex: split on comma and trim each part */
        var parts = raw.split(',').map(function(s){ return s.trim(); })
                                  .filter(function(s){ return s.length > 0; });
        return parts.length > 0 ? parts : ['(blank)'];
    }

    /* Regex: collect all global matches, joining capture groups per match */
    try {
        var rx = new RegExp(rxStr, 'g');
        var keys = [], m;
        while ((m = rx.exec(raw)) !== null) {
            /* Collect all defined capture groups (m[1], m[2], …) */
            var groups = [];
            for (var g = 1; g < m.length; g++) {
                if (m[g] !== undefined) groups.push(m[g]);
            }
            /* Key = captured groups joined, or full match if no groups */
            keys.push(groups.length > 0 ? groups.join(', ') : m[0]);
            /* Guard against zero-width matches causing infinite loop */
            if (m[0].length === 0) { rx.lastIndex++; }
        }
        return keys.length > 0 ? keys : ['(no match)'];
    } catch(e) { return [raw]; }
}''',
new='''    function _prefixKeys(keys) {
        if (scDimPrefix === null) return keys;
        return keys.map(function(k){ return scDimPrefix + ' \u2502 ' + k; });
    }
    if (!rxStr) {
        /* No regex: split on comma and trim each part */
        var parts = raw.split(',').map(function(s){ return s.trim(); })
                                  .filter(function(s){ return s.length > 0; });
        return _prefixKeys(parts.length > 0 ? parts : ['(blank)']);
    }

    /* Regex: collect all global matches, joining capture groups per match */
    try {
        var rx = new RegExp(rxStr, 'g');
        var keys = [], m;
        while ((m = rx.exec(raw)) !== null) {
            /* Collect all defined capture groups (m[1], m[2], …) */
            var groups = [];
            for (var g = 1; g < m.length; g++) {
                if (m[g] !== undefined) groups.push(m[g]);
            }
            /* Key = captured groups joined, or full match if no groups */
            keys.push(groups.length > 0 ? groups.join(', ') : m[0]);
            /* Guard against zero-width matches causing infinite loop */
            if (m[0].length === 0) { rx.lastIndex++; }
        }
        return _prefixKeys(keys.length > 0 ? keys : ['(no match)']);
    } catch(e) { return _prefixKeys([raw]); }
}''')

# ─────────────────────────────────────────────────────────────────
# 10. populatePlotSelectors calls _gdimRebuildSelects
# ─────────────────────────────────────────────────────────────────
patch('populatePlotSelectors calls _gdimRebuildSelects',
old='''    tryDefault('color-col',['DUT','RESULT','PAGETYPE','Type']);''',
new='''    tryDefault('color-col',['DUT','RESULT','PAGETYPE','Type']);
    _gdimRebuildSelects(headers);''')

# ─────────────────────────────────────────────────────────────────
# 11. DOMContentLoaded calls _gdimInit (add after _tryRestoreSession)
# ─────────────────────────────────────────────────────────────────
patch('DOMContentLoaded calls _gdimInit',
old='''    /* Restore session after browser refresh */
    _tryRestoreSession();
});''',
new='''    /* Init multi-dim group builders */
    _gdimInit();
    /* Restore session after browser refresh */
    _tryRestoreSession();
});''')

# ─────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────
if errors:
    print('\n=== ERRORS (not applied) ===')
    for e in errors:
        print(' ', e)
    print('\n=== OK ===')
    for o in ok:
        print(' ', o)
    print('\nFile NOT written due to errors.')
else:
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(src)
    print('=== ALL PATCHES APPLIED ===')
    for o in ok:
        print(' ', o)
    print(f'\nWritten: {TARGET}')
    # Sanity checks
    checks = ['_colorDims','_splitDims','_scDims','_compoundKey','_gdimAdd',
              '_gdimRead','_gdimChanged','_gdimRebuildSelects','_gdimInit',
              'color-dims-wrap','split-dims-wrap','sc-dims-wrap',
              'hasColorDims','hasSplitDims','scDimPrefix','_prefixKeys']
    print('\nSanity checks:')
    for c in checks:
        print(f'  {"OK" if c in src else "MISSING"}: {c}')
