"""
patch_editable_title.py
Add inline-editable title bar above main chart + editable tile headers.
"""
import shutil

TARGET = r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart.html'
shutil.copy(TARGET, TARGET + '.bak_before_editable_title')

with open(TARGET, encoding='utf-8') as f:
    src = f.read()

ok, err = [], []

def patch(label, old, new, n=1):
    global src
    c = src.count(old)
    if c == 0:             err.append('NOT FOUND: ' + label); return
    if n == 1 and c > 1:   err.append(f'AMBIGUOUS ({c}x): ' + label); return
    src = src.replace(old, new, n)
    ok.append('OK: ' + label)

# ── 1. CSS: extend .split-tile-title + add .chart-title-bar ───────────
patch('CSS editable title',
old='''        .split-tile-title {
            font-size: 0.75em; font-weight: bold; color: #495057;
            padding: 2px 6px; background: #e9ecef;
            border-bottom: 1px solid #dee2e6; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis;
        }''',
new='''        .split-tile-title {
            font-size: 0.75em; font-weight: bold; color: #495057;
            padding: 2px 6px; background: #e9ecef;
            border-bottom: 1px solid #dee2e6; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis;
            cursor: text; outline: none;
        }
        .split-tile-title:hover  { background: #d0d8e8; }
        .split-tile-title:focus  { background: #fff; outline: 2px solid #17a2b8;
                                    white-space: normal; text-overflow: clip; }
        /* Main chart editable title bar */
        #chart-title-bar {
            width: 100%; text-align: center; font-size: 0.95em; font-weight: bold;
            color: #333; padding: 3px 8px 2px; min-height: 22px; cursor: text;
            border-radius: 3px; transition: background 0.15s; outline: none;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            display: block; box-sizing: border-box;
        }
        #chart-title-bar:hover { background: #f0f4ff; }
        #chart-title-bar:focus { background: #fff; outline: 2px solid #17a2b8;
                                  white-space: normal; text-overflow: clip; }
        #chart-title-bar:empty:before {
            content: "Click to add chart title\u2026";
            color: #bbb; font-weight: normal; font-style: italic;
        }''')

# ── 2. HTML: add #chart-title-bar above #plot-area ────────────────────
patch('HTML chart-title-bar',
old='    <div id="plot-area">',
new=('    <div id="chart-title-bar"\n'
     '         contenteditable="true"\n'
     '         onkeydown="_titleKeyDown(event,\'main\')"\n'
     '         onblur="_titleBlur(\'main\')"\n'
     '         title="Click to add / edit chart title"></div>\n'
     '    <div id="plot-area">'))

# ── 3. JS engine: insert before MULTI-DIMENSION GROUP BUILDER ─────────
patch('JS editable title engine',
old='/* ================================================================\n   MULTI-DIMENSION GROUP BUILDER',
new='''/* ================================================================
   EDITABLE CHART TITLE
   • #chart-title-bar above the main canvas is a contenteditable div.
   • .split-tile-title on each tile is also contenteditable.
   • _syncMainTitle(text) sets the bar only when the user has NOT typed
     a custom title (resets on every new drawPlot() call).
================================================================ */
var _mainChartTitle = '';   /* '' = use auto-title from drawXxx() */

function _syncMainTitle(autoTitle) {
    if (_mainChartTitle) return;            /* keep custom title */
    var bar = document.getElementById('chart-title-bar');
    if (bar) bar.textContent = autoTitle || '';
}

function _titleKeyDown(e, ctx) {
    if (e.key === 'Enter')  { e.preventDefault(); e.target.blur(); }
    if (e.key === 'Escape') {
        if (ctx === 'main') {
            var bar = document.getElementById('chart-title-bar');
            if (bar) bar.textContent = _mainChartTitle;
        }
        e.target.blur();
    }
}

function _titleBlur(ctx, tileIdx) {
    if (ctx === 'main') {
        var bar = document.getElementById('chart-title-bar');
        _mainChartTitle = bar ? bar.textContent.trim() : '';
    } else {
        /* tile — tileIdx is the position in _splitInsts[] */
        var titleEls = document.querySelectorAll('.split-tile-title');
        var el = (tileIdx != null && titleEls[tileIdx]) ? titleEls[tileIdx] : null;
        if (!el) return;
        var newTitle = el.textContent.trim();
        var inst = _splitInsts[tileIdx];
        if (!inst) return;
        inst.options.plugins.title.text    = newTitle;
        inst.options.plugins.title.display = !!newTitle;
        try { inst.update('none'); } catch(e2) {}
    }
}

/* Reset title bar on every new plot so auto-title takes over */
function _resetTitleBar() {
    _mainChartTitle = '';
    var bar = document.getElementById('chart-title-bar');
    if (bar) bar.textContent = '';
}

/* Wire tile title contenteditable after split-chart renders */
function _wireTileTitles() {
    document.querySelectorAll('.split-tile-title').forEach(function(el, i) {
        el.contentEditable = 'true';
        el.title = 'Click to edit tile title';
        el.setAttribute('onkeydown', "_titleKeyDown(event,'tile')");
        el.setAttribute('onblur', "_titleBlur('tile'," + i + ")");
    });
}

/* ================================================================
   MULTI-DIMENSION GROUP BUILDER''')

# ── 4. Reset title bar at start of each draw function ────────────────
for fn in ['function drawScatterLine() {',
           'function drawHistogram() {',
           'function drawCumProba() {',
           'function drawBoxPlot() {']:
    patch('reset in ' + fn, fn,
          fn + '\n    _resetTitleBar();')

# ── 5. _syncMainTitle at end of each draw (before filtNote/_spinStop) ─
patch('sync scatter title',
old="    var filtNote = hasActiveFilters() ? ' \\u2014 filtered' : '';\n"
    "    var sampNote = sampled",
new="    _syncMainTitle(xCol + ' vs ' + yCol"
    " + (colCol ? '  \u2502  ' + colCol : ''));\n"
    "    var filtNote = hasActiveFilters() ? ' \u2014 filtered' : '';\n"
    "    var sampNote = sampled")

patch('sync histogram title',
old="    var filtNote = hasActiveFilters() ? ' \\u2014 filtered' : '';\n"
    "    status.textContent = vals.length.toLocaleString() + ' values, ' + nBins",
new="    _syncMainTitle('Histogram \u2502 ' + xCol);\n"
    "    var filtNote = hasActiveFilters() ? ' \u2014 filtered' : '';\n"
    "    status.textContent = vals.length.toLocaleString() + ' values, ' + nBins")

patch('sync cumproba title',
old="    var filtNote  = hasActiveFilters() ? ' \\u2014 filtered' : '';\n"
    "    var nGroups   = Object.keys(groups).length;",
new="    _syncMainTitle('Cum Proba \u2502 ' + xCol"
    " + (splitCol ? '  \u2502  ' + splitCol : ''));\n"
    "    var filtNote  = hasActiveFilters() ? ' \u2014 filtered' : '';\n"
    "    var nGroups   = Object.keys(groups).length;")

patch('sync boxplot title',
old="    var filtNote = hasActiveFilters() ? ' \\u2014 filtered' : '';\n"
    "    var capNote  = capped",
new="    _syncMainTitle('Box & Whisker \u2502 ' + xCol + ' vs ' + yCol);\n"
    "    var filtNote = hasActiveFilters() ? ' \u2014 filtered' : '';\n"
    "    var capNote  = capped")

# ── 6. Call _wireTileTitles() after split-chart renders ───────────────
patch('wire tile titles after split-chart',
old="    var filtNote  = hasActiveFilters() ? ' \\u2014 filtered' : '';\n"
    "    var totalBucketed = keys.reduce",
new="    _wireTileTitles();\n"
    "    var filtNote  = hasActiveFilters() ? ' \u2014 filtered' : '';\n"
    "    var totalBucketed = keys.reduce")

# ── Write ──────────────────────────────────────────────────────────────
if err:
    print('\n=== ERRORS — file NOT written ===')
    for e in err: print(' ', e)
    print('\n=== OK so far ===')
    for o in ok: print(' ', o)
else:
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(src)
    print('=== ALL OK — file written ===')
    for o in ok: print(' ', o)
    checks = ['chart-title-bar', '_syncMainTitle', '_titleBlur', '_titleKeyDown',
              '_resetTitleBar', '_wireTileTitles', '_mainChartTitle',
              'EDITABLE CHART TITLE', 'split-tile-title:hover', 'split-tile-title:focus']
    print('\nSanity:')
    with open(TARGET, encoding='utf-8') as f:
        c = f.read()
    for x in checks:
        print(f'  {"OK" if x in c else "MISSING"}: {x}')
