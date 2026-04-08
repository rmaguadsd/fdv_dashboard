#!/usr/bin/env python3
import sys

log_path = r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\fdv_chart_startup.log'
with open(log_path, 'w') as f:
    f.write("STARTUP_BEGIN\n")

print("PYTHON_START", flush=True)
sys.stderr.write("STDERR_START\n")
sys.stderr.flush()
"""
FDV Chart Parser - Web UI for filtering and parsing FDV log files
Based on process_fdv_poll.py structure but with user-configurable regex filtering

Features:
1. Upload a log file (.txt, .log, .csv)
2. Specify a regex pattern to include or exclude lines
3. Parse matching lines into structured CSV
4. Download results
"""

import os
import re
import csv
import json
import uuid
import io
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

# ── Local Ollama / LLM config ────────────────────────────────────────────────
_LLM_URL     = 'http://localhost:11434/v1/chat/completions'
_LLM_MODEL   = 'llama3'
_LLM_APIKEY  = ''    # leave empty for Ollama (no key needed)
_LLM_TIMEOUT      = 180   # seconds — llama3 can be slow; long conversations need more headroom
_CHAT_MAX_TURNS   = 20    # max user+assistant turn pairs kept in history (older pairs trimmed)

_LLM_SYSTEM_PROMPT = (
    'You are a data analysis assistant embedded in an engineering test-data viewer. '
    'The user gives you statistics extracted from parsed log file charts. '
    'Follow the TASK instruction in the prompt exactly. '
    'OUTPUT FORMAT — always respond in concise bullet points (•). '
    'Use one bullet per key finding. Never write long prose paragraphs. '
    'Each bullet must be a complete, self-contained observation. '
    'Always cover every requested aspect: do not stop early or truncate. '
    'When asked for a single-chart analysis cover: distribution shape, central tendency, '
    'spread, and outliers — one bullet each minimum. '
    'When asked for a comparative analysis cover every group — rank them by median or mean, '
    'note which have wider spread, higher extremes, or cross reference lines. '
    'Always use engineering language and concrete numbers from the statistics provided. '
    'Never pad with generic disclaimers or repeat the input data verbatim.\n\n'
    'MARKER COMMANDS — you may add or remove reference lines on the chart by emitting '
    'these special tokens anywhere in your reply (they will be executed automatically '
    'and hidden from the displayed text):\n'
    '  [MARKER: x=<value>:<label>]   — add a vertical line at X=value\n'
    '  [MARKER: y=<value>:<label>]   — add a horizontal line at Y=value\n'
    '  [CLEAR_MARKERS]               — remove all existing markers\n'
    'Examples: [MARKER: x=1000:spec_limit]  [MARKER: y=0.05:target]  [CLEAR_MARKERS]\n'
    'Use markers when the user asks to mark, highlight, or draw a line at a specific value, '
    'or when you identify a threshold worth highlighting. You may emit multiple MARKER '
    'commands in one reply. Always explain in a bullet what you marked and why.'
)

_OLLAMA_BASE = 'http://localhost:11434'

# ── Server config (can be overridden by command-line args) ───────────────────
_SERVER_PORT      = 5058
_SERVER_STORE_DIR = r'D:\FDV\recipes'   # auto-pushed to new clients via /store/default_dir

def _call_llm(messages, model=None):
    """POST a messages list to local Ollama and return the assistant reply string."""
    payload = json.dumps({
        'model':       model or _LLM_MODEL,
        'messages':    messages,
        'stream':      False,
        'temperature': 0.3
    }).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if _LLM_APIKEY:
        headers['Authorization'] = 'Bearer ' + _LLM_APIKEY
    req = urllib.request.Request(_LLM_URL, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as r:
        body = json.loads(r.read().decode('utf-8'))
    return body['choices'][0]['message']['content'].strip()

# ── Chat session store  {csv_id: [{"role":..,"content":..}, ...]} ─────
_chat_sessions      = {}
_chat_sessions_lock = threading.Lock()

with open(log_path, 'a') as f:
    f.write("IMPORTS_OK\n")

# Storage for parsed data
parsed_cache = {}

# Storage for async parse jobs: job_id -> {state, progress_lines, total_lines, error, csv_id}
parse_jobs = {}
parse_jobs_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a separate thread so long parses don't block."""
    daemon_threads = True


def parse_log_file(file_path, regex_pattern=None, include_mode=True, source_name=None):
    """
    Parse log file with regex filtering.
    Supports both FDV OUTPUT (functional) and FDV POLL (char/array) lines.
    Each line is auto-detected and parsed into the appropriate columns.
    Columns not applicable to a line type are left empty.

    Args:
        file_path: Path to log file
        regex_pattern: Optional regex to match lines
        include_mode: True to include matches, False to exclude

    Returns:
        tuple: (headers, data_rows)
    """
    # ── Unified headers covering both FDV OUTPUT and FDV POLL ──────────────
    HEADERS = [
        'Line#', 'Type', 'DUT',
        'tname', 'testname',
        'BLK', 'PAGE', 'PAGETYPE', 'PAGEMAP', 'WL', 'SB', 'BL',
        'STEP', 'DECK',
        'VCC', 'VCCQ', 'TEMP', 'TAC', 'TM',
        # FDV OUTPUT specific
        'Result',
        'nBytes', 'nFailBytes', 'BYBER', 'nFailBits', 'RBER', 'RBER_Limit',
        'FailData',
        # FDV POLL specific
        'SpecName', 'Status', 'PlaneOp', 'Measurement',
        # common
        'FDV_File',
        'SourceFile',
    ]

    rows = []
    line_num = 0

    # ── FDV OUTPUT regex ───────────────────────────────────────────────────
    _RE_OUT = re.compile(
        r'FDV OUTPUT \[(?P<path>[^\]]+)\.FDV::(?P<tname>[^,\]]+)'
        r'(?:,(?P<conds>[^\]]*))?\]:\s*'
        r'(?P<dut>\S+?),(?P<result>[^,]+),'
        r'(?P<nBytes>[^,]*),'
        r'(?P<nFailBytes>[^,]*),'
        r'(?P<byber>[^,]*),'
        r'(?P<nFailBits>[^,]*),'
        r'(?P<rber>[^,]*),'
        r'(?P<rber_lim>[^,]*),'
        r'(?P<faildata>.*)'
    )

    # ── FDV POLL regex ─────────────────────────────────────────────────────
    # Format: FDV POLL [...path/file.FDV::tname,conds]: DUT1 0,<measurement>,<rest ignored>
    _RE_POLL = re.compile(
        r'FDV POLL \[(?P<path>[^\]]+)\.FDV::(?P<tname>[^,\]]+)'
        r'(?:,(?P<conds>[^\]]*))?\]:\s*'
        r'(?P<dut>\S+?)\s+\d+\s*,'          # DUT1 0,
        r'(?P<measurement>[^,]+)'            # measurement value
    )

    # ── Shared helpers ─────────────────────────────────────────────────────
    def _cond(conds_str, key):
        m = re.search(r'(?:^|,)' + key + r'=([^,]*)', conds_str or '')
        return m.group(1) if m else ''

    def _tparam(tname, key):
        m = re.search(r'(?:^|_)' + key + r'[_:](\d+)', tname, re.I)
        return m.group(1) if m else ''

    def _pagetype(tname):
        m = re.search(r'PGTYPE[_:]([A-Za-z]+)', tname, re.I)
        return m.group(1).upper() if m else ''

    def _pagemap(tname):
        for pm in ('QLC', 'TLC', 'MLC', 'SSLC', 'DSLC', 'SLC'):
            if re.search(r'(?:^|[_\W])' + pm + r'(?:[_\W]|$)', tname, re.I):
                return pm
        return ''

    def _deck(tname):
        m = re.search(r'DECK[_:]([A-Za-z]+)', tname, re.I)
        return m.group(1).upper() if m else ''

    def _testname(tname):
        stop = re.search(
            r'(?:^|_)(?:BLK|PAGE|PG|PGTYPE|WL|SB|BL|STEP|DECK|QLC|TLC|MLC|SSLC|DSLC|SLC)[_:\d]',
            tname, re.I)
        base = tname[:stop.start()] if stop else tname
        return base.strip('_')

    def _specname(tname):
        """Extract spec name (e.g. TR, C0, E0) from POLL tname."""
        # tname pattern: POLL_<SPEC>_<STATUS>_<PLANEOP>_BLK_... or similar
        # SpecName is the part before array params, after known prefix
        m = re.match(r'POLL_([A-Z0-9]+)', tname, re.I)
        return m.group(1) if m else ''

    def _status(tname):
        """Extract status token: C0, E0, F0, E4, E1, 80 etc."""
        m = re.search(r'(?:^|_)(C0|E0|F0|E4|E1|80)(?:_|$)', tname, re.I)
        return m.group(1).upper() if m else ''

    def _planeop(tname):
        """Extract plane operation: SP, MP, 2P, 3P etc."""
        m = re.search(r'(?:^|_)((?:SP|MP|[2-6]P))(?:_|$)', tname, re.I)
        return m.group(1).upper() if m else ''

    def _fdvfile(path):
        base = path.split('/')[-1].split('\\')[-1]
        return base + '.FDV' if not base.upper().endswith('.FDV') else base

    # ── Empty row template (index matches HEADERS) ─────────────────────────
    _EMPTY = [''] * len(HEADERS)
    # Map header name → index for quick access
    _IDX = {h: i for i, h in enumerate(HEADERS)}

    def _parse_output_line(line, lnum):
        m = _RE_OUT.search(line)
        if not m:
            return None
        tname = m.group('tname')
        conds = m.group('conds') or ''
        row = list(_EMPTY)
        row[_IDX['Line#']]      = str(lnum)
        row[_IDX['Type']]       = 'OUTPUT'
        row[_IDX['DUT']]        = m.group('dut').strip()
        row[_IDX['tname']]      = tname
        row[_IDX['testname']]   = _testname(tname)
        row[_IDX['BLK']]        = _tparam(tname, 'BLK')
        row[_IDX['PAGE']]       = _tparam(tname, r'(?:PAGE|PG)')
        row[_IDX['PAGETYPE']]   = _pagetype(tname)
        row[_IDX['PAGEMAP']]    = _pagemap(tname)
        row[_IDX['WL']]         = _tparam(tname, 'WL')
        row[_IDX['SB']]         = _tparam(tname, 'SB')
        row[_IDX['BL']]         = _tparam(tname, 'BL')
        row[_IDX['STEP']]       = _tparam(tname, 'STEP')
        row[_IDX['DECK']]       = _deck(tname)
        row[_IDX['VCC']]        = _cond(conds, 'VCC')
        row[_IDX['VCCQ']]       = _cond(conds, 'VCCQ')
        row[_IDX['TEMP']]       = _cond(conds, 'TEMP')
        row[_IDX['TAC']]        = _cond(conds, 'TAC')
        row[_IDX['TM']]         = _cond(conds, 'TM')
        row[_IDX['Result']]     = m.group('result').strip()
        row[_IDX['nBytes']]     = m.group('nBytes').strip()
        row[_IDX['nFailBytes']] = m.group('nFailBytes').strip()
        row[_IDX['BYBER']]      = m.group('byber').strip()
        row[_IDX['nFailBits']]  = m.group('nFailBits').strip()
        row[_IDX['RBER']]       = m.group('rber').strip()
        row[_IDX['RBER_Limit']] = m.group('rber_lim').strip()
        row[_IDX['FailData']]   = m.group('faildata').strip().rstrip(',')
        row[_IDX['FDV_File']]   = _fdvfile(m.group('path'))
        return row

    def _parse_poll_line(line, lnum):
        m = _RE_POLL.search(line)
        if not m:
            return None
        tname = m.group('tname')
        conds = m.group('conds') or ''
        row = list(_EMPTY)
        row[_IDX['Line#']]       = str(lnum)
        row[_IDX['Type']]        = 'POLL'
        row[_IDX['DUT']]         = m.group('dut').strip()
        row[_IDX['tname']]       = tname
        row[_IDX['testname']]    = _testname(tname)
        row[_IDX['BLK']]         = _tparam(tname, 'BLK')
        row[_IDX['PAGE']]        = _tparam(tname, r'(?:PAGE|PG)')
        row[_IDX['PAGETYPE']]    = _pagetype(tname)
        row[_IDX['PAGEMAP']]     = _pagemap(tname)
        row[_IDX['WL']]          = _tparam(tname, 'WL')
        row[_IDX['SB']]          = _tparam(tname, 'SB')
        row[_IDX['BL']]          = _tparam(tname, 'BL')
        row[_IDX['STEP']]        = _tparam(tname, 'STEP')
        row[_IDX['DECK']]        = _deck(tname)
        row[_IDX['VCC']]         = _cond(conds, 'VCC')
        row[_IDX['VCCQ']]        = _cond(conds, 'VCCQ')
        row[_IDX['TEMP']]        = _cond(conds, 'TEMP')
        row[_IDX['TAC']]         = _cond(conds, 'TAC')
        row[_IDX['TM']]          = _cond(conds, 'TM')
        row[_IDX['SpecName']]    = _specname(tname)
        row[_IDX['Status']]      = _status(tname)
        row[_IDX['PlaneOp']]     = _planeop(tname)
        row[_IDX['Measurement']] = m.group('measurement').strip()
        row[_IDX['FDV_File']]    = _fdvfile(m.group('path'))
        return row

    # ── Main parse loop ────────────────────────────────────────────────────
    try:
        compiled_regex = None
        if regex_pattern:
            try:
                compiled_regex = re.compile(regex_pattern)
            except re.error as e:
                raise ValueError("Invalid regex: " + str(e))

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_num += 1
                line_stripped = line.rstrip('\n')

                if not line_stripped.strip():
                    continue

                # Apply regex filter
                if compiled_regex:
                    matches = compiled_regex.search(line_stripped)
                    if include_mode and not matches:
                        continue
                    if not include_mode and matches:
                        continue

                # Auto-detect and parse line type
                if 'FDV OUTPUT' in line_stripped:
                    row = _parse_output_line(line_stripped, line_num)
                elif 'FDV POLL' in line_stripped:
                    row = _parse_poll_line(line_stripped, line_num)
                else:
                    continue   # not a parseable FDV line — skip

                if row:
                    rows.append(row)

    except Exception as e:
        raise Exception("Error parsing file: " + str(e))

    # Stamp every row with the source log filename
    src_name = source_name if source_name else Path(file_path).name
    src_idx  = _IDX['SourceFile']
    for row in rows:
        row[src_idx] = src_name

    return HEADERS, rows


def get_html():
    """Return the HTML interface from external file"""
    html_file = Path(__file__).parent / 'fdv_chart.html'
    return html_file.read_text(encoding='utf-8')


# File extensions considered when scanning a directory
_DIR_EXTENSIONS = {'.txt', '.log', '.csv'}


def _run_parse_job(job_id, file_path, regex_pattern, include_mode, temp_path=None, source_name=None):
    """Run parse_log_file in a background thread, updating parse_jobs[job_id]."""
    try:
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'running'

        headers, rows = parse_log_file(file_path, regex_pattern, include_mode, source_name=source_name)

        if not rows:
            raise ValueError('No matching rows found')

        csv_id = 'csv_' + uuid.uuid4().hex[:8]
        parsed_cache[csv_id] = {'headers': headers, 'rows': rows}

        PREVIEW = 500
        result = {
            'success': True, 'csv_id': csv_id,
            'headers': headers, 'rows': rows[:PREVIEW],
            'total_rows': len(rows)
        }
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'done'
            parse_jobs[job_id]['result'] = result

    except Exception as e:
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'error'
            parse_jobs[job_id]['error'] = str(e)
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _run_parse_multi_job(job_id, file_paths, regex_pattern, include_mode, temp_paths=None, orig_names=None):
    """Parse multiple files and concatenate results into a single dataset."""
    try:
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'running'

        all_rows = []
        headers  = None
        errors   = []

        for i, fp in enumerate(file_paths):
            src_name = orig_names[i] if orig_names and i < len(orig_names) else None
            try:
                h, rows = parse_log_file(fp, regex_pattern, include_mode, source_name=src_name)
                if headers is None:
                    headers = h
                all_rows.extend(rows)
            except Exception as e:
                display_name = src_name or Path(fp).name
                errors.append(f'{display_name}: {e}')

        if not all_rows:
            msg = 'No matching rows found in any file'
            if errors:
                msg += ' — errors: ' + '; '.join(errors)
            raise ValueError(msg)

        csv_id = 'csv_' + uuid.uuid4().hex[:8]
        parsed_cache[csv_id] = {'headers': headers, 'rows': all_rows}

        PREVIEW = 500
        result = {
            'success': True, 'csv_id': csv_id,
            'headers': headers, 'rows': all_rows[:PREVIEW],
            'total_rows': len(all_rows),
            'file_count': len(file_paths),
            'errors': errors,
        }
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'done'
            parse_jobs[job_id]['result'] = result

    except Exception as e:
        with parse_jobs_lock:
            parse_jobs[job_id]['state'] = 'error'
            parse_jobs[job_id]['error'] = str(e)
    finally:
        for tp in (temp_paths or []):
            try:
                Path(tp).unlink(missing_ok=True)
            except Exception:
                pass


def _send_json(handler, status, obj):
    body = json.dumps(obj).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', len(body))
    handler.end_headers()
    handler.wfile.write(body)


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the web interface"""

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve HTML interface
            body = get_html().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith('/parse_status'):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            job_id = qs.get('job', [''])[0]
            with parse_jobs_lock:
                job = parse_jobs.get(job_id)
            if not job:
                _send_json(self, 404, {'success': False, 'error': 'Job not found'})
                return
            state = job['state']
            if state == 'done':
                _send_json(self, 200, job['result'])
                # Clean up job entry after delivery
                with parse_jobs_lock:
                    parse_jobs.pop(job_id, None)
            elif state == 'error':
                _send_json(self, 400, {'success': False, 'error': job['error']})
                with parse_jobs_lock:
                    parse_jobs.pop(job_id, None)
            else:
                # Still running — return pending status
                _send_json(self, 202, {'success': False, 'state': state, 'job_id': job_id})

        elif self.path.startswith('/download/'):
            # Download CSV
            csv_id = self.path.split('/')[-1]
            if csv_id not in parsed_cache:
                self.send_error(404)
                return
            
            data = parsed_cache[csv_id]
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(data['headers'])
            writer.writerows(data['rows'])
            csv_content = output.getvalue().encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename=fdv_parse_' + csv_id + '.csv')
            self.send_header('Content-Length', len(csv_content))
            self.end_headers()
            self.wfile.write(csv_content)

        elif self.path.startswith('/plot_data'):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            def qs1(k, d=''):
                return qs.get(k, [d])[0]
            csv_id   = qs1('csv_id')
            x_col    = qs1('x_col')
            y_col    = qs1('y_col')
            x_regex  = qs1('x_regex')
            y_regex  = qs1('y_regex')
            color_col= qs1('color_col')
            max_pts  = int(qs1('max_pts', '5000'))

            if csv_id not in parsed_cache:
                resp = json.dumps({'success': False, 'error': 'csv_id not found'}).encode()
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(resp))
                self.end_headers()
                self.wfile.write(resp)
                return

            cached = parsed_cache[csv_id]
            headers = cached['headers']
            rows    = cached['rows']

            def col_idx(name):
                try: return headers.index(name)
                except ValueError: return None

            xi = col_idx(x_col)
            yi = col_idx(y_col)
            ci = col_idx(color_col) if color_col else None

            if xi is None or yi is None:
                resp = json.dumps({'success': False, 'error': 'Column not found'}).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(resp))
                self.end_headers()
                self.wfile.write(resp)
                return

            def extract_num(val, rx):
                if rx:
                    try:
                        m = re.search(rx, str(val))
                        if m:
                            val = m.group(1) if m.lastindex else m.group(0)
                    except re.error:
                        pass
                try:
                    return float(re.sub(r'[^\d.eE+\-]', '', str(val)))
                except (ValueError, TypeError):
                    return None

            points = []
            skipped = 0
            step = max(1, len(rows) // max_pts) if len(rows) > max_pts else 1
            for row in rows[::step]:
                xv = extract_num(row[xi], x_regex)
                yv = extract_num(row[yi], y_regex)
                if xv is None or yv is None:
                    skipped += 1
                    continue
                pt = {'x': xv, 'y': yv}
                if ci is not None and ci < len(row):
                    pt['group'] = row[ci]
                points.append(pt)

            resp = json.dumps({'success': True, 'points': points, 'skipped': skipped}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(resp))
            self.end_headers()
            self.wfile.write(resp)

        elif self.path.startswith('/rows'):
            # Paginated row fetch: /rows?csv_id=...&offset=0&limit=1000
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            csv_id = qs.get('csv_id', [''])[0]
            offset = int(qs.get('offset', ['0'])[0])
            limit  = int(qs.get('limit',  ['1000'])[0])

            if csv_id not in parsed_cache:
                resp = json.dumps({'success': False, 'error': 'csv_id not found'}).encode()
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(resp))
                self.end_headers()
                self.wfile.write(resp)
                return

            all_rows = parsed_cache[csv_id]['rows']
            chunk = all_rows[offset:offset + limit]
            resp = json.dumps({
                'success': True,
                'rows': chunk,
                'offset': offset,
                'total': len(all_rows),
                'has_more': (offset + limit) < len(all_rows)
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(resp))
            self.end_headers()
            self.wfile.write(resp)

        elif self.path == '/models':
            # Return list of locally pulled Ollama models
            try:
                req  = urllib.request.Request(_OLLAMA_BASE + '/api/tags')
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read().decode('utf-8'))
                names = sorted(m['name'] for m in data.get('models', []))
                _send_json(self, 200, {'success': True, 'models': names})
            except Exception as ex:
                _send_json(self, 200, {'success': False, 'models': [_LLM_MODEL], 'error': str(ex)})

        elif self.path.startswith('/store/check'):
            # Validate a store directory: /store/check?dir=<path>
            try:
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                d  = qs.get('dir', [''])[0].strip()
                if not d:
                    _send_json(self, 200, {'success': False, 'error': 'No directory specified'})
                elif os.path.isdir(d):
                    _send_json(self, 200, {'success': True, 'dir': d})
                else:
                    _send_json(self, 200, {'success': False, 'error': 'Directory not found: ' + d})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path.startswith('/store/default_dir'):
            # Return the server's configured default store directory
            d = _SERVER_STORE_DIR.strip()
            if d and os.path.isdir(d):
                _send_json(self, 200, {'success': True,  'dir': d})
            elif d:
                _send_json(self, 200, {'success': False, 'dir': d,
                                       'error': 'Default store dir not found on server: ' + d})
            else:
                _send_json(self, 200, {'success': False, 'dir': '',
                                       'error': 'No default store dir configured'})

        elif self.path.startswith('/store/list'):
            # List recipe/session files in directory: /store/list?dir=<path>
            try:
                from urllib.parse import urlparse, parse_qs
                qs  = parse_qs(urlparse(self.path).query)
                d   = qs.get('dir', [''])[0].strip()
                if not d or not os.path.isdir(d):
                    _send_json(self, 200, {'success': False, 'error': 'Invalid directory', 'files': []})
                else:
                    files = []
                    for fname in sorted(os.listdir(d)):
                        if fname.endswith('.fdv_recipe') or fname.endswith('.fdv_session'):
                            fpath = os.path.join(d, fname)
                            files.append({
                                'name': fname,
                                'size': os.path.getsize(fpath),
                                'mtime': os.path.getmtime(fpath)
                            })
                    _send_json(self, 200, {'success': True, 'files': files})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e), 'files': []})

        elif self.path.startswith('/store/load'):
            # Load a recipe/session file: /store/load?dir=<path>&file=<name>
            try:
                from urllib.parse import urlparse, parse_qs
                qs    = parse_qs(urlparse(self.path).query)
                d     = qs.get('dir', [''])[0].strip()
                fname = qs.get('file', [''])[0].strip()
                if not d or not fname:
                    _send_json(self, 200, {'success': False, 'error': 'dir and file required'})
                elif not (fname.endswith('.fdv_recipe') or fname.endswith('.fdv_session')):
                    _send_json(self, 200, {'success': False, 'error': 'Invalid file type'})
                else:
                    fpath = os.path.join(d, os.path.basename(fname))
                    if not os.path.isfile(fpath):
                        _send_json(self, 200, {'success': False, 'error': 'File not found: ' + fname})
                    elif fname.endswith('.fdv_session'):
                        # Validate the file is readable JSON before streaming.
                        file_size = os.path.getsize(fpath)
                        if file_size == 0:
                            _send_json(self, 200, {'success': False, 'error': 'Session file is empty: ' + fname})
                        else:
                            # Quick validity check — try parsing header bytes
                            with open(fpath, 'rb') as _fv:
                                head = _fv.read(1).strip()
                            if head not in (b'{', b'['):
                                _send_json(self, 200, {'success': False, 'error': 'Session file appears corrupt (bad header): ' + fname})
                            else:
                                # Stream session files directly — they can be very large (many rows).
                                # Wrap the raw file bytes in {"success":true,"data": ... }
                                # by writing the envelope prefix/suffix around the file content.
                                prefix    = b'{"success":true,"data":'
                                suffix    = b'}'
                                total_len = len(prefix) + file_size + len(suffix)
                                self.send_response(200)
                                self.send_header('Content-Type', 'application/json; charset=utf-8')
                                self.send_header('Content-Length', total_len)
                                self.end_headers()
                                self.wfile.write(prefix)
                                with open(fpath, 'rb') as f:
                                    while True:
                                        chunk = f.read(256 * 1024)
                                        if not chunk:
                                            break
                                        self.wfile.write(chunk)
                                self.wfile.write(suffix)
                    else:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        _send_json(self, 200, {'success': True, 'data': data})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/parse_path':
            # Server-side path parse — starts async job, returns job_id immediately
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(content_len).decode('utf-8'))
                file_path    = body.get('path', '').strip()
                regex_filter = body.get('regex', '').strip()
                mode         = body.get('mode', 'include').strip()

                if not file_path:
                    raise ValueError('No file path provided')
                if not os.path.isfile(file_path):
                    raise ValueError(f'File not found: {file_path}')

                job_id = 'job_' + uuid.uuid4().hex[:8]
                with parse_jobs_lock:
                    parse_jobs[job_id] = {'state': 'pending', 'result': None, 'error': None}

                threading.Thread(
                    target=_run_parse_job,
                    args=(job_id, file_path, regex_filter if regex_filter else None, mode == 'include'),
                    daemon=True
                ).start()

                _send_json(self, 202, {'success': False, 'state': 'pending', 'job_id': job_id})

            except Exception as e:
                _send_json(self, 400, {'success': False, 'error': str(e)})

        elif self.path == '/parse':
            # File upload parse — save to temp, start async job
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                if content_len > 2 * 1024 * 1024 * 1024:
                    raise ValueError('File too large (>2 GB)')

                body = self.rfile.read(content_len)

                # Parse multipart form data
                boundary = None
                for name, value in self.headers.items():
                    if 'Content-Type' in name:
                        parts = value.split('boundary=')
                        if len(parts) > 1:
                            boundary = parts[1].strip()

                if not boundary:
                    raise ValueError('Missing multipart boundary')

                boundary_bytes = ('--' + boundary).encode()
                parts_list = body.split(boundary_bytes)

                file_content  = None
                orig_filename = ''
                regex_filter  = ''
                mode          = 'include'

                for part in parts_list:
                    if b'name="file"' in part and b'filename=' in part:
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if i == 0: continue
                            if line == b'':
                                file_content = b'\r\n'.join(lines[i+1:-1])
                                disp = lines[1].decode('utf-8', errors='ignore') if len(lines) > 1 else ''
                                m = re.search(r'filename="([^"]+)"', disp)
                                if m:
                                    orig_filename = Path(m.group(1)).name
                                break
                    elif b'name="regex"' in part:
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if i == 0: continue
                            if line == b'':
                                regex_filter = b'\r\n'.join(lines[i+1:-1]).decode('utf-8', errors='ignore').strip()
                                break
                    elif b'name="mode"' in part:
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if i == 0: continue
                            if line == b'':
                                mode = b'\r\n'.join(lines[i+1:-1]).decode('utf-8', errors='ignore').strip()
                                break

                if not file_content:
                    raise ValueError('No file provided')

                temp_path = Path(tempfile.gettempdir()) / ('fdv_upload_' + uuid.uuid4().hex + '.log')
                temp_path.write_bytes(file_content)
                del body, file_content  # free RAM immediately

                job_id = 'job_' + uuid.uuid4().hex[:8]
                with parse_jobs_lock:
                    parse_jobs[job_id] = {'state': 'pending', 'result': None, 'error': None}

                threading.Thread(
                    target=_run_parse_job,
                    args=(job_id, str(temp_path), regex_filter if regex_filter else None,
                          mode == 'include', str(temp_path), orig_filename or None),
                    daemon=True
                ).start()

                _send_json(self, 202, {'success': False, 'state': 'pending', 'job_id': job_id})

            except Exception as e:
                _send_json(self, 400, {'success': False, 'error': str(e)})

        elif self.path == '/parse_dir':
            # Server-side directory parse — all matching files under a directory
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(content_len).decode('utf-8'))
                dir_path     = body.get('path', '').strip()
                regex_filter = body.get('regex', '').strip()
                mode         = body.get('mode', 'include').strip()
                recursive    = bool(body.get('recursive', True))

                if not dir_path:
                    raise ValueError('No directory path provided')
                dp = Path(dir_path)
                if not dp.is_dir():
                    raise ValueError(f'Directory not found: {dir_path}')

                glob_iter = dp.rglob('*') if recursive else dp.glob('*')
                file_paths = sorted(
                    str(p) for p in glob_iter
                    if p.is_file() and p.suffix.lower() in _DIR_EXTENSIONS
                )
                if not file_paths:
                    raise ValueError(f'No .txt/.log/.csv files found under: {dir_path}')

                job_id = 'job_' + uuid.uuid4().hex[:8]
                with parse_jobs_lock:
                    parse_jobs[job_id] = {'state': 'pending', 'result': None, 'error': None}

                threading.Thread(
                    target=_run_parse_multi_job,
                    args=(job_id, file_paths, regex_filter if regex_filter else None, mode == 'include'),
                    daemon=True
                ).start()

                _send_json(self, 202, {
                    'success': False, 'state': 'pending', 'job_id': job_id,
                    'file_count': len(file_paths)
                })

            except Exception as e:
                _send_json(self, 400, {'success': False, 'error': str(e)})

        elif self.path == '/parse_multi':
            # Multi-file upload — save each to temp, parse all, concatenate
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                if content_len > 4 * 1024 * 1024 * 1024:
                    raise ValueError('Upload too large (>4 GB)')

                body = self.rfile.read(content_len)

                boundary = None
                for name, value in self.headers.items():
                    if 'Content-Type' in name:
                        parts = value.split('boundary=')
                        if len(parts) > 1:
                            boundary = parts[1].strip()

                if not boundary:
                    raise ValueError('Missing multipart boundary')

                boundary_bytes = ('--' + boundary).encode()
                parts_list = body.split(boundary_bytes)

                file_contents = []   # list of (filename, bytes)
                regex_filter  = ''
                mode          = 'include'

                for part in parts_list:
                    if b'name="file"' in part and b'filename=' in part:
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if i == 0: continue
                            if line == b'':
                                content = b'\r\n'.join(lines[i+1:-1])
                                fname = ''
                                disp = lines[1].decode('utf-8', errors='ignore') if len(lines) > 1 else ''
                                m = re.search(r'filename="([^"]+)"', disp)
                                if m:
                                    fname = m.group(1)
                                file_contents.append((fname, content))
                                break
                    elif b'name="regex"' in part:
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if i == 0: continue
                            if line == b'':
                                regex_filter = b'\r\n'.join(lines[i+1:-1]).decode('utf-8', errors='ignore').strip()
                                break
                    elif b'name="mode"' in part:
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if i == 0: continue
                            if line == b'':
                                mode = b'\r\n'.join(lines[i+1:-1]).decode('utf-8', errors='ignore').strip()
                                break

                del body

                if not file_contents:
                    raise ValueError('No files provided')

                temp_paths = []
                file_paths = []
                orig_names = []
                for fname, content in file_contents:
                    ext = Path(fname).suffix or '.log'
                    tp  = Path(tempfile.gettempdir()) / ('fdv_upload_' + uuid.uuid4().hex + ext)
                    tp.write_bytes(content)
                    temp_paths.append(str(tp))
                    file_paths.append(str(tp))
                    orig_names.append(Path(fname).name if fname else None)
                del file_contents

                job_id = 'job_' + uuid.uuid4().hex[:8]
                with parse_jobs_lock:
                    parse_jobs[job_id] = {'state': 'pending', 'result': None, 'error': None}

                threading.Thread(
                    target=_run_parse_multi_job,
                    args=(job_id, file_paths, regex_filter if regex_filter else None,
                          mode == 'include', temp_paths, orig_names),
                    daemon=True
                ).start()

                _send_json(self, 202, {
                    'success': False, 'state': 'pending', 'job_id': job_id,
                    'file_count': len(file_paths)
                })

            except Exception as e:
                _send_json(self, 400, {'success': False, 'error': str(e)})

        elif self.path == '/analyze':
            # Single-turn AI analysis — no session history
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length).decode('utf-8'))
                prompt = body.get('prompt', '').strip()
                model  = body.get('model', '').strip() or None
                if not prompt:
                    raise ValueError('Empty prompt')
                messages = [
                    {'role': 'system', 'content': _LLM_SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt}
                ]
                summary = _call_llm(messages, model=model)
                _send_json(self, 200, {'success': True, 'summary': summary})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/chat':
            # Multi-turn chat — maintains per-csv_id conversation history
            try:
                length  = int(self.headers.get('Content-Length', 0))
                body    = json.loads(self.rfile.read(length).decode('utf-8'))
                csv_id  = body.get('csv_id', 'default') or 'default'
                message = body.get('message', '').strip()
                context = body.get('context', '').strip()   # chart stats injected on first turn
                model   = body.get('model', '').strip() or None
                if not message:
                    raise ValueError('Empty message')

                with _chat_sessions_lock:
                    if csv_id not in _chat_sessions:
                        # Start new session with system prompt + optional context
                        sess = [{'role': 'system', 'content': _LLM_SYSTEM_PROMPT}]
                        if context:
                            sess.append({
                                'role': 'system',
                                'content': 'Current chart context:\n' + context
                            })
                        _chat_sessions[csv_id] = sess
                    elif context:
                        # Context re-injected — REPLACE the existing context message
                        # (search for it by prefix; if not found, append once)
                        sess = _chat_sessions[csv_id]
                        replaced = False
                        for i, m in enumerate(sess):
                            if m['role'] == 'system' and (
                                    m['content'].startswith('Current chart context:') or
                                    m['content'].startswith('Updated chart context:')):
                                sess[i] = {'role': 'system',
                                           'content': 'Updated chart context:\n' + context}
                                replaced = True
                                break
                        if not replaced:
                            sess.append({'role': 'system',
                                         'content': 'Updated chart context:\n' + context})

                    # Append user message
                    _chat_sessions[csv_id].append({'role': 'user', 'content': message})

                    # ── Trim history to avoid unbounded growth ──────────────
                    # Keep all system messages + last _CHAT_MAX_TURNS turn pairs.
                    sess = _chat_sessions[csv_id]
                    system_msgs = [m for m in sess if m['role'] == 'system']
                    conv_msgs   = [m for m in sess if m['role'] != 'system']
                    # Each turn = one user + one assistant message = 2 items
                    max_conv    = _CHAT_MAX_TURNS * 2
                    if len(conv_msgs) > max_conv:
                        conv_msgs = conv_msgs[-max_conv:]
                    _chat_sessions[csv_id] = system_msgs + conv_msgs

                    messages_snapshot = list(_chat_sessions[csv_id])

                # Call LLM outside the lock (slow network I/O)
                reply = _call_llm(messages_snapshot, model=model)

                with _chat_sessions_lock:
                    _chat_sessions[csv_id].append({'role': 'assistant', 'content': reply})
                    turn = sum(1 for m in _chat_sessions[csv_id] if m['role'] == 'user')

                _send_json(self, 200, {'success': True, 'reply': reply, 'turn': turn})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/chat_stream':
            # Streaming chat via Server-Sent Events — tokens arrive as they are generated.
            # Same session management as /chat but uses Ollama's stream=True native API.
            try:
                length  = int(self.headers.get('Content-Length', 0))
                body    = json.loads(self.rfile.read(length).decode('utf-8'))
                csv_id  = body.get('csv_id', 'default') or 'default'
                message = body.get('message', '').strip()
                context = body.get('context', '').strip()
                model   = body.get('model', '').strip() or _LLM_MODEL
                if not message:
                    raise ValueError('Empty message')

                # ── Build / update session history (same logic as /chat) ──
                with _chat_sessions_lock:
                    if csv_id not in _chat_sessions:
                        sess = [{'role': 'system', 'content': _LLM_SYSTEM_PROMPT}]
                        if context:
                            sess.append({'role': 'system',
                                         'content': 'Current chart context:\n' + context})
                        _chat_sessions[csv_id] = sess
                    elif context:
                        sess = _chat_sessions[csv_id]
                        replaced = False
                        for i, m in enumerate(sess):
                            if m['role'] == 'system' and (
                                    m['content'].startswith('Current chart context:') or
                                    m['content'].startswith('Updated chart context:')):
                                sess[i] = {'role': 'system',
                                           'content': 'Updated chart context:\n' + context}
                                replaced = True
                                break
                        if not replaced:
                            sess.append({'role': 'system',
                                         'content': 'Updated chart context:\n' + context})

                    _chat_sessions[csv_id].append({'role': 'user', 'content': message})

                    # Trim history
                    sess        = _chat_sessions[csv_id]
                    system_msgs = [m for m in sess if m['role'] == 'system']
                    conv_msgs   = [m for m in sess if m['role'] != 'system']
                    max_conv    = _CHAT_MAX_TURNS * 2
                    if len(conv_msgs) > max_conv:
                        conv_msgs = conv_msgs[-max_conv:]
                    _chat_sessions[csv_id] = system_msgs + conv_msgs
                    messages_snapshot = list(_chat_sessions[csv_id])

                # ── Send SSE headers ──
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()

                # ── Stream from Ollama native /api/chat ──
                payload = json.dumps({
                    'model':       model,
                    'messages':    messages_snapshot,
                    'stream':      True,
                    'temperature': 0.3
                }).encode('utf-8')
                req = urllib.request.Request(
                    _OLLAMA_BASE + '/api/chat',
                    data=payload,
                    headers={'Content-Type': 'application/json'}
                )
                full_reply = []
                with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as r:
                    for raw_line in r:
                        line = raw_line.decode('utf-8').strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        token = chunk.get('message', {}).get('content', '')
                        if token:
                            full_reply.append(token)
                            data_str = json.dumps({'token': token})
                            self.wfile.write(('data: ' + data_str + '\n\n').encode('utf-8'))
                            self.wfile.flush()
                        if chunk.get('done'):
                            break

                # Save assistant reply to history
                reply = ''.join(full_reply)
                with _chat_sessions_lock:
                    _chat_sessions[csv_id].append({'role': 'assistant', 'content': reply})

                self.wfile.write(b'data: {"done":true}\n\n')
                self.wfile.flush()

            except Exception as e:
                try:
                    err_data = json.dumps({'error': str(e)})
                    self.wfile.write(('data: ' + err_data + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                except Exception:
                    pass

        elif self.path == '/pull':
            # Pull a model from Ollama registry — blocking until complete
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length).decode('utf-8'))
                model  = body.get('model', '').strip()
                if not model:
                    raise ValueError('No model name provided')
                # Use Ollama's /api/pull (stream=false for simplicity)
                payload = json.dumps({'name': model, 'stream': False}).encode('utf-8')
                req = urllib.request.Request(
                    _OLLAMA_BASE + '/api/pull',
                    data=payload,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp_body = r.read().decode('utf-8')
                # Ollama returns multiple JSON lines for stream=false; last line has status
                last = [l for l in resp_body.strip().splitlines() if l.strip()][-1]
                result = json.loads(last)
                if result.get('status') == 'success':
                    _send_json(self, 200, {'success': True, 'model': model})
                else:
                    _send_json(self, 200, {'success': False,
                                           'error': result.get('status', 'unknown status')})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/chat_reset':
            # Delete the session history for a csv_id
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length).decode('utf-8'))
                csv_id = body.get('csv_id', 'default') or 'default'
                with _chat_sessions_lock:
                    _chat_sessions.pop(csv_id, None)
                _send_json(self, 200, {'success': True})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/store/save':
            # Save a recipe or session JSON file to disk
            # Body: { dir, name, type: 'recipe'|'session', data: {...} }
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length).decode('utf-8'))
                d      = (body.get('dir', '') or '').strip()
                name   = (body.get('name', '') or '').strip()
                ftype  = (body.get('type', '') or '').strip()   # 'recipe' or 'session'
                data   = body.get('data')
                if not d or not name or ftype not in ('recipe', 'session') or data is None:
                    raise ValueError('dir, name, type (recipe|session), and data are required')
                if not os.path.isdir(d):
                    raise ValueError('Directory not found: ' + d)
                ext   = '.fdv_recipe' if ftype == 'recipe' else '.fdv_session'
                fname = os.path.basename(name) + ext
                fpath = os.path.join(d, fname)
                with open(fpath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                _send_json(self, 200, {'success': True, 'file': fname,
                                       'size': os.path.getsize(fpath)})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/store/save_session':
            # Save a full session file server-side using rows already in parsed_cache.
            # Payload option A (normal parse): { dir, name, csv_id, snap, fname }
            # Payload option B (re-save of loaded session): { dir, name, session_csv_id, snap, fname }
            #   where session_csv_id was returned by /store/register_session at load time.
            # The server writes { headers, rows, fname, snap } to disk — no JS serialisation limit.
            try:
                length  = int(self.headers.get('Content-Length', 0))
                body    = json.loads(self.rfile.read(length).decode('utf-8'))
                d       = (body.get('dir', '') or '').strip()
                name    = (body.get('name', '') or '').strip()
                csv_id  = (body.get('csv_id', '') or '').strip()
                snap    = body.get('snap', {})
                fname   = (body.get('fname', '') or name).strip()
                if not d or not name or not csv_id:
                    raise ValueError('dir, name and csv_id are required')
                if not os.path.isdir(d):
                    raise ValueError('Directory not found: ' + d)
                cached = parsed_cache.get(csv_id)
                if cached is None:
                    raise ValueError('csv_id not found in cache — re-parse the file first')
                entry = {
                    'headers': cached['headers'],
                    'rows':    cached['rows'],
                    'fname':   fname,
                    'snap':    snap,
                }
                fpath = os.path.join(d, os.path.basename(name) + '.fdv_session')
                with open(fpath, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False)
                _send_json(self, 200, {'success': True, 'file': os.path.basename(fpath),
                                       'size': os.path.getsize(fpath)})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/store/register_session':
            # Load a session file's rows+headers into parsed_cache and return a csv_id.
            # Called by JS when a session is loaded so that re-saving works via /store/save_session.
            # Body: { dir, file }
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length).decode('utf-8'))
                d      = (body.get('dir', '') or '').strip()
                fname  = os.path.basename((body.get('file', '') or '').strip())
                if not d or not fname:
                    raise ValueError('dir and file are required')
                fpath = os.path.join(d, fname)
                if not os.path.isfile(fpath):
                    raise ValueError('File not found: ' + fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                headers = entry.get('headers')
                rows    = entry.get('rows')
                if not headers or rows is None:
                    raise ValueError('Session file missing headers or rows')
                csv_id = 'csv_' + uuid.uuid4().hex[:8]
                parsed_cache[csv_id] = {'headers': headers, 'rows': rows}
                _send_json(self, 200, {'success': True, 'csv_id': csv_id,
                                       'total_rows': len(rows), 'headers': headers})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        elif self.path == '/store/delete':
            # Delete a recipe or session file from disk
            # Body: { dir, file }
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length).decode('utf-8'))
                d      = (body.get('dir', '') or '').strip()
                fname  = os.path.basename((body.get('file', '') or '').strip())
                if not d or not fname:
                    raise ValueError('dir and file are required')
                if not (fname.endswith('.fdv_recipe') or fname.endswith('.fdv_session')):
                    raise ValueError('Invalid file type')
                fpath = os.path.join(d, fname)
                if not os.path.isfile(fpath):
                    raise ValueError('File not found: ' + fname)
                os.remove(fpath)
                _send_json(self, 200, {'success': True})
            except Exception as e:
                _send_json(self, 200, {'success': False, 'error': str(e)})

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """Suppress request logging"""
        pass


def main():
    """Start the web server.
    Usage: fdv_chart.py [PORT] [STORE_DIR]
      PORT       — TCP port to listen on (default 5058)
      STORE_DIR  — default store directory pushed to clients via /store/default_dir
    Examples:
      fdv_chart.py                       → port 5058, store D:\\FDV\\recipes
      fdv_chart.py 5059                  → port 5059 (dev), same store
      fdv_chart.py 5059 D:\\FDV\\dev_store → port 5059, different store
    """
    global _SERVER_PORT, _SERVER_STORE_DIR

    # Parse optional positional args: [port] [store_dir]
    args = sys.argv[1:]
    for a in args:
        try:
            _SERVER_PORT = int(a)
        except ValueError:
            _SERVER_STORE_DIR = a.strip()

    log_file = r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\server.log'
    with open(log_file, 'w') as f:
        f.write("Starting FDV Chart Parser...\n")
        f.flush()

    print("Starting FDV Chart Parser...", file=sys.stderr, flush=True)
    print("Port      : " + str(_SERVER_PORT), file=sys.stderr, flush=True)
    print("Store dir : " + (_SERVER_STORE_DIR or '(none)'), file=sys.stderr, flush=True)
    try:
        server = ThreadedHTTPServer(('0.0.0.0', _SERVER_PORT), RequestHandler)
        server.socket.settimeout(None)   # no accept() timeout; threads handle per-request I/O
        with open(log_file, 'a') as f:
            f.write("FDV Chart Parser is running at http://0.0.0.0:{} (all interfaces)\n".format(_SERVER_PORT))
            f.flush()
        print("FDV Chart Parser is running at http://0.0.0.0:{} (all interfaces)".format(_SERVER_PORT), file=sys.stderr, flush=True)
        print("Press Ctrl+C to stop", file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...", file=sys.stderr, flush=True)
            server.shutdown()
    except Exception as e:
        print("ERROR: " + str(e), file=sys.stderr, flush=True)
        with open(log_file, 'a') as f:
            f.write("ERROR: " + str(e) + "\n")
            f.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)


if __name__ == '__main__':
    main()
