#!/usr/bin/env python3
"""
FDV Chart Parser
4 Features:
1. Select an input file
2. Provide a regex filter to include or exclude from parsing
3. Use guideline from guide_to_fdvlog.txt to parse and generate a table
4. Generate a table which is downloadable to a .csv file
"""

import os
import re
import csv
import json
import uuid
import io
import tempfile
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# In-memory storage
parsed_data = {}


def parse_fdv_log(file_path, regex_filter=None, include_mode=True):
    """
    Parse FDV log file according to guide_to_fdvlog.txt format.
    
    Args:
        file_path: Path to log file
        regex_filter: Optional regex pattern to filter lines
        include_mode: True to include matching lines, False to exclude
    
    Returns:
        tuple: (headers, rows)
    """
    headers = [
        'DUT', 'Test Name', 'Test Conditions', 'Pagetype', 'WL', 'BLK',
        'RBER', 'Value', 'VCC', 'VCCQ', 'TEMP', 'Status', 'Notes'
    ]
    
    rows = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Apply regex filter if provided
                if regex_filter:
                    try:
                        # Use re.search with MULTILINE flag to respect ^ and $ anchors
                        matches = re.search(regex_filter, line, re.MULTILINE)
                        if include_mode and not matches:
                            continue
                        if not include_mode and matches:
                            continue
                    except re.error as e:
                        continue
                
                # Parse FDV log line
                # Format: DUT | Test Name | Conditions | Pagetype | WL | BLK | RBER/Value | VCC | VCCQ | TEMP | Status | Notes
                parts = [p.strip() for p in line.split('|')]
                
                # If not pipe-separated, try whitespace-separated
                if len(parts) < 2:
                    parts = line.split()
                
                # Pad with empty strings to match header count
                while len(parts) < len(headers):
                    parts.append('')
                
                # Truncate if too many columns
                parts = parts[:len(headers)]
                
                rows.append(parts)
    
    except Exception as e:
        raise Exception(f"Error parsing file: {str(e)}")
    
    return headers, rows


def make_response(status, content_type, body):
    """Create HTTP response."""
    if isinstance(body, dict):
        body = json.dumps(body)
    if isinstance(body, str):
        body = body.encode('utf-8')
    
    response = f"HTTP/1.1 {status}\r\n"
    response += f"Content-Type: {content_type}\r\n"
    response += f"Content-Length: {len(body)}\r\n"
    response += "Access-Control-Allow-Origin: *\r\n"
    response += "\r\n"
    
    return response.encode('utf-8') + body


def get_html():
    """Return HTML page."""
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>FDV Chart Parser</title>
</head>
<body>
    <h1>FDV Chart Parser</h1>
    
    <h2>Step 1: Select Input File</h2>
    <input type="file" id="file-input" accept=".txt,.log,.csv">
    
    <h2>Step 2: Regex Filter (Optional)</h2>
    <textarea id="regex-input" placeholder="e.g., PASS or ERROR or ^FDV" style="width: 100%; height: 60px;"></textarea>
    
    <h2>Step 3: Filter Mode</h2>
    <label><input type="radio" name="filter-mode" value="include" checked> Include matching lines</label><br>
    <label><input type="radio" name="filter-mode" value="exclude"> Exclude matching lines</label>
    
    <h2>Step 4: Parse</h2>
    <button onclick="parseFile()">Parse File</button>
    <button onclick="reset()">Reset</button>
    
    <div id="message"></div>
    
    <div id="result-section" style="display: none;">
        <h2>Results</h2>
        <p>Total Rows: <span id="row-count">0</span></p>
        <p>Total Columns: <span id="col-count">0</span></p>
        
        <button onclick="downloadCSV()">Download as CSV</button>
        <button onclick="reset()">Parse Another File</button>
        
        <h2>Parsed Data</h2>
        <div id="table-container" style="overflow: auto; max-height: 600px; border: 1px solid #ccc; margin: 10px 0;">
            <p>No data</p>
        </div>
    </div>

    <script>
        let currentData = { headers: [], rows: [], csv_id: null };
        
        function msg(text, type) {
            const el = document.getElementById('message');
            el.textContent = text;
            el.style.display = 'block';
            if (type === 'error') el.style.color = 'red';
            else if (type === 'success') el.style.color = 'green';
            else el.style.color = 'blue';
        }
        
        async function parseFile() {
            const file = document.getElementById('file-input').files[0];
            if (!file) {
                msg('Please select a file', 'error');
                return;
            }
            
            const regex = document.getElementById('regex-input').value.trim();
            const mode = document.querySelector('input[name="filter-mode"]:checked').value;
            
            const form = new FormData();
            form.append('file', file);
            form.append('regex', regex);
            form.append('mode', mode);
            
            try {
                const resp = await fetch('/parse', { method: 'POST', body: form });
                const data = await resp.json();
                
                if (!data.success) {
                    msg(data.error, 'error');
                    return;
                }
                
                currentData = data;
                document.getElementById('row-count').textContent = data.total_rows || data.rows.length;
                document.getElementById('col-count').textContent = data.headers.length;
                
                // Show info if data is truncated for display
                if (data.total_rows && data.displayed_rows < data.total_rows) {
                    msg(`Parsed ${data.total_rows} rows (showing first ${data.displayed_rows}). Full data available in CSV download.`, 'success');
                } else {
                    msg(`Parsed ${data.rows.length} rows`, 'success');
                }
                
                renderTable();
                document.getElementById('result-section').style.display = 'block';
                msg(`Parsed ${data.rows.length} rows`, 'success');
            } catch (e) {
                msg(`Error: ${e}`, 'error');
            }
        }
        
        function renderTable() {
            if (!currentData.rows.length) {
                document.getElementById('table-container').innerHTML = '<p>No data</p>';
                return;
            }
            
            let html = '<table border="1" style="width: 100%; border-collapse: collapse;"><thead><tr>';
            
            currentData.headers.forEach(h => {
                html += `<th>${h}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            currentData.rows.forEach(row => {
                html += '<tr>';
                row.forEach((cell, i) => {
                    html += `<td>${cell || ''}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            document.getElementById('table-container').innerHTML = html;
        }
        
        async function downloadCSV() {
            if (!currentData.csv_id) {
                msg('No data to download', 'error');
                return;
            }
            
            window.location.href = `/download/${currentData.csv_id}`;
        }
        
        function reset() {
            document.getElementById('file-input').value = '';
            document.getElementById('regex-input').value = '';
            document.getElementById('message').style.display = 'none';
            document.getElementById('result-section').style.display = 'none';
            currentData = { headers: [], rows: [], csv_id: null };
        }
    </script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    """HTTP request handler."""
    
    def do_GET(self):
        """Handle GET requests."""
        url = urlparse(self.path)
        path = url.path
        
        # Main page
        if path == '/':
            body = get_html().encode('utf-8')
            response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(body)}\r\n\r\n"
            self.wfile.write(response.encode('utf-8') + body)
        
        # Download CSV
        elif path.startswith('/download/'):
            csv_id = path.split('/')[-1]
            if csv_id in parsed_data:
                data = parsed_data[csv_id]
                
                # Check if CSV is stored on disk
                if 'csv_path' in data and Path(data['csv_path']).exists():
                    # Read from disk
                    with open(data['csv_path'], 'rb') as f:
                        csv_content = f.read()
                else:
                    # Create CSV from memory (for backward compatibility)
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(data['headers'])
                    if 'rows' in data:
                        writer.writerows(data['rows'])
                    csv_content = output.getvalue().encode('utf-8')
                
                body = csv_content
                response = f"HTTP/1.1 200 OK\r\n"
                response += f"Content-Type: text/csv\r\n"
                response += f"Content-Disposition: attachment; filename=fdv_parsed_{csv_id}.csv\r\n"
                response += f"Content-Length: {len(body)}\r\n"
                response += "\r\n"
                self.wfile.write(response.encode('utf-8') + body)
            else:
                response = make_response('404 Not Found', 'application/json', json.dumps({'error': 'Not found'}))
                self.wfile.write(response)
        
        else:
            response = make_response('404 Not Found', 'text/plain', 'Not found')
            self.wfile.write(response)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/parse':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len)
                
                # Parse multipart form data
                boundary = None
                for name, value in self.headers.items():
                    if 'Content-Type' in name:
                        parts = value.split('boundary=')
                        if len(parts) > 1:
                            boundary = parts[1].strip()
                
                if not boundary:
                    raise ValueError('No boundary found')
                
                # Extract file, regex, and mode
                boundary_bytes = f'--{boundary}'.encode()
                parts = body.split(boundary_bytes)
                
                file_content = None
                regex_filter = ''
                include_mode = True
                
                for part in parts:
                    if b'name="file"' in part and b'filename=' in part:
                        # File content
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if line == b'':
                                file_content = b'\r\n'.join(lines[i+1:-1])
                                break
                    elif b'name="regex"' in part:
                        # Regex filter
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if line == b'':
                                regex_filter = b'\r\n'.join(lines[i+1:-1]).decode('utf-8').strip()
                                break
                    elif b'name="mode"' in part:
                        # Filter mode
                        lines = part.split(b'\r\n')
                        for i, line in enumerate(lines):
                            if line == b'':
                                mode_val = b'\r\n'.join(lines[i+1:-1]).decode('utf-8').strip()
                                include_mode = mode_val == 'include'
                                break
                
                if not file_content:
                    raise ValueError('No file found')
                
                # Save temp file (use Windows-compatible path)
                temp_dir = Path(tempfile.gettempdir())
                temp_path = temp_dir / f'fdv_upload_{uuid.uuid4().hex}.log'
                with open(temp_path, 'wb') as f:
                    f.write(file_content)
                
                # Parse file
                try:
                    headers, rows = parse_fdv_log(str(temp_path), regex_filter if regex_filter else None, include_mode)
                finally:
                    temp_path.unlink(missing_ok=True)
                
                if not rows:
                    raise ValueError('No rows matched the filter criteria')
                
                # Store data with CSV on disk
                csv_id = f"parsed_{uuid.uuid4().hex}"
                
                # Write CSV to disk to avoid storing 100K+ rows in memory
                csv_path = Path(tempfile.gettempdir()) / f'fdv_{csv_id}.csv'
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                    writer.writerows(rows)
                
                # Store only metadata and path
                parsed_data[csv_id] = {
                    'headers': headers,
                    'csv_path': str(csv_path),
                    'total_rows': len(rows)
                }
                
                # Only return first 1000 rows for display
                display_rows = rows[:1000] if len(rows) > 1000 else rows
                
                response = make_response('200 OK', 'application/json',
                    json.dumps({
                        'success': True,
                        'csv_id': csv_id,
                        'headers': headers,
                        'rows': display_rows,
                        'total_rows': len(rows),
                        'displayed_rows': len(display_rows)
                    })
                )
                self.wfile.write(response)
            
            except Exception as e:
                response = make_response('400 Bad Request', 'application/json',
                    json.dumps({'success': False, 'error': str(e)})
                )
                self.wfile.write(response)
        else:
            response = make_response('404 Not Found', 'text/plain', 'Not found')
            self.wfile.write(response)
    
    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def run_server(port=5058):
    """Run HTTP server."""
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"FDV Chart Parser running on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    run_server(5058)
