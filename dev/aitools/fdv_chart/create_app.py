#!/usr/bin/env python3
"""Create complete fdv_chart.py"""

code = '''#!/usr/bin/env python3
import os, re, csv, json, uuid, io, sys, tempfile
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

LOG_FILE = str(Path(tempfile.gettempdir()) / 'fdv_chart_server.log')
parsed_cache = {}

def parse_log_file(file_path, regex_pattern=None, include_mode=True):
    headers = ['Line#', 'DUT', 'Test Name', 'Status', 'Value', 'VCC', 'VCCQ', 'TEMP', 'WL', 'BLK', 'Notes']
    rows = []
    try:
        compiled_regex = None
        if regex_pattern:
            try:
                compiled_regex = re.compile(regex_pattern)
            except re.error as e:
                raise Exception("Invalid regex: " + str(e))
        
        max_lines = 500000
        line_count = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                if line_count > max_lines:
                    rows.append(['...', '...', 'File truncated at 500k lines', '...', '...', '...', '...', '...', '...', '...', '...'])
                    break
                
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                
                if compiled_regex:
                    try:
                        matches = compiled_regex.search(line_stripped)
                        if include_mode and not matches:
                            continue
                        if not include_mode and matches:
                            continue
                    except Exception:
                        continue
                
                parts = [p.strip() for p in line_stripped.split('|')]
                if len(parts) < 2 or all(not p for p in parts):
                    parts = line_stripped.split()
                
                while len(parts) < len(headers):
                    parts.append('')
                
                row_data = parts[:len(headers)]
                row_data[0] = str(line_num)
                rows.append(row_data)
    except Exception as e:
        raise Exception("Error parsing: " + str(e))
    return headers, rows

def get_html():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FDV Chart Parser</title>
</head>
<body>
<h1>FDV Chart Parser</h1>
<p>Parse FDV log files with regex filtering</p>

<h2>Step 1: Select File</h2>
<input type="file" id="file-input" accept=".txt,.log,.csv,.dat">

<h2>Step 2: Regex Filter (Optional)</h2>
<p>Use ^ for line start, $ for line end, .* for any characters</p>
<textarea id="regex-input" placeholder="Example: ^FDV OUT" style="width:100%;height:50px"></textarea>

<h2>Step 3: Filter Mode</h2>
<label><input type="radio" name="filter-mode" value="include" checked> Include matching lines</label>
<label style="margin-left:20px"><input type="radio" name="filter-mode" value="exclude"> Exclude matching lines</label>

<h2>Step 4: Action</h2>
<button onclick="parseFile()" style="padding:10px 20px;cursor:pointer">Parse File</button>
<button onclick="reset()" style="padding:10px 20px;cursor:pointer">Clear</button>

<div id="message" style="margin:15px 0;padding:10px;border:1px solid #ccc"></div>
<div id="result-info" style="margin:10px 0;display:none"></div>

<div id="table-container" style="display:none;margin:20px 0">
<table id="result-table" border="1" cellpadding="5" cellspacing="0">
</table>
</div>

<button id="download-btn" onclick="downloadCSV()" style="padding:10px 20px;cursor:pointer;display:none">Download CSV</button>

<script>
let currentCsvId=null;

function showMessage(text){
    const el=document.getElementById('message');
    el.textContent=text;
}

async function parseFile(){
    const file=document.getElementById('file-input').files[0];
    if(!file){
        showMessage('Please select a file');
        return;
    }
    
    showMessage('Parsing...');
    const form=new FormData();
    form.append('file',file);
    form.append('regex',document.getElementById('regex-input').value.trim());
    form.append('mode',document.querySelector('input[name="filter-mode"]:checked').value);
    
    try{
        const resp=await fetch('/parse',{method:'POST',body:form});
        const result=await resp.json();
        
        if(!result.success){
            showMessage('Error: '+result.error);
            return;
        }
        
        currentCsvId=result.csv_id;
        showMessage('Parsed '+result.total_rows+' rows (showing first '+result.rows.length+')');
        document.getElementById('result-info').innerHTML='Total rows: '+result.total_rows+' | Columns: '+result.headers.length;
        document.getElementById('result-info').style.display='block';
        
        renderTable(result.headers,result.rows);
        document.getElementById('download-btn').style.display='block';
        
    }catch(e){
        showMessage('Error: '+e);
    }
}

function renderTable(headers,rows){
    const table=document.getElementById('result-table');
    let html='<thead><tr>';
    headers.forEach(h=>html+='<th>'+h+'</th>');
    html+='</tr></thead><tbody>';
    
    rows.forEach(row=>{
        html+='<tr>';
        row.forEach(cell=>html+='<td>'+(cell||'')+'</td>');
        html+='</tr>';
    });
    
    html+='</tbody>';
    table.innerHTML=html;
    document.getElementById('table-container').style.display='block';
}

function downloadCSV(){
    if(!currentCsvId){
        showMessage('No data to download');
        return;
    }
    window.location.href='/download/'+currentCsvId;
}

function reset(){
    document.getElementById('file-input').value='';
    document.getElementById('regex-input').value='';
    document.getElementById('message').textContent='';
    document.getElementById('result-info').style.display='none';
    document.getElementById('table-container').style.display='none';
    document.getElementById('download-btn').style.display='none';
    currentCsvId=null;
}
</script>
</body>
</html>"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body = get_html().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html;charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith('/download/'):
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
            self.send_header('Content-Type', 'text/csv;charset=utf-8')
            self.send_header('Content-Disposition', 'attachment;filename=fdv_' + csv_id + '.csv')
            self.send_header('Content-Length', len(csv_content))
            self.end_headers()
            self.wfile.write(csv_content)
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path != '/parse':
            self.send_error(404)
            return
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            if content_len > 500 * 1024 * 1024:
                raise ValueError('File too large')
            body = self.rfile.read(content_len)
            boundary = None
            for name, value in self.headers.items():
                if 'Content-Type' in name:
                    parts = value.split('boundary=')
                    if len(parts) > 1:
                        boundary = parts[1].strip().strip('"')
                        break
            if not boundary:
                raise ValueError('Missing boundary')
            boundary_bytes = ('--' + boundary).encode()
            parts_list = body.split(boundary_bytes)
            file_content = None
            regex_filter = ''
            mode = 'include'
            for part in parts_list:
                if b'name="file"' in part and b'filename=' in part:
                    lines = part.split(b'\\r\\n')
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue  # skip leading empty element before first header
                        if line == b'':
                            file_content = b'\\r\\n'.join(lines[i+1:-1])
                            break
                elif b'name="regex"' in part:
                    lines = part.split(b'\\r\\n')
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue
                        if line == b'':
                            regex_filter = b'\\r\\n'.join(lines[i+1:-1]).decode('utf-8', errors='ignore').strip()
                            break
                elif b'name="mode"' in part:
                    lines = part.split(b'\\r\\n')
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue
                        if line == b'':
                            mode = b'\\r\\n'.join(lines[i+1:-1]).decode('utf-8', errors='ignore').strip()
                            break
            if not file_content:
                raise ValueError('No file')
            temp_path = Path(tempfile.gettempdir()) / ('fdv_' + uuid.uuid4().hex + '.log')
            temp_path.write_bytes(file_content)
            try:
                headers, rows = parse_log_file(str(temp_path), regex_filter if regex_filter else None, include_mode=(mode == 'include'))
            finally:
                temp_path.unlink(missing_ok=True)
            if not rows:
                raise ValueError('No matching rows')
            csv_id = 'csv_' + uuid.uuid4().hex[:8]
            parsed_cache[csv_id] = {'headers': headers, 'rows': rows}
            display_rows = rows[:1000] if len(rows) > 1000 else rows
            response = {'success': True, 'csv_id': csv_id, 'headers': headers, 'rows': display_rows, 'total_rows': len(rows)}
            body_out = json.dumps(response).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body_out))
            self.end_headers()
            self.wfile.write(body_out)
        except Exception as e:
            response = {'success': False, 'error': str(e)}
            body_out = json.dumps(response).encode('utf-8')
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body_out))
            self.end_headers()
            self.wfile.write(body_out)
    
    def log_message(self, format, *args):
        pass

def main():
    try:
        server = HTTPServer(('127.0.0.1', 5058), RequestHandler)
        print("FDV Chart Parser running at http://localhost:5058", flush=True)
        server.serve_forever()
    except Exception as e:
        print("Error: " + str(e), flush=True)

if __name__ == '__main__':
    main()
'''

with open(r'd:\\FDV\\git\\fdv_dashboard\\dev\\aitools\\fdv_chart\\fdv_chart.py', 'w') as f:
    f.write(code)

print("fdv_chart.py created successfully")
