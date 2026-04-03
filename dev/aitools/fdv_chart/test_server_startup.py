#!/usr/bin/env python3
import sys
print("Test START")
sys.stderr.write("STDERR Test\n")

with open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\test_startup.log', 'w') as f:
    f.write("Startup test\n")
    
print("Test END")

# Try to start server
from http.server import HTTPServer, BaseHTTPRequestHandler

class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')
    
    def log_message(self, format, *args):
        pass

try:
    with open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\test_startup.log', 'a') as f:
        f.write("Creating server...\n")
    
    server = HTTPServer(('127.0.0.1', 5058), TestHandler)
    
    with open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\test_startup.log', 'a') as f:
        f.write("Server created, starting...\n")
    
    server.serve_forever()
    
except Exception as e:
    with open(r'd:\FDV\git\fdv_dashboard\dev\aitools\fdv_chart\test_startup.log', 'a') as f:
        f.write("ERROR: " + str(e) + "\n")
        import traceback
        f.write(traceback.format_exc())
