#!/usr/bin/env python3
"""Test server to verify chat fix is deployed"""
import http.server
import socketserver
import json
import os

PORT = 5099

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/test':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Server is running\n")
        else:
            super().do_GET()
    
    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {format % args}")

if __name__ == '__main__':
    os.chdir('d:\\FDV\\git\\fdv_dashboard\\dev\\aitools\\fdv_chart')
    with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
        print(f"Test server running at http://localhost:{PORT}")
        print(f"Serving HTML from: {os.getcwd()}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
