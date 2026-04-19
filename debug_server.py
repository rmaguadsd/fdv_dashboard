#!/usr/bin/env python3
"""Minimal chat server for debugging"""
import http.server
import socketserver
import json
import os
import sys

sys.path.insert(0, 'd:\\FDV\\git\\fdv_dashboard\\dev\\aitools\\fdv_chart')

PORT = 5099
HTML_DIR = 'd:\\FDV\\git\\fdv_dashboard\\dev\\aitools\\fdv_chart'

class DebugHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/chat_debug.html':
            try:
                with open(os.path.join(HTML_DIR, 'chat_debug.html'), 'r') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f.read().encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/chat_stream':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                
                # Send a test response
                response = "Hello! You said: " + data.get('message', 'nothing')
                tokens = response.split()
                for token in tokens:
                    chunk = json.dumps({'message': {'content': token + ' '}})
                    self.wfile.write(('data: ' + chunk + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                
                # Send done signal
                done_chunk = json.dumps({'done': True})
                self.wfile.write(('data: ' + done_chunk + '\n\n').encode('utf-8'))
                self.wfile.flush()
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {format % args}")

if __name__ == '__main__':
    os.chdir(HTML_DIR)
    with socketserver.TCPServer(("", PORT), DebugHandler) as httpd:
        print(f"Debug server running at http://localhost:{PORT}")
        print(f"Visit http://localhost:{PORT}/chat_debug.html to test")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
