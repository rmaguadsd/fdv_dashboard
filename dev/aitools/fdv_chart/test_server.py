#!/usr/bin/env python3
print("Starting test...", flush=True)
from http.server import HTTPServer, BaseHTTPRequestHandler
print("Imported HTTPServer", flush=True)

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Hello')

print("Creating server...", flush=True)
server = HTTPServer(('127.0.0.1', 5058), SimpleHandler)
print("Server created on http://localhost:5058", flush=True)
print("Starting serve...", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Stopping...", flush=True)
    server.shutdown()
