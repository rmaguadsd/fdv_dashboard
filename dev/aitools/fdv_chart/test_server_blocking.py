#!/usr/bin/env python3
import sys
print("START")

from http.server import HTTPServer, BaseHTTPRequestHandler

def main():
    print("IN MAIN", flush=True)
    try:
        server = HTTPServer(('127.0.0.1', 5058), BaseHTTPRequestHandler)
        print("Created server!", flush=True)
        print("About to serve_forever...", flush=True)
        sys.stdout.flush()
        server.serve_forever()
        print("After serve_forever", flush=True)
    except Exception as e:
        print("Error: " + str(e), flush=True)

if __name__ == '__main__':
    print("About to call main", flush=True)
    main()
    print("After main", flush=True)
