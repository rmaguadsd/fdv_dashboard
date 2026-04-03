#!/usr/bin/env python3
print("START")

from http.server import HTTPServer, BaseHTTPRequestHandler
print("Imported HTTPServer")

def main():
    print("IN MAIN")
    try:
        server = HTTPServer(('127.0.0.1', 5058), BaseHTTPRequestHandler)
        print("Created server!")
    except Exception as e:
        print("Error: " + str(e))

if __name__ == '__main__':
    print("About to call main")
    main()
    print("After main")
