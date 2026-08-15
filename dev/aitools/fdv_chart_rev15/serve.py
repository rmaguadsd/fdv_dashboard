"""
No-cache static file server for fdv_chart_rev4.

Use this instead of `python -m http.server` so that browsers always pull the
latest fdv_chart.html (and any sibling assets) instead of serving a stale
cached copy.

Usage (PowerShell):
    cd D:\\FDV\\git\\fdv_dashboard\\dev\\aitools\\fdv_chart_rev4
    py -3 serve.py 5058    # or: python serve.py 5058 (if 'python' is Python 3)

If no port is given it defaults to 5058. Requires Python 3.
"""
import os
import sys
import http.server
import socketserver


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that disables HTTP caching and routes / to fdv_chart.html."""

    def do_GET(self):
        # Auto-redirect bare "/" (and "/index.html") to the actual app page.
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/fdv_chart.html")
            self.end_headers()
            return
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        http.server.SimpleHTTPRequestHandler.end_headers(self)


def main():
    port = 5058
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port: {!r} (using default {})".format(sys.argv[1], port))

    # Always serve from the directory this script lives in.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # allow_reuse_address avoids "address already in use" after a quick restart
    socketserver.TCPServer.allow_reuse_address = True

    httpd = socketserver.TCPServer(("", port), NoCacheHandler)
    url = "http://localhost:{}/fdv_chart.html".format(port)
    print("Serving (no-cache) from {}".format(os.getcwd()))
    print("Open: {}".format(url))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
