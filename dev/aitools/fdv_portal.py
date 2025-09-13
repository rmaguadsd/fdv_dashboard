#!/usr/bin/env python
"""
FDV Portal
- Tiny top-level Flask app that links to FDV Report v2 and FDV POLL webapps.
- Use env to configure downstream hosts/ports.

Env (with defaults):
    FDV_REPORT2_HOST=127.0.0.1
    FDV_REPORT2_PORT=5057
    FDV_POLL_HOST=127.0.0.1
    FDV_POLL_PORT=5055
  FDV_PORTAL_HOST=127.0.0.1
  FDV_PORTAL_PORT=5050
"""
from __future__ import annotations
import os
from flask import Flask, render_template, redirect, url_for
from pathlib import Path

_HERE = Path(__file__).parent

app = Flask(__name__, template_folder=str(_HERE / 'templates'))

@app.route('/')
def home():
    r2_host = os.environ.get('FDV_REPORT2_HOST', '127.0.0.1').strip() or '127.0.0.1'
    try:
        r2_port = int(os.environ.get('FDV_REPORT2_PORT', '5057'))
    except Exception:
        r2_port = 5057
    poll_host = os.environ.get('FDV_POLL_HOST', '127.0.0.1').strip() or '127.0.0.1'
    try:
        poll_port = int(os.environ.get('FDV_POLL_PORT', '5055'))
    except Exception:
        poll_port = 5055
    links = {
        'fdv_report2': f"http://{r2_host}:{r2_port}/",
        'fdv_poll': f"http://{poll_host}:{poll_port}/",
    }
    return render_template('fdv_portal.html', links=links)

if __name__ == '__main__':
    host = os.environ.get('FDV_PORTAL_HOST', '127.0.0.1').strip() or '127.0.0.1'
    try:
        port = int(os.environ.get('FDV_PORTAL_PORT', '5050'))
    except Exception:
        port = 5050
    app.run(host=host, port=port, debug=True, use_reloader=False)
