#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch both FDV apps:
- FDV Report v2 on port 5057            cmd = [PY, script]
            print("[{0}] starting: {1}".format(name, cmd)) FDV POLL on port 5055

Features:
- Uses workspace .venv if found; falls back to current interpreter.
- Skips a service if its port is already in use.
- Streams child output with prefixes and handles Ctrl+C to stop both.
"""
import os
import sys
import time
import socket
import threading
import subprocess as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # workspace root (…/dev)
VENV_PY = os.path.join(ROOT, ".venv", "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable
DEFAULT_TMP = "D:/fdv_tmp" if os.name == 'nt' else "/tmp/fdv_tmp"
try:
    os.makedirs(DEFAULT_TMP)
except OSError:
    pass

APPS = [
    {
        "name": "REPORT",
        # Use minimal runner variant with updated limit/none logic
        "path": os.path.join(os.path.dirname(__file__), "fdv_report2_runner_min.py"),
        "port": 5057,
        "env": {
            # Single process (no reloader) is controlled by the app's use_reloader=False
            # Direct temps to D:\fdv_tmp by default
            "FDV_REPORT2_TMPDIR": DEFAULT_TMP,
            "TMP": DEFAULT_TMP,
            "TEMP": DEFAULT_TMP,
            # Optional: FDV_REPORT2_HOST/PORT could be respected if implemented in the app
        },
    },
    {
        "name": "POLL",
        "path": os.path.join(os.path.dirname(__file__), "fdv_poll_webapp.py"),
        "port": 5055,
        "env": {
            "FDV_POLL_DEBUG": "1",
            "FDV_POLL_HOST": "0.0.0.0",
            "FDV_POLL_PORT": "5055",
            # Direct temps to D:\fdv_tmp by default
            "FDV_POLL_TMPDIR": DEFAULT_TMP,
            "TMP": DEFAULT_TMP,
            "TEMP": DEFAULT_TMP,
        },
    },
]


def port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def stream_pipe(prefix, pipe):
    try:
        for line in iter(pipe.readline, b""):
            if not line:
                break
            try:
                txt = line.decode(errors="replace").rstrip("\r\n")
            except Exception:
                txt = str(line).rstrip()
            print("[{0}] {1}".format(prefix, txt))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def main():
    print("Using Python: {0}".format(PY))
    procs = []
    threads = []
    try:
        for app in APPS:
            name = app["name"]
            script = app["path"]
            port = app["port"]
            if not os.path.exists(script):
                print("[{0}] skip: not found -> {1}".format(name, script))
                continue
            if port_in_use(port):
                print("[{0}] port {1} already in use; skipping launch.".format(name, port))
                continue
            env = os.environ.copy()
            env.setdefault("MPLBACKEND", "Agg")
            for k, v in app.get("env", {}).items():
                env[k] = v
            cmd = [PY, script]
            print("[{0}] starting: {1}".format(name, cmd))
            p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, env=env)
            procs.append(p)
            t = threading.Thread(target=stream_pipe, args=(name, p.stdout))
            t.daemon = True
            t.start()
            threads.append(t)
        if not procs:
            print("Nothing to launch. Exiting.")
            return 0
        print("Both launch attempts issued. Press Ctrl+C to stop.")
        while True:
            # If any process exits unexpectedly, break and report
            for p in list(procs):
                ret = p.poll()
                if ret is not None:
                    print("[LAUNCHER] process exited with code {0}".format(ret))
                    procs.remove(p)
            if not procs:
                break
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Ctrl+C received, stopping children…")
        return 0
    finally:
        # Terminate any remaining children
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        # Give them a moment, then force kill if needed
        deadline = time.time() + 5.0
        for p in procs:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
