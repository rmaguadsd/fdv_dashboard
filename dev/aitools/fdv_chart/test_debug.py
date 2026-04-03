#!/usr/bin/env python3
print("START")

import os
print("imported os")

import re
print("imported re")

import csv
print("imported csv")

import json
print("imported json")

import uuid
print("imported uuid")

import io
print("imported io")

import tempfile
print("imported tempfile")

from pathlib import Path
print("imported Path")

from http.server import HTTPServer, BaseHTTPRequestHandler
print("imported HTTPServer, BaseHTTPRequestHandler")

from urllib.parse import urlparse
print("imported urlparse")

print("ALL IMPORTS OK - about to start server...")

if __name__ == '__main__':
    print("In main()")
    try:
        server = HTTPServer(('127.0.0.1', 5058), lambda: None)
        print("HTTPServer created!")
    except Exception as e:
        print("Error creating server: " + str(e))
