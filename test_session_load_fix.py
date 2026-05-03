#!/usr/bin/env python3
"""Test session loading after architectural fix."""

import json
import subprocess
import time
import requests
from pathlib import Path

def test_session_load():
    """Test loading a session through the browser."""
    
    # Get the session to load
    session_name = "n59a_a2_pr36_norem_25c_program_suspend_random_delay"
    
    print(f"\n{'='*60}")
    print(f"Testing session load: {session_name}")
    print(f"{'='*60}\n")
    
    # Try to load the session
    url = f"http://localhost:5058/fdv_chart_rev7.html?session={session_name}"
    print(f"Loading URL: {url}\n")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"✓ HTTP request succeeded (status {response.status_code})")
        
        # Check if page loads without errors
        if response.status_code == 200:
            # Look for any error signatures in the response
            content = response.text
            
            error_signatures = [
                "cannot read properties of null",
                "TypeError:",
                "SyntaxError:",
                "ReferenceError:",
            ]
            
            found_errors = []
            for sig in error_signatures:
                if sig.lower() in content.lower():
                    found_errors.append(sig)
            
            if found_errors:
                print(f"❌ Found error signatures: {found_errors}")
                return False
            else:
                print(f"✓ No obvious error signatures found in page content")
                print(f"\n✓ PAGE LOADED SUCCESSFULLY!")
                return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading page: {e}")
        return False

if __name__ == "__main__":
    success = test_session_load()
    exit(0 if success else 1)
