#!/usr/bin/env python3
"""Quick test to verify session loads without errors."""

import time
import requests

def test_session():
    """Test loading session and check for errors."""
    
    url = "http://localhost:5059/fdv_chart_rev7.html?session=n59a_a2_pr36_norem_25c_program_suspend_random_delay"
    
    print(f"Loading: {url}")
    print("Waiting 2 seconds for server response...")
    time.sleep(2)
    
    try:
        response = requests.get(url, timeout=5)
        print(f"✓ HTTP {response.status_code}")
        
        # Check for error patterns
        errors = [
            "Cannot read properties of null",
            "TypeError:",
            "ReferenceError:",
            "Uncaught"
        ]
        
        content_lower = response.text.lower()
        found = [e for e in errors if e.lower() in content_lower]
        
        if found:
            print(f"❌ Found errors: {found}")
            # Print a snippet around the error
            for error in found:
                idx = response.text.lower().find(error.lower())
                if idx >= 0:
                    snippet = response.text[max(0, idx-100):idx+200]
                    print(f"\nSnippet: ...{snippet}...")
            return False
        else:
            print("✓ No error signatures found")
            print("\n✓✓✓ SESSION LOADING SUCCESSFUL! ✓✓✓")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_session()
    sys.exit(0 if success else 1)
