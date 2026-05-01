#!/usr/bin/env python3
"""
Test if the server is running and has the correct code.
"""
import urllib.request
import re
import sys

url = "http://localhost:5059"
print(f"[TEST] Fetching {url}...")

try:
    with urllib.request.urlopen(url, timeout=5) as response:
        html = response.read().decode('utf-8')
        print(f"✅ Server is responding. HTML size: {len(html)} bytes")
        
        # Test 1: Check if "Point Size" is selected by default
        if 'value="point" selected' in html:
            print("✅ TEST 1 PASS: Point Size has default selection")
        else:
            print("❌ TEST 1 FAIL: Point Size does NOT have default selection")
            print(f"   Font item select near: {html[html.find('font-item-select'):html.find('font-item-select')+500]}")
        
        # Test 2: Check if _selectedFontItem is initialized to 'point'
        if "var _selectedFontItem = 'point'" in html:
            print("✅ TEST 2 PASS: _selectedFontItem is initialized to 'point'")
        else:
            print("❌ TEST 2 FAIL: _selectedFontItem is NOT initialized to 'point'")
        
        # Test 3: Check if DOMContentLoaded calls _onFontItemSelect
        if "setTimeout(function() {\n        _onFontItemSelect();" in html or \
           "setTimeout(function(){_onFontItemSelect();" in html or \
           "_onFontItemSelect()" in html and "DOMContentLoaded" in html:
            print("✅ TEST 3 PASS: DOMContentLoaded calls _onFontItemSelect")
        else:
            print("❌ TEST 3 FAIL: DOMContentLoaded does NOT call _onFontItemSelect")
        
        # Test 4: Check if _applyFontSize has the early exit check
        if "if (!_selectedFontItem) {" in html and "_applyFontSize" in html:
            print("✅ TEST 4 PASS: _applyFontSize has early exit check")
        else:
            print("❌ TEST 4 FAIL: _applyFontSize does NOT have early exit check")
        
        # Test 5: Check if split charts update code exists
        if "_splitInsts.forEach" in html:
            print("✅ TEST 5 PASS: Split charts update logic exists")
        else:
            print("❌ TEST 5 FAIL: Split charts update logic missing")
        
        # Test 6: Check if console.log for INIT exists
        if "[INIT] Font controls initialized" in html:
            print("✅ TEST 6 PASS: Initialization console.log exists")
        else:
            print("❌ TEST 6 FAIL: Initialization console.log missing")
        
        print("\n" + "="*60)
        print("Server code verification complete!")
        print("="*60)
        
except urllib.error.URLError as e:
    print(f"❌ ERROR: Cannot connect to server at {url}")
    print(f"   Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
